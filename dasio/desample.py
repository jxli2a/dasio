"""Desample pipeline + file selector + CLI.

Produces a 1 Hz DASdata from a window of raw 100 Hz files. Matches the
behavior of DAS-utilities/Desample_DAS.py:

1. Identify target files + pad files (contiguous).
2. Read all with the appropriate raw reader.
3. Concat along time.
4. Bandpass over the full padded span.
5. Decimate by fsRatio.
6. Trim to target window via timestamp-to-index.
"""
import argparse
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import default_nthreads

from .dasdata import DASdata
from .dasfile import DASFile
from .readers.proc import write_data_proc
from .signal import bandpass2d, preprocess_unwrap
from .dasdb import DASdb
from .schema import RawWindow


def _dasdb_path_loadable(path: Optional[Path]) -> bool:
    """True iff `path` exists and is a non-empty file we can pass to
    `DASdb.from_file`.

    Why this exists: an interrupted write (power loss, kernel panic,
    SIGKILL) leaves a 0-byte parquet behind. `path.exists()` returns
    True for that case, but `pq.read_schema(path)` then raises
    `ArrowInvalid: Parquet file size is 0 bytes` — wedging every
    subsequent cron tick. Treat 0-byte files as 'missing' so the
    caller falls into the from-scratch rebuild branch and the
    pipeline self-heals on the next tick.

    Note: a partially-written parquet (>0 bytes but missing the
    footer) would still fail to load. Covering that case requires an
    actual `pq.read_schema` probe, which is more expensive on the
    hot path. We accept the 0-byte fix here as the high-leverage,
    low-cost win for the realistic failure mode we observed in
    production.
    """
    if path is None:
        return False
    try:
        return path.exists() and path.stat().st_size > 0
    except OSError:
        return False


# --- Output naming convention ---
#
# Lives here (with the writer) rather than in dasdb so the existence
# check and the actual write share one definition; iter_unprocessed
# below and desample_and_write_window both call this. dasdb stays
# free of any desample-output knowledge.

def _proc_out_path(out_dir: Path, system: str, begin: datetime,
                    date_subdir: bool) -> Path:
    """Resolve the on-disk path for a desampled window.

    Filename always begins with 'Proc' so a desampled file is
    unambiguously distinguishable from raw input regardless of system;
    re-running desample on Proc inputs (system='Proc') keeps the bare
    'Proc-' prefix instead of producing 'ProcProc-'. `date_subdir=True`
    nests by UTC begin date so `Processed/` does not accumulate
    millions of flat siblings. The directory is *not* auto-created
    here — writers must mkdir before save.
    """
    prefix = system if system.startswith('Proc') else f'Proc{system}'
    name = f"{prefix}-{begin.strftime('%Y-%m-%dT%H%M%SZ')}.h5"
    if date_subdir:
        return out_dir / begin.strftime('%Y%m%d') / name
    return out_dir / name


# --- Backlog selectors ---
#
# These walk a DASdb's segments and tell the desample pipeline which
# 60 s windows still need writing. They live here (not on DASdb) so
# the catalog stays focused on file-list operations and doesn't have
# to know about the desample-output naming convention.

def iter_unprocessed(
        db: DASdb, out_dir: Path, file_len: int,
        *,
        date_subdir: bool = False,
    ):
    """Yield every `(begin, end)` that still needs desampling, in order.

    Walks each segment independently; a window never straddles an
    acquisition gap. Anchors each window's `begin` at the first file
    whose `begin_time >= previous_end`, so non-uniform file durations
    inside a segment never produce overlapping windows. Stops yielding
    in a segment once `begin + file_len` would advance past the
    segment's last file's begin_time, so a pad-after file is always
    available to `select_window`. `date_subdir=True` checks
    `<out_dir>/<YYYYMMDD>/<file>` instead of flat — must match the
    writer's layout.
    """
    if file_len <= 0:
        # Without this guard, file_len == 0 would infinite-loop:
        # searchsorted(begin) returns the index of begin itself, so
        # next_begin == begin and the loop never advances.
        raise ValueError(f'file_len must be positive, got {file_len}')
    if db.df.empty:
        return
    out_dir = Path(out_dir)

    for seg in db.segments():
        if len(seg) < 2:
            continue
        seg_begins = seg['begin_time'].reset_index(drop=True)
        last_file_begin = seg_begins.iloc[-1].to_pydatetime()
        begin = seg_begins.iloc[0].to_pydatetime()
        while True:
            end = begin + timedelta(seconds=file_len)
            if end > last_file_begin:
                break
            out_path = _proc_out_path(out_dir, db.system, begin, date_subdir)
            if not out_path.exists():
                yield begin, end
            idx = seg_begins.searchsorted(pd.Timestamp(end))
            if idx >= len(seg_begins):
                break
            begin = seg_begins.iloc[idx].to_pydatetime()


def next_unprocessed(
        db: DASdb, out_dir: Path, file_len: int,
        *,
        date_subdir: bool = False,
    ) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Oldest `file_len`-second window not yet written to `out_dir`.

    Returns `(None, None)` when no window is ready; callers unpack
    directly (`begin, end = next_unprocessed(...)`) and check
    `begin is None`.
    """
    for begin, end in iter_unprocessed(
            db, out_dir, file_len, date_subdir=date_subdir):
        return begin, end
    return None, None


def list_unprocessed(
        db: DASdb, out_dir: Path, file_len: int,
        *,
        date_subdir: bool = False,
    ) -> List[Tuple[datetime, datetime]]:
    """Materialize `iter_unprocessed` as a list — convenient when the
    caller wants to see the count or feed `joblib.Parallel`."""
    return list(iter_unprocessed(
        db, out_dir, file_len, date_subdir=date_subdir,
    ))


# --- Pipeline ---

def _trim_to_target(
        rw: RawWindow,
        reads: List[DASdata],
        dt_raw: float,
        fsRatio: int,
    ) -> Tuple[np.ndarray, int, int]:
    """Output-axis trim bounds for the target segment of a padded window.

    Returns `(timeAx, it_left, it_right)` where `timeAx` is the
    decimated time axis (seconds since epoch) spanning all reads and
    [it_left, it_right) is the slice covering the target files only
    (pad files excluded). Matches the legacy Desample_DAS convention:

    * T_left  = first target file's start — `reads[1]` when
        `has_pad_before`, else `rw.begin_time`.
    * T_right = one output sample before the first file after the
        target — `reads[-1].begin - dt_out` when `has_pad_after`, else
        `rw.end_time`. The `-dt_out` (not `-dt_raw`) makes the last
        decimated sample inclusive at the target boundary.
    """
    dt_out = fsRatio * dt_raw
    total_nt = sum(r.data.shape[1] for r in reads)
    timeAx = (
        reads[0].begin_time.timestamp() + np.arange(total_nt) * dt_raw
    )[::fsRatio]

    if rw.has_pad_before:
        t_left = reads[1].begin_time.timestamp()
        it_left = int(np.argmin(np.abs(timeAx - t_left)))
    else:
        it_left = 0

    if rw.has_pad_after:
        t_right = reads[-1].begin_time.timestamp() - dt_out
        it_right = int(np.argmin(np.abs(timeAx - t_right))) + 1
    else:
        it_right = timeAx.shape[0]

    return timeAx, it_left, it_right


def desample_window(
        rw: RawWindow,
        system: str,
        fmax: float = 0.4,
        order: int = 14,
        min_ch: int = 0,
        max_ch: Optional[int] = None,
        nchbuffer: int = 2000,
        nthreads: Optional[int] = None,
    ) -> DASdata:
    """Read padded window, bandpass over full span, decimate, trim to target.

    Accepts `system` in {'ASN', 'OptaSense', 'APSensing', 'Proc'}. With 'Proc' the
    input files are themselves the output of an earlier desample run
    and the OptaSense phase-count unwrap is skipped (Proc payloads
    are already strain).

    The channel axis is processed in chunks of `nchbuffer` to bound
    peak memory on wide acquisitions — a 30 min padded 100 Hz window
    at 50 000 channels would otherwise materialize a 72 GB float64
    buffer. Default 2000 matches the legacy `nChbuffer`. Set to a
    very large value (or `max_ch - min_ch`) to process all channels
    in one pass.

    `nthreads` controls the channel-parallel OpenMP fan-out inside
    the C++ bandpass kernel. `None` (default) resolves to all CPUs
    visible to the process (scheduler affinity if available, total
    cpu_count otherwise) — appropriate for a single-window-at-a-time
    pipeline that has the host to itself. Pass an explicit integer
    to throttle on shared hosts.
    """
    if nthreads is None:
        nthreads = default_nthreads()
    if max_ch is None:
        max_ch = int(rw.rows['nx'].iloc[0])

    # rw from db.select_window, ensures continuity in time and acq params.
    dt_raw = 1.0 / float(rw.rows['fs'].iloc[0])
    raw_fs = float(rw.rows['fs'].iloc[0])
    fsRatio = 1
    if fmax > 0:
        fsRatio = max(1, int(raw_fs / (2.5 * fmax)))
    dt_out = fsRatio * dt_raw
    fs_out = raw_fs / fsRatio

    out_chunks: list = []
    first_data: Optional[DASdata] = None
    begin_time_out: Optional[datetime] = None
    it_left_cached: Optional[int] = None
    it_right_cached: Optional[int] = None

    # Cascading Proc -> Proc must read raw strain with convert=False
    read_kwargs_extra = {'convert': False} if system == 'Proc' else {}

    for c0 in range(min_ch, max_ch, nchbuffer):
        # read nch=nchbuffer from each file
        c1 = min(c0 + nchbuffer, max_ch)
        reads = [
            DASFile(Path(r['file']), system=system).read(
                min_ch=c0, max_ch=c1,
                first_sample=int(r['first_sample']),
                n_samples=int(r['nt']),
                **read_kwargs_extra,
            )
            for _, r in rw.rows.iterrows()
        ]

        # Promote to float64 and force C-contiguous:
        # pybind11 C++ filter reads memory assuming C-contiguous layout
        data = np.ascontiguousarray(
            np.concatenate([d.data for d in reads], axis=1),
            dtype=np.float64,
        )

        # Unwrap OptaSense phase counts across the full concatenated buffer
        if system == 'OptaSense':
            data = preprocess_unwrap(data, factor=1)

        if fmax > 0:
            data = bandpass2d(
                data, freqmin=0.0, freqmax=fmax, dt=dt_raw, order=order,
                zerophase=True, nThreads=nthreads,
            )
            data = data[:, ::fsRatio]

        # compute trim bounds and reuse for following chunks
        # begin_time matches the time of the first data sample
        if it_left_cached is None:
            timeAx, it_left_cached, it_right_cached = _trim_to_target(
                rw, reads, dt_raw, fsRatio,
            )
            trimmed = timeAx[it_left_cached:it_right_cached]
            begin_time_out = (
                datetime.fromtimestamp(trimmed[0], tz=timezone.utc)
                if trimmed.size else rw.begin_time
            )

        data = data[:, it_left_cached:it_right_cached]
        out_chunks.append(data.astype(np.float32, copy=False))
        if first_data is None:
            first_data = reads[0]

    data_out = (out_chunks[0] if len(out_chunks) == 1
                else np.concatenate(out_chunks, axis=0))
    nt = data_out.shape[1]
    # Metadata (including /Acquisition_origin) comes from the first file
    end_time_out = (begin_time_out + timedelta(seconds=(nt - 1) / fs_out)
                    if nt > 0 else begin_time_out)
    return DASdata(
        data=data_out,
        fs=fs_out, dt=1.0 / fs_out, nt=nt, nx=data_out.shape[0],
        dx=first_data.dx,
        begin_time=begin_time_out,
        end_time=end_time_out,
        gauge_length_m=first_data.gauge_length_m, system=system,
        raw_meta=first_data.raw_meta,
    )


def desample_and_write_window(
        db: DASdb,
        begin: datetime,
        end: datetime,
        out_dir: Path,
        system: str,
        fmax: float = 0.4,
        order: int = 14,
        min_ch: int = 0,
        max_ch: Optional[int] = None,
        nchbuffer: int = 2000,
        nthreads: Optional[int] = None,
        pad: bool = True,
        date_subdir: bool = False,
    ) -> Optional[Path]:
    """Desample one window end-to-end and persist the result.

    Thin orchestration layer on top of `db.select_window` +
    `desample_window` + `write_data_proc`. Exposed at module scope
    so batch drivers and notebooks can reuse the full pipeline
    without re-implementing the window→file naming or the
    already-written skip check.

    Returns the written output path, or `None` when a file with the
    same canonical name already exists (the CLI uses this as a
    skip signal).

    Raises `RuntimeError` if `[begin, end)` straddles an acquisition
    gap — caller must split such windows into per-segment passes.
    """
    rws = db.select_window(begin, end, pad=pad)
    if len(rws) > 1:
        raise RuntimeError(
            f'window [{begin}, {end}) spans {len(rws)} contiguous '
            f'segments (acquisition gap); desample each segment separately'
        )
    d = desample_window(
        rws[0], system, fmax=fmax, order=order,
        min_ch=min_ch, max_ch=max_ch, nchbuffer=nchbuffer,
        nthreads=nthreads,
    )
    out_path = _proc_out_path(out_dir, system, d.begin_time, date_subdir)
    if out_path.exists():
        print(f'exists, skip: {out_path}')
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_data_proc(out_path, d)
    print(f'wrote: {out_path}  shape={d.shape}  fs={d.fs} Hz')
    return out_path


# --- CLI ---

def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m realTimeMonitor.dasio.desample')
    ap.add_argument('--from', dest='raw_dir', required=True, type=Path)
    ap.add_argument('--to', dest='out_dir', required=True, type=Path)
    ap.add_argument(
        '--system',
        choices=['ASN', 'OptaSense', 'APSensing', 'Proc'], required=True,
    )
    ap.add_argument('--fmax', type=float, default=0.4)
    ap.add_argument('--order', type=int, default=14)
    ap.add_argument(
        '--since', help='ISO begin (inclusive); omit with --until to auto-pick',
    )
    ap.add_argument(
        '--until', help='ISO end (exclusive); omit with --since to auto-pick',
    )
    ap.add_argument(
        '--file-len', type=int, default=60,
        help='output window length in seconds (auto mode only; default 60)',
    )
    ap.add_argument(
        '--all', action='store_true',
        help='in auto mode, process every unprocessed window instead of just one',
    )
    ap.add_argument('--no-pad', action='store_true')
    ap.add_argument(
        '--out-date-subdir', action='store_true',
        help='write output as <out_dir>/<YYYYMMDD>/<file>.h5 (UTC begin '
            'date) so Processed/ does not accumulate millions of flat '
            'siblings. Both the writer and the next_unprocessed skip '
            'check use the nested layout when this flag is set; if '
            'existing files were never moved into date subdirs they '
            'will be reprocessed.',
    )
    ap.add_argument('--min-ch', type=int, default=0)
    ap.add_argument('--max-ch', type=int, default=None)
    ap.add_argument(
        '--nchbuffer', type=int, default=2000, metavar='N',
        help='channel-axis chunk size used to bound peak memory in '
            'desample_window (default 2000, matches legacy nChbuffer)',
    )
    ap.add_argument(
        '--nthreads', '-nTh', type=int, default=None, metavar='N',
        help='OpenMP threads for the channel-parallel C++ bandpass '
            'kernel. Default: physical cores visible to the process. '
            'When --nworkers > 1 this auto-defaults to '
            'physical_cores // nworkers so the total budget '
            'nworkers * nthreads stays within the host.',
    )
    ap.add_argument(
        '--nworkers', '-nW', type=int, default=1, metavar='N',
        help='Process-level parallelism over unprocessed windows '
            '(joblib). Default 1 (live cron behavior). For backlog '
            'drain set to 2-8 to overlap I/O across windows; combine '
            'with a smaller --nthreads (or rely on the auto-default).',
    )
    ap.add_argument(
        '--dasdb', type=Path, default=None,
        help='persist the raw-file catalog to this file so subsequent '
            'invocations can refresh incrementally instead of '
            'rescanning --from every time. Format chosen from extension '
            '(.parquet / .csv). Created on first run.',
    )
    ap.add_argument(
        '--progress', action=argparse.BooleanOptionalAction, default=None,
        help='show a tqdm bar during the dasdb scan. Default: auto — on '
            'when stderr is a TTY (interactive), off when redirected '
            '(cron, systemd). Use --progress / --no-progress to override.',
    )
    ap.add_argument(
        '--scan-workers', type=int, default=1, metavar='N',
        help='ProcessPoolExecutor workers for the dasdb metadata scan '
            '(stage 1 — opening every HDF5 to read header/begin_time/'
            'sampleSkew/etc.). Distinct from --nthreads (OMP threads in '
            'the bandpass C++ kernel, stage 3) and --nworkers (joblib '
            'processes parallel-desampling multiple windows, stage 3). '
            'Default 1 = single-threaded scan. For a first 1 M-file '
            'archive scan, set 4-8 to drop the wallclock from ~10 min '
            'to ~2-3 min. HDF5 has an internal global lock so scaling '
            'is sublinear above ~8.',
    )
    ap.add_argument(
        '--dry-run', action='store_true',
        help='Print the output paths that would be written, then exit '
            'without opening any HDF5, running the bandpass, or '
            'writing output. Honours --all (lists the full backlog) '
            'and --out-date-subdir. Useful for previewing catch-up '
            'workload before committing.',
    )
    ap.add_argument(
        '--proc-dasdb', type=Path, default=None,
        help='Catalog of files already in --to (system=Proc). On '
            'startup, load (or scan --to to build) and use the '
            'latest end_time as a resume marker: raw files with '
            'begin_time <= cutoff are dropped before the segment '
            'walker runs, so the new desample never reprocesses '
            'time ranges the legacy run already covered. Cleaner '
            'than the default filename-existence skip-check when '
            'the legacy output used a different segment anchor or '
            'a different filename prefix.',
    )
    args = ap.parse_args(argv)

    # Auto-detect: on for interactive shells, off for cron / systemd /
    # any redirected stderr. Per-tick progress bars in cron logs would
    # spam thousands of carriage-return frames; per-startup interactive
    # bars are essential for the 1M-file first scan.
    progress = (args.progress if args.progress is not None
                else sys.stderr.isatty())

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.since) != bool(args.until):
        ap.error(
            '--since and --until must be given together '
            '(or neither, for auto mode)'
        )

    # dasdb build / load+update, consumed by `db.select_window` and
    # `db.next_unprocessed` below. Built / updated catalog is
    # written to the file specified by --dasdb.
    if _dasdb_path_loadable(args.dasdb):
        db = DASdb.from_file(args.dasdb, system=args.system)
        n_before = db.n_files
        n_new = db.update_from_dir(
            args.raw_dir, workers=args.scan_workers, progress=progress,
        )
        print(
            f'[dasdb] loaded {n_before} files from {args.dasdb}, {n_new} updated'
        )
        if n_new > 0:
            db.to_file(args.dasdb)
    else:
        db = DASdb.from_dir(
            args.raw_dir, args.system,
            workers=args.scan_workers, progress=progress,
        )
        if args.dasdb is not None:
            print(
                f'[dasdb] built fresh from {args.raw_dir}: ({db.n_files} files)'
            )
            db.to_file(args.dasdb)

    # Proc dasdb resume marker. The proc dasdb's only purpose here is
    # to carry a "stop sign" — the latest end_time across already-
    # desampled files. We trim the raw catalog in-place so the segment
    # walker re-runs on a clean post-cutoff view; no special anchor
    # logic, no in-segment shifting. The first window that lands gets
    # whatever 60-s grid the standard walker would produce on a fresh
    # deployment seeded with raw files starting just after the cutoff.
    proc_db = None
    if args.proc_dasdb is not None:
        if _dasdb_path_loadable(args.proc_dasdb):
            proc_db = DASdb.from_file(args.proc_dasdb, system='Proc')
            n_pre = proc_db.n_files
            n_new = proc_db.update_from_dir(
                args.out_dir, workers=args.scan_workers, progress=progress,
            )
            if n_new > 0:
                proc_db.to_file(args.proc_dasdb)
            print(
                f'[proc-dasdb] loaded {n_pre} files from {args.proc_dasdb}, '
                f'{n_new} new'
            )
        else:
            proc_db = DASdb.from_dir(
                args.out_dir, 'Proc',
                workers=args.scan_workers, progress=progress,
            )
            proc_db.to_file(args.proc_dasdb)
            print(
                f'[proc-dasdb] built fresh from {args.out_dir}: '
                f'({proc_db.n_files} files)'
            )
        if not proc_db.df.empty:
            cutoff = proc_db.df['end_time'].max()
            if pd.notna(cutoff):
                before = len(db.df)
                db.df = db.df[db.df['begin_time'] > cutoff].reset_index(drop=True)
                print(
                    f'[proc-dasdb] resume cutoff = {cutoff} → '
                    f'{before:,} → {len(db.df):,} raw files remain'
                )

    # When --nworkers > 1 and the user didn't pin --nthreads, divide the
    # host's physical cores evenly across workers so the joint
    # nworkers * nthreads budget stays within the box. Individual
    # bandpass calls under joblib processes have separate OMP contexts,
    # so this is a real partition (no nesting concerns).
    nworkers = max(1, args.nworkers)
    if args.nthreads is None:
        nthreads = max(1, default_nthreads() // nworkers)
    else:
        nthreads = args.nthreads

    if args.dry_run:
        # Walk the same iter_unprocessed the wet path uses; print the
        # output paths without ever opening the input HDF5s. Honours
        # --since/--until, --all, --out-date-subdir.
        if args.since and args.until:
            since = datetime.fromisoformat(args.since)
            until = datetime.fromisoformat(args.until)
            windows = [(since, until)]
        else:
            windows = list_unprocessed(
                db, args.out_dir, args.file_len,
                date_subdir=args.out_date_subdir,
            )
            if not args.all:
                windows = windows[:1]
        if not windows:
            print('dry-run: nothing unprocessed')
            return 0
        for begin, end in windows:
            out_path = _proc_out_path(
                args.out_dir, args.system, begin, args.out_date_subdir,
            )
            print(f'would write: {out_path}  '
                  f'window=[{begin.isoformat()}, {end.isoformat()})')
        print(f'dry-run: {len(windows)} window(s) total')
        return 0

    def _run_one(begin, end):
        return desample_and_write_window(
            db, begin, end, args.out_dir, args.system,
            fmax=args.fmax, order=args.order,
            min_ch=args.min_ch, max_ch=args.max_ch,
            nchbuffer=args.nchbuffer,
            nthreads=nthreads,
            pad=not args.no_pad,
            date_subdir=args.out_date_subdir,
        )

    def _finalize_proc_dasdb():
        # Refresh the proc-dasdb's tail from out_dir after the run, so
        # the next invocation's resume cutoff covers everything just
        # written. Cheap — only NEW Proc files get opened.
        if proc_db is None or args.proc_dasdb is None:
            return
        n_new = proc_db.update_from_dir(
            args.out_dir, workers=args.scan_workers, progress=False,
        )
        if n_new > 0:
            proc_db.to_file(args.proc_dasdb)
            print(f'[proc-dasdb] appended {n_new} new entries')

    if args.since and args.until:
        _run_one(
            datetime.fromisoformat(args.since),
            datetime.fromisoformat(args.until),
        )
        _finalize_proc_dasdb()
        return 0

    # Auto mode: enumerate the backlog up front so the work partition
    # is decided before any worker runs. joblib hands disjoint slices
    # to each worker — no race, no out_path.exists() collisions.
    if args.all and nworkers > 1:
        windows = list_unprocessed(
            db, args.out_dir, args.file_len,
            date_subdir=args.out_date_subdir,
        )
        if not windows:
            print('auto: nothing to process')
            return 0
        print(
            f'auto: {len(windows)} unprocessed window(s); '
            f'nworkers={nworkers}, nthreads={nthreads}'
        )
        from joblib import Parallel, delayed
        Parallel(n_jobs=nworkers, backend='loky', verbose=10)(
            delayed(_run_one)(b, e) for b, e in windows
        )
        _finalize_proc_dasdb()
        return 0

    # Sequential path (single worker — live cron, or --all not set).
    n = 0
    while True:
        begin, end = next_unprocessed(
            db, args.out_dir, args.file_len,
            date_subdir=args.out_date_subdir,
        )
        if begin is None:
            if n == 0:
                print('auto: nothing to process')
            break
        _run_one(begin, end)
        n += 1
        if not args.all:
            break
    _finalize_proc_dasdb()
    return 0


if __name__ == '__main__':
    sys.exit(main())
