"""`DASdb`: catalog of raw DAS files + `RawWindow` read-plan helpers.

Model:

DASmeta	one contiguous readable piece of raw data (one ASN / Proc
		file, or one RawDataTime chunk within an OptaSense file).
		Defined in `dasdata.py` as a TypedDict so scanners can
		emit a list of dicts and pandas consumes them directly.
DASdb		DataFrame of DASmeta rows for one vendor. Time-contiguous
		runs are recomputed on the fly by `segments`;
		nothing is persisted as a segment_id column.
RawWindow	selected read plan that `desample.desample_window`
		consumes — files + per-row slice offsets + pad flags.

Vendor-specific logic lives only in `list_das_files` and the per-
file metadata readers. The catalog and all queries (`select_window`,
`read`, plus the desample-side `next_unprocessed` selector that
consumes DASdb) work off DataFrame rows regardless of vendor.

Time convention: `begin_time` and `end_time` are last-sample-inclusive
timestamps, matching legacy DAS-utilities (`DAS_db.py`). Continuity
between two rows is therefore `next.begin_time - prev.end_time ≈ 1/fs`,
not `≈ 0`.

File metadata (fs, nt, channel count, etc.) is populated at scan time
from each file's HDF5 metadata — filename parsing is never used as a
correctness fallback; a file that can't be read is skipped with a
stderr warning.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .dasdata import DASdata, DASmeta
from .dasfile import DASFile
from .schema import RawWindow, _CSV_TO_DF, _DF_COLUMNS, _DF_TO_CSV
from .utils import atomic_write, list_data_files


def _format_for(file: Union[str, Path]) -> str:
    """Return 'parquet' or 'csv' from file extension; raise on unknown."""
    s = Path(file).suffix.lower()
    if s in ('.parquet', '.parq'):
        return 'parquet'
    if s in ('.csv', '.txt', ''):
        return 'csv'
    raise ValueError(
        f'Unsupported DASdb file extension {Path(file).suffix!r}; '
        f"expected one of '.parquet', '.parq', '.csv', '.txt'"
    )


# Canonical ASN layout: raw_dir/YYYYMMDD/HHMMSS.hdf5, with a Santorini
# variant that inserts a channel-name subdir (YYYYMMDD/<ch>/HHMMSS.hdf5)
# — covered by the '*/*.hdf5' branch. OptaSense and Proc both flat-
# glob '*.h5'.
_ASN_DAY_DIR_RE = re.compile(r'^\d{8}$')


def list_das_files(raw_dir: Path, system: str) -> List[Path]:
    """Enumerate candidate DAS files under `raw_dir` for `system`.

    Pure filesystem work — no HDF5 opens. Paired with
    `DASFile.metadata()` to drive incremental catalog refresh without
    reopening files already in the DASdb.
    """
    raw_dir = Path(raw_dir)
    if system == 'ASN':
        files: List[Path] = []
        for sub in raw_dir.iterdir():
            if sub.is_dir() and _ASN_DAY_DIR_RE.match(sub.name):
                files.extend(list_data_files(sub, ('*.hdf5', '*/*.hdf5')))
        return files
    if system == 'Proc':
        # Two layouts coexist: flat (legacy desample output, before
        # --out-date-subdir landed) and <YYYYMMDD>/ProcASN-*.h5 (the
        # new layout the cron uses on Santorini-scale deployments to
        # keep readdir on Processed/ from blowing up). Cover both —
        # without the date-subdir branch, building a Proc dasdb from
        # the new layout returns zero files.
        files = list(list_data_files(raw_dir, '*.h5'))
        for sub in raw_dir.iterdir():
            if sub.is_dir() and _ASN_DAY_DIR_RE.match(sub.name):
                files.extend(list_data_files(sub, '*.h5'))
        return files
    if system in ('OptaSense', 'APSensing'):
        return list_data_files(raw_dir, '*.h5')
    raise ValueError(f'Unknown system {system!r}')


def _read_metadata(file: Path, system: str) -> Optional[DASmeta]:
    """Module-level worker for `_read_metadata_batch` process pool.

    Lives at module scope (not as a closure) so it is picklable for
    ProcessPoolExecutor; the local-closure form would fail with
    'Can't pickle local object'.
    """
    return DASFile(file, system=system).metadata()


def _read_metadata_batch(
        files: List[Path], system: str,
        workers: int = 1, progress: bool = False,
    ) -> List[DASmeta]:
    """Read DASFile(f, system=system).metadata() for each file.

    Parallelizes via ProcessPoolExecutor (HDF5 holds an internal
    global lock that serializes thread-pool workers). progress=True
    wraps the iteration with a tqdm bar; silently disabled when tqdm
    is not installed so the flag stays safe to leave on in scripts.
    """
    def _pbar(iterable):
        if not progress:
            return iterable
        try:
            from tqdm import tqdm
        except ImportError:
            import sys
            print(
                '[dasdb] --progress needs tqdm; continuing without bar',
                file=sys.stderr,
            )
            return iterable
        return tqdm(iterable, total=len(files), unit='file', desc='scan')

    if workers <= 1:
        metas = [_read_metadata(f, system) for f in _pbar(files)]
    else:
        from concurrent.futures import ProcessPoolExecutor
        from functools import partial
        worker = partial(_read_metadata, system=system)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            metas = list(_pbar(ex.map(worker, files)))

    rows: List[DASmeta] = []
    for m in metas:
        if m is None:
            continue
        rows.extend(m if isinstance(m, list) else [m])
    return rows


def scan_metadata(
        raw_dir: Path, system: str,
        workers: int = 1, progress: bool = False,
    ) -> pd.DataFrame:
    """Scan `raw_dir` for all `system` files and return a DASmeta DataFrame.

    Enumerates files via `list_das_files`, reads each through the
    `DASFile.metadata()` facade, concatenates OptaSense RawDataTime
    splits, and sorts by begin_time. Files that fail metadata read
    are skipped (the vendor metadata readers emit a stderr warning).
    `workers > 1` parallelizes the per-file reads; `progress=True`
    adds a tqdm bar.

    `raw_dir` is resolved to an absolute path before enumeration so
    every row's `file` field is absolute regardless of the caller's
    cwd. Without this, a catalog written from a relative `--from`
    would only resolve correctly when re-read from the same working
    dir.
    """
    raw_dir = Path(raw_dir).resolve()
    rows = _read_metadata_batch(
        list_das_files(raw_dir, system), system,
        workers=workers, progress=progress,
    )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('begin_time').reset_index(drop=True)
    return df


# --- DASdb ---------------------------------------------------------------

class DASdb:
    """Catalog of DAS file metadata rows, partitioned on the fly.

    Parameters
    ----------
    df :
        DataFrame with the `_DF_COLUMNS` schema (one row per
        DASmeta). Missing columns fall back to NaN / 0 where a default
        is meaningful.
    system :
        ``ASN``, ``OptaSense``, or ``Proc`` (matches the scanner used
        to build ``df``).
    """

    def __init__(self, df: pd.DataFrame, system: str):
        if df is None or len(df) == 0:
            df = pd.DataFrame({c: pd.Series(dtype=object) for c in _DF_COLUMNS})
        else:
            df = df.copy()
            if 'first_sample' not in df.columns:
                df['first_sample'] = 0
            missing = [c for c in _DF_COLUMNS if c not in df.columns]
            for c in missing:
                df[c] = np.nan
            df = df[_DF_COLUMNS]
        self.df = df.reset_index(drop=True)
        self.system = system

    # ------------------------------------------------------------------ repr

    def __repr__(self) -> str:
        if self.df.empty:
            return f'DASdb({self.system}, empty)'
        n = len(self.df)
        n_seg = self.n_segments
        t0 = self.df['begin_time'].iloc[0]
        t1 = self.df['end_time'].iloc[-1]
        return (f'DASdb({self.system}, {n} files, {n_seg} segment'
                f'{"s" if n_seg != 1 else ""}, {t0} .. {t1})')

    # ------------------------------------------------------------- properties

    @property
    def n_files(self) -> int:
        return len(self.df)

    @property
    def n_segments(self) -> int:
        """Number of time-contiguous segments — recomputed on access."""
        if self.df.empty:
            return 0
        return sum(1 for _ in self.segments())

    # --------------------------------------------------------- continuity

    def segments(
            self, gap_factor: float = 1.5,
        ) -> Iterator[pd.DataFrame]:
        """Yield (begin_time-sorted) DataFrames of time-contiguous rows.

        One segment = one continuous run of rows with no acquisition
        gap. Break criteria between consecutive rows (prev → cur):

        * timing break  — |cur.begin_time - prev.end_time - 1/fs| >
        gap_factor x 1/fs  (default tolerates roughly one sample of
        jitter; raise gap_factor to fuse near-contiguous segments).
        * fs or nx change — acquisition parameters changed.
        * same file path — OptaSense intra-file RawDataTime breaks
        always start a new segment regardless of gap size.

        Recomputed on every call; DASdb never persists a segment_id.

        Implementation note: the break-detection rules are evaluated
        vectorized over the whole catalog with `Series.shift(1)` —
        on Santorini's 1.13 M-row production catalog the 24 h slice
        (~8.6 k rows) goes from ~1 s of GIL-held Python in the
        pre-vectorize row-by-row loop to ~5 ms. The slow loop was
        amplifying every /api/state poll's `_read_gaps_24h` call,
        saturating one CPU and freezing the dashboard whenever a
        user clicked the gaps tab. See
        test/test_dasdb_segments_perf.py for the regression guard.
        """
        if self.df.empty:
            return
        df = self.df.sort_values('begin_time').reset_index(drop=True)
        n = len(df)
        if n == 1:
            yield df
            return
        # Treat fs=NaN as fs=0 so 1/fs falls back to 0 — matches the
        # scalar reference's `if pd.notna(...) else 0.0` branch.
        fs_prev = df['fs'].shift(1).fillna(0.0).to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            expected = np.where(fs_prev > 0, 1.0 / fs_prev, 0.0)
        gap = (df['begin_time']
               - df['end_time'].shift(1)).dt.total_seconds().to_numpy()
        # Use np.isclose directly (array-valued atol via broadcasting)
        # so NaN gaps — possible if end_time has NaT mid-stream — are
        # treated as "not close" and produce a break, matching the
        # scalar reference's `not np.isclose(NaN, ...) is True` behavior.
        # np.abs() based comparison would silently swallow NaN as False.
        timing_break = ~np.isclose(
            gap, expected, atol=gap_factor * expected,
        )
        fs_change = (df['fs'] != df['fs'].shift(1)).to_numpy()
        nx_change = (df['nx'] != df['nx'].shift(1)).to_numpy()
        same_file = (df['file'] == df['file'].shift(1)).to_numpy()
        break_mask = fs_change | nx_change | same_file | timing_break
        # Position 0 is NOT a break — it's the start of the first
        # segment, never compared against a "previous" row in the
        # scalar reference (the loop runs from i=1).
        break_mask[0] = False
        break_idx = np.flatnonzero(break_mask)
        starts = np.concatenate(([0], break_idx))
        ends = np.concatenate((break_idx, [n]))
        for s, e in zip(starts, ends):
            yield df.iloc[int(s):int(e)].reset_index(drop=True)

    # --------------------------------------------------------------- builders

    @classmethod
    def from_dir(
            cls, raw_dir: Path, system: str,
            workers: int = 1, progress: bool = False,
        ) -> 'DASdb':
        """Scan `raw_dir` and build a catalog for the given `system`."""
        return cls(
            scan_metadata(raw_dir, system, workers=workers, progress=progress),
            system,
        )

    @classmethod
    def from_csv(
            cls, file: Union[str, Path],
            system: Optional[str] = None,
        ) -> 'DASdb':
        """Load a legacy DAS_db CSV (whitespace-separated).

        Translates legacy column names (begTime, endTime, nChannels,
        dCh, GaugeLen, firstSample) into the internal snake_case
        schema. Drops any `segment_id` or `system` columns the CSV
        might carry — neither is part of the in-memory schema. If
        `system` is given and the CSV has a `system` column, rows are
        filtered to that system before the column is dropped.
        """
        file = Path(file)
        df = pd.read_csv(file, sep=r'\s+', engine='python')
        df = df.rename(columns=_CSV_TO_DF)
        # format='ISO8601' so rows with and without fractional seconds mix freely;
        df['begin_time'] = pd.to_datetime(df['begin_time'], utc=True, format='ISO8601')
        df['end_time'] = pd.to_datetime(df['end_time'], utc=True, format='ISO8601')
        if 'system' in df.columns:
            if system is not None:
                df = df[df['system'] == system]
            elif len(df):
                system = df['system'].iloc[0]
            df = df.drop(columns=['system'])
        if system is None:
            raise ValueError(
                f"DASdb.from_csv: cannot infer system from {file.name!r} "
                "(no `system` column and no `system=` kwarg). Pass "
                "system='ASN' | 'OptaSense' | 'APSensing' | 'Proc' "
                "explicitly. Catalogs from desample.py are 'Proc'."
            )
        if 'segment_id' in df.columns:
            df = df.drop(columns=['segment_id'])
        df = df.sort_values('begin_time').reset_index(drop=True)
        if 'first_sample' not in df.columns:
            df['first_sample'] = 0
        keep = [c for c in _DF_COLUMNS if c in df.columns]
        return cls(df[keep], system)

    def to_csv(self, file: Union[str, Path]) -> None:
        """Write a DAS_db CSV (whitespace-separated, one row per DASmeta).

        Emits legacy column names (begTime, endTime, nChannels, dCh,
        GaugeLen, firstSample) so external DAS-utilities tooling can
        still read the output. Missing numeric fields get the legacy
        -1 sentinel (rather than empty, which would collapse under
        whitespace-separation). Sorted by begin_time; atomic write.
        """
        sorted_df = self.df.sort_values('begin_time').reset_index(drop=True)
        out = sorted_df[_DF_COLUMNS].copy()
        out = out.rename(columns=_DF_TO_CSV)
        out['begTime'] = out['begTime'].apply(
            lambda t: t.isoformat() if pd.notna(t) else '-1'
        )
        out['endTime'] = out['endTime'].apply(
            lambda t: t.isoformat() if pd.notna(t) else '-1'
        )
        for col in ('fs', 'nt', 'nChannels', 'dCh', 'GaugeLen', 'firstSample'):
            out[col] = out[col].fillna(-1)
        atomic_write(file, lambda tmp: out.to_csv(tmp, sep=' ', index=False))

    @classmethod
    def from_parquet(
            cls, file: Union[str, Path],
            system: Optional[str] = None,
        ) -> 'DASdb':
        """Load a DASdb Parquet file.

        system is recovered from the b'system' file-level metadata
        key written by to_parquet; the kwarg is a fallback for
        foreign parquet files without that key, and a conflict
        between the two raises.
        """
        import pyarrow.parquet as pq
        file = Path(file)
        meta = pq.read_schema(file).metadata or {}
        meta_system = meta.get(b'system')
        if meta_system is not None:
            system_in_file = meta_system.decode('utf-8')
            if system is not None and system != system_in_file:
                raise ValueError(
                    f"DASdb.from_parquet: system kwarg {system!r} "
                    f"contradicts file metadata {system_in_file!r}"
                )
            system = system_in_file
        if system is None:
            raise ValueError(
                f"DASdb.from_parquet: cannot infer system from {file.name!r} "
                "(no 'system' file metadata and no system= kwarg). Pass "
                "system='ASN' | 'OptaSense' | 'APSensing' | 'Proc' "
                "explicitly."
            )
        df = pd.read_parquet(file)
        df = df.sort_values('begin_time').reset_index(drop=True)
        return cls(df, system)

    def to_parquet(self, file: Union[str, Path]) -> None:
        """Write a DASdb to a Parquet file (snake_case schema, snappy).

        system rides along in file-level metadata under b'system'.
        Sorted by begin_time; atomic write.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        sorted_df = self.df.sort_values('begin_time').reset_index(drop=True)
        table = pa.Table.from_pandas(sorted_df, preserve_index=False)
        existing = table.schema.metadata or {}
        merged = {**existing, b'system': self.system.encode('utf-8')}
        table = table.replace_schema_metadata(merged)
        atomic_write(
            file,
            lambda tmp: pq.write_table(table, tmp, compression='snappy'),
        )

    @classmethod
    def from_file(
            cls, file: Union[str, Path],
            system: Optional[str] = None,
        ) -> 'DASdb':
        """Load a catalog by path; format chosen from extension."""
        reader = cls.from_parquet if _format_for(file) == 'parquet' else cls.from_csv
        return reader(file, system=system)

    def to_file(self, file: Union[str, Path]) -> None:
        """Persist the catalog; format chosen from extension."""
        writer = self.to_parquet if _format_for(file) == 'parquet' else self.to_csv
        writer(file)

    # --------------------------------------------------------- incremental

    def append(self, df_new: pd.DataFrame) -> None:
        """Concatenate `df_new` rows into the catalog.

        De-duplicates on (`file`, `first_sample`) — OptaSense emits
        multiple rows per file (one per RawDataTime chunk), so `file`
        alone is not unique. Sort order is preserved by `begin_time`.
        """
        if df_new is None or len(df_new) == 0:
            return
        combined = pd.concat([self.df, df_new], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=['file', 'first_sample'], keep='last'
        ).sort_values('begin_time').reset_index(drop=True)
        self.df = combined[_DF_COLUMNS]

    def update_from_dir(
            self, raw_dir: Path,
            workers: int = 1, progress: bool = False,
        ) -> int:
        """Incrementally add any new files from `raw_dir`.

        Lists the candidate files on disk and only opens the ones
        whose path isn't already in `self.df['file']` — turns a full
        O(n_total) rescan into O(n_new) HDF5 opens. Safe to call on
        every cron tick; continuity is recomputed on the fly by
        `segments`, so gaps that open or close in the newly
        discovered tail are reflected immediately. `workers > 1`
        parallelizes the per-file metadata reads; `progress=True`
        wraps the iteration with a tqdm bar.

        Returns the number of rows added (0 if nothing was new).
        """
        raw_dir = Path(raw_dir).resolve()
        known = set(self.df['file']) if not self.df.empty else set()
        new_files = [
            f for f in list_das_files(raw_dir, self.system)
            if str(f) not in known
        ]
        rows = _read_metadata_batch(
            new_files, self.system, workers=workers, progress=progress,
        )
        if rows:
            self.append(pd.DataFrame(rows))
        return len(rows)

    @classmethod
    def load_or_build(
            cls, file: Optional[Path], raw_dir: Path, system: str,
            overwrite: bool = False,
            workers: int = 1, progress: bool = False,
        ) -> 'DASdb':
        """Load an existing catalog (+ update) or scan raw_dir fresh.

        * file=None              — fresh scan, no persistence.
        * file exists + !overwrite → load (csv or parquet by extension) + update_from_dir.
        * file missing, or overwrite → fresh from_dir.

        Never writes; caller persists via db.to_csv / db.to_parquet / db.to_file.
        """
        if file is not None and file.exists() and not overwrite:
            db = cls.from_file(file, system=system)
            db.update_from_dir(raw_dir, workers=workers, progress=progress)
            return db
        return cls.from_dir(
            raw_dir, system, workers=workers, progress=progress,
        )

    # ---------------------------------------------------------------- queries

    def select_window(
            self, begin: datetime, end: datetime,
            pad: bool = True,
        ) -> List[RawWindow]:
        """One RawWindow per segment intersecting [begin, end).

        Rows whose `[begin_time, end_time]` intersects `[begin, end)`
        are the target set. Pad files are added from the SAME segment
        only — neighbours across a gap are never pulled in, so the
        returned list makes each segment's own pad-before/pad-after
        state explicit instead of silently dropping one.

        Most callers (the `next_unprocessed` → `desample_window` path)
        ask for a window that falls entirely in one segment and just
        take `[0]`; multi-segment callers iterate.

        Raises `RuntimeError` when no segment intersects.
        """
        if self.df.empty:
            raise RuntimeError('DASdb is empty')

        windows: List[RawWindow] = []
        for seg in self.segments():
            target_idx = seg.index[
                (seg['begin_time'] < end) & (seg['end_time'] > begin)
            ].tolist()
            if not target_idx:
                continue
            sel_idx = list(target_idx)
            has_pad_before = False
            has_pad_after = False
            if pad:
                if target_idx[0] > 0:
                    sel_idx.insert(0, target_idx[0] - 1)
                    has_pad_before = True
                if target_idx[-1] < len(seg) - 1:
                    sel_idx.append(target_idx[-1] + 1)
                    has_pad_after = True
            windows.append(RawWindow(
                rows=seg.iloc[sel_idx].reset_index(drop=True),
                begin_time=begin,
                end_time=end,
                has_pad_before=has_pad_before,
                has_pad_after=has_pad_after,
            ))
        if not windows:
            raise RuntimeError(f'No files in window [{begin}, {end})')
        return windows

    def read(
            self, 
            begin_time: datetime, 
            end_time: datetime,
            min_ch: int = 0,
            max_ch: Optional[int] = None,
            fill_gap: bool = True,
        ) -> DASdata:
        """Read a concatenated DASdata covering [begin, end).

        Vendor-agnostic time-range query. Walks every segment that
        overlaps the window; for each intersecting row the exact
        `first_sample` offset and `n_samples` count are computed from
        the row's begin_time / fs / nt, so `DASFile.read` returns
        only the samples that fall inside the window — no post-read
        trim.

        If the window straddles acquisition gaps:

        * `fill_gap=True` (default) — each inter-segment gap is
        filled with zeros so the returned data has a uniformly-
        spaced time axis running from the first kept sample onward.
        * `fill_gap=False` — gaps are dropped and the output is
        shorter than `(end - begin) x fs`. `end_time` reports the
        last sample's true timestamp, so callers can detect the
        shortening.

        Raises `RuntimeError` if no files intersect the window.
        """
        if self.df.empty:
            raise RuntimeError('DASdb is empty')

        blocks: List[np.ndarray] = []
        block_begins: List[datetime] = []
        first_read: Optional[DASdata] = None
        dt = fs = None

        # read and concatenate within each continuous segment
        for seg in self.segments():
            mask = (seg['begin_time'] < end_time) & (seg['end_time'] > begin_time)
            if not mask.any():
                continue
            reads: List[DASdata] = []
            for _, row in seg[mask].iterrows():
                row_dt = 1.0 / float(row['fs'])
                row_begin = row['begin_time'].to_pydatetime()
                row_end_exclusive = row_begin + timedelta(
                    seconds=int(row['nt']) * row_dt
                )
                clip_begin = max(begin_time, row_begin)
                clip_end = min(end_time, row_end_exclusive)
                offset = int(round(
                    (clip_begin - row_begin).total_seconds() / row_dt
                ))
                n = int(round(
                    (clip_end - clip_begin).total_seconds() / row_dt
                ))
                if n <= 0:
                    continue
                d = DASFile(Path(row['file']), system=self.system).read(
                    min_ch=min_ch, max_ch=max_ch,
                    first_sample=int(row['first_sample']) + offset,
                    n_samples=n,
                )
                reads.append(d)
            if not reads:
                continue
            if first_read is None:
                first_read = reads[0]
                dt, fs = first_read.dt, first_read.fs
            block = (reads[0].data if len(reads) == 1
                    else np.concatenate([d.data for d in reads], axis=1))
            blocks.append(block)
            block_begins.append(reads[0].begin_time)

        if not blocks:
            raise RuntimeError(f'No files in window [{begin_time}, {end_time})')

        # fillin zeros for gaps between continuous segment data
        nx = blocks[0].shape[0]
        dtype = blocks[0].dtype
        if fill_gap and len(blocks) > 1:
            pieces = [blocks[0]]
            for prev_block, prev_begin, cur_block, cur_begin in zip(
                    blocks[:-1], block_begins[:-1],
                    blocks[1:], block_begins[1:]):
                prev_end = prev_begin + timedelta(
                    seconds=prev_block.shape[1] * dt
                )
                gap_samples = int(round(
                    (cur_begin - prev_end).total_seconds() / dt
                ))
                if gap_samples > 0:
                    pieces.append(np.zeros((nx, gap_samples), dtype=dtype))
                pieces.append(cur_block)
            data = np.concatenate(pieces, axis=1)
        elif len(blocks) == 1:
            data = blocks[0]
        else:
            data = np.concatenate(blocks, axis=1)

        out_begin = block_begins[0]
        nt = data.shape[1]
        out_end = out_begin + timedelta(seconds=(nt - 1) * dt) if nt else out_begin
        return DASdata(
            data=data, fs=fs, dt=dt, nt=nt, nx=nx,
            dx=first_read.dx,
            begin_time=out_begin, end_time=out_end,
            gauge_length_m=first_read.gauge_length_m, system=self.system,
            raw_meta=first_read.raw_meta,
        )

    # ------------------------------------------------------------- visualization

    def plot_timeline(self, ax=None):
        """Gantt-style timeline: one green bar per segment, red bar per gap.

        The x-axis carries pandas Timestamps; matplotlib auto-formats
        the tick labels. Bar widths are `timedelta` values — letting
        matplotlib handle the datetime → axis-unit conversion avoids
        the "accidentally in hours instead of days" trap that a manual
        `.total_seconds() / 86400.0` invites.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                'plot_timeline needs matplotlib; already a core dep for '
                'the monitor but was not found in this env'
            ) from e

        if ax is None:
            _, ax = plt.subplots(figsize=(12, 1.6))

        segments = list(self.segments())
        for seg in segments:
            t0 = seg['begin_time'].iloc[0]
            t1 = seg['end_time'].iloc[-1]
            ax.barh(
                0, t1 - t0, left=t0, height=0.6,
                color='#66bb66', edgecolor='#2e7d32', alpha=0.9,
            )
        # Red bars: the acquisition gap between consecutive segments,
        # drawn with the same height so the gap visibly sits between
        # (not inside) the segments that bracket it.
        for s0, s1 in zip(segments[:-1], segments[1:]):
            t_prev = s0['end_time'].iloc[-1]
            t_next = s1['begin_time'].iloc[0]
            if t_next <= t_prev:
                continue
            ax.barh(
                0, t_next - t_prev, left=t_prev, height=0.6,
                color='#d55', alpha=0.4,
            )

        ax.set_yticks([])
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('time')
        ax.set_title(repr(self))
        ax.figure.autofmt_xdate()
        return ax


# --- CLI ---

def main(argv=None):
    """`python -m dasio.dasdb` — build or update a DASdb catalog.

    Format dispatch is by file extension: .parquet/.parq writes
    parquet, .csv/.txt/no-suffix writes legacy whitespace CSV.
    Default behaviour is incremental: if the target file already
    exists we load it and only open the HDF5 files that are new on
    disk. Pass --overwrite to force a full rescan.

    One-shot format conversion of an existing catalog is a one-liner:
        python -c "from dasio.dasdb import DASdb; \\
            DASdb.from_file('old.csv', system='ASN').to_file('new.parquet')"
    """
    import argparse
    ap = argparse.ArgumentParser(prog='python -m dasio.dasdb')
    ap.add_argument(
        '--from', dest='raw_dir', required=True, type=Path,
        help='directory of raw DAS files to scan',
    )
    ap.add_argument(
        '--dasdb', required=True, type=Path,
        help='catalog path (.parquet recommended for >few-thousand-file '
            'catalogs; .csv stays for legacy / human-inspection use). '
            'Created on first run, updated incrementally on subsequent runs.',
    )
    ap.add_argument(
        '--system',
        choices=['ASN', 'OptaSense', 'APSensing', 'Proc'], required=True,
    )
    ap.add_argument(
        '--overwrite', action='store_true',
        help='rescan raw_dir from scratch, ignoring any existing catalog',
    )
    ap.add_argument(
        '--nworkers', '-nW', type=int, default=1, metavar='N',
        help='parallelize per-file HDF5 metadata reads across N '
            'worker processes (default: 1 = in-process serial). Worth '
            'raising to ~4-8 (or more on a network FS) for thousand-'
            'plus-file catalogs.',
    )
    ap.add_argument(
        '--progress', action='store_true',
        help='show a tqdm progress bar during the metadata scan '
            '(silently disabled if tqdm is not installed)',
    )
    ap.add_argument(
        '--plot', nargs='?', const='', default=None, metavar='PATH',
        help='also save `plot_timeline` as an image; without an '
            'explicit path, writes <dasdb>.png next to the catalog',
    )
    ap.add_argument(
        '--quiet', action='store_true',
        help='skip the end-of-run summary',
    )
    args = ap.parse_args(argv)

    pre_existed = args.dasdb.exists()
    if pre_existed and not args.overwrite:
        db = DASdb.from_file(args.dasdb, system=args.system)
        n_new = db.update_from_dir(
            args.raw_dir, workers=args.nworkers, progress=args.progress,
        )
        action, tail = 'updated', f'+{n_new} new'
    else:
        db = DASdb.from_dir(
            args.raw_dir, args.system,
            workers=args.nworkers, progress=args.progress,
        )
        action = 'rebuilt' if pre_existed else 'created'
        tail = f'{db.n_files} files'

    db.to_file(args.dasdb)

    plot_path = None
    if args.plot is not None:
        plot_path = (
            Path(args.plot) if args.plot
            else args.dasdb.with_suffix('.png')
        )
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        # Force a headless backend for the CLI write path — autodetect
        # blocks indefinitely when there's no $DISPLAY and Qt/Tk
        # initialization stalls. Library callers of plot_timeline
        # keep their own backend choice.
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        ax = db.plot_timeline()
        ax.figure.savefig(plot_path, bbox_inches='tight')
        plt.close(ax.figure)

    if not args.quiet:
        msg = f'[dasdb] {action} {args.dasdb}: {db!r}  {tail}'
        if plot_path is not None:
            msg += f'  [plot: {plot_path}]'
        print(msg)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
