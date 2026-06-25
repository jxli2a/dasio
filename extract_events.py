"""Extract one HDF5 event-data file per catalog event.

For each event, read [event_time - before, event_time + after] from
continuous DAS via DASdb and write <event_id>.h5 in the existing
scalar-attr event-data layout (reusing readers.event.write_event).
"""
from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

from .readers.event import write_event


def _maybe_bar(iterable, total, progress):
    """Wrap an iterable in a tqdm bar when progress is on. Silently
    skipped if tqdm is unavailable (it is a declared dependency)."""
    if not progress:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable
    return tqdm(iterable, total=total, unit='event', desc='extract')


def extract_event(
        row,
        dasdb,
        out_dir,
        before,
        after,
        *,
        overwrite=False,
        min_ch=0,
        max_ch=None,
    ):
    """Read one event's window and write <event_id>.h5.

    Returns a status dict: {event_id, file, status, event_time_index}.
    status is one of 'extract' | 'skip' | 'fail'.
    """
    out_dir = Path(out_dir)
    event_id = str(row['event_id'])
    fpath = out_dir / f'{event_id}.h5'
    if fpath.exists() and not overwrite:
        return {'event_id': event_id, 'file': str(fpath), 'status': 'skip'}

    event_time = pd.Timestamp(row['event_time']).to_pydatetime()
    begin = event_time - timedelta(seconds=before)
    end = event_time + timedelta(seconds=after)
    try:
        d = dasdb.read(begin, end, min_ch=min_ch, max_ch=max_ch, fill_gap=True)
    except RuntimeError:
        return {'event_id': event_id, 'file': str(fpath), 'status': 'fail'}

    # Index relative to the ACTUAL returned begin_time (first real sample).
    idx = int(round((event_time - d.begin_time).total_seconds() / d.dt))
    meta = {
        'event_id': event_id,
        'event_time': event_time.isoformat(),
        'event_time_index': idx,
        'time_before': float(before), 'time_after': float(after),
        'latitude': float(row['latitude']), 'longitude': float(row['longitude']),
        'depth_km': float(row['depth_km']), 'magnitude': float(row['magnitude']),
        'unit': 'microstrain/s',
    }
    for opt in ('magnitude_type', 'source', 'time_reference'):
        if opt in row and pd.notna(row[opt]):
            meta[opt] = row[opt]

    out_dir.mkdir(parents=True, exist_ok=True)
    write_event(fpath, d, meta, overwrite=overwrite)
    return {'event_id': event_id, 'file': str(fpath), 'status': 'extract',
            'event_time_index': idx}


_REQUIRED = ('event_id', 'event_time', 'latitude', 'longitude', 'depth_km', 'magnitude')


def extract_events(
        catalog,
        dasdb,
        out_dir,
        *,
        before=30.0,
        after=90.0,
        min_ch=0,
        max_ch=None,
        overwrite=False,
        n_jobs=1,
        progress=None,
    ):
    """Extract one <event_id>.h5 per catalog row. Returns a status DataFrame.

    progress: None (default) shows a tqdm bar only when stderr is a TTY
    (interactive); True/False forces it on/off.
    """
    missing = [c for c in _REQUIRED if c not in catalog.columns]
    if missing:
        raise ValueError(f'catalog missing required columns {missing}')
    rows = catalog.to_dict('records')
    if progress is None:
        progress = sys.stderr.isatty()

    def _do(r):
        return extract_event(r, dasdb, out_dir, before, after,
                             overwrite=overwrite, min_ch=min_ch, max_ch=max_ch)

    if n_jobs and n_jobs > 1:
        from joblib import Parallel, delayed
        # return_as='generator' yields results as they COMPLETE, so the
        # bar tracks real progress rather than dispatch order.
        gen = Parallel(n_jobs=n_jobs, return_as='generator')(
            delayed(_do)(r) for r in rows)
        results = list(_maybe_bar(gen, len(rows), progress))
    else:
        results = [_do(r) for r in _maybe_bar(rows, len(rows), progress)]
    return pd.DataFrame(results)


def main():
    import argparse
    from .dasdb import DASdb
    p = argparse.ArgumentParser(
        description='Extract single-event DAS files from continuous data',
    )
    p.add_argument('catalog_csv')
    p.add_argument('dasdb')
    p.add_argument('out_dir')
    p.add_argument('--system', default=None)
    p.add_argument('--before', type=float, default=30.0)
    p.add_argument('--after', type=float, default=90.0)
    p.add_argument('--min-ch', type=int, default=0)
    p.add_argument('--max-ch', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    p.add_argument('--n-jobs', type=int, default=1)
    p.add_argument(
        '--progress', action=argparse.BooleanOptionalAction, default=None,
        help='show a tqdm bar. Default: auto — on when stderr is a TTY '
            '(interactive), off when redirected (cron, systemd).',
    )
    a = p.parse_args()
    cat = pd.read_csv(a.catalog_csv)
    db = DASdb.from_file(a.dasdb, system=a.system)
    m = extract_events(
        cat, db, a.out_dir,
        before=a.before, after=a.after,
        min_ch=a.min_ch, max_ch=a.max_ch,
        overwrite=a.overwrite, n_jobs=a.n_jobs, progress=a.progress,
    )
    print(m['status'].value_counts().to_string())


if __name__ == '__main__':
    main()
