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

    The window is converted with `to_physical()` before writing, so the `unit`
    attr describes the payload rather than asserting a hardcoded
    'microstrain/s' over whatever the catalog happened to hold.

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
    # Event files are consumed as microstrain; readers no longer convert, so
    # scale here rather than writing a unit label the payload does not match.
    d = d.to_physical()

    # Index relative to the ACTUAL returned begin_time (first real sample).
    idx = int(round((event_time - d.begin_time).total_seconds() / d.dt))
    meta = {
        'event_id': event_id,
        'event_time': event_time.isoformat(),
        'event_time_index': idx,
        'time_before': float(before), 'time_after': float(after),
        'latitude': float(row['latitude']), 'longitude': float(row['longitude']),
        'depth_km': float(row['depth_km']), 'magnitude': float(row['magnitude']),
        # Whatever the db actually returned — hardcoding 'microstrain/s' here
        # labeled raw-count catalogs wrongly, and `read_event` trusts this attr.
        'unit': d.units,
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


def main(argv=None):
    import argparse
    from .dasdb import DASdb
    ap = argparse.ArgumentParser(
        prog='python -m dasio.extract_events',
        description='Extract single-event DAS files from continuous data',
    )
    ap.add_argument(
        '--catalog', required=True,
        help=f'event catalog CSV; needs the columns {_REQUIRED}',
    )
    ap.add_argument('--dasdb', required=True, help='catalog of the data to cut')
    ap.add_argument(
        '--to', dest='out_dir', required=True,
        help='output directory, one <event_id>.h5 per event',
    )
    ap.add_argument('--format', default=None)
    ap.add_argument('--before', type=float, default=30.0)
    ap.add_argument('--after', type=float, default=90.0)
    ap.add_argument('--min-ch', type=int, default=0)
    ap.add_argument('--max-ch', type=int, default=None)
    ap.add_argument('--overwrite', action='store_true')
    ap.add_argument('--n-jobs', type=int, default=1)
    ap.add_argument(
        '--progress', action=argparse.BooleanOptionalAction, default=None,
        help='show a tqdm bar. Default: auto — on when stderr is a TTY '
            '(interactive), off when redirected (cron, systemd).',
    )
    args = ap.parse_args(argv)
    catalog = pd.read_csv(args.catalog)
    db = DASdb.from_file(args.dasdb, format=args.format)
    m = extract_events(
        catalog, db, args.out_dir,
        before=args.before, after=args.after,
        min_ch=args.min_ch, max_ch=args.max_ch,
        overwrite=args.overwrite, n_jobs=args.n_jobs, progress=args.progress,
    )
    print(m['status'].value_counts().to_string())


if __name__ == '__main__':
    main()
