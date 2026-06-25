"""Reader + writer for DAS event-data HDF5 files.

Event-data layout — distinct from Proc. `/data` has shape (nx, nt)
float32, with attrs (tab-separated columns below):

    begin_time / end_time	ISO-8601 absolute window bounds
    dt_s	sample period (seconds)
    dx_m	channel spacing (m)
    event_id	catalog ID (e.g. 'nc71113514')
    event_time	event origin time (ISO-8601)
    event_time_index	sample index where the event origin lands
    time_before / time_after	pre / post event padding in seconds
    magnitude / magnitude_type	from the source catalog
    latitude / longitude / depth_km	hypocenter
    source	catalog source (e.g. 'ncdd')
    unit	payload units, typically 'microstrain/s'

The catalog convention is "-time_before s before event, +time_after s
after". This reader sets `DASdata.t0_sec = -event_time_index · dt_s`
so the in-memory seconds frame has t = 0 at the event origin —
`d.truncate(t_range=(-2, 10))` then reads as "2 s before to 10 s
after the event." `begin_time` / `end_time` stay absolute.

`write_event` writes the same layout, schema-compatible with
`das_utilities.event_data.write_event_data_HDF5` so files written by
either function round-trip through `read_event`.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import h5py
import numpy as np

from ..dasdata import DASdata, DASmeta


# Structural attrs live on DASdata fields, not raw_meta.
_STRUCTURAL_ATTRS = ('begin_time', 'end_time', 'dt_s', 'dx_m')

REQUIRED_EVENT_ATTRS = _STRUCTURAL_ATTRS + (
    'event_id', 'event_time', 'event_time_index',
    'time_before', 'time_after',
    'latitude', 'longitude', 'depth_km',
    'magnitude', 'unit',
)

OPTIONAL_EVENT_ATTRS = (
    'magnitude_type', 'source', 'version',
)


def read_event(
        file: Union[str, Path],
        min_ch: int = 0,
        max_ch: Optional[int] = None,
        first_sample: int = 0,
        n_samples: Optional[int] = None,
    ) -> DASdata:
    """Read one DAS event-data file and return a `DASdata`.

    `t0_sec` is set so the event origin time is at t = 0 in the
    seconds frame. The full event-attr set (magnitude, lat/lon,
    depth, etc.) lands in `raw_meta` for downstream analysis.
    """
    file = Path(file)
    with h5py.File(file, 'r') as f:
        dset = f['data']
        attrs = dict(dset.attrs)
        total_nx, total_nt = dset.shape
        if max_ch is None:
            max_ch = total_nx
        if n_samples is None:
            n_samples = total_nt - first_sample
        data = dset[
            int(min_ch):int(max_ch),
            int(first_sample):int(first_sample) + int(n_samples),
        ]

    data = np.ascontiguousarray(data, dtype=np.float32)
    nx, nt = data.shape

    fs = 1.0 / float(attrs['dt_s'])
    dt = float(attrs['dt_s'])
    dx = float(attrs['dx_m'])
    begin_time_file = datetime.fromisoformat(attrs['begin_time'])
    # Slice may shift begin_time forward by `first_sample * dt`.
    begin_time = begin_time_file + timedelta(seconds=int(first_sample) * dt)
    end_time = (begin_time + timedelta(seconds=(nt - 1) * dt)
                if nt else begin_time)

    # event_time_index is the sample where origin lands in the FILE's
    # original sample 0. After `first_sample` shift, the event lands
    # at index `(event_time_index - first_sample)` in the returned
    # array. t0_sec = − that index · dt.
    event_idx_file = int(attrs.get('event_time_index', 0))
    event_idx_view = event_idx_file - int(first_sample)
    t0_sec = -event_idx_view * dt

    # Catalog + foreign attrs land in raw_meta; structural attrs already on DASdata.
    raw_meta = {k: v for k, v in attrs.items() if k not in _STRUCTURAL_ATTRS}

    return DASdata(
        data=data,
        fs=fs, dt=dt, nt=nt, nx=nx, dx=dx,
        begin_time=begin_time, end_time=end_time,
        gauge_length_m=None, system='Event',
        raw_meta=raw_meta, t0_sec=t0_sec,
    )


def read_event_metadata(file: Union[str, Path]) -> Optional[DASmeta]:
    """Read one event-data file's metadata as a `DASmeta` dict (no payload).

    Returns None (with a stderr warning) for files that can't be
    opened or lack the expected `/data` attributes.
    """
    import sys
    file = Path(file)
    try:
        with h5py.File(file, 'r') as f:
            if 'data' not in f:
                return None
            attrs = f['data'].attrs
            dt = float(attrs['dt_s'])
            fs = 1.0 / dt
            nx, nt = f['data'].shape
            dx = float(attrs.get('dx_m', np.nan))
            begin_time = datetime.fromisoformat(attrs['begin_time'])
            end_time = datetime.fromisoformat(attrs['end_time'])
    except (OSError, KeyError, ValueError) as e:
        print(f'[dasio.event] skipping {file}: {e}', file=sys.stderr)
        return None
    return DASmeta(
        file=str(file),
        begin_time=begin_time, end_time=end_time,
        fs=fs, nt=int(nt), nx=int(nx),
        dx=(None if np.isnan(dx) else dx),
        gauge_length_m=None,
        first_sample=0,
    )


def write_event(
        file: Union[str, Path],
        d: DASdata,
        event_meta: Optional[Mapping[str, Any]] = None,
        *,
        overwrite: bool = False,
        compress: bool = True,
        shuffle: bool = True,
    ) -> Path:
    """Write a DAS event-data HDF5.

    Structural attrs come from `d`; catalog attrs from `event_meta`
    (falls back to `d.raw_meta` when None, so `write_event(p, read_event(p))`
    round-trips). Schema is documented at the top of this module.
    """
    file = Path(file)
    if file.exists() and not overwrite:
        raise IOError(f'File {file!s} already exists (overwrite=False)')

    arr = np.ascontiguousarray(d.data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f'd.data must be 2-D (nx, nt); got {arr.shape}')

    # DASdata fields override event_meta — guards against stale dt_s/dx_m.
    catalog = dict(event_meta if event_meta is not None else (d.raw_meta or {}))
    structural = dict(zip(_STRUCTURAL_ATTRS, (
        d.begin_time.isoformat(timespec='milliseconds'),
        d.end_time.isoformat(timespec='milliseconds'),
        float(d.dt),
        float(d.dx) if d.dx is not None else float('nan'),
    )))
    meta = {**catalog, **structural}
    missing = [k for k in REQUIRED_EVENT_ATTRS if k not in meta]
    if missing:
        raise ValueError(f'event_meta missing required attrs {missing}')

    kwargs = {'chunks': True}
    if compress:
        kwargs.update(compression='gzip', compression_opts=9)
    if shuffle:
        kwargs['shuffle'] = True

    # Write known schema first, then any foreign keys, for deterministic on-disk order.
    known = REQUIRED_EVENT_ATTRS + OPTIONAL_EVENT_ATTRS
    ordered_keys = [k for k in known if k in meta] + [k for k in meta if k not in known]

    file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(file, 'w') as f:
        dset = f.create_dataset('data', data=arr, **kwargs)
        for k in ordered_keys:
            v = meta[k]
            dset.attrs[k] = v.isoformat() if isinstance(v, datetime) else v
    return file
