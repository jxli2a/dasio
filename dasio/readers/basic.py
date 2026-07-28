"""Minimal self-describing DAS format: `/data` plus one attr per `DASdata` field.

For analysis products — anything past `to_physical()`, filtering or picking —
where Proc is the wrong tool. Proc exists to concatenate and downsample raw
vendor data, so it stores raw payloads and re-derives units from
`/Acquisition_origin`; that inference is only sound while nothing has been
converted, which is why `write_data_proc` now refuses converted input.

Basic infers nothing. Every scalar `DASdata` carries is written verbatim and
read straight back, so `read_basic(write_basic(d))` round-trips exactly —
including `units`, `ch0`, `dch` and `t0_sec`, the fields the other formats drop.

No `raw_meta`: preserving the vendor tree is the Proc cascade's job. `origin`
is kept, since which interrogator recorded the samples stays true however far
the data is processed. `format` doubles as the format stamp — it always reads
'Basic', which is how `detect_format` recognizes the file without guessing
at group names (`/data` alone is ambiguous with ASN and Event).
"""
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np

from ..dasdata import DASdata, DASmeta, normalize_unit
from ..utils import iso_timestamp, parse_iso


# DASdata scalar fields stored verbatim. Handled separately: `data` is the
# payload, `begin_time` / `end_time` need ISO serialization, `raw_meta` is
# deliberately not kept, and `format` is written as the literal 'Basic' rather
# than copied from the DASdata — the file IS Basic whatever it was read from,
# and stamping a source 'Proc' there would make it undetectable.
_ATTRS = ('fs', 'dt', 'dx', 'gauge_length_m', 'origin',
          't0_sec', 'ch0', 'dch', 'units', 'physical_factor')


def _text(v, default: str = 'unknown') -> str:
    """h5py hands back `str` or `bytes` depending on how the attr was written."""
    if v is None:
        return default
    return v.decode('utf-8') if isinstance(v, bytes) else str(v)


def read_basic(
        file: Union[str, Path],
        min_ch: int = 0,
        max_ch: Optional[int] = None,
        first_sample: int = 0,
        n_samples: Optional[int] = None,
    ) -> DASdata:
    """Read a Basic HDF5 file.

    Slicing moves both anchors: `ch0` advances by `min_ch * dch` and
    `begin_time` / `t0_sec` by `first_sample * dt`, so a sliced read still
    reports the true fiber channels and the same seconds frame as the whole.
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

    data = np.ascontiguousarray(data)
    nx, nt = data.shape
    dt = float(attrs['dt'])
    dch = int(attrs.get('dch', 1))
    offset = timedelta(seconds=int(first_sample) * dt)
    begin_time = parse_iso(attrs['begin_time']) + offset
    end_time = begin_time + timedelta(seconds=(nt - 1) * dt) if nt else begin_time
    gauge_length_m = float(attrs.get('gauge_length_m', np.nan))

    return DASdata(
        data=data,
        fs=float(attrs['fs']), dt=dt, nt=nt, nx=nx, dx=float(attrs['dx']),
        begin_time=begin_time, end_time=end_time,
        gauge_length_m=None if np.isnan(gauge_length_m) else gauge_length_m,
        format=_text(attrs.get('format'), 'Basic'),
        origin=_text(attrs.get('origin')),
        raw_meta=None,
        t0_sec=float(attrs.get('t0_sec', 0.0)) + int(first_sample) * dt,
        ch0=int(attrs.get('ch0', 0)) + int(min_ch) * dch,
        dch=dch,
        units=normalize_unit(_text(attrs.get('units'))),
        physical_factor=float(attrs.get('physical_factor', 1.0)),
    )


def write_basic(
        file: Union[str, Path],
        d: DASdata,
        *,
        overwrite: bool = False,
        compress: Optional[str] = 'gzip',
    ) -> Path:
    """Write a DASdata as a Basic HDF5 file. Accepts any units.

    Written to `<file>.lock` and renamed, so a reader never sees a partial
    file. `compress=None` skips gzip when write speed matters more than size.
    """
    file = Path(file)
    if file.exists() and not overwrite:
        raise IOError(f'File {file!s} already exists (overwrite=False)')
    kwargs = {'chunks': True}
    if compress:
        kwargs['compression'] = compress

    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix(file.suffix + '.lock')
    with h5py.File(tmp, 'w') as f:
        dset = f.create_dataset('data', data=d.data, **kwargs)
        dset.attrs['format'] = 'Basic'
        dset.attrs['begin_time'] = iso_timestamp(d.begin_time)
        dset.attrs['end_time'] = iso_timestamp(d.end_time)
        for k in _ATTRS:
            v = getattr(d, k)
            # h5py has no null attr; NaN is the sentinel `read_basic` maps back
            # to None, matching how read_metadata_proc handles a missing dx.
            dset.attrs[k] = np.nan if v is None else v
    tmp.replace(file)
    return file


def read_basic_metadata(file: Union[str, Path]) -> Optional[DASmeta]:
    """Read one Basic file's metadata as a DASmeta dict (no payload load).

    Returns None (with a stderr warning) for files that can't be opened or
    lack the expected `/data` attributes.
    """
    file = Path(file)
    try:
        with h5py.File(file, 'r') as f:
            if 'data' not in f:
                return None
            attrs = f['data'].attrs
            nx, nt = f['data'].shape
            fs = float(attrs['fs'])
            dx = float(attrs.get('dx', np.nan))
            gauge_length_m = float(attrs.get('gauge_length_m', np.nan))
            begin_time = parse_iso(attrs['begin_time'])
            end_time = parse_iso(attrs['end_time'])
    except (OSError, KeyError, ValueError) as e:
        print(f'[dasio.basic] skipping {file}: {e}', file=sys.stderr)
        return None
    return DASmeta(
        file=str(file),
        begin_time=begin_time, end_time=end_time,
        fs=fs, nt=int(nt), nx=int(nx),
        dx=(None if np.isnan(dx) else dx),
        gauge_length_m=(None if np.isnan(gauge_length_m) else gauge_length_m),
        first_sample=0,
    )
