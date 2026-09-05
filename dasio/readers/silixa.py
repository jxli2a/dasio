"""Reader for Silixa iDAS HDF5 files.

One root dataset `/Acoustic`, shape (nt, nx) int16, with every acquisition
parameter stored as an attribute on it. The payload is the iDAS "Differential"
output: counts proportional to strain rate. `silixa_count2strainrate_factor`
is the counts -> strain/s scalar that `DASFile.read` attaches as
`physical_factor`.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np

from ..dasdata import DASdata, DASmeta


def silixa_count2strainrate_factor(f: h5py.File) -> float:
    """One count is Unit Calibration (nm) / 2^13 of fibre extension per sample,
    over the gauge length. Reads the raw attrs or the Proc-flattened copy."""
    attrs = f['Acoustic'].attrs if 'Acoustic' in f else f['Acquisition_origin'].attrs
    try:
        nm = float(attrs['Unit Calibration (nm)'])
        fs = float(attrs['SamplingFrequency[Hz]'])
        gauge = float(attrs['GaugeLength'])
    except KeyError as e:
        print(f'WARNING! silixa_count2strainrate_factor: missing {e}; returning 1.0',
              file=sys.stderr)
        return 1.0
    return nm * 1e-9 / 8192 * fs / gauge


def _text(v) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def _begin_time(attrs) -> datetime:
    iso = attrs.get('ISO8601 Timestamp')
    if iso:
        return datetime.fromisoformat(_text(iso)).astimezone(timezone.utc)
    # older files only carry "12/11/2020 00:00:32.602 (UTC)", day first;
    # the CPU clock stands in when GPS was not locked
    s = _text(attrs.get('GPSTimeStamp') or attrs['CPUTimeStamp']).replace(' (UTC)', '')
    return datetime.strptime(s, '%d/%m/%Y %H:%M:%S.%f').replace(tzinfo=timezone.utc)


def _dx(attrs) -> float:
    # optical spacing: header resolution times the fibre length multiplier
    return float(attrs['SpatialResolution[m]']) * float(attrs['Fibre Length Multiplier'])


def read_silixa_raw(
        file: Union[str, Path],
        min_ch: int = 0,
        max_ch: Optional[int] = None,
        first_sample: int = 0,
        n_samples: Optional[int] = None,
    ) -> DASdata:
    """Read one Silixa iDAS file as raw counts, (nx, nt) float32."""
    with h5py.File(file, 'r') as f:
        ds = f['Acoustic']
        attrs = dict(ds.attrs)
        total_nt, total_nx = ds.shape
        if max_ch is None:
            max_ch = total_nx
        if n_samples is None:
            n_samples = total_nt - first_sample
        raw = ds[first_sample:first_sample + n_samples, min_ch:max_ch]

    data = np.ascontiguousarray(raw.T, dtype=np.float32)
    nx, nt = data.shape
    fs = float(attrs['SamplingFrequency[Hz]'])
    dt = 1.0 / fs
    begin_time = _begin_time(attrs) + timedelta(seconds=first_sample * dt)
    return DASdata(
        data=data,
        channels={'raw': int(min_ch) + np.arange(nx)},
        fs=fs, dt=dt, nt=nt, nx=nx, dx=_dx(attrs),
        begin_time=begin_time,
        end_time=begin_time + timedelta(seconds=(nt - 1) * dt),
        gauge_length_m=float(attrs['GaugeLength']),
        format='Silixa', origin='Silixa',
        raw_meta=attrs,
        units='count/s',
    )


def read_silixa_metadata(file: Union[str, Path]) -> Optional[DASmeta]:
    """One DASmeta row per file; None (with a stderr note) if it cannot be read."""
    file = Path(file)
    try:
        with h5py.File(file, 'r') as f:
            ds = f['Acoustic']
            attrs = dict(ds.attrs)
            nt, nx = ds.shape
    except (OSError, KeyError) as e:
        print(f'[dasio.silixa] skipping {file}: {e}', file=sys.stderr)
        return None
    fs = float(attrs['SamplingFrequency[Hz]'])
    begin_time = _begin_time(attrs)
    return DASmeta(
        file=str(file),
        begin_time=begin_time,
        end_time=begin_time + timedelta(seconds=(nt - 1) / fs),
        fs=fs, nt=int(nt), nx=int(nx),
        dx=_dx(attrs), gauge_length_m=float(attrs['GaugeLength']),
        first_sample=0,
    )
