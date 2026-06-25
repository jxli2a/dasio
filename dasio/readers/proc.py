"""Read/write/scan the Proc HDF5 format (Data + Acquisition_origin).

`Proc` is our processed data format: a `/Data` payload in strain (or
strain-rate for legacy OptaSense) units, plus a flattened
`/Acquisition_origin` group carrying the full native-vendor metadata
tree. `read_data_proc` / `write_data_proc` are the in-memory reader /
writer; `read_metadata_proc` supplies the per-file catalog row so
Proc is a first-class system for `DASdb.from_dir` and `read_das_data`.

Preserves the on-disk attr names (`nCh`, `dCh`, `startTime`, `endTime`,
`GaugeLength`) that existing external readers use. In-memory DASdata
uses the cleaner snake_case names (`nx`, `dx`, `begin_time`,
`end_time`, `gauge_length_m`).
"""
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np

from .apsensing import apsensing_radians2strain_factor
from .detector import detect_origin
from .optasense import optasense_count2strain_factor
from ..dasdata import DASdata, DASmeta
from ..utils import iso_timestamp, parse_iso


def read_data_proc(
        file: Union[str, Path],
        min_ch: int = 0,
        max_ch=None,
        first_sample: int = 0,
        n_samples: Optional[int] = None,
        convert: bool = True,
    ) -> DASdata:
    """Read a Proc HDF5 file and return a DASdata.

    When `convert=True` we bring the payload into microstrain / sec —
    the unit the monitor and viewers expect. ASN-origin Proc files
    already store strain and just need a 1e6 factor; OptaSense-origin
    Proc files store raw phase counts and need `count2strain * 1e6`,
    computed from the on-disk /Acquisition_origin attrs. The origin
    system is detected from /Acquisition_origin so callers always get
    an accurate `system` field.

    ``first_sample`` / ``n_samples`` are accepted for signature parity
    with the raw readers (`read_asn_raw`, `read_optasense_raw`) — they
    let `desample_window` treat all three vendors uniformly. Proc
    files have no intra-file time gaps, so `first_sample` is a plain
    slice offset into the time axis (default 0 = read from the start).
    """
    file = Path(file)
    with h5py.File(file, 'r') as f:
        dset = f['Data']
        attrs = dict(dset.attrs)
        nCh = int(attrs['nCh'])
        if max_ch is None:
            max_ch = nCh
        # Detect layout the way legacy DASutils._read_data_proc did: if
        # the last-axis length matches the nt attribute, the file is
        # (channel, time); otherwise it's (time, channel) and we slice
        # then transpose. Older Proc files from DAS-Utilities
        # Desample_DAS.py occasionally landed time-first.
        nt_total = int(attrs['nt'])
        t_end = nt_total if n_samples is None else first_sample + n_samples
        if dset.shape[-1] == nt_total:
            data = dset[int(min_ch):int(max_ch), int(first_sample):int(t_end)]
        else:
            data = dset[int(first_sample):int(t_end), int(min_ch):int(max_ch)].T
        system = detect_origin(f)
        factor = 1.0
        if convert:
            if system == 'OptaSense':
                factor = optasense_count2strain_factor(f) * 1e6
            elif system == 'APSensing':
                # Proc payload is already radians/sec (saved by
                # Desample_DAS); convert to strain/sec → microstrain/sec.
                factor = apsensing_radians2strain_factor(f) * 1e6
            else:
                # ASN, Sintela, or Unknown (files produced by
                # write_data_proc without raw_meta carry no system
                # marker in /Acquisition_origin). Legacy readFile_HDF
                # multiplied by 1e6 unconditionally for everything that
                # wasn't OptaSense / APSensing raw, so keep that as
                # the default.
                factor = 1e6

    data = data.astype(np.float32, copy=False)
    if convert and factor != 1.0:
        data = data * np.float32(factor)

    if convert:
        # TODO: verify against a real OptaSense Proc file — the module docstring
        # mentions "strain-rate for legacy OptaSense", so confirm whether
        # OptaSense-origin Proc (convert=True) stores strain or strain-rate.
        # Current assumption: "microstrain" (strain scale, not strain-rate).
        units = "microstrain" if system == "OptaSense" else "microstrain/s"
    else:
        units = {"OptaSense": "count", "APSensing": "radian/s"}.get(system, "strain/s")

    nt_out = data.shape[1]
    dt = float(attrs['dt'])
    begin_time = parse_iso(attrs['startTime']) + timedelta(seconds=first_sample * dt)
    end_time = begin_time + timedelta(seconds=(nt_out - 1) * dt) if nt_out else begin_time
    return DASdata(
        data=data,
        fs=float(attrs['fs']),
        dt=dt,
        nt=nt_out,
        nx=int(max_ch - min_ch),
        dx=float(attrs.get('dCh', 0.0)),
        begin_time=begin_time,
        end_time=end_time,
        gauge_length_m=float(attrs['GaugeLength']) if 'GaugeLength' in attrs else None,
        system=system,
        raw_meta=None,
        units=units,
    )


def read_metadata_proc(file: Union[str, Path]) -> Optional[DASmeta]:
    """Read one Proc file's metadata as a DASmeta dict (no payload load).

    Returns None (with a stderr warning) for files that can't be
    opened or lack the expected `/Data` attributes.
    """
    file = Path(file)
    try:
        with h5py.File(file, 'r') as f:
            if 'Data' not in f:
                return None
            attrs = f['Data'].attrs
            fs = float(attrs['fs'])
            nt = int(attrs['nt'])
            nx = int(attrs['nCh'])
            dx = float(attrs.get('dCh', np.nan))
            gauge_length_m = (
                float(attrs['GaugeLength'])
                if 'GaugeLength' in attrs else None
            )
            begin_time = parse_iso(attrs['startTime'])
            end_time = parse_iso(attrs['endTime'])
    except (OSError, KeyError, ValueError) as e:
        print(f'[dasio.proc] skipping {file}: {e}', file=sys.stderr)
        return None
    return DASmeta(
        file=str(file),
        begin_time=begin_time, end_time=end_time,
        fs=fs, nt=nt, nx=nx,
        dx=(None if np.isnan(dx) else dx),
        gauge_length_m=gauge_length_m,
        first_sample=0,
    )



def _flatten_dict(d: dict, parent_key: str = '', sep: str = '.') -> dict:
    """Flatten nested dict with dot-separated keys. Mirrors Desample_DAS.flatten_dict."""
    items = {}
    for k, v in d.items():
        full = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(_flatten_dict(v, full, sep=sep))
        elif isinstance(v, str):
            items[full] = v.encode('utf-8')
        elif isinstance(v, (list, np.ndarray)) and all(
            isinstance(x, (str, bytes, np.str_, np.bytes_)) for x in np.asarray(v).ravel()
        ):
            # String array, or any empty array (vacuous all()) → list of bytes.
            # Legacy treats empty arrays the same way, which is why fields
            # like processingChain.step-0.errors land in /Acquisition_origin.
            items[full] = [
                (x.encode('utf-8') if isinstance(x, str) else bytes(x))
                for x in np.asarray(v).ravel()
            ]
        else:
            items[full] = v
    return items


def write_data_proc(
        file: Union[str, Path],
        d: DASdata,
        compress: str = 'gzip',
    ) -> None:
    """Write a DASdata as a Proc HDF5 file.

    Writes /Data + /Acquisition_origin, gzip compression, .lock-then-rename.
    On-disk attrs use nCh/dCh names for backward compat with existing readers.
    """
    file = Path(file)
    tmp = file.with_suffix(file.suffix + '.lock')
    # libver=('earliest', 'latest') lifts the 64 KB object-header cap
    # (which otherwise overflows when the full Acquisition_origin
    # flattened tree with 300+ attrs is written) while staying portable
    # across h5py builds. The legacy Desample_DAS.py value 'v200' is
    # only understood by HDF5 ≥ 2.0; h5py 3.15 / HDF5 1.14.6 accepts
    # only earliest, latest, v108, v110, v112, v114.
    with h5py.File(tmp, 'w', libver=('earliest', 'latest')) as hf:
        ds = hf.create_dataset('Data', data=d.data, chunks=True, compression=compress)
        ds.attrs['fs'] = d.fs
        ds.attrs['dt'] = d.dt
        ds.attrs['nt'] = d.nt
        ds.attrs['nCh'] = d.nx
        ds.attrs['dCh'] = d.dx
        ds.attrs['startTime'] = iso_timestamp(d.begin_time)
        ds.attrs['endTime'] = iso_timestamp(d.end_time)
        if d.gauge_length_m is not None:
            ds.attrs['GaugeLength'] = d.gauge_length_m
        # Scalar float32 placeholder; matches legacy Desample_DAS.py exactly
        # (it calls create_dataset('Acquisition_origin', ()) which defaults
        # to float32 zero).
        acq = hf.create_dataset('Acquisition_origin', data=np.float32(0.0))
        if d.raw_meta:
            for k, v in _flatten_dict(d.raw_meta).items():
                # Skip arrays (legacy does the same) — keeps the object header
                # under the HDF5 limit and matches the 300-ish attr set size.
                if isinstance(v, np.ndarray):
                    continue
                acq.attrs[k] = v
    tmp.replace(file)
