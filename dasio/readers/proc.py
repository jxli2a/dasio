"""Read/write/scan the Proc HDF5 format (Data + Acquisition_origin).

`Proc` is the concatenate / downsample intermediate: a `/Data` payload in
whatever the origin vendor recorded (OptaSense counts, AP Sensing radian/s,
else strain/s), plus a flattened `/Acquisition_origin` group carrying the full
native-vendor metadata tree. `read_data_proc` / `write_data_proc` are the
in-memory reader / writer; `read_metadata_proc` supplies the per-file catalog
row so Proc is a first-class format for `DASdb.from_dir` and `read_das_data`.

Units are re-derived from the origin on read rather than stored, which is only
sound while the payload is raw — so `write_data_proc` refuses converted data
and `dasio.write_basic` is where anything past `to_physical()` belongs.

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

from .detector import detect_origin
from ..dasdata import DASdata, DASmeta
from ..utils import iso_timestamp, parse_iso


# What each origin's raw payload is stored as. ASN, Sintela and Unknown all
# fall through to strain/s: files written by `write_data_proc` without raw_meta
# carry no origin marker, and legacy readFile_HDF likewise treated everything
# non-OptaSense / non-AP Sensing as already-strain.
_ORIGIN_UNITS = {"OptaSense": "count", "APSensing": "radian/s"}
_DEFAULT_UNITS = "strain/s"

# Proc re-derives units from /Acquisition_origin on read, so a payload is only
# storable here in the units its origin implies. "unknown" passes — it makes no
# claim, and synthetic payloads legitimately carry it. Converted data does not:
# it would read back mislabeled and pick up a spurious 1e6 in `to_physical`.
_WRITABLE_UNITS = frozenset(_ORIGIN_UNITS.values()) | {_DEFAULT_UNITS, "unknown"}


def _gauge_length(f, data_attrs) -> Optional[float]:
    """`GaugeLength` from /Data, falling back to /Acquisition_origin.

    `write_data_proc` puts it on /Data, but legacy Desample_DAS.py wrote it
    only into the flattened origin group — so every iceland_quantx window came
    back with `gauge_length_m=None` despite the value (102.095 m) sitting in
    the file. Both places are checked because both layouts are in the archive.
    """
    if 'GaugeLength' in data_attrs:
        return float(data_attrs['GaugeLength'])
    origin = f.get('Acquisition_origin')
    if origin is not None and 'GaugeLength' in origin.attrs:
        return float(origin.attrs['GaugeLength'])
    return None


def read_data_proc(
        file: Union[str, Path],
        min_ch: int = 0,
        max_ch=None,
        first_sample: int = 0,
        n_samples: Optional[int] = None,
    ) -> DASdata:
    """Read a Proc HDF5 file and return a DASdata in its stored units.

    Payload units follow the origin recorded in /Acquisition_origin:
    OptaSense-origin files hold raw phase counts, AP Sensing radian/s,
    everything else strain/s. Call `.to_physical()` for microstrain,
    exactly as with the raw vendor readers — this reader used to apply
    the conversion itself behind a `convert=True` flag, which meant the
    vendor factor was chosen in two places and a Proc read came back in a
    different unit from every other read.

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
        origin = detect_origin(f)
        gauge_length_m = _gauge_length(f, attrs)
        # Carried forward, already flattened, so it writes back unchanged.
        # Dropping it made a Proc -> Proc desample emit an origin-less file,
        # which the next generation reads as 'Unknown' and mis-infers units.
        origin_grp = f.get('Acquisition_origin')
        raw_meta = dict(origin_grp.attrs) if origin_grp is not None else None

    # Contiguous: the time-first branch above transposes, and `astype` would
    # keep that F layout (order='K'). See `signal.bandpass2d`.
    data = np.ascontiguousarray(data, dtype=np.float32)
    units = _ORIGIN_UNITS.get(origin, _DEFAULT_UNITS)

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
        gauge_length_m=gauge_length_m,
        # Proc is the format; the vendor beneath it comes from
        # /Acquisition_origin and is what the conversion factor keys on.
        format='Proc', origin=origin,
        raw_meta=raw_meta,
        units=units,
        channels={'raw': int(min_ch) + np.arange(int(max_ch - min_ch))},
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
            gauge_length_m = _gauge_length(f, attrs)
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

    Raises `ValueError` on a converted payload. Proc is the concatenate /
    downsample intermediate: it stores raw vendor units and `read_data_proc`
    re-derives them from the origin, which is only sound while nothing has been
    scaled. Use `write_basic` for anything past `to_physical()`.
    """
    if d.units not in _WRITABLE_UNITS:
        raise ValueError(
            f"write_data_proc: Proc stores raw vendor payloads, but units="
            f"{d.units!r} has already been converted. On read the units would "
            f"be re-derived from /Acquisition_origin and come back wrong. Use "
            f"dasio.write_basic for converted data."
        )
    file = Path(file)
    tmp = file.with_suffix(file.suffix + '.lock')
    # The legacy Desample_DAS.py wrote libver='v200', which h5py rejects here
    # (HDF5 1.14); ('earliest', 'latest') is the portable spelling and lets the
    # newest object-header format hold the large flattened Acquisition_origin.
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
