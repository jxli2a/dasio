"""Reader for raw 100 Hz ASN/OptoDAS HDF5 files (Iceland native format)."""
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np

from ..dasdata import DASdata, DASmeta


# Filename conventions used by the ASN YYYYMMDD/HHMMSS.hdf5 layout.
# Kept here so catalog code (db.py) and any external caller can re-use
# them without reaching into the HDF5 reader.
_ASN_FILE_RE  = re.compile(r'(\d{6})\.hdf5$')
_ASN_DIR_RE   = re.compile(r'^(\d{8})$')


# ----- skip-logging policy ----------------------------------------------
# Catalog rebuilds + desample crawls can hit hundreds-to-thousands of
# truncated / zero-byte / partially-written HDF5 files. Logging every
# one of them spams desample.log and webview.log to uselessness. Policy:
#   - print the first N skips verbatim (so the operator sees what's
#     going wrong)
#   - then suppress; emit one summary line every M additional skips
#   - set env DASIO_ASN_QUIET_SKIPS=0 to log all skips verbatim
_SKIP_VERBOSE_FIRST = 5
_SKIP_SUMMARY_EVERY = 100
_skip_count = 0


def _log_skip(file, exc):
    """Bounded-rate stderr log for HDF5 files that won't open."""
    global _skip_count
    _skip_count += 1
    quiet = os.environ.get('DASIO_ASN_QUIET_SKIPS', '1') != '0'
    if (not quiet) or _skip_count <= _SKIP_VERBOSE_FIRST:
        print(f'[dasio.asn] skipping {file}: {exc}', file=sys.stderr)
    elif _skip_count % _SKIP_SUMMARY_EVERY == 0:
        print(f'[dasio.asn] {_skip_count} files skipped so far '
              f'(use DASIO_ASN_QUIET_SKIPS=0 to log each)',
              file=sys.stderr)


def asn_filename_time(file: Union[str, Path]) -> Optional[datetime]:
    """Cheap, rough begin_time from the YYYYMMDD/HHMMSS.hdf5 convention.

    Utility for quick timestamp estimates without opening the file —
    logging, filename-based sorting, progress reporting. Returns None
    if the path doesn't match the layout. NOT a correctness fallback:
    catalog construction always reads the authoritative time from the
    HDF5 metadata; a file that fails that read is considered garbage
    and skipped.
    """
    file = Path(file)
    for parent in file.parents:
        dm = _ASN_DIR_RE.match(parent.name)
        if dm:
            fm = _ASN_FILE_RE.search(file.name)
            if not fm:
                return None
            day = datetime.strptime(dm.group(1), '%Y%m%d').replace(tzinfo=timezone.utc)
            hms = fm.group(1)
            return day + timedelta(
                hours=int(hms[0:2]),
                minutes=int(hms[2:4]),
                seconds=int(hms[4:6]),
            )
    return None


def _load_h5_tree(group, skip=('data',)) -> dict:
    """Walk an HDF5 group into a nested dict, mirroring DASutils.h5pydict.load_dict.

    Scalar string datasets are decoded; numeric scalars are returned as Python
    scalars; arrays stay as numpy arrays. The top-level `data` dataset (raw
    strain) is skipped by default so the metadata stays small.
    """
    out = {}
    for key, val in group.items():
        if key in skip:
            continue
        if isinstance(val, h5py.Group):
            out[key] = _load_h5_tree(val, skip=skip)
        else:
            dtype_char = val.dtype.char
            if dtype_char in 'SO':
                if val.shape == ():
                    raw = val[()]
                    if raw == b'None':
                        out[key] = None
                    else:
                        try:
                            out[key] = raw.decode('utf-8')
                        except Exception:
                            out[key] = raw
                else:
                    out[key] = np.asarray(val).astype(str)
            elif val.shape == ():
                out[key] = val[()].item()
            else:
                out[key] = np.asarray(val[()])
    return out


def read_asn_raw(
        file: Union[str, Path],
        min_ch: int = 0,
        max_ch: Optional[int] = None,
        first_sample: int = 0,
        n_samples: Optional[int] = None,
    ) -> DASdata:
    """Read one raw ASN HDF5 file, apply sensitivity scaling + polarity flip.

    Returns data in strain units (sign-flipped relative to raw). On-disk
    layout is (n_time_samples, n_channels); output DASdata is (nx, nt).
    """
    file = Path(file)
    with h5py.File(file, 'r') as f:
        dset = f['data']
        total_nt, total_nx = dset.shape
        if max_ch is None:
            max_ch = total_nx
        if n_samples is None:
            n_samples = total_nt - first_sample
        raw = dset[first_sample:first_sample + n_samples, min_ch:max_ch]

        header = f['header']
        sens_arr = np.asarray(header['sensitivities'])
        data_scale = float(header['dataScale'][()])
        # header/dt reports the acquisition-side rate (e.g. 500 Hz for Santorini),
        # NOT the rate of the data actually written to disk after any
        # TemporalDecimation step in the processingChain. The true per-sample
        # time step is recorded at header/dimensionRanges/dimension0/unitScale
        # and accounts for all processing-chain decimations. Iceland has
        # unitScale == header.dt (no late decimation) so both sources agree;
        # Santorini has unitScale = 5 * header.dt.
        unit_scale = f.get('header/dimensionRanges/dimension0/unitScale')
        if unit_scale is not None:
            dt = float(unit_scale[()])
        else:
            dt = float(header['dt'][()])
        # dx is per raw-channel spacing; the on-disk channel axis can be a
        # decimated subset identified by the `channels` index array.
        # True inter-channel spacing is diff(channels)[0] * dx.
        raw_dx = float(header['dx'][()])
        channels = np.asarray(header['channels'])
        ch_stride = int(channels[1] - channels[0]) if channels.size > 1 else 1
        dx = ch_stride * raw_dx
        gauge_length_m = float(header['gaugeLength'][()])
        # /timing/sampleSkew carries a sub-sample offset the FPGA applies
        # between acquisition and the timestamp it writes to /header/time.
        # Catalog scan applies it (read_asn_metadata) so the raw reader
        # must too — otherwise a single file's begin_time disagrees
        # between the raw read and the DASdb row by a few microseconds.
        try:
            sample_skew = float(f['timing/sampleSkew'][()])
        except KeyError:
            sample_skew = 0.0
        file_start = float(header['time'][()]) + sample_skew
        # Full nested metadata tree (every group/scalar except the raw data).
        # Flattened downstream into /Acquisition_origin attrs for parity with
        # Desample_DAS.py, which writes the same 300+ keys.
        raw_meta = _load_h5_tree(f, skip=('data',))

    # Per-channel scaling + polarity flip. Iceland writes a scalar in
    # /header/sensitivities; Santorini writes a per-channel array of
    # length total_nx. Broadcast either way: scale has shape () or
    # (nx_slice,), both broadcast against raw (n_samples, nx_slice).
    if sens_arr.size == total_nx:
        sens = sens_arr[min_ch:max_ch].astype(np.float32, copy=False)
    else:
        sens = np.float32(sens_arr.flat[0])
    scale = np.float32(-data_scale) / sens
    data = (raw.astype(np.float32, copy=False) * scale).T  # (nx, nt)

    nx = data.shape[0]
    nt = data.shape[1]
    fs = 1.0 / dt
    begin_time = datetime.fromtimestamp(file_start + first_sample * dt, tz=timezone.utc)
    end_time = begin_time + timedelta(seconds=(nt - 1) * dt)

    return DASdata(
        data=data,
        fs=fs, dt=dt, nt=nt, nx=nx, dx=dx,
        begin_time=begin_time, end_time=end_time,
        gauge_length_m=gauge_length_m, system='ASN',
        raw_meta=raw_meta,
    )


# --- Catalog metadata helpers ---------------------------------------------

def read_asn_metadata(file: Union[str, Path]) -> Optional[DASmeta]:
    """Read one ASN file's metadata as a DASmeta dict.

    Mirrors legacy DAS_db.py: `/header/time` + `/timing/sampleSkew`
    give the authoritative begin_time;
    `/header/dimensionRanges/dimension0/unitScale` is dt; `/data.shape`
    yields (nt, nx). Returns None (with a stderr warning) if the file
    can't be opened or the expected attrs are missing — the catalog
    treats such files as garbage rather than substituting a
    filename-derived timestamp.
    """
    file = Path(file)
    try:
        with h5py.File(file, 'r') as f:
            ts = float(f['header/time'][()]) + float(f['timing/sampleSkew'][()])
            dt = float(f['header/dimensionRanges/dimension0/unitScale'][()])
            nt = int(f['data'].shape[0])
            nx = int(f['data'].shape[1])
            fs = 1.0 / dt
            dx = gauge_length_m = None
            try:
                channels = f['header/channels'][:2]
                dx_attr = float(f['header/dx'][()])
                dx = (
                    float(channels[1] - channels[0]) * dx_attr
                    if channels.size > 1 else dx_attr
                )
            except (KeyError, IndexError):
                pass
            try:
                gauge_length_m = float(f['header/gaugeLength'][()])
            except KeyError:
                pass
            begin_time = datetime.fromtimestamp(ts, tz=timezone.utc)
            end_time = datetime.fromtimestamp(ts + (nt - 1) * dt, tz=timezone.utc)
    except (OSError, KeyError, ValueError) as e:
        _log_skip(file, e)
        return None
    return DASmeta(
        file=str(file),
        begin_time=begin_time, end_time=end_time,
        fs=fs, nt=nt, nx=nx,
        dx=dx, gauge_length_m=gauge_length_m,
        first_sample=0,
    )


