"""Reader for raw 100 Hz OptaSense/QuantX HDF5 files (Mammoth native format)."""
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Union

import h5py
import numpy as np

from ..dasdata import DASdata, DASmeta


# The only hardcoded vendor physics in the package — every other conversion
# constant is read off the file. Copied verbatim from legacy
# DASutils._parse_raw2strain_factor_optasense.
_ETA = 0.78                        # photo-elastic scaling, isotropic material
_COUNT2PHASE = np.pi / 2 ** 15     # raw count -> radians


def optasense_count2strain_factor(f: h5py.File) -> float:
    """Compute raw-counts → strain conversion factor for an OptaSense file.

    Pulls GaugeLength, refractive index, laser wavelength and FPGA
    polarity attrs from either the native-raw path (`/Acquisition` +
    `/Acquisition/Custom`) or the Proc-flattened path
    (`/Acquisition_origin`), whichever the file carries. Mirrors
    DASutils._parse_raw2strain_factor_optasense, which also handled
    both layouts.

    Returns 1.0 with a stderr warning when any required attribute is
    missing — matches legacy fallback semantics (better to pass raw
    counts through than fail the read outright).
    """
    # Pick the attribute source based on which group is present.
    if 'Acquisition_origin' in f:              # Proc file
        attrs = f['Acquisition_origin'].attrs
        G_attrs = attrs
        custom_attrs = attrs                    # Proc flattens Custom into origin
    elif 'Acquisition' in f:                    # Raw OptaSense file
        acq = f['Acquisition']
        G_attrs = acq.attrs
        custom_attrs = acq['Custom'].attrs
    else:
        print(
            'WARNING! optasense_count2strain_factor: neither /Acquisition '
            'nor /Acquisition_origin present; returning 1.0',
            file=sys.stderr,
        )
        return 1.0

    try:
        G = float(G_attrs['GaugeLength'])
        n = float(custom_attrs['Fibre Refractive Index'])
        lamd = float(custom_attrs['Laser Wavelength (nm)']) * 1e-9  # nm → m
        fpgaDRN = int(custom_attrs['FPGA Drawing Number'])
        fpga_raw = custom_attrs['FPGA Version']
        fpgaVersion = float(
            fpga_raw.decode('utf-8')
            if isinstance(fpga_raw, bytes) else fpga_raw
        )
    except (KeyError, TypeError, ValueError) as e:
        print(
            f'WARNING! optasense_count2strain_factor: cannot compute factor '
            f'({e}); returning 1.0',
            file=sys.stderr,
        )
        return 1.0
    polarity = -1.0 if (fpgaDRN == 7804701 and fpgaVersion <= 2.0) else 1.0
    phase2strain = lamd / (4.0 * np.pi * _ETA * n * G)
    return polarity * _COUNT2PHASE * phase2strain


def _time_major(raw_ds, f) -> bool:
    """True when `RawData` is stored as (time, channel).

    The `Dimensions` attribute is authoritative and present on every file seen
    so far. Where it is missing, fall back to matching the first axis against
    the length of `RawDataTime`, which is one value per time sample.
    """
    dims = raw_ds.attrs.get('Dimensions')
    if dims is not None:
        first = dims[0]
        first = first.decode() if isinstance(first, bytes) else str(first)
        return first.strip().lower().startswith('time')
    n_time = f['Acquisition/Raw[0]/RawDataTime'].shape[0]
    return raw_ds.shape[0] == n_time and raw_ds.shape[1] != n_time


def read_optasense_raw(
        file: Union[str, Path],
        min_ch: int = 0,
        max_ch: Optional[int] = None,
        first_sample: int = 0,
        n_samples: Optional[int] = None,
    ) -> DASdata:
    """Read one raw OptaSense HDF5 file as unwrapped-but-unscaled phase counts.

    Unwrap across adjacent-file boundaries is deferred to the pipeline
    (desample_window) so results match legacy Desample_DAS.py, which unwraps
    the whole concatenated trace_buffer at once. Strain conversion is also
    deferred — legacy writes unscaled phase counts on disk in its default
    (no --compressZFP) mode. Returned DASdata is (nx, nt) float32 with NO
    unwrap applied.

    On-disk `RawData` comes in **either** axis order depending on the
    acquisition — `[locus, time]` (mammoth_south) or `[time, locus]`
    (iceland_quantx) — and the file says which via its `Dimensions` attribute.
    Assuming one order silently transposes the other: nx and nt come back
    swapped and every value is taken from the wrong axis, so the result is
    garbage rather than merely mis-shaped.
    """
    file = Path(file)
    with h5py.File(file, 'r') as f:
        raw_ds = f['Acquisition/Raw[0]/RawData']
        time_first = _time_major(raw_ds, f)
        total_nt, total_nx = raw_ds.shape if time_first else raw_ds.shape[::-1]
        if max_ch is None:
            max_ch = total_nx
        if n_samples is None:
            n_samples = total_nt - first_sample
        ch, smp = slice(min_ch, max_ch), slice(first_sample, first_sample + n_samples)
        # Slice on the axes as stored, then transpose — never the reverse, or
        # h5py reads the whole dataset before the selection applies.
        raw = (raw_ds[smp, ch].T if time_first else raw_ds[ch, smp]).astype(np.float64)

        time_ds = f['Acquisition/Raw[0]/RawDataTime']
        # RawDataTime is microseconds since epoch
        t_us = int(time_ds[first_sample])
        begin_time = datetime.fromtimestamp(t_us * 1e-6, tz=timezone.utc)

        acq = f['Acquisition']
        raw0 = acq['Raw[0]']
        fs = float(raw0.attrs['OutputDataRate'])
        dt = 1.0 / fs
        dx = float(acq.attrs.get('SpatialSamplingInterval', 0.0))
        gauge_length_m = float(acq.attrs['GaugeLength'])

        # Flat merge of /Acquisition + /Acquisition/Custom attrs, matching the
        # OptaSense branch in Desample_DAS.py which writes the union straight
        # into /Acquisition_origin without any prefix.
        raw_meta = {}
        for k, v in acq.attrs.items():
            raw_meta[k] = v
        if 'Custom' in acq:
            for k, v in acq['Custom'].attrs.items():
                raw_meta[k] = v

    # NOTE: unwrap is applied later by desample_window on the full concatenated
    # buffer, so it propagates across file boundaries. This reader just returns
    # the raw int32 counts cast to float32.
    data = raw.astype(np.float32)

    nx = data.shape[0]
    nt = data.shape[1]
    end_time = begin_time + timedelta(seconds=(nt - 1) * dt)

    return DASdata(
        data=data,
        ch0=int(min_ch),
        fs=fs, dt=dt, nt=nt, nx=nx, dx=dx,
        begin_time=begin_time, end_time=end_time,
        gauge_length_m=gauge_length_m, system='OptaSense', origin='OptaSense',
        raw_meta=raw_meta,
        units="count",
    )


# --- Catalog metadata helpers ---------------------------------------------

def read_optasense_metadata(
        file: Union[str, Path], rtol: float = 1e-3,
    ) -> List[DASmeta]:
    """Read one OptaSense file's metadata as a list of DASmeta dicts.

    A single `.h5` may hold a single gap-free run or several — legacy
    DAS_db split on any ``|stride - 1/fs|`` exceeding ``rtol`` (relative
    tolerance). We mirror that so the catalog carries one row per
    contiguous RawDataTime chunk, with `first_sample` / `nt` set to the
    offsets a reader must apply to skip past the intra-file gap.

    Returns `[]` if the file isn't an OptaSense capture (missing the
    /Acquisition/Raw[0]/RawData signature), can't be opened, or is
    empty. Non-capture files are silently ignored so callers can glob
    a mixed directory; true errors surface on stderr.
    """
    file = Path(file)
    try:
        with h5py.File(file, 'r') as f:
            if 'Acquisition' not in f or 'Raw[0]' not in f['Acquisition']:
                return []
            acq = f['Acquisition']
            raw0 = acq['Raw[0]']
            fs = float(raw0.attrs['OutputDataRate'])
            t_raw = raw0['RawDataTime'][:].astype(np.int64)
            t_us = t_raw.astype(np.float64) * 1e-6
            # shape[0] is the *time* axis on time-major files, so pick the
            # channel axis by layout rather than assuming the first one.
            _ax = 1 if _time_major(raw0['RawData'], f) else 0
            nx = int(acq['Custom'].attrs.get(
                'Num Output Channels', raw0['RawData'].shape[_ax]))
            dx = float(acq.attrs.get('SpatialSamplingInterval', 0.0))
            gauge_length_m = float(acq.attrs.get('GaugeLength', 0.0))
    except (OSError, KeyError) as e:
        print(f'[dasio.optasense] skipping {file}: {e}', file=sys.stderr)
        return []

    n_tot = t_us.size
    if n_tot == 0:
        return []
    if n_tot > 1:
        strides_us = np.diff(t_raw)
        dt_target_us = 1e6 / fs
        skip_idx = np.where(~np.isclose(strides_us, dt_target_us, rtol=rtol))[0] + 1
    else:
        skip_idx = np.array([], dtype=int)
    bounds = np.concatenate(([0], skip_idx, [n_tot]))
    metas: List[DASmeta] = []
    for i in range(len(bounds) - 1):
        i0, i1 = int(bounds[i]), int(bounds[i + 1])
        if i1 <= i0:
            continue
        begin_time = datetime.fromtimestamp(t_us[i0], tz=timezone.utc)
        # end_time is last-sample-inclusive (matches ASN metadata and
        # the schema contract); the DASdb continuity check expects
        # `next.begin_time - prev.end_time ≈ 1/fs`, not ≈ 0.
        end_time = datetime.fromtimestamp(t_us[i1 - 1], tz=timezone.utc)
        metas.append(DASmeta(
            file=str(file),
            begin_time=begin_time, end_time=end_time,
            fs=fs, nt=i1 - i0, nx=nx,
            dx=dx, gauge_length_m=gauge_length_m,
            first_sample=i0,
        ))
    return metas


