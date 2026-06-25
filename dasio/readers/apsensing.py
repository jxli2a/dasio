"""Reader for raw APSensing HDF5 files.

Top-level groups: `/ProcessingServer` (sampling + conversion factors),
`/Timestamps` (per-sample µs-since-epoch), `/Distances` (channel
positions), `/DAS` (payload in radians/sec, i.e. phase rate).

Payload units: radians/sec. The DASdb-scan returns raw values; conver-
sion to strain-rate via `apsensing_radians2strain_factor` = `1e-9 *
RadiansToNanoStrain`, applied by `read_data_proc` when the Proc file's
origin is APSensing, by analogy with the OptaSense count→strain chain.

No intra-file RawDataTime gap splitting here (legacy DASutils doesn't
split either) — each file contributes exactly one `DASmeta` row.
"""
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np

from ..dasdata import DASdata, DASmeta


def apsensing_radians2strain_factor(f: h5py.File) -> float:
    """Radians/sec → strain/sec scalar for an APSensing file.

    Pulls ``RadiansToNanoStrain`` from either the native-raw path
    (``/ProcessingServer``) or the Proc-flattened path
    (``/Acquisition_origin``), whichever the file carries. Returns
    1.0 with a stderr warning when the attribute is missing — matches
    legacy fallback semantics (better to pass unscaled phase rate
    through than fail the read outright).
    """
    try:
        if 'Acquisition_origin' in f:                # Proc file
            ns_per_rad = float(
                f['Acquisition_origin'].attrs['ProcessingServer.RadiansToNanoStrain']
            )
        elif 'ProcessingServer' in f:                # raw APSensing file
            ns_per_rad = float(f['ProcessingServer']['RadiansToNanoStrain'][0])
        else:
            print(
                'WARNING! apsensing_radians2strain_factor: neither '
                '/ProcessingServer nor /Acquisition_origin present; '
                'returning 1.0',
                file=sys.stderr,
            )
            return 1.0
    except (KeyError, TypeError, ValueError) as e:
        print(
            f'WARNING! apsensing_radians2strain_factor: missing '
            f'RadiansToNanoStrain ({e}); returning 1.0',
            file=sys.stderr,
        )
        return 1.0
    return 1e-9 * ns_per_rad


def read_apsensing_raw(
        file: Union[str, Path],
        min_ch: int = 0,
        max_ch: Optional[int] = None,
        first_sample: int = 0,
        n_samples: Optional[int] = None,
    ) -> DASdata:
    """Read one raw APSensing HDF5 file as unscaled phase-rate (rad/s).

    Strain-rate conversion is deferred (`apsensing_radians2strain_factor`
    is the scalar); this matches OptaSense's convention of returning
    unscaled counts so the pipeline can stage filter / decimate on a
    single float array and apply the factor once at the end.

    On-disk layout: `/DAS` is either (n_time_samples, n_channels) or
    (n_channels, n_time_samples) — determined by which axis matches
    `/Timestamps/DataTimestamps.shape[0]`. Output DASdata is always
    (nx, nt) float32.
    """
    file = Path(file)
    with h5py.File(file, 'r') as f:
        t_us = f['Timestamps']['DataTimestamps'][:, 0].astype(np.float64)
        total_nt = t_us.size
        dset = f['DAS']
        # time-first layout: DAS.shape[0] == total_nt ⇒ (nt, nx)
        time_first = dset.shape[0] == total_nt
        total_nx = dset.shape[1] if time_first else dset.shape[0]

        if max_ch is None:
            max_ch = total_nx
        if n_samples is None:
            n_samples = total_nt - first_sample
        if time_first:
            raw = dset[first_sample:first_sample + n_samples, min_ch:max_ch].T
        else:
            raw = dset[min_ch:max_ch, first_sample:first_sample + n_samples]

        ps = f['ProcessingServer']
        fs = float(ps['DataRate'][0])
        dx = float(ps['SpatialSampling'][0])
        try:
            gauge_length_m: Optional[float] = float(ps['GaugeLength'][0])
        except (KeyError, IndexError):
            gauge_length_m = None

        # Mirror the Desample_DAS convention for /Acquisition_origin:
        # every scalar dataset under /ProcessingServer becomes
        # `ProcessingServer.<key>` and every attr on
        # /Distances/MeterPositions becomes `Distances.<key>`. proc.py's
        # _flatten_dict turns the nested dict into the dotted attr
        # names downstream.
        ps_meta = {}
        for k in ps.keys():
            try:
                ps_meta[k] = ps[k][0]
            except (IndexError, TypeError):
                pass
        dist_meta = {}
        if 'Distances' in f:
            mp = f['Distances'].get('MeterPositions')
            if mp is not None:
                for k, v in mp.attrs.items():
                    try:
                        dist_meta[k] = v[0]
                    except (IndexError, TypeError):
                        dist_meta[k] = v
        raw_meta = {'ProcessingServer': ps_meta, 'Distances': dist_meta}

    data = raw.astype(np.float32, copy=False)  # (nx, nt)
    nx = data.shape[0]
    nt = data.shape[1]
    dt = 1.0 / fs
    begin_time = datetime.fromtimestamp(
        t_us[first_sample] * 1e-6, tz=timezone.utc,
    )
    end_time = begin_time + timedelta(seconds=(nt - 1) * dt)

    return DASdata(
        data=data,
        fs=fs, dt=dt, nt=nt, nx=nx, dx=dx,
        begin_time=begin_time, end_time=end_time,
        gauge_length_m=gauge_length_m, system='APSensing',
        raw_meta=raw_meta,
    )


# --- Catalog metadata helpers ---------------------------------------------

def read_apsensing_metadata(file: Union[str, Path]) -> Optional[DASmeta]:
    """Read one APSensing file's metadata as a DASmeta dict.

    Returns None (with a stderr warning) if the file isn't an
    APSensing capture (missing `/ProcessingServer`), can't be opened,
    or is empty. Unlike OptaSense this yields a single row per file
    — legacy DASutils doesn't split APSensing RawDataTime either.
    """
    file = Path(file)
    try:
        with h5py.File(file, 'r') as f:
            if 'ProcessingServer' not in f or 'Timestamps' not in f:
                return None
            ps = f['ProcessingServer']
            fs = float(ps['DataRate'][0])
            dx = float(ps['SpatialSampling'][0])
            try:
                gauge_length_m: Optional[float] = float(ps['GaugeLength'][0])
            except (KeyError, IndexError):
                gauge_length_m = None
            t_us = f['Timestamps']['DataTimestamps'][:, 0].astype(np.float64)
            nt = int(t_us.size)
            if nt == 0:
                return None
            das = f['DAS']
            total_nx = das.shape[1] if das.shape[0] == nt else das.shape[0]
    except (OSError, KeyError, ValueError) as e:
        print(f'[dasio.apsensing] skipping {file}: {e}', file=sys.stderr)
        return None

    begin_time = datetime.fromtimestamp(t_us[0] * 1e-6, tz=timezone.utc)
    # end_time is last-sample-inclusive (schema contract), matching
    # ASN / Proc / OptaSense metadata.
    end_time = datetime.fromtimestamp(t_us[-1] * 1e-6, tz=timezone.utc)
    return DASmeta(
        file=str(file),
        begin_time=begin_time, end_time=end_time,
        fs=fs, nt=nt, nx=total_nx,
        dx=dx, gauge_length_m=gauge_length_m,
        first_sample=0,
    )
