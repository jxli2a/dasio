"""High-level DAS processing — `DASdata` in, `DASdata` out.

Thin wrappers around the numeric kernels in `signal.py` that preserve
the `DASdata` envelope (fs, dt, nx, nt, dx, begin_time, end_time,
gauge_length_m, system, raw_meta) so pipelines read naturally:

        d = read_das_data(path, 'Proc')
        d = bandpass(d, 0.01, 0.4)
        d = integrate(d)

Each function returns a new `DASdata` via `dataclasses.replace`. Pass
`copy=False` when you know the input's `data` can be thrown away — the
kernels themselves return fresh arrays either way, but `copy=True`
(the default) guards against later aliasing surprises when the
caller mutates the input in place.
"""
from dataclasses import replace

from typing import Optional

from .dasdata import DASdata
from .signal import (
    bandpass2d, detrend_time, diff_time, integrate_time,
    preprocess_unwrap, taper_time,
    subtract_common_mode as _subtract_common_mode_kernel,
)


def bandpass(
        d: DASdata,
        fmin: float, fmax: float,
        order: int = 14, zerophase: bool = True, copy: bool = True,
    ) -> DASdata:
    """Butterworth bandpass along the time axis."""
    data = d.data.copy() if copy else d.data
    data = bandpass2d(
        data, fmin, fmax, d.dt, order=order, zerophase=zerophase,
    )
    return replace(d, data=data)


def detrend(d: DASdata, copy: bool = True) -> DASdata:
    """Subtract a per-channel linear trend along the time axis.

    Standard pre-processing step before bandpass — removes any DC
    offset and slow drift that would otherwise leak into the filter
    transition band. Mirrors legacy DASutils.detrend_2D.
    """
    data = d.data.copy() if copy else d.data
    data = detrend_time(data)
    return replace(d, data=data)


def taper(d: DASdata, alpha: float = 0.4, copy: bool = True) -> DASdata:
    """Tukey edge taper along the time axis.

    `alpha` is the fraction of the window covered by the cosine
    transition at each end (0 = no taper, 1 = full Hann). Default
    0.4 matches legacy DASutils.readFile_HDF. Apply between
    `detrend` and `bandpass` to suppress filter ringing at the
    segment boundaries.
    """
    data = d.data.copy() if copy else d.data
    data = taper_time(data, alpha=alpha)
    return replace(d, data=data)


def differentiate(d: DASdata, copy: bool = True) -> DASdata:
    """Differentiate along the time axis (d/dt).

    First time sample is forced to zero (same convention as
    `signal.diff_time`, matching DASutils.preprocess_diff).
    """
    data = d.data.copy() if copy else d.data
    data = diff_time(data, d.dt)
    return replace(d, data=data)


def integrate(d: DASdata, copy: bool = True) -> DASdata:
    """Integrate along the time axis (cumulative sum × dt).

    First time sample is forced to zero so the trace is pinned to
    the window's start — otherwise a DC offset in `d.data` would run
    off linearly and dominate the result.
    """
    data = d.data.copy() if copy else d.data
    data = integrate_time(data, d.dt)
    return replace(d, data=data)


def subtract_common_mode(
        d: DASdata,
        ch_min: int = 0,
        ch_max: Optional[int] = None,
        copy: bool = True,
    ) -> DASdata:
    """Per-time-sample common-mode noise subtraction across channels.

    For each time sample we take the median across channels
    [ch_min, ch_max) and subtract it from every channel of `d`. The
    slice bounds let callers estimate from a quiet stretch of fiber
    rather than the whole array — biases the median toward background
    rather than active signal. Default bounds use all channels
    (matches legacy DASutils.preprocess_medfilt).
    """
    if ch_max is None:
        ch_max = d.nx
    data = d.data.copy() if copy else d.data
    data = _subtract_common_mode_kernel(data, ch_min, ch_max)
    return replace(d, data=data)


def unwrap(d: DASdata, factor: int = 1, copy: bool = True) -> DASdata:
    """Unwrap int32 rollover along the time axis.

    Needed for raw OptaSense phase counts; a no-op for data that's
    already strain. `factor` scales the 2**32 wrap increment (default
    1, matching standard OptaSense int32).
    """
    data = d.data.copy() if copy else d.data
    data = preprocess_unwrap(data, factor=factor)
    return replace(d, data=data)
