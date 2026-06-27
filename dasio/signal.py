"""Signal-processing kernels for DAS data.

Time-axis kernels (operate per channel along the time axis; input
shape `(nchan, nt)`, output is a new array of the same shape):

- bandpass2d: thin wrapper around the vendored pybind11 Butterworth filter
- diff_time / gradient_time / integrate_time: numba-JIT'd backward- and
  central-difference derivatives / cumsum
- detrend_time: numba-JIT'd per-channel least-squares linear detrend
- taper_time: Tukey (cosine) edge taper, used before bandpass to suppress
  filter ringing at the segment boundaries
- preprocess_unwrap: int32 wrap correction (needed for OptaSense raw)

Cross-channel kernel:

- subtract_common_mode: per-time median across a channel band,
  subtracted from every channel — removes the common-mode noise
  track shared by the cable / interrogator electronics. Replaces
  the misnamed `preprocess_medfilt` from legacy DASutils.
"""
import numpy as np
from numba import njit, prange
from scipy.signal.windows import tukey

from .cpp import lfilter, lfilter_double


def bandpass2d(data, freqmin, freqmax, dt, order=6, zerophase=False, nThreads=1):
    """2D bandpass along the fast axis via the vendored pybind11 C++ extension."""
    phase = 0 if zerophase else 1
    if data.dtype == np.float32:
        return lfilter(data, freqmin * dt, order, freqmax * dt, order, phase, nThreads)
    if data.dtype == np.float64:
        return lfilter_double(data, freqmin * dt, order, freqmax * dt, order, phase, nThreads)
    raise ValueError(f"Array dtype not supported by bandpass2d: {data.dtype}")


@njit(parallel=True, cache=True)
def detrend_time(data):
    """Subtract a per-channel least-squares linear fit along the time axis.

    Returns a new array; the input is not modified.
    """
    nchan, nt = data.shape
    out = np.empty_like(data)
    x = np.arange(nt).astype(np.float64)
    x_mean = x.mean()
    x_centered = x - x_mean
    denom = (x_centered * x_centered).sum()
    for ich in prange(nchan):
        y = data[ich, :]
        y_mean = y.mean()
        numer = (x_centered * (y - y_mean)).sum()
        m = numer / denom
        b = y_mean - m * x_mean
        for it in range(nt):
            out[ich, it] = y[it] - (m * x[it] + b)
    return out


def taper_time(data, alpha=0.4):
    """Apply a Tukey (cosine-tapered) window along the time axis.

    `alpha` is the fraction of the window covered by the cosine
    transition at each end (0 → no taper / rectangular, 1 → full
    Hann). Default 0.4 matches legacy DASutils.readFile_HDF's default.
    Returns a new array; the input is not modified.
    """
    nt = data.shape[1]
    w = tukey(nt, alpha).astype(data.dtype, copy=False)
    return data * w


@njit(parallel=True, cache=True)
def diff_time(data, dt):
    """Backward-difference time derivative (axis=-1). Mirrors DASutils.preprocess_diff.

    First-order ``(x[i] - x[i-1]) / dt`` with the first sample forced to zero.
    """
    nchan, nt = data.shape
    out = np.empty_like(data)
    for ich in prange(nchan):
        out[ich, 0] = 0.0
        for it in range(1, nt):
            out[ich, it] = (data[ich, it] - data[ich, it - 1]) / dt
    return out


@njit(parallel=True, cache=True)
def gradient_time(data, dt):
    """Central-difference time derivative (axis=-1), matching ``np.gradient``.

    Interior samples use the second-order central difference
    ``(x[i+1] - x[i-1]) / (2*dt)``; the two end samples use a first-order
    one-sided difference (``np.gradient`` with the default ``edge_order=1``).
    Bit-for-bit identical to ``np.gradient(data, axis=-1) / dt`` but parallel
    over channels and allocation-free, so several times faster.
    """
    nchan, nt = data.shape
    out = np.empty_like(data)
    inv, inv2 = 1.0 / dt, 1.0 / (2.0 * dt)
    for ich in prange(nchan):
        out[ich, 0] = (data[ich, 1] - data[ich, 0]) * inv
        for it in range(1, nt - 1):
            out[ich, it] = (data[ich, it + 1] - data[ich, it - 1]) * inv2
        out[ich, nt - 1] = (data[ich, nt - 1] - data[ich, nt - 2]) * inv
    return out


@njit(parallel=True, cache=True)
def integrate_time(data, dt):
    """Integrate along the time axis. cumsum(data)*dt, first time sample zero."""
    nchan, nt = data.shape
    out = np.empty_like(data)
    for ich in prange(nchan):
        acc = data[ich, 0]
        out[ich, 0] = 0.0
        for it in range(1, nt):
            acc += data[ich, it]
            out[ich, it] = acc * dt
    return out


@njit(parallel=True, cache=True)
def subtract_common_mode(data, ch_min, ch_max):
    """Estimate per-time common-mode noise from a channel band and remove it.

    For each time sample the median across channels [ch_min, ch_max)
    is computed and subtracted from every channel of `data`. 
    """
    nchan, nt = data.shape
    out = np.empty_like(data)
    for it in prange(nt):
        m = np.median(data[ch_min:ch_max, it])
        for ich in range(nchan):
            out[ich, it] = data[ich, it] - m
    return out


@njit(parallel=True, cache=True)
def preprocess_unwrap(data, factor=1):
    """Unwrap int32 rollover along the time axis. Mirrors DASutils.preprocess_unwrap.

    factor scales the 2**32 wrap increment (factor=1 for standard OptaSense int32).
    """
    nchan, nt = data.shape
    wrap = 2.0 ** 32 * factor
    half = wrap / 2.0
    out = data.astype(np.float64).copy()
    for ich in prange(nchan):
        offset = 0.0
        for it in range(1, nt):
            diff = out[ich, it] + offset - out[ich, it - 1]
            if diff > half:
                offset -= wrap
            elif diff < -half:
                offset += wrap
            out[ich, it] = out[ich, it] + offset
    return out
