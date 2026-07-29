"""Signal-processing kernels for DAS data.

Time-axis kernels (operate per channel along the time axis; input
shape `(nchan, nt)`, output is a new array of the same shape):

- bandpass2d: thin wrapper around the vendored pybind11 Butterworth filter
- diff_time / gradient_time / integrate_time: numba-JIT'd backward- and
  central-difference derivatives / cumsum
- detrend_time: numba-JIT'd per-channel least-squares linear detrend
- taper_time: Tukey (cosine) edge taper, used before bandpass to suppress
  filter ringing at the segment boundaries
- subtract_common_mode: per-time median across a channel band,
  subtracted from every channel — removes the common-mode noise
- unwrap_int32: int32 overflow correction: OptaSense-origin data
- median_filter_1d: running median along time or channel axes, reflection-padded
"""
import numpy as np
from numba import njit, prange
from scipy.signal.windows import tukey

from .cpp import lfilter, lfilter_double


def bandpass2d(data, freqmin, freqmax, dt, order=6, zerophase=False, nThreads=1):
    """2D bandpass along the fast axis via the vendored pybind11 C++ extension.

    The extension reads the array's raw buffer assuming a C-contiguous layout,
    so a transposed or strided input is walked in the wrong order and comes
    back as a plausible-looking periodic pattern rather than an error. ASN and
    AP Sensing readers transpose to reach `(nx, nt)`, so this is reachable from
    a plain `DASdb.read`; the copy is skipped when the input is already
    contiguous.
    """
    data = np.ascontiguousarray(data)
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


@njit(cache=True, inline='always')
def _median_1d(src, dst, k):
    """Running median of one line into `dst`.

    The line is reflect-padded once so the window is a contiguous slice; each
    step then drops `ext[o-1]` and inserts `ext[o+k-1]`, shifting only across
    the ranks between them. That keeps the cost near-flat in `k` — 7 ms at
    k=3 and 27 ms at k=101 on 1200x12000 — where re-selecting the window every
    step is not: torch 29 and 384 ms, scipy 307 ms and 14.3 s.

    Reflection does not repeat the edge sample, matching torch's
    `F.pad(mode='reflect')` and scipy's `mode='mirror'` (scipy's own
    `'reflect'` duplicates it and differs over the first and last k // 2).
    """
    n = src.shape[0]
    h = k // 2
    ext = np.empty(n + 2 * h, src.dtype)
    ext[:h] = src[h:0:-1]
    ext[h:h + n] = src
    ext[h + n:] = src[n - 2:n - h - 2:-1]

    win = np.sort(ext[:k])
    dst[0] = win[h]
    for o in range(1, n):
        old, new = ext[o - 1], ext[o + k - 1]
        if old != new:
            i = np.searchsorted(win, old)           # the rank `old` vacates
            if new > old:
                while i + 1 < k and win[i + 1] <= new:
                    win[i] = win[i + 1]
                    i += 1
            else:
                while i > 0 and win[i - 1] > new:
                    win[i] = win[i - 1]
                    i -= 1
            win[i] = new
        dst[o] = win[h]


@njit(parallel=True, cache=True)
def _median_along_time(data, k):
    out = np.empty_like(data)
    for i in prange(data.shape[0]):
        _median_1d(data[i], out[i], k)
    return out


@njit(parallel=True, cache=True)
def _median_along_channel(data, k):
    """Filter down the channel axis, a gathered column at a time.

    Channels are axis 0 of a C-contiguous `(nx, nt)` array, so they stride by
    `nt` and `_median_1d` would read a cache line per sample off the raw view.
    Gathering the column into a contiguous buffer first costs ~14 % against a
    tiled variant and is far simpler; transposing the array is ~6x slower.
    """
    nchan, nt = data.shape
    out = np.empty_like(data)
    for j in prange(nt):
        col = np.empty(nchan, data.dtype)
        col[:] = data[:, j]
        _median_1d(col, out[:, j], k)
    return out


def median_filter_1d(data, kernel_size, axis='t'):
    """Running median along time (`axis='t'`) or channels (`axis='x'`).

    Removes spikes narrower than the kernel while leaving real edges intact —
    what a band-pass cannot do, since it smears an impulse across its passband
    instead. Returns a new array; the input is untouched.

    Edges reflect without repeating the edge sample, so results are
    bit-identical to `scipy.ndimage.median_filter(mode='mirror')` — at a
    fraction of its cost, which is O(n*k) per sample.
    """
    if kernel_size % 2 == 0:
        raise ValueError(f'kernel_size must be odd, got {kernel_size}')
    if axis not in ('t', 'x'):
        raise ValueError(f"axis must be 't' or 'x', got {axis!r}")

    data = np.ascontiguousarray(data)
    along_time = axis == 't'
    n = data.shape[1] if along_time else data.shape[0]
    if kernel_size // 2 >= n:
        raise ValueError(
            f'kernel_size {kernel_size} is wider than the {axis!r} axis ({n}); '
            f'reflection is undefined unless k // 2 < n'
        )

    if along_time:
        return _median_along_time(data, kernel_size)
    return _median_along_channel(data, kernel_size)


@njit(parallel=True, cache=True)
def unwrap_int32(data, factor=1, threshold=0.99):
    """Undo int32 phase rollover along time, in place. Returns `data`.

    Port of `DASutils.preprocess_unwrap`, bit-identical on real windows: per
    channel, mark each step exceeding a wrap, cumsum the marks into a running
    multiple of 2**32, add it back. Copied rather than imported because
    DASutils drags in utm, tdms_reader, segyio and das_utilities.

    OptaSense only — ASN and AP Sensing return floats near 1e-1 and can never
    reach an int32 rail. The name refers to the on-disk type; readers have
    already widened to float by the time this runs. Proc payloads of OptaSense
    origin hold the same counts and are handled the same way.

    Call it on the **concatenated** array, never per file: a channel that
    wrapped in file N carries an offset file N+1 knows nothing about, leaving a
    2**32 step at the boundary (4 of 3000 channels across two iceland files).
    Idempotent, so a second pass over already-unwrapped data is a no-op.

    `threshold` is the fraction of a wrap a step must exceed to count. A
    rollover appears as `2**32 - (true change)`, so the 0.99 default misses one
    whose true change exceeds ~4.3e7 counts; real data peaks near 1.9e5. Pass
    0.5 for the half-period convention and a wider margin.
    """
    clip = 2.0 ** 32 * factor
    clip_threshold = clip * threshold
    nx, nt = data.shape
    for ix in prange(nx):
        correction = np.zeros(nt - 1, dtype=data.dtype)
        d = np.diff(data[ix, :])
        correction[d < -clip_threshold] = 1.0
        correction[d > clip_threshold] = -1.0
        data[ix, 1:] += np.cumsum(correction) * clip
    return data
