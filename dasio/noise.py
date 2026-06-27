"""Ambient-noise cross-correlation for DAS.

`preprocess` runs the standard CC preparation chain on a freshly loaded
(continuous, unprocessed) `DASdata` — differentiate, detrend, band-pass,
decimate, common-mode removal, temporal normalization — reusing the core dasio
ops. `temporal_normalization` and `spectral_whitening` are the individual
amplitude-flattening steps; `xcorr` is the FFT cross-correlation.

The channel-pair gathers (`common_shot_pairs`, `common_offset_pairs`) build the
``(ch1, ch2)`` index arrays that say which channels to correlate.
`xcorr_stack` cross-correlates those pairs over the 60-s-style segments of
one `DASdata` and stacks; `xcorr_dataset` walks a whole `DASdb`,
preprocessing and stacking every segment. Both return a `CCGather` — the
``(npair, lag)`` stacked correlation plus its lag axis, with a `.plot()` method.

PyTorch is an optional dependency (``pip install 'dasio[noise]'``); it is
imported lazily inside `xcorr` / `spectral_whitening`, so `preprocess`,
`temporal_normalization` and the gather builders (all CPU) work without it.
"""
from dataclasses import dataclass, replace
from datetime import timedelta
from typing import Optional

from matplotlib.mlab import detrend
import numpy as np

from dasio.signal import subtract_common_mode

from .dasdata import DASdata


def preprocess(
        d: DASdata, *,
        differentiate: bool = True,
        detrend: bool = True,
        diff_method: str = "central",
        freqmin: Optional[float] = None,
        freqmax: Optional[float] = None,
        taper_alpha: Optional[float] = None,
        order: int = 14,
        zerophase: bool = True,
        decimate: int = 1,
        subtract_common_mode: bool = True,
        norm_window_sec: Optional[float] = None,
    ) -> DASdata:
    """Standard ambient-noise CC preparation for a raw-loaded `DASdata`.

    Runs the canonical chain on continuous data straight off the reader and
    returns a fresh `DASdata`: differentiate -> detrend -> (taper ->) band-pass
    -> decimate -> common-mode removal -> temporal normalization. Each step
    reuses an existing dasio op; compose them yourself for a non-standard recipe.

    differentiate : strain -> strain rate (the usual CC quantity). Turn off if
        the data is already a rate.
    diff_method : "central" (default, second-order central difference, bit-
        identical to ``np.gradient``) or "backward" (first-order, first sample
        zeroed). Only used when ``differentiate`` is True.
    freqmin, freqmax : band-pass corners in Hz; skipped unless both are given.
    taper_alpha : Tukey edge-taper fraction applied just before the band-pass
        (None skips it). Matches the legacy ``filter(data) = tukey * data ->
        butter`` recipe; e.g. 0.05 tapers 2.5 % at each end before filtering.
    order, zerophase : Butterworth band-pass order and zero-phase flag (order 4
        + ``zerophase=True`` reproduces a ``butter(4)`` + ``filtfilt`` pass).
    decimate : integer time-decimation factor, applied as a plain stride AFTER
        the band-pass — the band-pass is the anti-alias filter, so choose
        ``freqmax < fs / (2 * decimate)``. 1 (default) means no decimation.
    norm_window_sec : temporal normalization. None skips it; 0 = one-bit;
        >0 = running-absolute-mean window in seconds.

    detrend and common-mode removal are always applied — they are part of the
    standard recipe. CPU only (no PyTorch needed; that's just `xcorr`).
    """
    if differentiate:
        d = d.differentiate(method=diff_method)
    if detrend:
        d = d.detrend()
    if freqmin is not None and freqmax is not None:
        if taper_alpha is not None:
            d = d.taper(alpha=taper_alpha)                # tukey * data, then filter
        d = d.bandpass(freqmin, freqmax, order=order, zerophase=zerophase)
    if decimate > 1:
        d = d.skip_t(decimate)                            # stride; band-pass already band-limited
    if subtract_common_mode:
        d = d.subtract_common_mode()
    if norm_window_sec is not None:
        d = temporal_normalization(d, norm_window_sec)
    return d


def temporal_normalization(d: DASdata, window_sec: Optional[float] = None) -> DASdata:
    """Per-channel temporal amplitude normalization.

    ``window_sec`` None or 0 -> one-bit (the sign of each sample). Otherwise
    running-absolute-mean: divide each sample by a moving average of
    ``|amplitude|`` over ``window_sec`` seconds (a common choice is ~half the
    longest period of interest). Flattens the amplitude envelope so transients
    don't dominate the cross-correlation. Returns a fresh `DASdata`.
    """
    data = d.data
    if not window_sec:                                    # one-bit
        return replace(d, data=np.sign(data).astype(data.dtype, copy=False))
    from scipy.ndimage import uniform_filter1d
    nwin = max(1, int(round(d.fs * window_sec)))
    env = uniform_filter1d(np.abs(data).astype(np.float32, copy=False),
                            size=nwin, axis=-1, mode="nearest")
    return replace(d, data=np.divide(
        data, env, out=np.zeros(data.shape, dtype=np.float32), where=env > 0,
    ))


def spectral_whitening(spec, smooth_bins=1, eps=1e-3, band=None, df=None):
    """Flatten a spectrum's amplitude while keeping its phase (PyTorch).

    ``smooth_bins == 1`` is phase-only whitening: divide by ``|spec|`` in every
    bin. ``smooth_bins > 1`` is running-absolute-mean (RAM) whitening: divide
    by a moving average of ``|spec|`` over that many frequency bins, which
    keeps the broad spectral shape that phase-only erases. The denominator is
    floored at ``eps`` x the row's mean magnitude so empty (out-of-band) bins
    stay ~0 instead of blowing up.

    band : optional ``(f1, f2)`` Hz pass-band. After whitening, bins below
        ``f1`` and above ``f2`` are smoothly suppressed with a ``cos**2`` edge
        taper (1 at the band edge, 0 at DC / Nyquist), reproducing the legacy
        spectral_whitening band-limit so whitening doesn't re-flatten
        out-of-band noise. Needs `df` (the rFFT bin spacing, ``fs / nfft``).

    `spec` is a torch rFFT tensor (last axis = frequency); returns a tensor.
    """
    import torch
    mag = spec.abs()
    k = max(1, int(smooth_bins))
    if k == 1:
        denom = mag                                       # phase-only
    else:
        n = mag.shape[-1]                                 # RAM: moving average of |spec|
        sm = torch.nn.functional.avg_pool1d(
            mag.reshape(-1, 1, n), k, stride=1, padding=k // 2, count_include_pad=False,
        )
        denom = sm[..., :n].reshape(mag.shape)
    floor = eps * mag.mean(dim=-1, keepdim=True).clamp_min(1e-30)
    out = spec / torch.maximum(denom, floor)
    if band is not None:                                  # cos^2 band-limit (legacy)
        if df is None:
            raise ValueError("spectral_whitening(band=...) needs df= (fs / nfft)")
        n = out.shape[-1]
        i1 = int(np.floor(band[0] / df))
        i2 = int(np.ceil(band[1] / df))
        if i1 > 0:
            ramp = torch.cos(torch.linspace(
                np.pi / 2, np.pi, i1, device=out.device, dtype=mag.dtype)) ** 2
            out[..., :i1] = out[..., :i1] * ramp
        if i2 < n:
            ramp = torch.cos(torch.linspace(
                np.pi, np.pi / 2, n - i2, device=out.device, dtype=mag.dtype)) ** 2
            out[..., i2:] = out[..., i2:] * ramp
    return out


def xcorr(a, b, *, whiten=False, fs=None, max_lag=None, eps=1e-3,
            whiten_band=None, device=None):
    """FFT cross-correlation along the time axis (PyTorch).

    Correlates `a` and `b` per channel. `a` is ``(nch, npts)``; `b` is either
    the same shape (channel-wise) or a single ``(npts,)`` trace, broadcast
    against every channel of `a` (a virtual-source gather). Linear
    (zero-padded, not circular) correlation; the lag axis is centered, so
    column ``max_lag`` — or ``npts-1`` when ``max_lag`` is None — is zero lag.

    whiten : spectral whitening before correlating. ``False`` -> none; ``True``
        -> phase-only; a float -> running-absolute-mean whitening with that
        smoothing bandwidth in **Hz** (requires `fs`). See `spectral_whitening`.
        Assumes inputs are already band-limited (bandpass first).
    fs : sampling rate, needed when `whiten` is a Hz bandwidth or `whiten_band`
        is set.
    whiten_band : optional ``(f1, f2)`` Hz pass-band for the whitening (needs
        `fs`); applies the legacy ``cos**2`` band-limit so whitening doesn't
        re-flatten out-of-band noise. See `spectral_whitening`.
    max_lag : keep only +/- this many lag samples (returns ``2*max_lag+1``);
        None returns the full ``2*npts-1``.
    eps : whitening floor, as a fraction of each row's mean magnitude.
    device : torch device for the FFTs (e.g. ``'cuda'``); None uses the inputs'.

    Returns a numpy array ``(nch, 2*npts-1)`` or ``(nch, 2*max_lag+1)``.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError("dasio xcorr needs PyTorch: pip install 'dasio[noise]'") from e

    ta = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
    tb = torch.as_tensor(np.ascontiguousarray(b), dtype=torch.float32, device=device)
    npts = ta.shape[-1]
    nfft = 1 << int(np.ceil(np.log2(2 * npts - 1)))       # next pow2 >= 2N-1 -> linear, no wrap

    fa = torch.fft.rfft(ta, nfft, dim=-1)
    fb = torch.fft.rfft(tb, nfft, dim=-1)
    if whiten:
        if whiten is True:
            smooth_bins = 1                               # phase-only
        elif fs is None:
            raise ValueError("xcorr(whiten=<Hz>) needs fs=; pass whiten=True for phase-only")
        else:
            smooth_bins = max(1, round(float(whiten) * nfft / fs))   # Hz -> bins
        df = None
        if whiten_band is not None:
            if fs is None:
                raise ValueError("xcorr(whiten_band=...) needs fs=")
            df = fs / nfft
        fa = spectral_whitening(fa, smooth_bins, eps, band=whiten_band, df=df)
        fb = spectral_whitening(fb, smooth_bins, eps, band=whiten_band, df=df)

    cc = torch.fft.irfft(fa.conj() * fb, nfft, dim=-1)    # zero lag at index 0; negatives wrap to tail
    cc = torch.roll(cc, npts - 1, dims=-1)[..., :2 * npts - 1]   # center zero lag at index npts-1
    if max_lag is not None:
        c = npts - 1
        cc = cc[..., c - max_lag: c + max_lag + 1]
    return cc.cpu().numpy()


# --- channel-pair gathers ----------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lon/lat points (vectorized).

    A lightweight stand-in for ``obspy.geodetics`` — spherical-earth haversine,
    accurate to a few tenths of a percent, which is plenty for picking
    common-offset channel pairs. All four arguments broadcast, so you can pass
    one point against an array of channels (``haversine_km(lat[i], lon[i], lat,
    lon)``) to get that channel's distance to every other.
    """
    r = 6371.0088                                         # mean earth radius, km
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(a, dtype=float))
                                for a in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def common_shot_pairs(source, channels):
    """Pair one source channel against every channel — a common-shot gather.

    `source` is the fixed source channel index. `channels` is the list of
    receiver indices, or an int ``n`` meaning ``range(n)``. Returns
    ``(ch1, ch2)`` integer arrays with ``ch1`` filled with `source`, suitable
    for `xcorr_stack` / `xcorr_dataset`.
    """
    ch2 = np.arange(channels) if np.isscalar(channels) else np.asarray(channels, dtype=int)
    ch1 = np.full(ch2.shape, int(source), dtype=int)
    return ch1, ch2


def common_offset_pairs(lat, lon, offset_km, tol_km=None):
    """Channel pairs separated by ~`offset_km` — a common-offset gather.

    For each channel `i`, find the forward channel `j > i` whose great-circle
    distance to `i` is closest to `offset_km` (via `haversine_km`), and keep the
    pair when that distance is within `tol_km`. Forward-only search keeps a
    consistent pair orientation and avoids duplicate (i, j)/(j, i). `lat`/`lon`
    are per-channel coordinate arrays (degrees).

    `tol_km` None keeps every channel's nearest-to-offset partner — including
    channels near the far end that have no true match; pass a tolerance (e.g.
    half the channel spacing) to drop those. Returns ``(ch1, ch2)``.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    n = lat.size
    ch1, ch2 = [], []
    for i in range(n - 1):
        d = haversine_km(lat[i], lon[i], lat[i + 1:], lon[i + 1:])
        j = int(np.argmin(np.abs(d - offset_km)))
        if tol_km is None or abs(d[j] - offset_km) <= tol_km:
            ch1.append(i)
            ch2.append(i + 1 + j)
    return np.asarray(ch1, dtype=int), np.asarray(ch2, dtype=int)


# --- segment cross-correlation + stacking ------------------------------------

@dataclass
class CCGather:
    """A stacked cross-correlation gather — what `xcorr_stack` / `xcorr_dataset` return.

    `cc` is the ``(npair, nlag)`` mean cross-correlation over `nseg` stacked
    segments, zero lag at the centre column; `lag_s` is the matching lag axis in
    seconds and `fs` the (post-`preprocess`) sample rate. `plot()` images it via
    `dasio.plot.plot_xcorr` — the same DASdata-style "the object knows how to draw
    itself" convenience, so callers don't juggle `cc` and `lag_s` separately.
    """
    cc: np.ndarray
    lag_s: np.ndarray
    fs: float
    nseg: int

    def __repr__(self):
        npair, nlag = self.cc.shape
        return (f'CCGather({npair} pairs x {nlag} lags, '
                f'fs={self.fs:g} Hz, {self.nseg} segments)')

    def plot(self, **kwargs):
        """Image the gather (lag vs pair); see `dasio.plot.plot_xcorr`."""
        from .plot import plot_xcorr
        return plot_xcorr(self.cc, lag_s=self.lag_s, **kwargs)


def xcorr_stack(
        d: DASdata, ch1, ch2, *,
        seg_sec: float, max_lag_sec: float,
        whiten=False, whiten_band=None, eps: float = 1e-3, chunk: int = 1250, device=None,
    ):
    """Cross-correlate channel pairs over `seg_sec` segments and stack.

    Splits `d` into back-to-back windows of `seg_sec` seconds (the trailing
    remainder is dropped), cross-correlates each pair's matching segments with
    `xcorr`, and averages over the segments — the segment stack. `ch1`/`ch2` are
    equal-length channel-index arrays (e.g. from `common_shot_pairs`); they
    index the channel axis of `d.data`.

    seg_sec, max_lag_sec : segment length and kept lag half-width, in seconds;
        converted to samples with `d.fs`, so this is decimation-agnostic.
    whiten, whiten_band : forwarded to `xcorr` (whitening mode and optional
        ``(f1, f2)`` Hz band-limit).
    chunk : process this many pairs per `xcorr` call to bound memory.

    Returns a `CCGather` (mean cross-correlation over its `nseg` segments), or
    ``None`` when `d` is shorter than one segment. `xcorr_dataset` weights these
    by `nseg` to accumulate across windows.
    """
    ch1 = np.asarray(ch1, dtype=int)
    ch2 = np.asarray(ch2, dtype=int)
    npts_seg = int(round(seg_sec * d.fs))
    max_lag = int(round(max_lag_sec * d.fs))
    npts = (d.data.shape[1] // npts_seg) * npts_seg
    nseg = npts // npts_seg
    if nseg == 0:
        return None

    nlag = 2 * max_lag + 1
    out = np.zeros((ch1.size, nlag), dtype=np.float32)
    for s in range(0, ch1.size, chunk):
        i1 = ch1[s:s + chunk]
        i2 = ch2[s:s + chunk]
        a = d.data[i1, :npts].reshape(-1, npts_seg)       # (npair_chunk*nseg, npts_seg)
        b = d.data[i2, :npts].reshape(-1, npts_seg)
        cc = xcorr(a, b, whiten=whiten, whiten_band=whiten_band, fs=d.fs,
                   max_lag=max_lag, eps=eps, device=device)
        out[s:s + i1.size] = cc.reshape(i1.size, nseg, nlag).mean(axis=1)   # stack segments
    lag_s = np.arange(-max_lag, max_lag + 1) / d.fs
    return CCGather(cc=out, lag_s=lag_s, fs=d.fs, nseg=nseg)


def xcorr_dataset(
        db, ch1, ch2, *,
        max_lag_sec: float, seg_sec: float = 60.0,
        begin=None, end=None, window_sec: float = 3600.0,
        min_ch: int = 0, max_ch: Optional[int] = None,
        prep: Optional[dict] = None,
        whiten=False, whiten_band=None, eps: float = 1e-3, chunk: int = 1250, device=None,
    ):
    """Cross-correlate channel pairs across a whole `DASdb` and stack.

    Walks the catalog's time-contiguous `segments`, reads each segment in
    continuous `window_sec` chunks (`DASdb.read`), runs `preprocess` on each
    chunk, then `xcorr_stack`, accumulating the per-segment sums. The final
    result is the **mean** cross-correlation over every segment of every chunk
    — the dataset-level stack.

    db : a `DASdb` catalog of the raw files to correlate.
    ch1, ch2 : channel-index pair arrays (e.g. from `common_shot_pairs` /
        `common_offset_pairs`); they index the read range ``[min_ch, max_ch)``
        (default = all channels, so indices are absolute). Channel selection is
        the caller's job — restrict the read with `min_ch`/`max_ch` and only
        reference good channels in `ch1`/`ch2` (bad channels read but unpaired).
    min_ch, max_ch : contiguous channel range to read each window.
    begin, end : tz-aware UTC datetimes bounding the run (default = full
        catalog span). Group by day by calling once per day's [begin, end).
    window_sec : continuous read-chunk length; a multiple of `seg_sec` wastes no
        samples at chunk boundaries. Iterating per segment means a segment never
        straddles an acquisition gap.
    prep : kwargs forwarded to `preprocess` (e.g. ``dict(freqmin=0.1,
        freqmax=20, decimate=5, norm_window_sec=0)``); ``{}`` runs the
        `preprocess` defaults.
    whiten, whiten_band, eps, chunk, device : forwarded down to `xcorr` via
        `xcorr_stack`.

    Returns a `CCGather` whose `cc` is the mean cross-correlation over the
    `nseg` segments of every window. Raises if no data falls in the requested
    range. Assumes a uniform sample rate across the catalog.
    """
    ch1 = np.asarray(ch1, dtype=int)
    ch2 = np.asarray(ch2, dtype=int)
    prep = {} if prep is None else dict(prep)

    cc_total = None                                       # nseg-weighted sum of per-window means
    nseg_total = 0
    lag_out = None
    fs_out = None
    for seg in db.segments():
        dt0 = 1.0 / float(seg['fs'].iloc[0])
        s_begin = seg['begin_time'].iloc[0].to_pydatetime()
        s_end = seg['end_time'].iloc[-1].to_pydatetime() + timedelta(seconds=dt0)  # exclusive
        if begin is not None and s_begin < begin:
            s_begin = begin
        if end is not None and s_end > end:
            s_end = end

        t = s_begin
        while t < s_end:
            t2 = min(t + timedelta(seconds=window_sec), s_end)
            d = db.read(t, t2, min_ch=min_ch, max_ch=max_ch, fill_gap=False)
            d = preprocess(d, **prep)
            g = xcorr_stack(
                d, ch1, ch2, seg_sec=seg_sec, max_lag_sec=max_lag_sec,
                whiten=whiten, whiten_band=whiten_band, eps=eps, chunk=chunk, device=device,
            )
            if g is not None:
                if cc_total is None:
                    cc_total = g.cc.astype(np.float64) * g.nseg   # weight by segment count
                    lag_out, fs_out = g.lag_s, g.fs
                else:
                    cc_total += g.cc * g.nseg
                nseg_total += g.nseg
            t = t2

    if cc_total is None:
        raise RuntimeError('xcorr_dataset: no data in the requested range')
    return CCGather(cc=(cc_total / nseg_total).astype(np.float32),
                    lag_s=lag_out, fs=fs_out, nseg=nseg_total)
