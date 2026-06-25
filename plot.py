"""Minimal plotting helpers for DASdata.

Two free functions (`imshow`, `wiggle`) and a bound `_PlotAccessor`
exposed via `DASdata.plot`. Matplotlib is imported lazily inside each
function body so a bare `import dasio` (or `import dasdata`) stays
free of the matplotlib startup cost — the load only fires when an
actual plot is requested.

For richer plot types (cc2d, fk, xcorr) keep using
`realTimeMonitor.lfdas.DASplot`; this module covers the everyday
"show me the data" case in ~60 lines without dragging that whole
library into dasio.
"""
from typing import Optional, Tuple

import numpy as np


def imshow(
        d,
        ax=None,
        ch_range: Optional[Tuple[int, int]] = None,
        t_range: Optional[Tuple] = None,
        style: str = 'seismic',
        usedatetime: bool = False,
        perc: float = 99.5,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        cmap: str = 'seismic',
        cbar: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
    """Color-mapped image of a `DASdata` (regular grid).

    Parameters
    ----------
    d : DASdata
    ax : matplotlib Axes, optional
        Plot target; one is created if not given.
    ch_range, t_range : tuples, optional
        Forwarded to `DASdata.truncate` for the displayed window.
    style : {'seismic', 'normal'}
        `'seismic'` (default, matches legacy DASutils) puts channels
        on the x-axis and time on the y-axis with time growing
        downward — the traditional seismic-record orientation.
        `'normal'` puts time on the x-axis and channels on the y-axis
        — easier to read as a stack of time-series.
    usedatetime : bool
        Show the time axis as `datetime64` ticks (True) instead of
        seconds-from-`t0_sec` (False, default).
    perc : float
        Percentile (of `|data|`) used to set symmetric `vmin` / `vmax`
        when the latter aren't provided. Default 99.5 trims out the
        loudest 0.5 % so background noise stays visible.
    vmin, vmax : floats, optional
        Manual color-scale bounds; bypass `perc` when both are given.
    cmap : str or Colormap
        Default `'seismic'` (red-white-blue), matches legacy DASutils.
    cbar : bool
    figsize : (w, h), optional
        Figure size when `ax` is created here. Defaults to (8, 4)
        for `style='seismic'` and (4, 8) for `style='normal'`.
        Ignored when `ax` is supplied.
    **kwargs : forwarded to `ax.imshow`

    Returns
    -------
    (ax, im) so callers can attach annotations / set titles / etc.
    """
    import matplotlib.pyplot as plt
    from matplotlib import dates as mdates

    if style not in ('seismic', 'normal'):
        raise ValueError(f"style must be 'seismic' or 'normal', got {style!r}")

    sub = d if (ch_range is None and t_range is None) else d.truncate(
        ch_range=ch_range, t_range=t_range,
    )

    if vmin is None or vmax is None:
        v = float(np.nanpercentile(np.abs(sub.data), perc))
        if vmin is None:
            vmin = -v
        if vmax is None:
            vmax = v

    if ax is None:
        if figsize is None:
            figsize = (8, 4) if style == 'seismic' else (4, 8)
        _, ax = plt.subplots(figsize=figsize)

    # Time-axis values for the extent, in either seconds or mpl date
    # units; we apply a date formatter on the chosen axis afterwards.
    if usedatetime:
        dt_axis = sub.datetime_axis
        t_left = mdates.date2num(dt_axis[0].astype('datetime64[us]').astype(object))
        t_right = mdates.date2num(dt_axis[-1].astype('datetime64[us]').astype(object))
    else:
        tax = sub.time_axis
        t_left, t_right = float(tax[0]), float(tax[-1])

    if style == 'normal':
        # data shape (nx, nt) — imshow it directly: rows=channels, cols=time.
        # extent = (left, right, bottom, top); channel 0 at top.
        extent = (t_left, t_right, sub.nx - 1, 0)
        im = ax.imshow(
            sub.data, aspect='auto', cmap=cmap,
            vmin=vmin, vmax=vmax, extent=extent, **kwargs,
        )
        ax.set_xlabel('time' if usedatetime else 't (s)')
        ax.set_ylabel('channel')
        date_axis = ax.xaxis if usedatetime else None
    else:
        # 'seismic': transpose so rows=time, cols=channels; flip y so
        # time grows downward (small-time at top, large-time at bottom).
        extent = (0, sub.nx - 1, t_right, t_left)
        im = ax.imshow(
            sub.data.T, aspect='auto', cmap=cmap,
            vmin=vmin, vmax=vmax, extent=extent, **kwargs,
        )
        ax.set_xlabel('channel')
        ax.set_ylabel('time' if usedatetime else 't (s)')
        date_axis = ax.yaxis if usedatetime else None

    if date_axis is not None:
        date_axis.axis_date() if hasattr(date_axis, 'axis_date') else None
        date_axis.set_major_locator(mdates.AutoDateLocator())
        date_axis.set_major_formatter(
            mdates.AutoDateFormatter(date_axis.get_major_locator())
        )

    if cbar:
        ax.figure.colorbar(im, ax=ax, label=getattr(d, 'unit', None))
    return ax, im


def wiggle(
        d,
        ax=None,
        ch_range: Optional[Tuple[int, int]] = None,
        t_range: Optional[Tuple] = None,
        n_max_ch: Optional[int] = 100,
        style: str = 'seismic',
        scale: float = 1.0,
        color: str = 'k',
        lw: float = 0.6,
        usedatetime: bool = False,
        normalize: bool = True,
        figsize: Optional[Tuple[float, float]] = None,
        **kwargs,
    ):
    """Per-channel wiggle traces of a `DASdata`.

    Each retained channel becomes a line offset by its (original)
    channel index, optionally amplitude-normalized so all traces
    share the same visual range.

    `style='seismic'` (default) places channels on the x-axis and
    time on the y-axis growing downward — the traditional seismic
    record orientation. `style='normal'` places time on the x-axis
    and channel on the y-axis (descending), which reads more like
    a stack of time-series.

    `n_max_ch` caps the number of traces drawn — when
    `sub.nx > n_max_ch` the channels are decimated with stride
    `ceil(sub.nx / n_max_ch)`. Defaults to 100; pass `None` to draw
    every channel. This stops a casual `d.plot.wiggle()` on a
    full-array DASdata from rendering thousands of overlapping lines.
    Use `ch_range` first if you want a specific window.
    """
    import matplotlib.pyplot as plt

    if style not in ('seismic', 'normal'):
        raise ValueError(f"style must be 'seismic' or 'normal', got {style!r}")

    sub = d if (ch_range is None and t_range is None) else d.truncate(
        ch_range=ch_range, t_range=t_range,
    )

    if n_max_ch is not None and sub.nx > n_max_ch:
        step = int(np.ceil(sub.nx / n_max_ch))
        ch_indices = np.arange(0, sub.nx, step)
    else:
        step = 1
        ch_indices = np.arange(sub.nx)

    data = sub.data[ch_indices]
    if normalize:
        peak = np.maximum(np.abs(data).max(axis=1, keepdims=True), 1e-30)
        data = data / peak
    # Scale by `step` so a peak ±1 normalized trace just fills the
    # gap between adjacent traces — keeps visual density constant
    # regardless of n_max_ch decimation.
    data = data * scale * step

    t = sub.datetime_axis if usedatetime else sub.time_axis

    if ax is None:
        if figsize is None:
            n_ch = len(ch_indices)
            figsize = (
                (max(2, n_ch * 0.25), 10) if style == 'seismic'
                else (10, max(2, n_ch * 0.25))
            )
        _, ax = plt.subplots(figsize=figsize)

    if style == 'seismic':
        # channel on x, time on y growing downward
        for i, ch in enumerate(ch_indices):
            ax.plot(data[i] + ch, t, color=color, lw=lw, **kwargs)
        ax.set_xlim(-step, int(ch_indices[-1]) + step)
        ax.set_ylim(t[-1], t[0])                          # inverted
        ax.set_xlabel('channel')
        ax.set_ylabel('time' if usedatetime else 't (s)')
    else:
        # 'normal': time on x, channel on y (descending so ch 0 is at top)
        for i, ch in enumerate(ch_indices):
            ax.plot(t, data[i] + ch, color=color, lw=lw, **kwargs)
        ax.set_xlim(t[0], t[-1])
        ax.set_ylim(int(ch_indices[-1]) + step, -step)    # inverted
        ax.set_xlabel('time' if usedatetime else 't (s)')
        ax.set_ylabel('channel')
    return ax


class _PlotAccessor:
    """Bound plot interface for a single DASdata.

    Reachable as `d.plot`. `d.plot()` is shorthand for `d.plot.imshow()`;
    `d.plot.imshow(...)` and `d.plot.wiggle(...)` reach the named
    plotters. `pcolormesh` will be added when a non-uniform channel
    axis use case shows up (truncate + uniform `dx` covers the rest).
    """
    __slots__ = ('_d',)

    def __init__(self, d):
        self._d = d

    def __call__(self, **kw):
        return self.imshow(**kw)

    def imshow(self, **kw):
        return imshow(self._d, **kw)

    def wiggle(self, **kw):
        return wiggle(self._d, **kw)
