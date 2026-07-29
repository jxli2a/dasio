"""Interactive in-memory viewer for a `DASdata`.

Rendering is delegated to fastplotlib (GPU, via pygfx/wgpu) and displayed in
JupyterLab; this module supplies only the data path. fastplotlib is imported
lazily inside `view()` so `pool_maxabs` and the chain helper stay importable —
and testable — on a bare dasio install.
"""
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .utils import default_nthreads

# Channel bands are pooled concurrently: numpy's reductions release the GIL, so
# threads scale here (measured 926 ms -> 108 ms for a 1250 x 180000 window).
# One module-level pool — a repool fires on every zoom, and per-call executor
# setup would show up against a ~100 ms budget.
_POOL = None


def _pool():
    global _POOL
    if _POOL is None:
        _POOL = ThreadPoolExecutor(min(32, default_nthreads()))
    return _POOL


def pool_maxabs(a: np.ndarray, ncol: int):
    """Reduce `a` along time to at most `ncol` columns, keeping each bin's peak.

    Returns `(pooled, kt)` where `kt` is the number of source samples per
    output column. Decimation for display has to preserve the extreme value,
    not sample the bin: a bare stride (and equally the GPU's own minification)
    drops short arrivals outright, which for an event-hunting waterfall is a
    correctness bug rather than a cosmetic one.

    The time axis is zero-padded up to a whole number of bins before the
    reshape, so the tail samples are pooled rather than truncated; 0 never wins
    an |x| argmax over real data.
    """
    nx, nt = a.shape
    kt = max(1, -(-nt // ncol))              # ceil, so the output never exceeds ncol
    nout = -(-nt // kt)
    pad = nout * kt - nt

    def band(sl):
        v = a[sl]
        # Pad per band rather than up front: padding the whole array first costs
        # a full-size copy, which on a 0.9 GB window dominates the pooling.
        if pad:
            v = np.pad(v, ((0, 0), (0, pad)))
        v = v.reshape(v.shape[0], nout, kt)
        return np.take_along_axis(v, np.abs(v).argmax(2)[:, :, None], 2)[:, :, 0]

    bands = [slice(i, min(i + 64, nx)) for i in range(0, nx, 64)]
    pooled = np.concatenate(list(_pool().map(band, bands)), axis=0)
    return pooled, kt


def clim_of(a: np.ndarray, perc: float = 99.5) -> float:
    """Symmetric colour limit: the `perc`-th percentile of |a|, as `dasio.plot` uses.

    Returns 1.0 rather than 0 for a flat window — an all-zero slice or a dead
    channel range otherwise divides the image by zero, and the resulting NaNs
    cast to uint8 are undefined. `nanpercentile` because a single NaN sample
    would otherwise poison the limit for the whole view.
    """
    v = float(np.nanpercentile(np.abs(a), perc))
    return v if np.isfinite(v) and v > 0 else 1.0


def default_band(fs: float) -> tuple:
    """Sensible (fmin, fmax) band-pass corners for a given sample rate.

    Both corners scale with `fs`. Pinning `fmin` at 1 Hz while `fmax` tracked
    the rate inverted the band below ~2.5 Hz — and the C kernel does not reject
    `fmin > fmax`, it quietly returns near-zero data, so a 1 Hz catalog opened
    to a blank panel. Normal seismic rates still get the familiar 1-20 Hz.
    """
    fmax = min(20.0, 0.4 * fs)
    return round(min(1.0, fmax / 20.0), 4), round(fmax, 4)


def apply_chain(d, differentiate=False, detrend=False, taper_sec=None,
                med_t=0, med_x=0, bandpass=None, integrate=False,
                common_mode=None):
    """Run the processing chain over a whole `DASdata`, returning a new one.

    Order is fixed: differentiate, detrend, taper, median (time then channel),
    bandpass, integrate, common-mode.

    The medians run *before* the band-pass on purpose — a band-pass smears a
    spike across its whole passband rather than removing it, so despiking
    afterwards is too late.

    The rest matches `noise.preprocess`, so the viewer and the noise pipeline
    agree on what "filtered" means, except that **integrate runs after
    detrend** — `integrate_time` accumulates in the array's own dtype, so
    integrating float32 that still carries its DC offset is catastrophic
    (115 % max relative error on real data, versus 3e-6 detrended first).

    `bandpass` is `(fmin, fmax, order, zerophase)`, `common_mode` is
    `(ch_min, ch_max)`, `taper_sec` is a taper *duration* — exposing Tukey
    `alpha` directly would silently scale down whole minutes of a long window.

    Every step passes `copy=False`: the kernels behind `dasio.processing` are
    already out-of-place, so nothing aliases the caller's array, and skipping
    the defensive copies takes a 4-step chain over a 0.9 GB window from 2.6 s
    to 0.9 s.
    """
    from . import processing as p

    if differentiate:
        d = p.differentiate(d, copy=False)
    if detrend:
        d = p.detrend(d, copy=False)
    if taper_sec:
        d = p.taper(d, alpha=min(1.0, 2.0 * taper_sec / (d.nt * d.dt)), copy=False)
    if med_t and med_t > 1:
        d = p.median_filter_1d(d, int(med_t), axis='t', copy=False)
    if med_x and med_x > 1:
        d = p.median_filter_1d(d, int(med_x), axis='x', copy=False)
    if bandpass:
        fmin, fmax, order, zerophase = bandpass
        # Checked here because the kernel does not: it returns near-zero data
        # for an inverted band and passes the signal through unchanged for a
        # corner past Nyquist, either of which looks like broken data rather
        # than a bad parameter.
        if not 0 <= fmin < fmax:
            raise ValueError(f'band-pass needs 0 <= fmin < fmax, got {fmin} and {fmax}')
        if fmax >= 0.5 * d.fs:
            raise ValueError(
                f'band-pass fmax={fmax} is at or above Nyquist ({0.5 * d.fs}) '
                f'for fs={d.fs}'
            )
        d = p.bandpass(d, fmin, fmax, order=order, zerophase=zerophase, copy=False)
    if integrate:
        d = p.integrate(d, copy=False)
    if common_mode:
        ch_min, ch_max = common_mode
        d = p.subtract_common_mode(d, ch_min=ch_min, ch_max=ch_max, copy=False)
    return d


# ipywidgets defaults every control to the full container width, which is what
# turns a dozen inputs into an unreadable stack — so widths are explicit here.
_NUM = dict(layout=dict(width='80px'), style={'description_width': '0px'})

# Ceiling on the pan-mode texture. The whole array is uploaded so that dragging
# always lands on real data, and this bounds what that costs in VRAM.
_MAX_TEXELS = 24_000_000

# Ceiling on texture rows. Generous against a panel of a few hundred pixels, so
# a browser zoom still has detail to reveal, while keeping the channel axis off
# the GPU's minifier — see `_draw`.
_MAX_ROWS = 2048


def window_bounds(d, t_lo, t_hi, ch_lo, ch_hi):
    """Requested view clamped to the array, as row/sample indices.

    Returns `(i0, i1, a0, a1)` — half-open, always at least one sample and one
    channel wide. `t_lo`/`t_hi` are seconds in `d`'s own frame and
    `ch_lo`/`ch_hi` are *fiber* channel numbers, so they map through `ch0`/`dch`;
    reading them as row indices silently mislabels any `min_ch` read.
    """
    i0 = max(0, int(round((t_lo - d.t0_sec) / d.dt)))
    i1 = min(d.nt, int(round((t_hi - d.t0_sec) / d.dt)))
    a0 = max(0, min((int(ch_lo) - d.ch0) // d.dch, d.nx - 1))
    a1 = max(a0 + 1, min(-(-(int(ch_hi) - d.ch0) // d.dch), d.nx))
    return i0, max(i0 + 1, i1), a0, a1


def clamp_to_rect(point, rect):
    """Clamp `(x, y)` into `(x0, x1, y0, y1)`; `None` rect leaves it alone."""
    if not rect:
        return point
    return (min(max(point[0], rect[0]), rect[1]),
            min(max(point[1], rect[2]), rect[3]))


def screen_to_world(subplot, ev):
    """Pointer event to world `(x, y)`, or None when it misses the subplot."""
    p = subplot.map_screen_to_world((ev.x, ev.y))
    return None if p is None else (float(p[0]), float(p[1]))


def view_rect(subplot):
    """Camera view as `(x0, x1, y0, y1)` in world units.

    The rulers already carry exactly this — fastplotlib recomputes their
    start/end each frame from the viewport corners — so read it off them
    rather than re-deriving it from the camera matrix.
    """
    return (float(subplot.axes.x.start_value), float(subplot.axes.x.end_value),
            float(subplot.axes.y.start_value), float(subplot.axes.y.end_value))


def fit_view(subplot, x0, x1, y0, y1, pan_enabled=False):
    """Show `(x0, x1, y0, y1)` with room for the tick labels.

    Margins are sized in *pixels*, not as a fraction of the data range: tick
    labels need a fixed ~52 px to the left and ~34 px below, so a proportional
    margin leaves acres of white space on a wide view and too little on a
    narrow one. Only those two sides get one, so the data reaches the other
    edges. Both are clamped to 15 %: the viewport is not always sized on the
    first draw, and a canvas still negotiating its width reports a few pixels,
    where 52/1 would span fifty times the data and shrink it to a sliver.

    The rulers are pinned to the data boundary, as matplotlib's spines are in
    `dasio.plot` — the margin exists to give their labels room, and a ruler
    floating out in it reads as an axis for data that is not there. Except in
    pan mode, where leaving `intersection` unset makes fastplotlib re-derive
    the placement from the camera each frame: that is what holds the axis
    still while the data moves under it.
    """
    vw, vh = subplot.viewport.logical_size
    fx = 52.0 / vw if vw and vw > 120 else 0.06
    fy = 34.0 / vh if vh and vh > 80 else 0.06
    mx, my = min(fx, 0.15) * (x1 - x0), min(fy, 0.15) * (y1 - y0)
    subplot.camera.show_rect(x0 - mx, x1, y0, y1 + my)
    subplot.axes.intersection = None if pan_enabled else (x0, y1, 0)


def sample_at(d, t_sec, channel):
    """Value at (time in seconds, fiber channel), or None if outside the array.

    The readout needs the actual sample, not the pooled display value, so this
    indexes the in-memory array directly — no GPU readback, no texture lookup.
    Returns `(row, col, amplitude)`.
    """
    row = int(round((channel - d.ch0) / d.dch))
    col = int(round((t_sec - d.t0_sec) / d.dt))
    if not (0 <= row < d.nx and 0 <= col < d.nt):
        return None
    return row, col, float(d.data[row, col])


def _lbl(text, px=None):
    import ipywidgets as w
    return w.HTML(f"<span style='font-size:11px;color:#555'>{text}</span>",
                  layout=w.Layout(width=f'{px}px') if px else w.Layout())


def _cbar_png(cmap_name: str, width: int = 150, height: int = 13) -> bytes:
    """Horizontal colormap ramp as PNG bytes, for the control panel.

    fastplotlib 0.6 ships no colour bar, and giving one its own subplot left a
    thin bar stranded in a wide empty column. A small image widget sits in the
    panel instead, beside the gain slider that changes it.

    Only the gradient lives here — the end labels are separate HTML, so
    dragging the gain slider updates text rather than re-encoding a PNG.
    """
    from io import BytesIO

    from matplotlib import colormaps
    from PIL import Image

    ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)
    rgba = (colormaps[cmap_name](ramp) * 255).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(np.repeat(rgba[None, :, :3], height, axis=0)).save(buf, 'PNG')
    return buf.getvalue()


def _row(title, *widgets):
    import ipywidgets as w
    return w.HBox([w.HTML(f"<b style='font-size:11px'>{title}</b>",
                          layout=w.Layout(width='74px'))] + list(widgets),
                  layout=w.Layout(align_items='center', margin='1px 0'))


def view(d, style: str = 'seismic', ncol: int = 2800, perc: float = 99.5,
         figsize=(1000, 720)):
    """Interactive viewer for an in-memory `DASdata`. Returns the canvas widget.

    Waterfall over wiggles, sharing one axis. Install the libraries with the
    `[viewer]` extra and run it from a Jupyter kernel you already have — the
    extra deliberately does not pull JupyterLab itself. `rendercanvas` picks
    its anywidget backend in a kernel, which works in JupyterLab and VSCode
    alike.

    `style` matches `dasio.plot`: `'seismic'` (default) puts channels on x and
    time on y running **downward**, the traditional seismic-record orientation,
    with wiggle traces hanging vertically side by side. `'normal'` puts time on
    x and channels on y — easier to read as a stack of time-series, and the
    better choice for a long window across few channels.

    **The view is set by typing ranges, not by dragging.** Mouse pan/zoom is
    off by default, on purpose: it silently desynchronises the range boxes from
    what is on screen, and a DAS window is naturally specified as "these
    seconds, these channels" rather than reached by dragging. Set the time and
    channel range and press Zoom, or Full to go back to everything. Ticking
    `pan` enables dragging and the scroll wheel; the range boxes then follow
    the camera when the gesture ends.

    Zoom is not a camera move — the requested slice is re-pooled at full
    resolution, so narrowing the time range genuinely resolves more detail
    rather than magnifying the same pixels.

    Cost of each control: waterfall gain, wiggle gain and colormap are GPU
    uniform updates (free, drag them). Zoom re-pools (tens of ms). Apply
    re-runs the processing chain over the whole array and is the only control
    that costs real time.
    """
    # Validate before the heavy import, so a typo fails immediately instead of
    # after fastplotlib and a GPU context have been brought up.
    if style not in ('seismic', 'normal'):
        raise ValueError(f"style must be 'seismic' or 'normal', got {style!r}")
    seismic = style == 'seismic'

    try:
        import fastplotlib as fpl
        import ipywidgets as w
        from fastplotlib.utils.types import SelectorColorStates as _PlaneColor
        from IPython.display import display
    except ImportError as e:
        raise ImportError(
            f"the viewer needs fastplotlib and ipywidgets ({e.name} is "
            f"missing): pip install 'dasio[viewer]'. They are an extra because "
            f"they pull a GPU stack, ~36 packages, and everything else in "
            f"dasio -- readers, catalogs, desample, picking -- is headless."
        ) from e
    from datetime import timedelta as _timedelta

    base = d
    t_end = base.t0_sec + base.nt * base.dt
    state = {'proc': d, 'img': None, 'stack': None, 'band': None,
          'lim': None, 'clim': 1.0, 'ka': 1}

    _white = (1.0, 1.0, 1.0, 1.0)
    fig = fpl.Figure(shape=(2, 1), size=figsize,
                     names=[[f'waterfall  ({style})'], [f'wiggles  ({style})']])
    wf, wig = fig[0, 0], fig[1, 0]
    for sp in (wf, wig):
        sp.camera.maintain_aspect = False
        sp.controller.enabled = False                  # typed ranges are authoritative
        sp.background_color = (1, 1, 1, 1)             # default is black
        # `Ruler.color` is singular; the plural spelling is silently accepted
        # as a new attribute and does nothing, leaving white ticks on white.
        for ruler in (sp.axes.x, sp.axes.y):
            ruler.color = '#222222'
        # The band behind the subplot title is a mesh in the figure's *underlay*
        # scene, not the subplot background — so `background_color` above leaves
        # it dark grey. Recolour that mesh, and darken the title text, which
        # ships white to suit the dark default.
        #
        # `plane_color` carries three states and the layout engine swaps to
        # `highlight` on hover and `action` on click, so setting only the
        # current colour makes the dark band reappear the moment the pointer
        # crosses the panel. Shadow the class-level states per instance.
        sp.frame.plane_color = _PlaneColor(idle=_white, highlight=_white,
                                           action=_white)
        sp.frame.plane.material.color = _white
        sp.frame.title_graphic.face_color = '#111111'

    # ---- widgets -----------------------------------------------------------
    t0 = w.FloatText(value=round(base.t0_sec, 3), **_NUM)
    t1 = w.FloatText(value=round(t_end, 3), **_NUM)
    c0 = w.IntText(value=base.ch0, **_NUM)
    c1 = w.IntText(value=base.ch0 + base.nx * base.dch, **_NUM)
    zoom_btn = w.Button(description='Zoom', button_style='info',
                        layout=w.Layout(width='72px'))
    full_btn = w.Button(description='Full', layout=w.Layout(width='60px'))
    pan_on = w.Checkbox(value=False, description='pan', indent=False,
                        layout=w.Layout(width='60px'))

    _fmin, _fmax = default_band(base.fs)
    f0 = w.FloatText(value=_fmin, **_NUM)
    f1 = w.FloatText(value=_fmax, **_NUM)
    order = w.IntText(value=4, **_NUM)
    zph = w.Checkbox(value=True, description='zero-phase', indent=False,
                     layout=w.Layout(width='104px'))
    bp_on = w.Checkbox(value=True, description='band-pass', indent=False,
                       layout=w.Layout(width='98px'))
    dt_on = w.Checkbox(value=True, description='detrend', indent=False,
                       layout=w.Layout(width='84px'))
    di_on = w.Checkbox(value=False, description='d/dt', indent=False,
                       layout=w.Layout(width='64px'))
    in_on = w.Checkbox(value=False, description='∫dt', indent=False,
                       layout=w.Layout(width='62px'))
    cm_on = w.Checkbox(value=False, description='common-mode', indent=False,
                       layout=w.Layout(width='120px'))
    cm0 = w.IntText(value=base.ch0, **_NUM)
    cm1 = w.IntText(value=base.ch0 + base.nx * base.dch, **_NUM)
    tap = w.FloatText(value=0.0, **_NUM)
    mt = w.IntText(value=0, **_NUM)
    mx = w.IntText(value=0, **_NUM)
    apply_btn = w.Button(description='Apply', button_style='primary',
                         layout=w.Layout(width='76px'))
    hover = w.HTML(layout=w.Layout(width='auto'))
    status = w.HTML()

    def _gain(vmax):
        return w.FloatLogSlider(value=1.0, base=10, min=-2, max=vmax, step=0.02,
                                readout=False, continuous_update=True,
                                layout=w.Layout(width='210px'),
                                style={'description_width': '0px'})

    img_gain, img_gain_n = _gain(3), w.FloatText(value=1.0, **_NUM)
    wig_gain, wig_gain_n = _gain(2), w.FloatText(value=1.0, **_NUM)
    w.jslink((img_gain, 'value'), (img_gain_n, 'value'))
    w.jslink((wig_gain, 'value'), (wig_gain_n, 'value'))
    cmap = w.Dropdown(options=['bwr', 'seismic', 'gray', 'viridis', 'plasma'],
                      value='bwr', layout=w.Layout(width='100px'),
                      style={'description_width': '0px'})
    trace_stride = w.IntText(value=max(1, base.nx // 30), **_NUM)
    cbar_img = w.Image(value=_cbar_png('bwr'), format='png',
                       layout=w.Layout(width='150px', height='13px',
                                       margin='0 3px'))
    cb_lo, cb_hi = w.HTML(), w.HTML()

    # ---- helpers -----------------------------------------------------------
    def _bounds():
        return window_bounds(state['proc'], t0.value, t1.value, c0.value, c1.value)

    def _set_clim(*_):
        if state['img'] is None:
            return
        c = state['clim'] / max(img_gain.value, 1e-6)
        state['img'].vmin, state['img'].vmax = -c, c
        cb_lo.value = f"<code style='font-size:10px'>-{c:.2e}</code>"
        cb_hi.value = (f"<code style='font-size:10px'>+{c:.2e} "
                       f"{state['proc'].units}</code>")

    def _set_wiggle_gain(*_):
        """Amplitude only — each trace stays pinned to its own channel row, so
        the y-axis reads channel numbers. Unit-normalised traces at gain 1 just
        touch their neighbours; higher gains overlap, which is what weak
        arrivals need."""
        if state['stack'] is None:
            return
        p = state['proc']
        step = max(1, int(trace_stride.value))
        _, _, a0, _ = _bounds()
        amp = 0.5 * step * p.dch * wig_gain.value
        for i, g in enumerate(state['stack'].graphics):
            ch = p.ch0 + (a0 + i * step) * p.dch
            if seismic:                      # traces run down the page, side by side
                g.world_object.local.position = (ch, 0.0, 0.0)
                g.world_object.local.scale = (amp, 1.0, 1.0)
            else:                            # traces run across, stacked vertically
                g.world_object.local.position = (0.0, ch, 0.0)
                g.world_object.local.scale = (1.0, amp, 1.0)

    def _draw(*_):
        """Re-pool and rebuild both panels.

        In pan mode the texture covers the WHOLE array rather than just the
        visible window, so dragging reveals real data instead of running off
        the edge into blank canvas. Resolution is then whatever fits the texel
        budget; press Zoom to come back to full detail for a window.
        """
        p = state['proc']
        i0, i1, a0, a1 = _bounds()
        if pan_on.value:
            # Load everything, at the finest resolution the budget allows —
            # scaled up for a zoomed-in view so panning does not also mean
            # losing detail.
            s_i0, s_i1, s_a0, s_a1 = 0, p.nt, 0, p.nx
            zoom = p.nt / max(1, i1 - i0)
            ncol_eff = int(min(ncol * zoom, _MAX_TEXELS / max(1, s_a1 - s_a0), 16384))
            ncol_eff = max(ncol_eff, 256)
        else:
            s_i0, s_i1, s_a0, s_a1 = i0, i1, a0, a1
            ncol_eff = ncol
        sub = np.ascontiguousarray(p.data[s_a0:s_a1, s_i0:s_i1])
        pooled, kt = pool_maxabs(sub, ncol_eff)
        # Pool channels by the same rule. Left to the GPU the channel axis gets
        # nearest-neighbour minification, which is what `pool_maxabs` exists to
        # avoid: on 15686 channels over an ~800 px panel it kept 5 % of the
        # strongest texels and lost 21 % of peak amplitude. It also pays for
        # itself — the extra pass runs on the already-reduced array (+13 ms) and
        # shrinks the texture 20x, 75 -> 3.8 MB, for ~70 ms less upload.
        if pooled.shape[0] > _MAX_ROWS:
            pooled, ka = pool_maxabs(pooled.T, _MAX_ROWS)
            pooled = np.ascontiguousarray(pooled.T)
        else:
            ka = 1
        # Colour limit from the samples, not from the texture: pooling keeps
        # per-bin maxima, so a percentile of `pooled` sits well above the same
        # percentile of the data — 6.7x on a 2 min x 15686 ch window — and the
        # waterfall washes out the more it pools. A strided subsample is ample
        # for a percentile and costs ~1 ms against a full pass over gigabytes.
        state['ka'] = ka
        state['clim'] = clim_of(
            sub[::max(1, sub.shape[0] // 256), ::max(1, sub.shape[1] // 4096)], perc)
        ts, te = p.t0_sec + s_i0 * p.dt, p.t0_sec + s_i1 * p.dt

        # The graphic is rebuilt rather than reassigned: a re-pool at a new
        # window generally changes the array shape, which a texture cannot
        # absorb in place.
        wf.clear()
        # 'seismic' transposes so image rows are time; since fastplotlib draws
        # row 0 at the top, time then runs down the page with no extra flip.
        state['img'] = wf.add_image(pooled.T if seismic else pooled, cmap=cmap.value,
                                 vmin=-state['clim'], vmax=state['clim'])
        wo = state['img'].world_object.local
        ch_img = p.ch0 + s_a0 * p.dch
        # One texel now spans `ka` channels and `kt` samples.
        wo.scale = ((p.dch * ka, p.dt * kt, 1.0) if seismic
                    else (p.dt * kt, p.dch * ka, 1.0))
        wo.position = (ch_img, ts, -1.0) if seismic else (ts, ch_img, -1.0)
        _set_clim()

        # Clamp the rubber band to what is actually loaded, not to the view.
        ch_lo = p.ch0 + s_a0 * p.dch
        ch_hi = p.ch0 + s_a1 * p.dch
        state['lim'] = ((ch_lo, ch_hi, ts, te) if seismic else (ts, te, ch_lo, ch_hi))
        state['band'] = None                # rubber band belongs to the old frame

        step = max(1, int(trace_stride.value))
        sel = p.data[a0:a1:step, i0:i1]
        tstep = max(1, sel.shape[1] // 4000)          # screen-resolution cap
        tr = sel[:, ::tstep].astype(np.float32)
        tr = tr / np.maximum(np.abs(tr).max(axis=1, keepdims=True), 1e-30)
        t_vis = p.t0_sec + i0 * p.dt
        tt = (t_vis + np.arange(0, sel.shape[1], tstep) * p.dt).astype(np.float32)
        wig.clear()
        state['stack'] = wig.add_line_stack(
            [np.column_stack([y, tt] if seismic else [tt, y]) for y in tr],
            separation=0.0, colors='black')
        _set_wiggle_gain()

        # `fit_view` adds the margin: show_rect fits the camera to exactly the
        # rectangle it is given, and the rulers are drawn over the viewport
        # edges, so an exact fit hides the first row/column behind the axes.
        v_lo, v_hi = p.ch0 + a0 * p.dch, p.ch0 + a1 * p.dch
        wig_lo = v_lo - step * p.dch * 0.6
        wig_hi = v_hi + step * p.dch * 0.6
        panning = pan_on.value
        if seismic:
            fit_view(wf, v_lo, v_hi, ts, te, panning)
            fit_view(wig, wig_lo, wig_hi, ts, te, panning)
        else:
            fit_view(wf, ts, te, v_lo, v_hi, panning)
            fit_view(wig, ts, te, wig_lo, wig_hi, panning)
        # Force the down-the-page direction on both panels. fastplotlib flips y
        # for image subplots but not for line subplots, so left alone the two
        # panels would disagree about which way time (or channel) grows.
        for sp in (wf, wig):
            sp.camera.local.scale_y = -abs(sp.camera.local.scale[1])

        status.value = (f"<code style='font-size:11px'>{te - ts:.3g} s x "
                        f"{a1 - a0} ch | {kt} smp/px | {state['ka']} ch/px | "
                        f"{state['proc'].units}</code>")

    def _full(_=None):
        t0.value, t1.value = round(base.t0_sec, 3), round(t_end, 3)
        c0.value, c1.value = base.ch0, base.ch0 + base.nx * base.dch
        _draw()

    def _apply(_=None):
        """Re-run the processing chain from the widget values, then redraw."""
        status.value = "<span style='color:#b60'>processing…</span>"
        try:
            state['proc'] = apply_chain(
                base,
                differentiate=di_on.value, detrend=dt_on.value,
                taper_sec=tap.value or None,
                med_t=mt.value, med_x=mx.value,
                bandpass=((f0.value, f1.value, order.value, zph.value)
                          if bp_on.value else None),
                integrate=in_on.value,
                common_mode=(((cm0.value - base.ch0) // base.dch,
                              -(-(cm1.value - base.ch0) // base.dch))
                             if cm_on.value else None),
            )
        except ValueError as e:                    # bad corner, inverted band…
            status.value = f"<span style='color:#c00'>{e}</span>"
            return
        _draw()

    def _apply_box(x0, x1, y0, y1):
        """Adopt a world-coordinate rectangle as the new view.

        Axis meaning depends on `style`, so the mapping onto the range boxes
        swaps: 'seismic' has channels on x and time on y, 'normal' the reverse.
        """
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if seismic:
            c0.value, c1.value = int(round(x0)), int(round(x1))
            t0.value, t1.value = round(y0, 3), round(y1, 3)
        else:
            t0.value, t1.value = round(x0, 3), round(x1, 3)
            c0.value, c1.value = int(round(y0)), int(round(y1))
        _draw()

    # ---- click-drag a box on the waterfall to zoom into it ------------------
    drag = {}

    def _world(ev):
        return screen_to_world(wf, ev)

    def _clamp(p):
        return clamp_to_rect(p, state.get('lim'))

    def _band(x0, x1, y0, y1):
        """Draw/refresh the rubber band as a plain closed line.

        Deliberately not a `RectangleSelector`: each one registers its own
        pointer_down/move/up/click handlers on the renderer and never releases
        them, so rebuilding one per zoom piles up handlers that then fight the
        drag.
        """
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]],
                       dtype=np.float32)
        if state.get('band') is None:
            state['band'] = wf.add_line(pts, colors='red', thickness=2.0)
        else:
            state['band'].data[:, :2] = pts

    def _view_rect():
        return view_rect(wf)

    def _sync_boxes():
        """Write the camera's view back into the range boxes.

        Without this the boxes keep describing the window that was last
        requested while the screen shows somewhere else entirely, so the next
        Zoom or Apply snaps back to the stale range.
        """
        x0, x1, y0, y1 = _view_rect()
        p = state['proc']
        t_lo, t_hi = (y0, y1) if seismic else (x0, x1)
        c_lo, c_hi = (x0, x1) if seismic else (y0, y1)
        t_end_ = p.t0_sec + p.nt * p.dt
        t0.value = round(max(p.t0_sec, min(t_lo, t_end_)), 3)
        t1.value = round(max(p.t0_sec, min(t_hi, t_end_)), 3)
        ch_max = p.ch0 + p.nx * p.dch
        c0.value = int(round(max(p.ch0, min(c_lo, ch_max))))
        c1.value = int(round(max(p.ch0, min(c_hi, ch_max))))

    def _pan_end():
        """After a pan or zoom gesture: adopt the new view and re-pool for it.

        This is what makes panning show *data for the range on screen* rather
        than sliding a fixed picture around — the boxes take the camera's
        range, then the redraw re-pools at the resolution that range deserves.
        """
        if not pan_on.value:
            return
        _sync_boxes()
        _draw()

    def _down(ev):
        if pan_on.value:            # the controller owns dragging in pan mode
            return
        p = _world(ev)
        if p is None:
            return
        drag['start'] = _clamp(p)
        # Subscribe to pointer_move only for the duration of the drag. Every
        # move event is a round trip from the browser to the kernel over the
        # widget comm, and the browser emits them continuously while the
        # pointer is anywhere over the canvas — listening the whole time floods
        # the comm and the notebook stops responding.
        fig.renderer.add_event_handler(_move, 'pointer_move')

    def _move(ev):
        if 'start' not in drag:
            return
        p = _world(ev)
        if p is None:
            return
        (sx, sy), (cx, cy) = drag['start'], _clamp(p)
        _band(min(sx, cx), max(sx, cx), min(sy, cy), max(sy, cy))

    def _up(ev):
        if pan_on.value:
            _pan_end()
            return
        if 'start' not in drag:
            return
        fig.renderer.remove_event_handler(_move, 'pointer_move')
        sx, sy = drag.pop('start')
        p = _world(ev)
        if state.get('band') is not None:
            wf.remove_graphic(state['band'])
            state['band'] = None
        if p is None:
            return
        cx, cy = _clamp(p)
        # A click with no real drag must not collapse the view to nothing.
        if abs(cx - sx) < 1e-9 or abs(cy - sy) < 1e-9:
            return
        _apply_box(sx, cx, sy, cy)

    def _hover(ev):
        """Report absolute time, channel and amplitude under the pointer.

        Costs no extra traffic: the frontend forwards pointer_move whether or
        not anything listens, and rendercanvas merges consecutive moves into
        one pending event, so a fast sweep does not queue up sixty of them.
        """
        q = _world(ev)
        if q is None:                       # pointer is outside the waterfall
            return
        x, y = q
        t_sec, ch = (y, x) if seismic else (x, y)
        p = state['proc']
        hit = sample_at(p, t_sec, ch)
        if hit is None:
            hover.value = ''
            return
        _, _, amp = hit
        stamp = p.begin_time + _timedelta(seconds=t_sec - p.t0_sec)
        hover.value = (
            f"<code style='font-size:11px'>{stamp:%Y-%m-%d %H:%M:%S}"
            f".{stamp.microsecond // 1000:03d} &nbsp; ch {int(round(ch))} "
            f"&nbsp; {amp:+.3e} {p.units}</code>"
        )

    fig.renderer.add_event_handler(_hover, 'pointer_move')
    fig.renderer.add_event_handler(_down, 'pointer_down')
    fig.renderer.add_event_handler(_up, 'pointer_up')

    def _toggle_pan(_=None):
        """Hand the mouse to the pan/zoom controller, and reload accordingly.

        The redraw is the point: pan mode swaps the visible-window texture for
        a whole-array one, which is what makes dragging show data rather than
        blank canvas.
        """
        for sp in (wf, wig):
            sp.controller.enabled = pan_on.value
        _draw()

    pan_on.observe(_toggle_pan, names='value')
    # Scroll-zoom has no release event, so only the boxes follow it live; the
    # re-pool waits for the next pointer release or an explicit Zoom, since
    # re-pooling on every wheel tick would stutter.
    fig.renderer.add_event_handler(
        lambda ev: _sync_boxes() if pan_on.value else None, 'wheel')
    apply_btn.on_click(_apply)
    zoom_btn.on_click(_draw)
    full_btn.on_click(_full)
    trace_stride.observe(_draw, names='value')
    img_gain.observe(_set_clim, names='value')
    wig_gain.observe(_set_wiggle_gain, names='value')
    def _set_cmap(change):
        state['img'].cmap = change['new']
        cbar_img.value = _cbar_png(change['new'])

    cmap.observe(_set_cmap, names='value')

    panel = w.VBox([
        _row('view', _lbl('t', 12), t0, _lbl('–'), t1, _lbl('s', 14),
             _lbl('ch', 20), c0, _lbl('–'), c1, zoom_btn, full_btn, pan_on,
             status),
        _row('filter', bp_on, f0, _lbl('–'), f1, _lbl('Hz', 20),
             _lbl('order', 34), order, zph),
        _row('', dt_on, di_on, in_on, _lbl('taper s', 44), tap,
             cm_on, cm0, _lbl('–'), cm1, apply_btn),
        _row('median', _lbl('time k', 44), mt, _lbl('chan k', 46), mx,
             _lbl('(odd, 0 = off)')),
        _row('cursor', hover),
        _row('waterfall', _lbl('gain', 28), img_gain, img_gain_n, cmap,
             cb_lo, cbar_img, cb_hi),
        _row('wiggles', _lbl('every', 34), trace_stride, _lbl('ch', 16),
             _lbl('gain', 28), wig_gain, wig_gain_n),
    ], layout=w.Layout(border='1px solid #ddd', padding='4px', margin='0 0 4px 0'))

    display(panel)
    # `fig.show()` resets every camera, which would silently undo the axis
    # orientation `_draw` sets — so realise the canvas first and populate it
    # after. Drawing before show() left time running up the page in 'seismic'.
    canvas = fig.show()
    # `_apply`, not `_draw`: the filter checkboxes start ticked, so the first
    # frame has to be the filtered data. Drawing raw here contradicted the
    # panel, and raw DAS strain carries per-channel offsets large enough that
    # the untouched array renders as meaningless banding.
    _apply()
    return canvas
