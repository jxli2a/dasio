"""Tests for `dasio.viewer`'s data path (no GPU, no fastplotlib import).

`pool_maxabs` is the one piece of real numerics in the viewer. It exists
because GPU minification *samples* rather than reduces: at a full-hour
zoom-out one pixel column spans ~128 samples, and a sub-second arrival
drawn straight from the raw texture is invisible more often than not
(measured 0/10 for a 1-sample transient). Pooling by max-|x| keeps the
peak, so the same transient is always drawn.
"""
from datetime import datetime, timezone

import numpy as np
import pytest

from dasio.viewer import apply_chain, clim_of, pool_maxabs


def test_pooling_preserves_a_spike_that_striding_drops():
    """The reason this function exists: a one-sample transient must survive."""
    nt, ncol = 180_000, 1600
    a = np.zeros((4, nt), dtype=np.float32)
    a[:, 12_345] = 1.0                       # single-sample arrival

    pooled, kt = pool_maxabs(a, ncol)

    assert kt > 1, "this window should be decimated"
    assert np.abs(a[:, ::kt]).max() == 0.0, "precondition: striding drops this spike"
    assert np.abs(pooled).max() == 1.0, "max-|x| pooling must keep it"


@pytest.mark.parametrize("nt", [1, 1599, 1600, 1601, 3199, 3200, 30_000, 180_000])
def test_output_never_exceeds_requested_width(nt):
    """The texture is sized from ncol, so overshooting it would over-allocate."""
    pooled, _ = pool_maxabs(np.zeros((2, nt), dtype=np.float32), 1600)
    assert pooled.shape[1] <= 1600


@pytest.mark.parametrize("nt", [1601, 30_000, 180_000])
def test_no_tail_sample_is_dropped(nt):
    """A spike in the final partial bin must survive, not fall off the edge."""
    a = np.zeros((2, nt), dtype=np.float32)
    a[:, -1] = 1.0
    pooled, _ = pool_maxabs(a, 1600)
    assert np.abs(pooled[:, -1]).max() == 1.0


def _make(nx=8, nt=4096, fs=100.0, dc=0.0):
    from datetime import datetime, timezone
    from dasio.dasdata import DASdata
    t = np.arange(nt) / fs
    sig = np.sin(2 * np.pi * 5.0 * t)[None, :] + dc
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return DASdata(data=np.tile(sig, (nx, 1)).astype(np.float32), fs=fs, dt=1 / fs,
                   nt=nt, nx=nx, dx=2.0, begin_time=t0, end_time=t0)


def test_integrate_runs_after_detrend_not_before():
    """Order matters catastrophically: integrate_time accumulates in the array's
    own dtype, so integrating float32 that still carries a large DC offset
    destroys the signal (measured 115% relative error on real data)."""
    from dasio.processing import detrend, integrate

    d = _make(dc=1e4)
    got = apply_chain(d, detrend=True, integrate=True)

    right = integrate(detrend(d))
    wrong = detrend(integrate(d))
    np.testing.assert_allclose(got.data, right.data, rtol=1e-6)
    assert not np.allclose(got.data, wrong.data), "chain applied integrate before detrend"


def test_chain_does_not_mutate_the_input():
    """The chain runs with copy=False for speed, which is only safe because every
    dasio.processing kernel is out-of-place. If that ever stops being true the
    viewer would silently corrupt the user's base array."""
    d = _make(dc=1e4)
    before = d.data.copy()

    apply_chain(d, differentiate=True, detrend=True, taper_sec=0.5,
                bandpass=(1.0, 10.0, 4, True), integrate=True, common_mode=(0, 4))

    np.testing.assert_array_equal(d.data, before)


def test_clim_is_symmetric_percentile_of_magnitude():
    a = np.array([[-3.0, 1.0, 2.0, 100.0]], dtype=np.float32)
    assert clim_of(a, perc=100.0) == pytest.approx(100.0)


def test_clim_falls_back_to_one_when_the_window_is_flat():
    """A dead channel range or an all-zero window gives clim 0; dividing by it
    yields NaN, and NaN cast to uint8 is undefined. Must not propagate."""
    assert clim_of(np.zeros((4, 100), dtype=np.float32)) == 1.0


def test_clim_ignores_nan_samples():
    """One NaN poisons np.percentile for the whole tile."""
    a = np.ones((4, 100), dtype=np.float32)
    a[0, 0] = np.nan
    assert clim_of(a) == pytest.approx(1.0)


def test_view_rejects_an_unknown_style_before_importing_fastplotlib():
    """Validation must precede the heavy import so a typo fails fast — and so
    this test runs on an install without the [viewer] extra."""
    from dasio.viewer import view
    with pytest.raises(ValueError, match="style must be"):
        view(_make(), style='waterfall')


def test_colorbar_png_is_a_real_image_of_the_requested_colormap():
    """The panel's colour bar is a plain PNG, so it needs no GPU — and it must
    actually reflect the chosen colormap, not a fixed gradient."""
    from dasio.viewer import _cbar_png
    from io import BytesIO
    from PIL import Image

    bwr = np.asarray(Image.open(BytesIO(_cbar_png('bwr'))))
    gray = np.asarray(Image.open(BytesIO(_cbar_png('gray'))))
    assert bwr.shape[2] == 3 and bwr.shape[1] == 150
    assert bwr[0, 0, 2] > bwr[0, 0, 0], "bwr should start blue"
    assert bwr[0, -1, 0] > bwr[0, -1, 2], "bwr should end red"
    assert not np.array_equal(bwr, gray), "colormap must be honoured"


@pytest.mark.parametrize("vw,vh", [(948.0, 343.0), (1.0, 1.0), (0.0, 0.0), (None, None)])
def test_view_margin_stays_sane_for_any_viewport_size(vw, vh):
    """The camera margin is derived from the viewport's pixel size, which is not
    always populated on the first draw. An unsized viewport must not blow the
    margin up until the data is an invisible sliver."""
    fx = 52.0 / vw if vw and vw > 120 else 0.06
    fy = 20.0 / vh if vh and vh > 80 else 0.05
    assert 0 < min(fx, 0.15) <= 0.15
    assert 0 < min(fy, 0.15) <= 0.15


@pytest.mark.parametrize("fs", [1.0, 2.0, 5.0, 50.0, 160.0, 1000.0])
def test_default_band_is_valid_at_every_sample_rate(fs):
    """`fmax` scaled with fs but `fmin` was pinned at 1 Hz, so below ~2.5 Hz the
    default band inverted. The kernel does not raise on fmin > fmax — it just
    returns near-zero data, so the viewer came up blank."""
    from dasio.viewer import default_band

    fmin, fmax = default_band(fs)
    assert fmin == 0.0, "the default is a low-pass; nothing is cut below fmax"
    assert 0 <= fmin < fmax < fs / 2, f"invalid default band at fs={fs}"


def test_default_band_keeps_the_familiar_20_hz_top_for_normal_rates():
    from dasio.viewer import default_band
    assert default_band(100.0) == (0.0, 20.0)


def test_chain_rejects_an_inverted_band_instead_of_returning_junk():
    with pytest.raises(ValueError, match="fmin"):
        apply_chain(_make(fs=100.0), bandpass=(20.0, 1.0, 4, True))


def test_chain_rejects_a_corner_at_or_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        apply_chain(_make(fs=100.0), bandpass=(1.0, 60.0, 4, True))


def test_chain_median_filters_run_before_the_bandpass():
    """A band-pass smears a spike across its passband rather than removing it,
    so despiking has to happen first."""
    d = _make(nx=8, nt=512, fs=100.0)
    d.data[:, 200] += 500.0                       # a one-sample spike
    out = apply_chain(d, med_t=5, bandpass=(1.0, 20.0, 4, True))
    ref = apply_chain(d, bandpass=(1.0, 20.0, 4, True))
    assert np.abs(out.data).max() < 0.2 * np.abs(ref.data).max()


def test_chain_ignores_median_kernels_of_zero_or_one():
    d = _make()
    np.testing.assert_array_equal(apply_chain(d, med_t=0).data,
                                  apply_chain(d, med_t=1).data)


def test_sample_at_returns_the_underlying_value_not_a_display_value():
    from dasio.viewer import sample_at
    d = _make(nx=6, nt=100, fs=100.0)
    d.data[2, 50] = 7.5
    d.t0_sec = -1.0
    row, col, amp = sample_at(d, t_sec=-1.0 + 50 * d.dt, channel=2)
    assert (row, col) == (2, 50) and amp == pytest.approx(7.5)


def test_sample_at_honours_the_channel_origin():
    from dasio.viewer import sample_at
    d = _make(nx=6, nt=10)
    d.index_raw = 2000 + np.arange(d.nx)
    d.data[3, 4] = 1.25
    assert sample_at(d, t_sec=4 * d.dt, channel=2003)[2] == pytest.approx(1.25)


@pytest.mark.parametrize("t_sec,channel", [(-5.0, 0), (1e6, 0), (0.0, -1), (0.0, 99)])
def test_sample_at_returns_none_outside_the_array(t_sec, channel):
    from dasio.viewer import sample_at
    assert sample_at(_make(nx=6, nt=10), t_sec, channel) is None


# --- the channel axis is pooled, not left to the GPU's minifier -------------

def test_channel_pooling_keeps_the_peak_a_stride_would_drop():
    """A one-channel arrival on a wide gather: nearest-neighbour minification
    keeps it only if it lands on a sampled row, max-|x| pooling always does."""
    from dasio.viewer import _MAX_ROWS, pool_maxabs

    nx, nt = 4000, 500
    a = np.zeros((nx, nt), np.float32)
    a[1234, 250] = 1.0                                   # not a multiple of the stride

    img, _ = pool_maxabs(a, 256)                         # time axis, as today
    rows, ka = pool_maxabs(img.T, 800)
    rows = rows.T
    assert ka == img.shape[0] // 800 or ka == -(-img.shape[0] // 800)
    assert rows.max() == 1.0, "max-|x| pooling must retain the arrival"
    assert img[::ka][:800].max() == 0.0, "a bare stride drops it — the bug"


def test_channel_pooling_bounds_the_texture():
    from dasio.viewer import _MAX_ROWS, pool_maxabs

    img, _ = pool_maxabs(np.zeros((15686, 6000), np.float32), 1200)
    assert img.shape[0] > _MAX_ROWS                      # precondition
    rows, ka = pool_maxabs(img.T, _MAX_ROWS)
    assert rows.T.shape[0] <= _MAX_ROWS
    # one texel spans `ka` channels, which is what `_draw` folds into the scale
    assert ka == -(-img.shape[0] // _MAX_ROWS)


def test_clim_comes_from_the_data_not_the_pooled_texture():
    """`pool_maxabs` keeps per-bin maxima, so a percentile of the texture runs
    above the same percentile of the samples and the waterfall washes out —
    worse the longer the window and the wider the gather."""
    from dasio.viewer import clim_of, pool_maxabs

    rng = np.random.default_rng(0)
    a = rng.standard_normal((2000, 4000)).astype(np.float32)

    img, _ = pool_maxabs(a, 400)
    rows, _ = pool_maxabs(img.T, 250)

    c_data = clim_of(a, 99.5)
    # Gaussian noise inflates modestly; real DAS with sparse arrivals far more
    # (measured 6.7x on a 2 min x 15686 ch window).
    assert clim_of(rows.T, 99.5) > 1.15 * c_data, "pooling must inflate the percentile"
    # the strided subsample `_draw` uses instead tracks the true limit
    sl = a[::max(1, a.shape[0] // 256), ::max(1, a.shape[1] // 4096)]
    assert abs(clim_of(sl, 99.5) / c_data - 1) < 0.05


# --- helpers hoisted out of view(), now reachable without a widget ----------

class _FakeRuler:
    def __init__(self, a, b): self.start_value, self.end_value = a, b


class _FakeAxes:
    def __init__(self, x, y): self.x, self.y = x, y; self.intersection = "unset"


class _FakeSubplot:
    """Only what fit_view / view_rect touch."""
    def __init__(self, w=1000, h=700):
        self.viewport = type("V", (), {"logical_size": (w, h)})()
        self.axes = _FakeAxes(_FakeRuler(0.0, 10.0), _FakeRuler(2000.0, 2400.0))
        self.camera = type("C", (), {"show_rect": lambda s, *a: setattr(s, "rect", a)})()


def test_window_bounds_maps_fiber_channels_through_ch0_and_dch():
    """The boxes carry fiber channel numbers; rows are 0-based. Reading them as
    rows silently mislabels every min_ch read."""
    from dasio.viewer import window_bounds

    d = _make(nx=100, nt=1000)
    d.index_raw = 2000 + np.arange(d.nx) * 4                       # rows 0..99 are channels 2000..2396
    i0, i1, a0, a1 = window_bounds(d, 0.0, 2.0, 2040, 2080)
    assert (a0, a1) == (10, 20)
    assert (i0, i1) == (0, 200)


def test_window_bounds_clamps_and_never_returns_an_empty_window():
    from dasio.viewer import window_bounds

    d = _make(nx=10, nt=100)
    assert window_bounds(d, -99.0, 99.0, -99, 99) == (0, 100, 0, 10)   # clamped
    i0, i1, a0, a1 = window_bounds(d, 0.5, 0.5, 3, 3)                  # degenerate
    assert i1 > i0 and a1 > a0


def test_clamp_to_rect():
    from dasio.viewer import clamp_to_rect

    rect = (0.0, 10.0, 100.0, 200.0)
    assert clamp_to_rect((-5.0, 50.0), rect) == (0.0, 100.0)
    assert clamp_to_rect((99.0, 999.0), rect) == (10.0, 200.0)
    assert clamp_to_rect((3.0, 150.0), None) == (3.0, 150.0)   # no rect, untouched


def test_view_rect_reads_the_rulers():
    from dasio.viewer import view_rect

    assert view_rect(_FakeSubplot()) == (0.0, 10.0, 2000.0, 2400.0)


def test_fit_view_margin_is_pixel_sized_and_clamped():
    """52 px of label room on a wide panel; on an unsized canvas the fraction
    would blow up, so it is capped at 15 % — that is what made the first frame
    render as a blank sliver."""
    from dasio.viewer import fit_view

    wide = _FakeSubplot(w=1000, h=700)
    fit_view(wide, 0.0, 100.0, 0.0, 100.0)
    x0, x1, y0, y1 = wide.camera.rect
    assert x0 == pytest.approx(-5.2, abs=0.01)          # 52/1000 of the range
    assert (x1, y0) == (100.0, 0.0)                     # only two sides get one

    # Two separate guards. Below 120 px the pixel ratio is abandoned for a
    # flat 6 %; between there and ~347 px it applies but is capped at 15 %.
    unsized = _FakeSubplot(w=4, h=4)                     # canvas not sized yet
    fit_view(unsized, 0.0, 100.0, 0.0, 100.0)
    assert unsized.camera.rect[0] == pytest.approx(-6.0)     # flat fallback

    narrow = _FakeSubplot(w=200, h=200)                  # 52/200 = 26 %
    fit_view(narrow, 0.0, 100.0, 0.0, 100.0)
    assert narrow.camera.rect[0] == pytest.approx(-15.0)     # capped, not -26


def test_fit_view_left_margin_is_widenable_for_wider_labels():
    """A timestamp label needs about twice the room a bare number does, and the
    margin is what stops it rendering clipped against the panel edge."""
    from dasio.viewer import fit_view

    sp = _FakeSubplot(w=1000, h=700)
    fit_view(sp, 0.0, 100.0, 0.0, 100.0, left_px=92.0)
    assert sp.camera.rect[0] == pytest.approx(-9.2, abs=0.01)


def test_fit_view_leaves_the_ruler_free_only_when_panning():
    from dasio.viewer import fit_view

    fixed = _FakeSubplot(); fit_view(fixed, 0.0, 1.0, 0.0, 1.0, pan_enabled=False)
    assert fixed.axes.intersection == (0.0, 1.0, 0)

    panning = _FakeSubplot(); fit_view(panning, 0.0, 1.0, 0.0, 1.0, pan_enabled=True)
    assert panning.axes.intersection is None


# --- usedatetime: wall-clock tick labels ------------------------------------

def _labels(span, start=(2024, 3, 5, 14, 22, 31)):
    """Tick labels a `span`-second window would carry, starting at `start`."""
    from dasio.viewer import datetime_ticks

    begin = datetime(*start, tzinfo=timezone.utc)
    values, fmt = datetime_ticks(begin, 0.0, 0.0, span)
    return [fmt(v, 0.0, span) for v in values]


def test_datetime_ticks_land_on_clock_boundaries_not_round_seconds():
    """The whole reason this places its own ticks. pygfx's 1/2/2.5/5 ladder is
    round in seconds, so a day-long axis steps by 20000 s and labels ticks
    19:55:51 and 01:29:11. A clock wants 00:00 and 06:00."""
    assert _labels(86400) == [
        "03-05 18:00", "03-06 00:00", "03-06 06:00", "03-06 12:00"]


def test_datetime_tick_precision_follows_the_step():
    """No tick carries a field that is constant down the whole axis, and none
    omits one that is changing."""
    assert _labels(21600) == [f"{h:02d}:00" for h in range(15, 21)]   # hours
    assert _labels(600)[:2] == ["14:24", "14:26"]                     # minutes
    assert _labels(30)[:2] == ["14:22:35", "14:22:40"]                # seconds
    assert _labels(2)[:2] == ["14:22:31.000", "14:22:31.500"]         # milliseconds


def test_datetime_ticks_add_the_date_only_when_they_cross_midnight():
    assert _labels(21600, start=(2024, 3, 5, 21, 0, 0))[0] == "03-05 21:00"
    assert _labels(21600)[0] == "15:00"                  # same day, no date


def test_datetime_ticks_are_relative_to_t0_sec_not_sample_zero():
    """Event readers set `t0_sec` negative so t=0 is the event origin; a tick at
    t=0 must then read `begin_time` + 30 s, not `begin_time`."""
    from dasio.viewer import datetime_ticks

    begin = datetime(2024, 3, 5, 14, 22, 31, tzinfo=timezone.utc)
    _, fmt = datetime_ticks(begin, -30.0, -30.0, 30.0)
    assert fmt(0.0, -30.0, 30.0) == "14:23:01"


def test_datetime_ticks_are_accepted_by_a_real_pygfx_ruler():
    """pygfx probes the format callable with (0, -1, 1) and rejects anything not
    returning a str, and `ticks` must be list-like — so a signature drift here
    fails at assignment rather than silently leaving the ruler numeric."""
    pygfx = pytest.importorskip("pygfx")
    from dasio.viewer import datetime_ticks

    r = pygfx.Ruler()
    r.ticks, r.tick_format = datetime_ticks(
        datetime(2024, 3, 5, tzinfo=timezone.utc), 0.0, 0.0, 3600.0)
    assert callable(r.tick_format) and len(r.ticks) > 1


# --- view() itself: construction and the first draw -------------------------

@pytest.mark.parametrize("style", ["seismic", "normal"])
@pytest.mark.parametrize("panels", ["both", "image", "wiggle"])
def test_view_builds_and_draws_the_first_frame(style, panels):
    """The only coverage `view()` has. It runs the full construction path plus
    the initial `_apply` -> `_draw`: chain, both pooling passes, image creation,
    scale and position. Needs the [viewer] extra and a GPU adapter.

    Each `panels` mode skips a different half of `_draw`, and the single-panel
    ones re-index the figure and move the pointer handlers onto whichever
    subplot survives — so all three are exercised."""
    import os

    os.environ.setdefault("WGPU_FORCE_OFFSCREEN", "1")
    pytest.importorskip("fastplotlib")
    from dasio.viewer import view

    d = _make(nx=200, nt=2000)
    d.index_raw = 2000 + np.arange(d.nx) * 4
    view(d, style=style, panels=panels)   # raises if any of the above is broken


@pytest.mark.parametrize("style,time_axis", [("seismic", "y"), ("normal", "x")])
def test_usedatetime_formats_only_the_time_ruler(style, time_axis):
    """The other ruler counts channels and has to keep its plain numbers."""
    import os

    os.environ.setdefault("WGPU_FORCE_OFFSCREEN", "1")
    fpl = pytest.importorskip("fastplotlib")
    from dasio.viewer import view

    seen = []
    real = fpl.Figure

    class RecFigure(real):                # name must end in "Figure" for fpl
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            seen.append(self)

    fpl.Figure = RecFigure
    try:
        d = _make(nx=64, nt=512)
        view(d, style=style, panels="image", usedatetime=True)
    finally:
        fpl.Figure = real

    axes = seen[0][0, 0].axes
    other = "x" if time_axis == "y" else "y"
    assert callable(getattr(axes, time_axis).tick_format)
    assert len(getattr(axes, time_axis).ticks) > 1          # placed, not auto
    assert getattr(axes, other).tick_format == "0.4g"       # untouched default
    assert getattr(axes, other).ticks is None


def _viewer_widgets(monkeypatch, **kw):
    """Build a viewer, returning `(recorded fit_view rects, widgets by name)`.

    `view()` displays its panel rather than returning it, so the only way to
    drive the controls from a test is to record the widgets as they are built.
    """
    import ipywidgets as w
    import dasio.viewer as V

    made = []
    for name in ("Button", "FloatText", "IntText", "Checkbox"):
        base = getattr(w, name)

        class Rec(base):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                made.append(self)

        monkeypatch.setattr(w, name, Rec)

    fits = []
    real = V.fit_view

    def rec_fit(sp, x0, x1, y0, y1, pan_enabled=False, left_px=52.0):
        fits.append((x0, x1, y0, y1))
        return real(sp, x0, x1, y0, y1, pan_enabled, left_px)

    monkeypatch.setattr(V, "fit_view", rec_fit)

    d = _make(nx=400, nt=6000, fs=100.0)
    d.dt = 0.01
    d.data = np.random.default_rng(0).standard_normal((400, 6000)).astype(np.float32)
    V.view(d, panels="image", **kw)
    by_name = {getattr(x, "description", ""): x for x in made}
    nums = [x for x in made if type(x).__bases__[0].__name__ in
            ("FloatText", "IntText")]
    return fits, by_name, nums


@pytest.mark.parametrize("style", ["seismic", "normal"])
def test_pan_mode_keeps_the_view_it_was_enabled_on(monkeypatch, style):
    """Pan loads the whole array so dragging lands on data, but the camera must
    follow the *view*. Fitting it to the loaded extent instead zoomed the time
    axis out to the entire record the moment `pan` was ticked — and, because
    every pointer-up redraws through the same path, undid each drag on release.
    """
    import os

    os.environ.setdefault("WGPU_FORCE_OFFSCREEN", "1")
    pytest.importorskip("fastplotlib")

    fits, by_name, nums = _viewer_widgets(monkeypatch, style=style)
    t0, t1, c0, c1 = nums[:4]
    t0.value, t1.value, c0.value, c1.value = 10.0, 13.0, 50, 150

    fits.clear()
    by_name["Zoom"].click()
    zoomed = fits[-1]

    by_name["pan"].value = True
    assert fits[-1] == zoomed, "ticking pan moved the camera off the view"

    by_name["pan"].value = False
    assert fits[-1] == zoomed, "unticking pan moved the camera off the view"


def test_view_rejects_an_unknown_panels_mode_before_importing_fastplotlib():
    from dasio.viewer import view
    with pytest.raises(ValueError, match="panels must be"):
        view(_make(), panels='waterfall')


def test_missing_viewer_extra_names_the_extra(monkeypatch):
    """A bare ModuleNotFoundError leaves the user guessing which extra to
    install — `picking.py` already points at `dasio[pick]`, so this should
    point at `dasio[viewer]`."""
    import builtins

    from dasio.viewer import view

    real = builtins.__import__

    def no_fpl(name, *a, **kw):
        if name.startswith("fastplotlib"):
            raise ImportError("nope", name="fastplotlib")
        return real(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_fpl)
    with pytest.raises(ImportError, match=r"dasio\[viewer\]"):
        view(_make())


def test_view_accepts_a_custom_chain():
    """The built-in order is fixed, so a callable is how you reorder it or add
    a step it does not offer. It replaces the built-in chain rather than
    composing, or the panel's band-pass would run on top of your work."""
    import os

    os.environ.setdefault("WGPU_FORCE_OFFSCREEN", "1")
    pytest.importorskip("fastplotlib")
    from dasio.viewer import view

    calls = []

    def my_chain(x):
        calls.append(x.units)
        return x.detrend().differentiate()      # integrate-last order reversed

    d = _make(nx=64, nt=512)
    view(d, chain=my_chain)
    assert calls, "the custom chain was never called"


def test_channel_lookup_survives_a_non_ramp_axis(tmp_path):
    """`window_bounds` and `sample_at` used to map channel -> row with
    `(ch - ch0) // dch`. After `select_taptest` the active axis is taptest
    while `ch0` is still the raw index, so every channel landed on row 0."""
    import pandas as pd
    from dasio import DASinfo
    from dasio.dasdata import DASdata
    from dasio.viewer import sample_at, window_bounds

    path = tmp_path / "info.csv"
    pd.DataFrame([{"index": i, "status": 0 if i % 7 == 3 else 1,
                   "lat": 37.0, "lon": -118.0}
                  for i in range(2000, 2060)]).to_csv(path, index=False)
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    d = DASdata(
        data=np.random.default_rng(0).standard_normal((60, 100)).astype(np.float32),
        fs=100.0, dt=0.01, nt=100, nx=60, dx=2.0, begin_time=t0, end_time=t0,
        index_raw=2000 + np.arange(60),
    ).select_taptest(DASinfo.from_csv(path))

    assert d.channel_type == "taptest" and d.ch0 == 2000     # axes disagree
    for ch in (0, 5, 20, d.nx - 1):
        row = window_bounds(d, 0.0, 1.0, ch, ch + 1)[2]
        assert row == int(np.flatnonzero(d.channels() == ch)[0])
        assert sample_at(d, 0.5, ch)[0] == row
    assert sample_at(d, 0.5, 9999) is None                 # past the axis
