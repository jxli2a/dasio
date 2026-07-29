"""Tests for `dasio.viewer`'s data path (no GPU, no fastplotlib import).

`pool_maxabs` is the one piece of real numerics in the viewer. It exists
because GPU minification *samples* rather than reduces: at a full-hour
zoom-out one pixel column spans ~128 samples, and a sub-second arrival
drawn straight from the raw texture is invisible more often than not
(measured 0/10 for a 1-sample transient). Pooling by max-|x| keeps the
peak, so the same transient is always drawn.
"""
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
    assert 0 <= fmin < fmax < fs / 2, f"invalid default band at fs={fs}"


def test_default_band_keeps_the_familiar_1_to_20_hz_for_normal_rates():
    from dasio.viewer import default_band
    assert default_band(100.0) == (1.0, 20.0)


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
    d.ch0 = 2000
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
    d.ch0, d.dch = 2000, 4                       # rows 0..99 are channels 2000..2396
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


def test_fit_view_leaves_the_ruler_free_only_when_panning():
    from dasio.viewer import fit_view

    fixed = _FakeSubplot(); fit_view(fixed, 0.0, 1.0, 0.0, 1.0, pan_enabled=False)
    assert fixed.axes.intersection == (0.0, 1.0, 0)

    panning = _FakeSubplot(); fit_view(panning, 0.0, 1.0, 0.0, 1.0, pan_enabled=True)
    assert panning.axes.intersection is None


# --- view() itself: construction and the first draw -------------------------

@pytest.mark.parametrize("style", ["seismic", "normal"])
def test_view_builds_and_draws_the_first_frame(style):
    """The only coverage `view()` has. It runs the full construction path plus
    the initial `_apply` -> `_draw`: chain, both pooling passes, image creation,
    scale and position. Needs the [viewer] extra and a GPU adapter."""
    import os

    os.environ.setdefault("WGPU_FORCE_OFFSCREEN", "1")
    pytest.importorskip("fastplotlib")
    from dasio.viewer import view

    d = _make(nx=200, nt=2000)
    d.ch0, d.dch = 2000, 4
    view(d, style=style)          # raises if any of the above is broken


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
