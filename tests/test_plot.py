"""dasio.plot.wiggle renders all traces as a single LineCollection."""
from datetime import datetime, timezone

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
from matplotlib.collections import LineCollection

from dasio.dasdata import DASdata


def make(nx=12, nt=80, fs=100.0):
    rng = np.random.default_rng(0)
    data = rng.standard_normal((nx, nt)).astype(np.float32)
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return DASdata(data=data, fs=fs, dt=1.0 / fs, nt=nt, nx=nx, dx=2.0,
                   begin_time=t0, end_time=t0, t0_sec=0.0)


def test_wiggle_single_linecollection_one_seg_per_channel():
    d = make(nx=12)
    ax = d.plot.wiggle(n_max_ch=None)
    lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(lcs) == 1                        # one artist, not 12 separate lines
    assert len(lcs[0].get_segments()) == 12     # one polyline per channel
    assert not ax.lines                         # no per-channel ax.plot lines


def test_wiggle_respects_n_max_ch_decimation():
    d = make(nx=200)
    ax = d.plot.wiggle(n_max_ch=50)
    lc = next(c for c in ax.collections if isinstance(c, LineCollection))
    assert len(lc.get_segments()) <= 50


def test_wiggle_decimates_wide_time_window():
    d = make(nx=4, nt=20000)
    ax = d.plot.wiggle(n_max_ch=None, max_nt=2000)
    lc = next(c for c in ax.collections if isinstance(c, LineCollection))
    segs = lc.get_segments()
    assert len(segs) == 4
    assert segs[0].shape[0] <= 2000            # time axis strided down


def test_wiggle_max_nt_none_keeps_all_samples():
    d = make(nx=2, nt=8000)
    ax = d.plot.wiggle(n_max_ch=None, max_nt=None)
    lc = next(c for c in ax.collections if isinstance(c, LineCollection))
    assert lc.get_segments()[0].shape[0] == 8000


def test_wiggle_normal_style_and_datetime_run():
    d = make()
    ax1 = d.plot.wiggle(style="normal")
    ax2 = d.plot.wiggle(usedatetime=True)
    assert any(isinstance(c, LineCollection) for c in ax1.collections)
    assert any(isinstance(c, LineCollection) for c in ax2.collections)


def test_imshow_skip_decimates_raster_but_keeps_full_extent():
    d = make(nx=100, nt=2000, fs=100.0)
    _, im_full = d.plot()                            # seismic style -> imshow(data.T)
    assert im_full.get_array().shape == (2000, 100)  # (nt, nx)

    _, im = d.plot(skip_ch=5, skip_t=10)
    assert im.get_array().shape == (200, 20)         # raster decimated (nt/10, nx/5)
    x0, x1, y0, y1 = im.get_extent()
    assert (x0, x1) == (0, 99)                        # channel axis spans full range
    assert y0 == pytest.approx(d.time_axis[-1])       # time axis spans full range
    assert y1 == pytest.approx(d.time_axis[0])


def test_plot_xcorr_extent_and_clim():
    from dasio.plot import plot_xcorr
    rng = np.random.default_rng(1)
    npair, nlag = 30, 401                              # zero lag at column 200
    cc = rng.standard_normal((npair, nlag)).astype(np.float32)
    lag_s = (np.arange(nlag) - (nlag - 1) // 2) / 50.0  # +/- 4 s at 50 Hz
    ax, im = plot_xcorr(cc, lag_s=lag_s)
    assert im.get_array().shape == (nlag, npair)       # transposed: lag on the vertical axis
    x0, x1, y0, y1 = im.get_extent()
    assert (x0, x1) == pytest.approx((0, npair - 1))   # pair index on the horizontal axis
    assert (y0, y1) == pytest.approx((lag_s[-1], lag_s[0]))  # lag on y, increasing downward
    v = float(np.nanpercentile(np.abs(cc), 99.0))     # symmetric perc clim
    assert im.get_clim() == pytest.approx((-v, v))


def test_ccgather_plot_delegates_to_plot_xcorr():
    from dasio.noise import CCGather                     # dataclass needs no torch
    cc = np.random.default_rng(0).standard_normal((10, 21)).astype(np.float32)
    g = CCGather(cc=cc, lag_s=np.linspace(-0.2, 0.2, 21), fs=50.0, nseg=3)
    ax, im = g.plot()                                    # -> plot_xcorr(self.cc, lag_s=self.lag_s)
    assert im.get_array().shape == (21, 10)              # transposed: lag (21) on y, pairs (10) on x
    _, _, y0, y1 = im.get_extent()
    assert (y0, y1) == pytest.approx((0.2, -0.2))        # stored lag axis on the vertical, downward
    assert "10 pairs" in repr(g) and "50 Hz" in repr(g)


# --- the channel axis is labelled in fiber channels, not row positions ------

def _anchored(nx=12, nt=200, ch0=2000, dch=4):
    """A window as `read(min_ch=2000)` then `skip_ch(4)` would return it."""
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return DASdata(
        data=np.random.default_rng(0).standard_normal((nx, nt)).astype(np.float32),
        fs=100.0, dt=0.01, nt=nt, nx=nx, dx=4.0,
        begin_time=t0, end_time=t0, units="strain/s", ch0=ch0, dch=dch,
    )


@pytest.mark.parametrize("style,axis", [("normal", "y"), ("seismic", "x")])
def test_imshow_extent_spans_the_channel_range(style, axis):
    d = _anchored()
    _, im = d.plot.imshow(style=style)
    left, right, bottom, top = im.get_extent()
    got = (top, bottom) if axis == "y" else (left, right)
    assert sorted(got) == [d.ch0, d.ch0 + (d.nx - 1) * d.dch]      # 2000 .. 2044


def test_imshow_channel_extent_survives_skip_ch():
    """skip_ch multiplies dch, so the extent must be computed before it runs
    or a decimated raster relabels the same fiber as a shorter span."""
    _, im = _anchored().plot.imshow(style="normal", skip_ch=3)
    left, right, bottom, top = im.get_extent()
    assert sorted((top, bottom)) == [2000, 2044]


def test_wiggle_places_traces_at_their_channel_numbers():
    ax = _anchored(nx=4, dch=4).plot.wiggle(
        style="normal", normalize=True, n_max_ch=None)
    centres = sorted(float(np.median(s[:, 1]))
                     for s in ax.collections[0].get_segments())
    assert centres == pytest.approx([2000, 2004, 2008, 2012], abs=1.0)
