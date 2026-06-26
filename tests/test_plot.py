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
