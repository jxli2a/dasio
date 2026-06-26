"""Tests for DASdata.select_channels (arbitrary channel pick) and
truncate (contiguous window only)."""
from datetime import datetime, timezone

import numpy as np
import pytest

from dasio.dasdata import DASdata


def make(nx=6, nt=20, fs=100.0):
    # channel c, sample t -> value c*1000 + t, so data[:, 0] // 1000 == channel id
    data = (np.arange(nx)[:, None] * 1000 + np.arange(nt)[None, :]).astype(np.float32)
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return DASdata(data=data, fs=fs, dt=1.0 / fs, nt=nt, nx=nx, dx=2.0,
                   begin_time=t0, end_time=t0, t0_sec=0.0)


def test_select_channels_int_array_order_preserved():
    d = make()
    out = d.select_channels([5, 0, 2])
    assert out.nx == 3
    assert out.nt == d.nt                       # time axis untouched
    assert out.begin_time == d.begin_time
    assert list((out.data[:, 0] // 1000).astype(int)) == [5, 0, 2]
    assert out.data.flags["C_CONTIGUOUS"]


def test_select_channels_boolean_mask():
    d = make()
    mask = np.array([True, False, True, False, True, False])
    out = d.select_channels(mask)
    assert out.nx == 3
    assert list((out.data[:, 0] // 1000).astype(int)) == [0, 2, 4]


def test_select_channels_then_truncate_time():
    d = make(fs=100.0)                           # dt = 0.01 s
    out = d.select_channels([1, 3]).truncate(t_range=(0.0, 0.05))
    assert out.nx == 2
    assert out.nt == 5                           # samples 0..4
    assert list((out.data[:, 0] // 1000).astype(int)) == [1, 3]


def test_truncate_contiguous_window_still_works():
    d = make()
    out = d.truncate(ch_range=(2, 5), t_range=(0.0, 0.05))
    assert out.nx == 3
    assert out.nt == 5
    assert list((out.data[:, 0] // 1000).astype(int)) == [2, 3, 4]


def test_truncate_no_longer_accepts_ch_index():
    d = make()
    with pytest.raises(TypeError):
        d.truncate(ch_index=[0, 1])


def test_skip_ch_decimates_and_scales_dx():
    d = make(nx=6, fs=100.0)                         # dx = 2.0
    out = d.skip_ch(2)
    assert out.nx == 3                               # channels 0, 2, 4
    assert list((out.data[:, 0] // 1000).astype(int)) == [0, 2, 4]
    assert out.dx == 4.0                             # 2.0 * step
    assert out.nt == d.nt and out.fs == d.fs and out.dt == d.dt   # time untouched
    assert out.data.flags["C_CONTIGUOUS"]


def test_skip_t_decimates_and_scales_dt_fs():
    d = make(nx=4, nt=20, fs=100.0)                  # dt = 0.01
    out = d.skip_t(5)
    assert out.nt == 4                               # samples 0, 5, 10, 15
    assert list((out.data[0] % 1000).astype(int)) == [0, 5, 10, 15]
    assert out.nx == d.nx and out.dx == d.dx         # channels untouched
    assert out.dt == pytest.approx(0.05)             # 0.01 * step
    assert out.fs == pytest.approx(20.0)             # 100 / step
    assert out.begin_time == d.begin_time            # sample 0 kept
    # end_time snaps to the last kept sample: begin + (nt-1)*new_dt
    assert (out.end_time - out.begin_time).total_seconds() == pytest.approx(0.15)
    assert out.data.flags["C_CONTIGUOUS"]


def test_skip_step_le_one_is_noop():
    d = make()
    for out in (d.skip_ch(1), d.skip_t(1), d.skip_ch(0)):   # 0 clamps to 1
        assert out.nx == d.nx and out.nt == d.nt
        assert out.dx == d.dx and out.dt == d.dt
        np.testing.assert_array_equal(out.data, d.data)
