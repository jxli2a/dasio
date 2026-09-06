"""Tests for `DASdata.select` — a channel range, an explicit channel list,
or both intersected, plus the time window."""
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


def test_select_int_array_order_preserved():
    d = make()
    out = d.select(ch_index=[5, 0, 2])
    assert out.nx == 3
    assert out.nt == d.nt                       # time axis untouched
    assert out.begin_time == d.begin_time
    assert list((out.data[:, 0] // 1000).astype(int)) == [5, 0, 2]
    assert out.data.flags["C_CONTIGUOUS"]


def test_select_boolean_mask():
    d = make()
    mask = np.array([True, False, True, False, True, False])
    out = d.select(ch_index=mask)
    assert out.nx == 3
    assert list((out.data[:, 0] // 1000).astype(int)) == [0, 2, 4]


def test_select_then_truncate_time():
    d = make(fs=100.0)                           # dt = 0.01 s
    out = d.select(ch_index=[1, 3]).truncate(t_range=(0.0, 0.05))
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


# --- selecting by channel number, on an axis that does not start at 0 --------
#
# `channels()` is the authority on which fiber channel a row holds, so it is
# also what `select` matches against. Both halves used to be broken
# once `ch0`/`dch` moved off (0, 1): the argument was read as a row index, and
# the result inherited the parent's anchor, which described the wrong channels.

def offset_make(nx=6, ch0=2000, dch=5):
    d = make(nx=nx)
    d.index_raw = ch0 + np.arange(nx) * dch
    return d


def test_select_takes_channel_numbers_not_row_indices():
    d = offset_make()                            # channels() 2000, 2005, ... 2025
    out = d.select(ch_index=[2005, 2015])
    assert list((out.data[:, 0] // 1000).astype(int)) == [1, 3]   # rows 1 and 3
    np.testing.assert_array_equal(out.channels(), [2005, 2015])


def test_select_preserves_the_requested_order():
    d = offset_make()
    out = d.select(ch_index=[2025, 2000, 2010])
    assert list((out.data[:, 0] // 1000).astype(int)) == [5, 0, 2]
    np.testing.assert_array_equal(out.channels(), [2025, 2000, 2010])


def test_select_rejects_a_channel_the_array_does_not_hold():
    """Silently landing on the neighbouring channel is the failure that hides:
    `searchsorted` always returns *some* row."""
    d = offset_make()
    with pytest.raises(ValueError, match="2007"):
        d.select(ch_index=[2005, 2007])


def test_uniform_pick_retunes_ch0_dch_to_the_pick():
    """imshow extents and the viewer's texture transform read `ch0`/`dch`, so a
    stale parent stride would stretch the picture over the wrong channels."""
    out = offset_make().select(ch_index=[2000, 2010, 2020])
    assert (out.ch0, out.dch) == (2000, 10)
    np.testing.assert_array_equal(out.channels(), [2000, 2010, 2020])


def test_non_uniform_pick_keeps_the_exact_axis():
    out = offset_make().select(ch_index=[2000, 2010, 2025])
    np.testing.assert_array_equal(out.channels(), [2000, 2010, 2025])
    # ch0/dch survive only as the ramp through the same span, for consumers
    # that can draw nothing else — deliberately not the exact axis.
    assert out.ch0 == 2000
    assert out.ch0 + (out.nx - 1) * out.dch == pytest.approx(2025, abs=2)


def test_truncate_and_skip_ch_carry_a_non_uniform_axis_through():
    d = offset_make().select(ch_index=[2000, 2010, 2025])
    np.testing.assert_array_equal(
        d.truncate(ch_range=(2010, 2026)).channels(), [2010, 2025])
    np.testing.assert_array_equal(d.skip_ch(2).channels(), [2000, 2025])
    np.testing.assert_array_equal(d.select(ch_index=[2025]).channels(), [2025])


def test_boolean_mask_stays_positional():
    """A mask is one entry per row, so it cannot be read as channel numbers."""
    d = offset_make()
    out = d.select(ch_index=np.array([True, False, True, False, False, True]))
    np.testing.assert_array_equal(out.channels(), [2000, 2010, 2025])
    with pytest.raises(ValueError, match="one entry per channel"):
        d.select(ch_index=np.array([True, False]))


def test_default_anchor_selection_is_unchanged():
    """With ch0=0 and dch=1 a channel number *is* the row index, so none of the
    above changes what the plain case does."""
    d = make()
    out = d.select(ch_index=[5, 0, 2])
    assert list((out.data[:, 0] // 1000).astype(int)) == [5, 0, 2]
    np.testing.assert_array_equal(out.channels(), [5, 0, 2])
