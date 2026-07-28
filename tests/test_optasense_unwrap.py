"""`unwrap_int32` — vendor-specific int32 rollover correction.

Vendor-specific — only OptaSense-origin data can wrap, since ASN returns
float32 rad/(s*m) at ~1e-1 and AP Sensing returns floats — but it lives in
`signal.py` with the other kernels: `readers/` never calls it, while `dasdb`,
`desample` and `DASdata.unwrap` all need it after concatenation.

Operates in place. The name refers to the on-disk type that rolls over; every
reader has already widened to float32 (or float64 in desample) before this runs,
so the array is always wide enough to hold the corrected values.
"""
from datetime import datetime, timezone

import numpy as np
import pytest

from dasio.signal import unwrap_int32

WRAP = 2.0 ** 32


def _wrapped(nx=6, nt=400, rows=(2, 4), dtype=np.float64):
    """Ramp with a genuine int32 rollover in `rows`."""
    a = np.linspace(0, 5e5, nt)[None, :].repeat(nx, 0)
    for r in rows:
        a[r, nt // 2:] -= WRAP
    return np.ascontiguousarray(a, dtype=dtype)


def test_removes_the_rollover():
    a = _wrapped()
    unwrap_int32(a)
    assert np.abs(np.diff(a, axis=1)).max() < WRAP / 2


def test_modifies_in_place_and_returns_the_same_array():
    a = _wrapped()
    out = unwrap_int32(a)
    assert out is a, "must be in place, not a copy"


def test_untouched_rows_are_bit_identical():
    a = _wrapped(rows=(2,))
    before = a[0].copy()
    unwrap_int32(a)
    np.testing.assert_array_equal(a[0], before)


def test_is_idempotent():
    """Proc data may or may not already be unwrapped; a second pass must be a
    no-op rather than double-correcting."""
    a = _wrapped()
    unwrap_int32(a)
    once = a.copy()
    unwrap_int32(a)
    np.testing.assert_array_equal(a, once)


def test_noop_when_nothing_wraps():
    a = _wrapped(rows=())
    before = a.copy()
    unwrap_int32(a)
    np.testing.assert_array_equal(a, before)


def test_float32_input_is_unwrapped():
    """float32 is what every reader hands over; spacing at 2.5e9 is 256 counts,
    so the result is correct but not exact."""
    a = _wrapped(dtype=np.float32)
    unwrap_int32(a)
    assert np.abs(np.diff(a.astype(np.float64), axis=1)).max() < WRAP / 2


# --- time axis: where the wrap sits, how many, which direction --------------

def _with_wraps(nx, nt, pattern, dtype=np.float64):
    """Ramp plus rollovers: {row: [(sample_index, sign), ...]}."""
    a = np.linspace(0, 3e5, nt)[None, :].repeat(nx, 0)
    for r, events in pattern.items():
        for idx, sign in events:
            a[r, idx:] += sign * WRAP
    return np.ascontiguousarray(a, dtype=dtype)


def _clean(a):
    return np.abs(np.diff(a.astype(np.float64), axis=1)).max() < WRAP / 2


@pytest.mark.parametrize("pattern,label", [
    ({2: [(200, -1)]},                    "middle"),
    ({2: [(1, -1)]},                      "first sample"),
    ({2: [(399, -1)]},                    "last sample"),
    ({2: [(200, +1)]},                    "underflow"),
    ({2: [(100, -1), (300, -1)]},         "two, same direction"),
    ({2: [(100, -1), (300, +1)]},         "wrap then back"),
    ({2: [(200, -1), (201, -1)]},         "adjacent samples"),
])
def test_time_axis_wrap_positions(pattern, label):
    a = _with_wraps(6, 400, pattern)
    unwrap_int32(a)
    assert _clean(a), f"not unwrapped: {label}"


def test_many_wraps_in_one_channel():
    a = _with_wraps(6, 4000,
                    {2: [(300 * i + 50, -1 if i % 2 else +1) for i in range(10)]})
    unwrap_int32(a)
    assert _clean(a)


# --- channel axis: coverage, and that channels stay independent -------------

@pytest.mark.parametrize("pattern,label", [
    ({},                                                  "no channel wraps"),
    ({r: [(150 + 7 * r, -1)] for r in range(8)},          "every channel"),
    ({r: [(100 + r, -1)] for r in range(0, 8, 3)},        "every third"),
])
def test_channel_axis_coverage(pattern, label):
    a = _with_wraps(8, 400, pattern)
    unwrap_int32(a)
    assert _clean(a), f"not unwrapped: {label}"


def test_channels_do_not_influence_each_other():
    """The correction runs along time; a wrap in one channel must never leak
    into its neighbours."""
    a = _with_wraps(64, 800, {r: [(100 + r, -1)] for r in range(0, 64, 3)})
    whole = a.copy()
    unwrap_int32(whole)

    row_by_row = a.copy()
    for r in range(row_by_row.shape[0]):
        row = np.ascontiguousarray(row_by_row[r:r + 1])
        unwrap_int32(row)
        row_by_row[r] = row[0]

    np.testing.assert_array_equal(whole, row_by_row)


def test_sparse_wraps_in_a_wide_array():
    a = _with_wraps(500, 400, {17: [(200, -1)], 333: [(90, +1)]})
    unwrap_int32(a)
    assert _clean(a)


@pytest.mark.parametrize("delta", [1e4, 1e7])
def test_caught_while_the_true_change_stays_small(delta):
    """A rollover is observed as 2**32 minus the true sample-to-sample change,
    so the default 0.99 threshold catches it only while that change is under
    (1 - 0.99) * 2**32. Real OptaSense data peaks at ~1.9e5, ~200x inside."""
    nt = 200
    a = np.zeros((1, nt))
    a[0, nt // 2:] = delta - WRAP
    unwrap_int32(a)
    assert _clean(a), f"missed a wrap whose true change was {delta:g}"


@pytest.mark.parametrize("delta", [5e7, 1e8])
def test_missed_once_the_true_change_exceeds_the_threshold_margin(delta):
    """Documents the cost of matching DASutils' 0.99 threshold: past
    ~4.3e7 counts the observed jump slips under it and the wrap is left in
    place, silently. Lowering `threshold` recovers these."""
    nt = 200
    a = np.zeros((1, nt))
    a[0, nt // 2:] = delta - WRAP
    missed = a.copy()
    unwrap_int32(missed)
    assert not _clean(missed), "0.99 threshold unexpectedly caught this"

    recovered = a.copy()
    unwrap_int32(recovered, threshold=0.5)
    assert _clean(recovered), "half-period threshold should catch it"
