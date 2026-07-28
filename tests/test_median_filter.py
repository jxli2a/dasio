"""Running median filter along time or channels.

Checked against `scipy.ndimage.median_filter`, which is bit-identical under
`mode='mirror'` — scipy's `'reflect'` duplicates the edge sample and does not
match. scipy is the reference rather than torch because it is already a hard
dependency, and because it is 21x to 552x slower (O(n*k) per sample), which is
why the kernel here exists at all.
"""
import numpy as np
import pytest
from scipy.ndimage import median_filter as nd_median

from dasio.signal import median_filter_1d


def ref(a, k, axis):
    """scipy equivalent. `mirror` is the mode whose edge handling matches."""
    size = (1, k) if axis == "t" else (k, 1)
    return nd_median(a, size=size, mode="mirror")


@pytest.mark.parametrize("k", [3, 5, 11, 31])
@pytest.mark.parametrize("axis", ["t", "x"])
def test_matches_scipy_exactly(k, axis):
    a = np.random.default_rng(0).standard_normal((40, 200)).astype(np.float32)
    np.testing.assert_array_equal(median_filter_1d(a, k, axis=axis), ref(a, k, axis))


@pytest.mark.parametrize("axis", ["t", "x"])
def test_shape_and_dtype_are_preserved(axis):
    a = np.random.default_rng(1).standard_normal((6, 9)).astype(np.float32)
    out = median_filter_1d(a, 3, axis=axis)
    assert out.shape == a.shape and out.dtype == a.dtype


@pytest.mark.parametrize("axis", ["t", "x"])
def test_heavy_ties_and_constant_input(axis):
    """Repeated values exercise the insert/delete path's tie handling."""
    a = np.repeat(np.arange(8, dtype=np.float32), 5).reshape(8, 5)
    np.testing.assert_array_equal(median_filter_1d(a, 3, axis=axis), ref(a, 3, axis))
    flat = np.full((8, 12), 2.5, dtype=np.float32)
    np.testing.assert_array_equal(median_filter_1d(flat, 5, axis=axis), flat)


def test_a_spike_narrower_than_the_kernel_is_removed():
    """The reason a median filter is used at all."""
    a = np.zeros((3, 50), dtype=np.float32)
    a[:, 25] = 100.0
    assert median_filter_1d(a, 5, axis="t").max() == 0.0


def test_even_kernel_is_rejected():
    a = np.zeros((3, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="odd"):
        median_filter_1d(a, 4, axis="t")


def test_kernel_wider_than_the_axis_is_rejected():
    """Reflection indexing is only defined while k // 2 < n."""
    a = np.zeros((3, 10), dtype=np.float32)
    with pytest.raises(ValueError, match="wider"):
        median_filter_1d(a, 31, axis="t")
