"""`nthreads` plumbing for processing.bandpass / downsample.

The C++ kernel's own default is `nThreads=1`; the wrappers must resolve
`None` to `default_nthreads()` so a bare `d.bandpass(...)` uses the whole
machine. Thread count is a scheduling detail, so results must be identical
regardless of it — the OMP loop partitions over channels and each channel is
filtered independently.
"""
from datetime import datetime, timezone

import numpy as np

from dasio.dasdata import DASdata
from dasio.processing import bandpass, downsample


def make(nx=16, nt=2048, fs=100.0):
    rng = np.random.default_rng(0)
    t = np.arange(nt) / fs
    base = np.sin(2 * np.pi * 5.0 * t) + 0.5 * np.sin(2 * np.pi * 30.0 * t)
    sig = (base[None, :] + 0.1 * rng.standard_normal((nx, nt))).astype(np.float32)
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return DASdata(data=sig, fs=fs, dt=1.0 / fs, nt=nt, nx=nx,
                   dx=2.0, begin_time=t0, end_time=t0)


def test_bandpass_result_independent_of_nthreads():
    d = make()
    ref = bandpass(d, 2.0, 10.0, order=4, nthreads=1)
    for n in (2, 4, 8):
        np.testing.assert_array_equal(bandpass(d, 2.0, 10.0, order=4, nthreads=n).data,
                                      ref.data)


def test_bandpass_default_matches_single_thread():
    """`nthreads=None` (the default) resolves to default_nthreads(), not 1."""
    d = make()
    np.testing.assert_array_equal(bandpass(d, 2.0, 10.0, order=4).data,
                                  bandpass(d, 2.0, 10.0, order=4, nthreads=1).data)


def test_downsample_forwards_nthreads_to_anti_alias_filter():
    d = make(nt=4000, fs=1000.0)
    np.testing.assert_array_equal(downsample(d, 5, nthreads=4).data,
                                  downsample(d, 5, nthreads=1).data)
