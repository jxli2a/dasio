"""Tests for DASdata.downsample / processing.downsample (anti-alias + stride)."""
from datetime import datetime, timezone

import numpy as np
import pytest

from dasio.dasdata import DASdata


def make(nx=4, nt=4000, fs=1000.0, signal=None):
    if signal is None:
        signal = np.zeros((nx, nt), dtype=np.float32)
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return DASdata(data=np.ascontiguousarray(signal, dtype=np.float32),
                   fs=fs, dt=1.0 / fs, nt=signal.shape[1], nx=signal.shape[0],
                   dx=2.0, begin_time=t0, end_time=t0, t0_sec=0.0)


def test_downsample_updates_time_metadata():
    d = make(nt=4000, fs=1000.0)
    out = d.downsample(5)
    assert out.nt == 800                            # 4000 // 5
    assert out.fs == pytest.approx(200.0)
    assert out.dt == pytest.approx(0.005)
    assert out.nx == d.nx and out.dx == d.dx        # channel axis untouched
    assert out.begin_time == d.begin_time           # sample 0 kept
    assert out.data.flags["C_CONTIGUOUS"]


def test_downsample_factor_one_is_noop():
    d = make(signal=np.random.default_rng(0).standard_normal((3, 100)).astype(np.float32))
    out = d.downsample(1)
    assert out.nt == d.nt and out.fs == d.fs
    np.testing.assert_array_equal(out.data, d.data)


def test_downsample_no_antialias_equals_skip_t():
    sig = np.random.default_rng(1).standard_normal((3, 100)).astype(np.float32)
    d = make(signal=sig, fs=100.0)
    a = d.downsample(4, anti_alias=False)
    b = d.skip_t(4)
    np.testing.assert_array_equal(a.data, b.data)
    assert a.fs == b.fs and a.nt == b.nt and a.dt == b.dt


def test_downsample_anti_alias_suppresses_above_new_nyquist():
    fs, nt = 1000.0, 4000
    t = np.arange(nt) / fs
    low = np.sin(2 * np.pi * 10 * t)                # 10 Hz, well below new Nyquist (100)
    high = np.sin(2 * np.pi * 350 * t)             # 350 Hz, aliases to 50 Hz at fs/5=200
    d = make(signal=np.tile((low + high).astype(np.float32), (2, 1)), fs=fs)

    aa = d.downsample(5, anti_alias=True)          # low-pass removes the 350 Hz first
    no = d.downsample(5, anti_alias=False)         # bare stride -> 350 Hz folds in

    # With anti-aliasing the surviving energy is ~the 10 Hz sine (std ~0.707);
    # without it, the aliased 350->50 Hz adds variance.
    assert np.std(aa.data) < np.std(no.data)
    assert 0.5 < np.std(aa.data) < 0.9