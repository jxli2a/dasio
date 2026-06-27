"""Tests for dasio.noise — temporal_normalization (CPU) and cross_correlate (torch)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from dasio.dasdata import DASdata
from dasio.noise import (
    common_offset_pairs, common_shot_pairs, haversine_km,
    preprocess, temporal_normalization,
)


def make(data, fs=100.0):
    data = np.ascontiguousarray(data, dtype=np.float32)
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return DASdata(data=data, fs=fs, dt=1.0 / fs, nt=data.shape[1], nx=data.shape[0],
                   dx=2.0, begin_time=t0, end_time=t0, t0_sec=0.0)


# ---- preprocess chain (CPU) -------------------------------------------------

def test_preprocess_full_chain_decimates_and_updates_fs():
    rng = np.random.default_rng(0)
    d = make(rng.standard_normal((6, 2000)).astype(np.float32), fs=250.0)
    out = preprocess(d, freqmin=0.1, freqmax=20.0, decimate=5, norm_window_sec=1.0)
    assert out.nx == 6
    assert out.nt == 2000 // 5                       # plain stride after band-pass
    assert out.fs == 50.0 and out.dt == 1.0 / 50.0
    assert out.data.dtype == np.float32 and np.isfinite(out.data).all()


def test_preprocess_one_bit_norm_gives_signs():
    rng = np.random.default_rng(1)
    d = make(rng.standard_normal((4, 500)).astype(np.float32), fs=100.0)
    out = preprocess(d, freqmin=1.0, freqmax=10.0, norm_window_sec=0)   # 0 -> one-bit
    assert set(np.unique(out.data)).issubset({-1.0, 0.0, 1.0})
    assert out.shape == (4, 500)                     # decimate=1 -> no stride


def test_preprocess_taper_and_filter_order_run():
    rng = np.random.default_rng(2)
    d = make(rng.standard_normal((4, 1000)).astype(np.float32), fs=100.0)
    out = preprocess(d, freqmin=0.5, freqmax=2.0, taper_alpha=0.05, order=4,
                     decimate=2, norm_window_sec=1.0)        # legacy-style recipe
    assert out.fs == 50.0 and out.nt == 500
    assert out.data.dtype == np.float32 and np.isfinite(out.data).all()


# ---- differentiate flavor (central default) ---------------------------------

def test_central_difference_bit_matches_numpy_gradient():
    from dasio.signal import gradient_time
    rng = np.random.default_rng(7)
    x = (rng.standard_normal((8, 5000)) * 0.05).astype(np.float32)
    fs = 100.0
    ref = (np.gradient(x, axis=-1) * fs).astype(np.float32)   # exactly what legacy PyCC computes
    np.testing.assert_array_equal(gradient_time(x, 1.0 / fs), ref)
    np.testing.assert_array_equal(make(x, fs=fs).differentiate().data, ref)   # default = central


def test_backward_method_keeps_diff_time_convention():
    from dasio.signal import diff_time
    rng = np.random.default_rng(8)
    x = (rng.standard_normal((4, 1000)) * 0.05).astype(np.float32)
    d = make(x, fs=100.0)
    out = d.differentiate(method="backward").data
    np.testing.assert_array_equal(out, diff_time(x, d.dt))
    assert np.all(out[:, 0] == 0.0)                          # backward zeroes the first sample
    with pytest.raises(ValueError, match="central"):
        d.differentiate(method="forward")


def test_preprocess_diff_method_threads_through():
    rng = np.random.default_rng(9)
    d = make((rng.standard_normal((5, 800)) * 0.05).astype(np.float32), fs=100.0)
    for m in ("central", "backward"):                        # no band-pass / decimate / norm
        ref = d.differentiate(method=m).detrend().subtract_common_mode()
        np.testing.assert_array_equal(preprocess(d, diff_method=m).data, ref.data)
    assert not np.array_equal(preprocess(d, diff_method="central").data,
                              preprocess(d, diff_method="backward").data)


# ---- temporal_normalization (CPU) -------------------------------------------

def test_one_bit_when_window_none_or_zero():
    d = make([[-3.0, 0.0, 2.0, -0.5]])
    for w in (None, 0):
        out = temporal_normalization(d, w)
        np.testing.assert_array_equal(out.data, np.array([[-1, 0, 1, -1]], dtype=np.float32))
        assert out.shape == d.shape


def test_ram_flattens_amplitude_envelope():
    rng = np.random.default_rng(0)
    nt = 2000
    base = rng.standard_normal(nt).astype(np.float32) * 0.1
    base[800:1000] *= 30.0                          # loud burst on quiet background
    d = make(base[None, :], fs=100.0)

    out = temporal_normalization(d, window_sec=0.5)  # 50-sample running window

    raw_ratio = np.std(base[800:1000]) / np.std(base[:200])
    nrm_ratio = np.std(out.data[0, 800:1000]) / np.std(out.data[0, :200])
    assert nrm_ratio < raw_ratio / 3                # burst no longer dominates
    assert out.shape == d.shape


def test_ram_output_is_fresh_float32():
    d = make(np.ones((2, 50), dtype=np.float32))
    out = temporal_normalization(d, window_sec=0.1)
    assert out.data.dtype == np.float32
    assert out.data is not d.data


# ---- xcorr (torch; skipped when torch is absent) ----------------------------

def test_xcorr_peak_at_known_lag():
    pytest.importorskip("torch")
    from dasio.noise import xcorr
    rng = np.random.default_rng(0)
    npts, shift, mlag = 512, 7, 50
    x = rng.standard_normal(npts).astype(np.float32)
    y = np.roll(x, shift)                            # y[t] = x[t - shift]
    cc = xcorr(np.stack([x, x]), np.stack([y, y]), max_lag=mlag)
    assert cc.shape == (2, 2 * mlag + 1)
    lags = np.arange(-mlag, mlag + 1)
    assert lags[np.argmax(cc[0])] == shift          # peak lands at the true shift


def test_xcorr_broadcasts_1d_reference():
    pytest.importorskip("torch")
    from dasio.noise import xcorr
    rng = np.random.default_rng(1)
    a = rng.standard_normal((5, 256)).astype(np.float32)
    ref = a[2].copy()                               # virtual source = channel 2
    cc = xcorr(a, ref, max_lag=40)
    assert cc.shape == (5, 81)
    # channel 2 vs itself is an autocorrelation -> peak at zero lag (center)
    assert np.argmax(cc[2]) == 40


def test_xcorr_whiten_phase_only_and_ram():
    pytest.importorskip("torch")
    from dasio.noise import xcorr
    a = np.random.default_rng(2).standard_normal((3, 200)).astype(np.float32)
    phase = xcorr(a, a, whiten=True)                 # phase-only
    ram = xcorr(a, a, whiten=2.0, fs=100.0)         # RAM, 2 Hz smoothing
    assert phase.shape == (3, 2 * 200 - 1)          # full lag axis when max_lag is None
    assert np.isfinite(phase).all() and np.isfinite(ram).all()


def test_xcorr_whiten_hz_requires_fs():
    pytest.importorskip("torch")
    from dasio.noise import xcorr
    a = np.random.default_rng(3).standard_normal((2, 128)).astype(np.float32)
    with pytest.raises(ValueError):
        xcorr(a, a, whiten=2.0)                      # Hz bandwidth without fs


def test_spectral_whitening_band_limit_cos2_taper():
    pytest.importorskip("torch")
    import torch
    from dasio.noise import spectral_whitening
    rng = np.random.default_rng(0)
    n = 65                                           # rfft bins; df=1 Hz -> bin index == Hz
    spec = torch.tensor(rng.standard_normal(n) + 1j * rng.standard_normal(n))[None]
    out = spectral_whitening(spec, smooth_bins=1, band=(10.0, 20.0), df=1.0)
    mag = out.abs().numpy()[0]
    assert mag[0] == pytest.approx(0.0, abs=1e-4)    # DC fully suppressed (cos^2 -> 0)
    assert mag[-1] == pytest.approx(0.0, abs=1e-4)   # Nyquist fully suppressed
    assert np.allclose(mag[10:20], 1.0, atol=1e-4)   # in-band phase-only -> unit magnitude


def test_xcorr_whiten_band_runs_and_requires_fs():
    pytest.importorskip("torch")
    from dasio.noise import xcorr
    a = np.random.default_rng(4).standard_normal((2, 256)).astype(np.float32)
    cc = xcorr(a, a, whiten=True, whiten_band=(5.0, 20.0), fs=100.0, max_lag=30)
    assert cc.shape == (2, 61) and np.isfinite(cc).all()
    with pytest.raises(ValueError):
        xcorr(a, a, whiten=True, whiten_band=(5.0, 20.0))   # band-limit needs fs


# ---- channel-pair gathers (CPU) ---------------------------------------------

def test_haversine_one_degree_latitude():
    # one degree of latitude is ~111.19 km on a sphere of this radius
    assert haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19, abs=0.1)
    # broadcasts a single point against an array of points
    d = haversine_km(0.0, 0.0, np.array([0.0, 1.0, 2.0]), np.zeros(3))
    np.testing.assert_allclose(d, [0.0, 111.19, 222.39], atol=0.1)


def test_common_shot_pairs_fixed_source():
    ch1, ch2 = common_shot_pairs(600, 5)                # int -> range(5)
    np.testing.assert_array_equal(ch1, [600, 600, 600, 600, 600])
    np.testing.assert_array_equal(ch2, [0, 1, 2, 3, 4])
    ch1, ch2 = common_shot_pairs(2, [0, 3, 7])          # explicit receiver list
    np.testing.assert_array_equal(ch1, [2, 2, 2])
    np.testing.assert_array_equal(ch2, [0, 3, 7])


def test_common_offset_pairs_picks_target_spacing():
    # channels along a meridian, ~1.1119 km apart (0.01 deg latitude)
    lat = np.arange(10) * 0.01
    lon = np.zeros(10)
    spacing = 111.19 * 0.01                              # km between adjacent channels
    ch1, ch2 = common_offset_pairs(lat, lon, offset_km=2 * spacing, tol_km=spacing / 2)
    # every kept pair spans two channels (the nearest match to 2x spacing)
    np.testing.assert_array_equal(ch2 - ch1, 2)
    # far-end channels with no partner at the offset are dropped by tol
    assert ch1.max() == 7 and ch2.max() == 9


# ---- xcorr_stack / xcorr_dataset (torch) ----------------------------

def test_xcorr_stack_peaks_and_segment_count():
    pytest.importorskip("torch")
    from dasio.noise import xcorr_stack
    rng = np.random.default_rng(0)
    fs, seg_sec, nseg, shift = 50.0, 2.0, 3, 5
    npts_seg = int(seg_sec * fs)                         # 100
    base = rng.standard_normal(npts_seg * nseg + 37).astype(np.float32)  # +remainder -> dropped
    d = make(np.stack([base, np.roll(base, shift)]), fs=fs)
    ch1, ch2 = common_shot_pairs(0, [0, 1])             # (0,0) auto, (0,1) cross
    g = xcorr_stack(d, ch1, ch2, seg_sec=seg_sec, max_lag_sec=0.4)
    mlag = int(0.4 * fs)                                 # 20
    assert g.nseg == nseg                                # remainder dropped, 3 whole segments
    assert g.cc.shape == (2, 2 * mlag + 1)
    assert g.fs == fs and g.lag_s.shape == (2 * mlag + 1,)
    lags = np.arange(-mlag, mlag + 1)
    assert lags[np.argmax(g.cc[0])] == 0               # autocorrelation peaks at zero lag
    assert lags[np.argmax(g.cc[1])] == shift           # cross peaks at the imposed shift


def test_xcorr_dataset_end_to_end(tmp_path):
    pytest.importorskip("torch")
    from dasio.dasdb import DASdb
    from dasio.noise import xcorr_dataset
    from dasio.readers.proc import write_data_proc

    fs, dt, nch, nt = 50.0, 1.0 / 50.0, 3, 200
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    sig = np.random.default_rng(0).standard_normal((nch, 2 * nt)).astype(np.float32)
    for k in range(2):                                   # two back-to-back contiguous files
        b = t0 + timedelta(seconds=k * nt * dt)
        d = DASdata(data=sig[:, k * nt:(k + 1) * nt].copy(), fs=fs, dt=dt, nt=nt,
                    nx=nch, dx=2.0, begin_time=b,
                    end_time=b + timedelta(seconds=(nt - 1) * dt), t0_sec=0.0)
        write_data_proc(tmp_path / f"proc_{k}.h5", d)

    db = DASdb.from_dir(tmp_path, system="Proc")
    assert db.n_segments == 1                            # the two files are time-contiguous

    ch1, ch2 = common_shot_pairs(0, nch)
    res = xcorr_dataset(db, ch1, ch2, max_lag_sec=0.4, seg_sec=2.0, window_sec=4.0,
                            prep=dict(differentiate=False, freqmin=1.0, freqmax=20.0))
    mlag = int(0.4 * fs)
    assert res.cc.shape == (nch, 2 * mlag + 1)
    assert res.nseg == 4                                 # 2 windows x 2 segments each
    assert res.fs == fs and res.lag_s.shape == (2 * mlag + 1,)
    assert np.argmax(res.cc[0]) == mlag                 # channel-0 autocorrelation centered