import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from dasio.cpp import lfilter


def _make_signal(nx=4, nt=2048, fs=100.0):
    rng = np.random.default_rng(0)
    t = np.arange(nt) / fs
    base = np.sin(2 * np.pi * 5.0 * t) + 0.5 * np.sin(2 * np.pi * 30.0 * t)
    return (base[None, :] + 0.1 * rng.standard_normal((nx, nt))).astype(np.float32)


def test_matches_scipy_butterworth_zerophase():
    fs, fmin, fmax, order = 100.0, 2.0, 10.0, 4
    data = _make_signal(fs=fs)
    dt = 1.0 / fs
    # dasio C++ zero-phase bandpass (phase=0): lfilter(data, flo, nlo, fhi, nhi, phase, nThreads)
    got = lfilter(data, fmin * dt, order, fmax * dt, order, 0, 1)
    sos = butter(order, [fmin, fmax], btype="band", fs=fs, output="sos")
    ref = sosfiltfilt(sos, data, axis=-1).astype(np.float32)
    # Custom C++ Butterworth vs scipy SOS differ at edges; compare interior RMS.
    interior = slice(200, -200)
    rms = np.sqrt(np.mean((got[:, interior] - ref[:, interior]) ** 2))
    sig = np.sqrt(np.mean(ref[:, interior] ** 2))
    assert rms / sig < 0.15, f"relative RMS {rms/sig:.3f} too large vs scipy"


def test_matches_realtimemonitor_reference_bitwise():
    pytest.importorskip("realTimeMonitor.dasio.cpp._bandpass")
    from realTimeMonitor.dasio.cpp import lfilter as ref_lfilter
    fs, fmin, fmax, order = 100.0, 2.0, 10.0, 6
    data = _make_signal(fs=fs)
    dt = 1.0 / fs
    got = lfilter(data, fmin * dt, order, fmax * dt, order, 1, 1)
    ref = ref_lfilter(data, fmin * dt, order, fmax * dt, order, 1, 1)
    # Same source -> expect bit-identical (allow ~8 ULP float32 slack).
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-6)


def test_bandpass_ignores_memory_layout():
    """The C++ extension reads the raw buffer, so a transposed array was walked
    in the wrong order and returned a plausible periodic pattern instead of an
    error. ASN and AP Sensing reach `(nx, nt)` by transposing, so this was
    reachable straight from DASdb.read: the error exceeded the signal."""
    import numpy as np
    from dasio.signal import bandpass2d

    rng = np.random.default_rng(0)
    c = rng.standard_normal((32, 2000)).astype(np.float32)
    f = np.asfortranarray(c)
    assert np.array_equal(c, f) and not f.flags["C_CONTIGUOUS"]

    kw = dict(order=8, zerophase=True, nThreads=2)
    np.testing.assert_array_equal(
        bandpass2d(c, 1.0, 20.0, 0.01, **kw),
        bandpass2d(f, 1.0, 20.0, 0.01, **kw),
    )


@pytest.mark.parametrize("reader,fixture", [("asn", "asn_file"), ("apsensing", "apsensing_file")])
def test_transposing_readers_return_contiguous_data(reader, fixture, request):
    """Both reach (nx, nt) via .T; an F-contiguous payload silently corrupts
    every downstream bandpass."""
    from dasio.dasfile import DASFile

    d = DASFile(request.getfixturevalue(fixture)).read()
    assert d.data.flags["C_CONTIGUOUS"], f"{reader} payload must be C-contiguous"
