"""Tests for the PASSCAL SEG-Y reader + DASFile suffix routing."""
import struct
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from dasio import DASFile, read_passcal_segy, read_passcal_segy_metadata


def write_segy(path, data, dt_us, *, zero_binhdr=False, gmt=None, fmt=5):
    """Minimal SEG-Y writer: 3600-byte header + (240-byte trace hdr + float32 BE data)*."""
    nx, ns = data.shape
    fhdr = bytearray(3600)
    if not zero_binhdr:
        struct.pack_into(">h", fhdr, 3200 + 16, dt_us)   # sample interval (us)
        struct.pack_into(">h", fhdr, 3200 + 20, ns)      # samples / trace
        struct.pack_into(">h", fhdr, 3200 + 24, fmt)     # format code (5 = IEEE float)
    with open(path, "wb") as fid:
        fid.write(fhdr)
        for i in range(nx):
            th = bytearray(240)
            if gmt is not None:
                struct.pack_into(">5h", th, 156, *gmt)   # year, doy, hh, mm, ss
            fid.write(th)
            fid.write(data[i].astype(">f4").tobytes())
    return path


def test_read_with_valid_header(tmp_path):
    data = np.random.default_rng(0).standard_normal((6, 100)).astype(np.float32)
    p = write_segy(tmp_path / "x.segy", data, dt_us=4000, gmt=(2020, 211, 14, 0, 0))
    d = read_passcal_segy(p)
    assert d.shape == (6, 100)                       # n_traces derived from file size
    assert d.fs == pytest.approx(250.0)             # 4000 us -> 250 Hz
    assert d.system == "PASSCAL_SEGY"
    np.testing.assert_allclose(d.data, data, rtol=1e-5)
    expect = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(days=210, hours=14)
    assert d.begin_time == expect


def test_headerless_needs_dims_then_derives_from_filesize(tmp_path):
    data = np.ones((4, 50), dtype=np.float32)
    p = write_segy(tmp_path / "h.segy", data, dt_us=0, zero_binhdr=True)
    with pytest.raises(ValueError):
        read_passcal_segy(p)                          # ns / dt unknowable from header
    d = read_passcal_segy(p, n_traces=4, sample_interval=1 / 250)
    assert d.shape == (4, 50)                          # ns recovered from file size
    assert d.fs == pytest.approx(250.0)


def test_conversion_factor_scales_data(tmp_path):
    data = np.ones((2, 10), dtype=np.float32)
    p = write_segy(tmp_path / "c.segy", data, dt_us=4000)
    d = read_passcal_segy(p, conversion_factor=3.0)
    np.testing.assert_allclose(d.data, 3.0, rtol=1e-6)


def test_dasfile_routes_segy_by_suffix(tmp_path):
    data = np.arange(3 * 20, dtype=np.float32).reshape(3, 20)
    p = write_segy(tmp_path / "r.segy", data, dt_us=4000)
    f = DASFile(p)
    assert f.system == "PASSCAL_SEGY"                  # detected by suffix, no h5py open
    assert f.origin == "PASSCAL_SEGY"
    d = f.read()
    assert d.shape == (3, 20)
    np.testing.assert_allclose(d.data, data, rtol=1e-5)


def test_endian_autodetect_little(tmp_path):
    # little-endian payload + empty binary header — mimics real PASSCAL DAS files
    data = (np.random.default_rng(2).standard_normal((4, 80)) * 1e4).astype(np.float32)
    p = tmp_path / "le.segy"
    with open(p, "wb") as fid:
        fid.write(bytearray(3600))
        for i in range(4):
            fid.write(bytearray(240))
            fid.write(data[i].astype("<f4").tobytes())
    d = read_passcal_segy(p, n_traces=4, sample_interval=1 / 250)   # endian='auto'
    assert d.raw_meta["endian"] == "<"
    np.testing.assert_allclose(d.data, data, rtol=1e-5)


def test_explicit_endian_override(tmp_path):
    data = np.ones((2, 20), dtype=np.float32)
    p = write_segy(tmp_path / "be.segy", data, dt_us=4000)         # big-endian payload
    d = read_passcal_segy(p, endian="big")
    assert d.raw_meta["endian"] == ">"
    np.testing.assert_allclose(d.data, 1.0, rtol=1e-6)


def test_metadata_matches_read(tmp_path):
    data = np.ones((5, 40), dtype=np.float32)
    p = write_segy(tmp_path / "m.segy", data, dt_us=2000)  # 500 Hz
    m = read_passcal_segy_metadata(p)
    assert m["nx"] == 5 and m["nt"] == 40
    assert m["fs"] == pytest.approx(500.0)
    assert m["first_sample"] == 0