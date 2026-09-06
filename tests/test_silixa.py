from datetime import datetime, timedelta, timezone
from pathlib import Path

import h5py
import numpy as np
import pytest

from dasio.dasfile import DASFile
from dasio.readers.detector import detect_format

T0 = datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)
FACTOR = 116e-9 / 8192 * 100.0 / 10.0     # Unit Calibration / 2^13 * fs / gauge length


def test_detect_format(silixa_file):
    with h5py.File(silixa_file, "r") as f:
        assert detect_format(f) == "Silixa"


def test_read_payload_and_axes(silixa_file):
    d = DASFile(silixa_file).read()
    with h5py.File(silixa_file, "r") as f:
        raw = f["Acoustic"][:]
    assert d.data.shape == (4, 256) and d.data.dtype == np.float32
    assert d.data.flags["C_CONTIGUOUS"]
    assert np.array_equal(d.data, raw.T)
    assert d.fs == 100.0 and d.gauge_length_m == 10.0
    assert np.isclose(d.dx, 4.0 * 1.02)
    assert d.begin_time == T0 and d.end_time == T0 + timedelta(seconds=2.55)
    assert d.format == "Silixa" and d.origin == "Silixa" and d.units == "count/s"
    assert list(d.index_raw) == [0, 1, 2, 3]


def test_time_from_gps_string_without_iso_attr(silixa_file):
    with h5py.File(silixa_file, "r+") as f:
        del f["Acoustic"].attrs["ISO8601 Timestamp"]
    assert DASFile(silixa_file).read().begin_time == T0


def test_subset_kwargs(silixa_file):
    d = DASFile(silixa_file).read(min_ch=1, max_ch=3, first_sample=10, n_samples=20)
    assert d.data.shape == (2, 20)
    assert list(d.index_raw) == [1, 2]
    assert d.begin_time == T0 + timedelta(seconds=0.1)


def test_metadata(silixa_file):
    m = DASFile(silixa_file).metadata()
    assert (m["fs"], m["nt"], m["nx"], m["first_sample"]) == (100.0, 256, 4, 0)
    assert m["begin_time"] == T0 and m["end_time"] == T0 + timedelta(seconds=2.55)
    assert np.isclose(m["dx"], 4.08) and m["gauge_length_m"] == 10.0


def test_factor_and_to_physical(silixa_file):
    d = DASFile(silixa_file).read()
    assert np.isclose(d.physical_factor, FACTOR)
    p = d.to_physical()
    assert p.units == "microstrain/s"
    assert np.allclose(p.data, d.data * FACTOR * 1e6)


def test_dasdb_scans_flat_directory(silixa_file):
    from dasio.dasdb import DASdb
    db = DASdb.from_dir(silixa_file.parent, progress=False)
    assert db.format == "Silixa" and len(db.df) == 1


def test_proc_roundtrip_keeps_origin(silixa_file, tmp_path):
    from dasio.readers.proc import write_data_proc
    out = tmp_path / "proc.h5"
    write_data_proc(out, DASFile(silixa_file).read())
    p = DASFile(out).read()
    assert p.origin == "Silixa" and p.units == "count/s"
    assert np.isclose(p.physical_factor, FACTOR)


IVDF = Path("/home/jxli/xsilica/projects/claude/20260904_download_SaltonSea_GDR/data/"
            "imperialvalleydas_v1.0.0/500Hz/data/DF__UTC_20201112_000032.602.h5")


@pytest.mark.skipif(not IVDF.exists(), reason="IVDF file not available")
def test_real_ivdf_file():
    d = DASFile(IVDF).read(min_ch=0, max_ch=64, first_sample=0, n_samples=500)
    assert d.data.shape == (64, 500)
    assert d.begin_time == datetime(2020, 11, 12, 0, 0, 32, 602000, tzinfo=timezone.utc)
    assert np.isclose(d.dx, 4.0838, atol=1e-4) and d.gauge_length_m == 10.0
    assert np.isclose(d.physical_factor, 116e-9 / 8192 * 500 / 10)
    m = DASFile(IVDF).metadata()
    assert (m["nx"], m["nt"], m["fs"]) == (6912, 30000, 500.0)
