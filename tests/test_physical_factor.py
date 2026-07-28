import numpy as np
import pytest
from dasio.dasfile import DASFile


def test_default_read_attaches_the_factor(apsensing_file):
    """The default is on, so `.to_physical()` works off a plain `read()`."""
    d = DASFile(apsensing_file).read()
    assert d.physical_factor != 1.0
    assert DASFile(apsensing_file).read(with_factor=False).physical_factor == 1.0


def test_with_factor_attaches_nontrivial_factor(apsensing_file):
    d = DASFile(apsensing_file).read(with_factor=True)
    # RadiansToNanoStrain=100 -> factor = 1e-9 * 100 = 1e-7
    assert np.isclose(d.physical_factor, 1e-7)
    assert d.units == "radian/s"


def test_asn_factor_is_one(asn_file):
    d = DASFile(asn_file).read(with_factor=True)
    assert d.physical_factor == 1.0                         # already strain/s


def test_to_physical_applies_factor_and_the_microstrain_scale(apsensing_file):
    d = DASFile(apsensing_file).read(with_factor=True)
    raw = d.data.copy()
    p = d.to_physical()
    assert np.allclose(p.data, raw * 1e-7 * 1e6)    # factor, then strain -> micro
    assert p.physical_factor == 1.0
    assert p.units == "microstrain/s"                       # radian/s -> microstrain/s


def test_to_physical_scales_a_vendor_that_needs_no_factor(asn_file):
    """ASN is already strain/s, so `factor()` is 1.0 — but the 1e6 still owes,
    or ASN windows would come back in a different unit from everyone else."""
    d = DASFile(asn_file).read(with_factor=True)
    raw = d.data.copy()
    p = d.to_physical()
    assert np.allclose(p.data, raw * 1e6)
    assert p.units == "microstrain/s"


def test_to_physical_raises_without_factor(optasense_file):
    # Counts with no factor attached: to_physical() must raise rather than
    # silently relabel them microstrain. `with_factor=False` is how you get here.
    d = DASFile(optasense_file).read(with_factor=False)
    assert d.units == "count"
    assert d.physical_factor == 1.0
    with pytest.raises(ValueError):
        d.to_physical()


# --- Proc files: the origin vendor decides the units ------------------------

@pytest.fixture
def proc_optasense_file(tmp_path):
    """Proc file whose origin is OptaSense, so `factor()` is a real number.

    The payload is stored as raw counts, so the round trip has to pick up the
    count->strain constant from /Acquisition_origin.
    """
    import h5py
    p = tmp_path / "proc_opta.hdf5"
    with h5py.File(p, "w") as f:
        d = f.create_dataset("Data", data=np.ones((4, 32), dtype=np.float32))
        d.attrs["nCh"] = 4
        d.attrs["nt"] = 32
        d.attrs["dt"] = 0.01
        d.attrs["fs"] = 100.0
        d.attrs["dCh"] = 2.0
        d.attrs["GaugeLength"] = 10.0
        d.attrs["startTime"] = "2023-11-14T22:13:20.000000+00:00"
        d.attrs["endTime"] = "2023-11-14T22:13:20.310000+00:00"
        o = f.create_dataset("Acquisition_origin", data=np.float32(0.0))
        o.attrs["AcquisitionId"] = "opta-test"      # what detect_origin keys on
        o.attrs["GaugeLength"] = 10.0
        o.attrs["Fibre Refractive Index"] = 1.468
        o.attrs["Laser Wavelength (nm)"] = 1550.0
        o.attrs["FPGA Drawing Number"] = 7804701
        o.attrs["FPGA Version"] = "2.0"
        o.attrs["Num Output Channels"] = 4
    return p


def test_optasense_origin_proc_reads_as_counts_and_converts(proc_optasense_file):
    """A Proc read is now indistinguishable from a raw read: stored units out,
    factor attached, `.to_physical()` finishes the job. The reader applying the
    factor itself is what once let it be applied twice — 118 microstrain
    collapsing to 1.2e-11 on a real 1 Hz catalog."""
    f = DASFile(proc_optasense_file, format="Proc")
    assert f.origin == "OptaSense" and f.factor() != 1.0   # precondition

    d = f.read()
    assert d.units == "count"
    assert d.physical_factor == f.factor()

    p = d.to_physical()
    assert p.units == "microstrain"
    assert np.allclose(p.data, d.data * f.factor() * 1e6)


def test_to_physical_is_idempotent_on_proc_data(proc_optasense_file):
    """Converted data must not pick up a second factor or 1e6 from a
    redundant call — callers should be able to apply it unconditionally."""
    p = DASFile(proc_optasense_file, format="Proc").read().to_physical()
    np.testing.assert_array_equal(p.to_physical().data, p.data)
    assert p.to_physical().units == p.units


# --- gauge length lives in one of two places depending on the writer --------

def test_gauge_length_falls_back_to_acquisition_origin(tmp_path):
    """`write_data_proc` puts GaugeLength on /Data, but legacy Desample_DAS.py
    wrote it only into /Acquisition_origin — reading just /Data left
    gauge_length_m None on every file from that era."""
    import h5py
    from dasio.readers.proc import read_data_proc, read_metadata_proc

    p = tmp_path / "legacy_layout.h5"
    with h5py.File(p, "w") as f:
        d = f.create_dataset("Data", data=np.ones((4, 32), dtype=np.float32))
        for k, v in (("nCh", 4), ("nt", 32), ("dt", 0.01), ("fs", 100.0),
                     ("dCh", 2.0)):
            d.attrs[k] = v
        d.attrs["startTime"] = "2023-11-14T22:13:20.000000+00:00"
        d.attrs["endTime"] = "2023-11-14T22:13:20.310000+00:00"
        assert "GaugeLength" not in d.attrs                 # the legacy shape
        o = f.create_dataset("Acquisition_origin", data=np.float32(0.0))
        o.attrs["GaugeLength"] = 102.09524

    assert read_data_proc(p).gauge_length_m == pytest.approx(102.09524)
    assert read_metadata_proc(p)["gauge_length_m"] == pytest.approx(102.09524)


def test_gauge_length_on_data_wins(proc_optasense_file):
    """When both carry it, /Data is authoritative — it is what the current
    writer emits, and a cascade could have changed it."""
    import h5py
    from dasio.readers.proc import read_data_proc

    with h5py.File(proc_optasense_file, "r+") as f:
        f["Data"].attrs["GaugeLength"] = 25.0
        f["Acquisition_origin"].attrs["GaugeLength"] = 999.0
    assert read_data_proc(proc_optasense_file).gauge_length_m == pytest.approx(25.0)
