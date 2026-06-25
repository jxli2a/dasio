import numpy as np
from dasio.dasfile import DASFile


def test_default_read_has_unit_factor(apsensing_file):
    d = DASFile(apsensing_file).read()                      # with_factor defaults False
    assert d.physical_factor == 1.0


def test_with_factor_attaches_nontrivial_factor(apsensing_file):
    d = DASFile(apsensing_file).read(with_factor=True)
    # RadiansToNanoStrain=100 -> factor = 1e-9 * 100 = 1e-7
    assert np.isclose(d.physical_factor, 1e-7)
    assert d.units == "radian/s"


def test_asn_factor_is_one(asn_file):
    d = DASFile(asn_file).read(with_factor=True)
    assert d.physical_factor == 1.0                         # already strain/s


def test_to_physical_applies_factor_and_advances_units(apsensing_file):
    d = DASFile(apsensing_file).read(with_factor=True)
    raw = d.data.copy()
    p = d.to_physical()
    assert np.allclose(p.data, raw * 1e-7)
    assert p.physical_factor == 1.0
    assert p.units == "strain/s"                            # radian/s -> strain/s


def test_to_physical_noop_when_factor_one(asn_file):
    d = DASFile(asn_file).read(with_factor=True)
    p = d.to_physical()
    assert np.array_equal(p.data, d.data)
    assert p.units == "strain/s"
