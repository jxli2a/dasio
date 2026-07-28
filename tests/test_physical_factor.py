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
