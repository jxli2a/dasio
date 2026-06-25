from dasio.dasfile import DASFile
from dasio.dasdata import VALID_UNITS, normalize_unit


def test_vocabulary_contents():
    assert VALID_UNITS == frozenset({
        "count", "radian", "radian/s", "strain", "strain/s",
        "microstrain", "microstrain/s",
    })


def test_normalize_unit():
    assert normalize_unit("microstrain/s") == "microstrain/s"
    assert normalize_unit("strain rate (microstrain/s)") == "microstrain/s"
    assert normalize_unit("MICROSTRAIN/S") == "microstrain/s"
    assert normalize_unit("nonsense") == "unknown"
    assert normalize_unit("microstrain/sec") == "microstrain/s"


def test_reader_units(optasense_file, asn_file, apsensing_file, proc_file, event_file):
    assert DASFile(optasense_file).read().units == "count"
    assert DASFile(asn_file).read().units == "strain/s"
    assert DASFile(apsensing_file).read().units == "radian/s"
    assert DASFile(proc_file).read().units == "microstrain/s"   # Unknown origin, convert=True
    assert DASFile(event_file).read().units == "microstrain/s"  # from file 'unit' attr
