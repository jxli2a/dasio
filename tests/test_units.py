from datetime import datetime, timezone

import numpy as np

from dasio.dasfile import DASFile
from dasio.dasdata import VALID_UNITS, normalize_unit, DASdata


def _d(units):
    x = np.ones((3, 50), np.float32)
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return DASdata(data=x, fs=100.0, dt=0.01, nt=50, nx=3, dx=1.0,
                   begin_time=t0, end_time=t0, units=units)


def test_differentiate_propagates_units():
    assert _d("strain").differentiate().units == "strain/s"
    assert _d("microstrain").differentiate().units == "microstrain/s"
    assert _d("radian").differentiate().units == "radian/s"
    assert _d("count").differentiate().units == "count"          # unmapped -> unchanged


def test_integrate_reverses_units():
    assert _d("strain/s").integrate().units == "strain"
    assert _d("radian/s").integrate().units == "radian"
    assert _d("unknown").integrate().units == "unknown"


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
