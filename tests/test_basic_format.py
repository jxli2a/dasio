"""The Basic format: `/data` plus one attr per DASdata field, no inference.

Exists because Proc re-derives units from `/Acquisition_origin`, which is only
sound while the payload is raw. Converted data needs a format that stores what
it is rather than deducing it, and `write_data_proc` now refuses it outright.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from dasio.dasdata import DASdata
from dasio.dasfile import DASFile
from dasio.readers.basic import read_basic, read_basic_metadata, write_basic
from dasio.readers.detector import detect_format
from dasio.readers.proc import write_data_proc

T0 = datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)


def make(units="microstrain", nx=6, nt=40, **kw):
    rng = np.random.default_rng(0)
    fields = dict(
        data=rng.standard_normal((nx, nt)).astype(np.float32),
        fs=100.0, dt=0.01, nt=nt, nx=nx, dx=2.5,
        begin_time=T0, end_time=T0 + timedelta(seconds=(nt - 1) * 0.01),
        gauge_length_m=10.0, format="Proc", origin="OptaSense", units=units,
        t0_sec=-3.0, physical_factor=1.0,
        channels={'raw': 2000 + np.arange(nx) * 4},
    )
    fields.update(kw)
    return DASdata(**fields)


def test_round_trip_preserves_every_field(tmp_path):
    """The point of the format: nothing is inferred, so nothing is lost."""
    d = make()
    back = read_basic(write_basic(tmp_path / "a.h5", d))

    np.testing.assert_array_equal(back.data, d.data)
    for field in ("fs", "dt", "nt", "nx", "dx", "gauge_length_m", "origin",
                  "t0_sec", "ch0", "dch", "units", "physical_factor"):
        assert getattr(back, field) == getattr(d, field), field
    assert back.begin_time == d.begin_time
    assert back.end_time == d.end_time
    # `format` is the format, so it becomes 'Basic' — the interrogator that
    # recorded the samples is what `origin` keeps.
    assert back.format == "Basic" and back.origin == "OptaSense"


@pytest.mark.parametrize("units", ["microstrain", "microstrain/s", "count",
                                   "strain/s", "unknown"])
def test_units_survive_whatever_they_are(tmp_path, units):
    """Proc infers units from origin and so cannot hold microstrain; Basic
    stores the string, so every unit round-trips."""
    d = make(units=units)
    assert read_basic(write_basic(tmp_path / "u.h5", d)).units == units


def test_none_gauge_length_round_trips(tmp_path):
    """HDF5 has no null attr; NaN is the sentinel."""
    d = make(gauge_length_m=None)
    assert read_basic(write_basic(tmp_path / "g.h5", d)).gauge_length_m is None


def test_slicing_moves_both_anchors(tmp_path):
    """A sliced read must still report true fiber channels and the same
    seconds frame, or picks and plots come out shifted by a constant."""
    d = make()
    p = write_basic(tmp_path / "s.h5", d)
    out = read_basic(p, min_ch=2, max_ch=5, first_sample=10, n_samples=20)

    assert out.shape == (3, 20)
    np.testing.assert_array_equal(out.data, d.data[2:5, 10:30])
    assert out.ch0 == d.ch0 + 2 * d.dch                  # 2000 + 2*4
    assert out.channel_axis[0] == out.ch0
    assert out.begin_time == d.begin_time + timedelta(seconds=10 * d.dt)
    assert out.t0_sec == pytest.approx(d.t0_sec + 10 * d.dt)


def test_detected_and_dispatched_by_dasfile(tmp_path):
    """The root stamp is what distinguishes it — `/data` alone is ambiguous
    with the ASN and Event layouts."""
    import h5py

    p = write_basic(tmp_path / "d.h5", make())
    with h5py.File(p) as f:
        assert detect_format(f) == "Basic"
    assert DASFile(p).format == "Basic"
    assert DASFile(p).read().units == "microstrain"      # no reader kwargs needed


def test_factor_is_one_so_to_physical_is_a_noop(tmp_path):
    """Basic data is already converted; nothing may rescale it."""
    d = make()
    back = DASFile(write_basic(tmp_path / "f.h5", d)).read()
    assert back.physical_factor == 1.0
    np.testing.assert_array_equal(back.to_physical().data, back.data)


def test_write_refuses_to_clobber(tmp_path):
    p = write_basic(tmp_path / "o.h5", make())
    with pytest.raises(IOError, match="already exists"):
        write_basic(p, make())
    write_basic(p, make(units="count"), overwrite=True)
    assert read_basic(p).units == "count"


def test_metadata_matches_the_payload(tmp_path):
    d = make()
    meta = read_basic_metadata(write_basic(tmp_path / "m.h5", d))
    assert meta["nx"] == d.nx and meta["nt"] == d.nt
    assert meta["fs"] == d.fs and meta["dx"] == d.dx
    assert meta["begin_time"] == d.begin_time


def test_metadata_skips_a_foreign_file(tmp_path):
    import h5py

    p = tmp_path / "not_basic.h5"
    with h5py.File(p, "w") as f:
        f.create_dataset("something_else", data=np.zeros(4))
    assert read_basic_metadata(p) is None


# --- the gate that sends converted data here in the first place -------------

@pytest.mark.parametrize("units", ["microstrain", "microstrain/s", "strain"])
def test_write_data_proc_rejects_converted_payloads(tmp_path, units):
    """Silently accepting these produced a file that read back as strain/s and
    picked up a spurious 1e6 in to_physical()."""
    with pytest.raises(ValueError, match="already been converted"):
        write_data_proc(tmp_path / "bad.h5", make(units=units))


@pytest.mark.parametrize("units", ["count", "radian/s", "strain/s", "unknown"])
def test_write_data_proc_accepts_raw_payloads(tmp_path, units):
    """Raw vendor units are what Proc exists to hold; "unknown" makes no claim
    and is what synthetic test payloads carry."""
    write_data_proc(tmp_path / f"ok_{units.replace('/', '_')}.h5", make(units=units))


def test_basic_round_trips_a_gappy_channel_axis(tmp_path):
    """The format persists `(ch0, dch)`, which is a ramp; a `select` can leave
    the axis gappy, and rounding it back onto a ramp relabels real channels."""
    from dasio.readers.basic import read_basic, write_basic

    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    opt = np.array([2000, 2004, 2009, 2015])
    src = DASdata(
        data=np.repeat(opt[:, None].astype(np.float32), 20, axis=1),
        fs=100.0, dt=0.01, nt=20, nx=4, dx=2.0, begin_time=t0, end_time=t0,
        units="strain/s",
        channels={"raw": opt, "taptest": np.arange(4)},
    )
    f = tmp_path / "gappy.h5"
    write_basic(f, src)

    back = read_basic(f)
    np.testing.assert_array_equal(back.channels["raw"], opt)
    np.testing.assert_array_equal(back.channels["taptest"], np.arange(4))
    # the labels must still describe the rows they came with
    np.testing.assert_array_equal(back.data[:, 0], opt)

    half = read_basic(f, min_ch=1, max_ch=3)
    np.testing.assert_array_equal(half.channels["raw"], [2004, 2009])
    np.testing.assert_array_equal(half.data[:, 0], [2004, 2009])


def test_basic_writes_no_label_arrays_for_a_plain_ramp(tmp_path):
    """A ramp is fully described by `(ch0, dch)`, so a normal read writes the
    same bytes it always did and an older reader loses nothing."""
    import h5py
    from dasio.readers.basic import write_basic

    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    d = DASdata(
        data=np.zeros((4, 20), np.float32), fs=100.0, dt=0.01, nt=20, nx=4,
        dx=2.0, begin_time=t0, end_time=t0, units="strain/s",
        channels={"raw": 2000 + np.arange(4)},
    )
    f = tmp_path / "ramp.h5"
    write_basic(f, d)
    with h5py.File(f) as h:
        assert "channels" not in h
        assert h["data"].attrs["ch0"] == 2000 and h["data"].attrs["dch"] == 1


def test_basic_read_defaults_channel_by_for_a_legacy_file(tmp_path):
    """Files written before `channel_axis_name` was persisted have no such attr."""
    import h5py
    from dasio.readers.basic import read_basic, write_basic

    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    d = DASdata(data=np.zeros((4, 20), np.float32), fs=100.0, dt=0.01, nt=20,
                nx=4, dx=2.0, begin_time=t0, end_time=t0, units="strain/s")
    f = tmp_path / "legacy.h5"
    write_basic(f, d)
    with h5py.File(f, "a") as h:
        del h["data"].attrs["channel_axis_name"]
    assert read_basic(f).channel_axis_name == "raw"
