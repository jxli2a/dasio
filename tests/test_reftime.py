"""The seconds frame: `times(reftime)` on demand, `reftime =` to move it."""
from datetime import datetime, timedelta, timezone

import numpy as np

from dasio.dasdata import DASdata


def record(nx=4, nt=500, fs=100.0):
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return DASdata(
        data=np.zeros((nx, nt), dtype=np.float32), fs=fs, dt=1 / fs, nt=nt,
        nx=nx, dx=8.0, begin_time=t0,
        end_time=t0 + timedelta(seconds=(nt - 1) / fs))


def test_times_counts_from_reftime_without_touching_the_record():
    d = record()
    t = d.times(reftime=d.begin_time + timedelta(seconds=3))
    assert t[0] == -3.0 and np.allclose(np.diff(t), d.dt)
    assert d.t0_sec == 0.0                       # unchanged, as in obspy


def test_reftime_moves_the_frame_and_the_float_t_range():
    d = record()
    origin = d.begin_time + timedelta(seconds=1.0)
    d.reftime = origin
    assert d.times()[0] == -1.0 and d.reftime == origin
    assert d.begin_time == record().begin_time   # absolute time untouched
    w = d.truncate(t_range=(0.0, 1.0))           # floats count from the origin now
    assert w.begin_time == origin and w.reftime == origin


def test_times_datetime_kind_is_absolute_and_ignores_the_frame():
    d = record(); d.reftime = d.begin_time + timedelta(seconds=5)
    t = d.times(type='datetime')
    assert t.dtype == np.dtype('datetime64[ns]')
    assert t[0] == np.datetime64(d.begin_time.replace(tzinfo=None))
    assert (t[1] - t[0]) == np.timedelta64(int(d.dt * 1e9), 'ns')
