"""`DASdata.ch0` — the channel-axis anchor, counterpart to `t0_sec`.

A read with `min_ch=2000` returns 3000 rows that are really fiber channels
2000..4999. Without an anchor that offset is lost, every consumer relabels them
0..2999, and plots disagree with the fiber by a constant no one can see.
"""
from datetime import datetime, timezone

import numpy as np
import pytest

from dasio.dasdata import DASdata


def make(nx=10, nt=20, ch0=0):
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return DASdata(data=np.zeros((nx, nt), dtype=np.float32), fs=100.0, dt=0.01,
                   nt=nt, nx=nx, dx=2.0, begin_time=t0, end_time=t0, ch0=ch0)


def test_channel_axis_starts_at_ch0():
    d = make(nx=5, ch0=2000)
    np.testing.assert_array_equal(d.channel_axis, [2000, 2001, 2002, 2003, 2004])


def test_default_ch0_is_zero_so_existing_behaviour_is_unchanged():
    assert make().ch0 == 0
    np.testing.assert_array_equal(make(nx=3).channel_axis, [0, 1, 2])


def test_truncate_shifts_ch0_by_the_slice_start():
    d = make(nx=100, ch0=2000).truncate(ch_range=(10, 40))
    assert d.ch0 == 2010 and d.nx == 30
    assert d.channel_axis[0] == 2010 and d.channel_axis[-1] == 2039


def test_truncate_on_time_only_leaves_ch0_alone():
    assert make(nx=10, ch0=2000).truncate(t_range=(0.0, 0.1)).ch0 == 2000


def test_skip_ch_keeps_the_first_channel_as_the_anchor():
    d = make(nx=20, ch0=2000).skip_ch(4)
    assert d.ch0 == 2000
    np.testing.assert_array_equal(d.channel_axis[:3], [2000, 2004, 2008])


def test_dasdb_read_records_min_ch(proc_file):
    """The read path is where the offset was being dropped."""
    import pandas as pd
    from dasio.dasdb import DASdb
    from dasio.readers.proc import read_metadata_proc

    meta = read_metadata_proc(proc_file)
    if meta["nx"] < 3:
        pytest.skip("fixture too narrow to slice channels")
    db = DASdb(pd.DataFrame([meta]), "Proc")
    out = db.read(meta["begin_time"], meta["end_time"], min_ch=1, max_ch=3)
    assert out.ch0 == 1
    assert out.channel_axis[0] == 1
