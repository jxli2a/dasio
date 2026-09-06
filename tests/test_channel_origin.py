"""`DASdata.ch0` — the channel-axis anchor, counterpart to `t0_sec`.

A read with `min_ch=2000` returns 3000 rows that are really fiber channels
2000..4999. Without an anchor that offset is lost, every consumer relabels them
0..2999, and plots disagree with the fiber by a constant no one can see.
"""
from datetime import datetime, timezone

import numpy as np
import pytest

from dasio.dasdata import DASdata


def make(nx=10, nt=20, ch0=0, dch=1):
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return DASdata(
        data=np.zeros((nx, nt), dtype=np.float32), fs=100.0, dt=0.01,
        nt=nt, nx=nx, dx=2.0, begin_time=t0, end_time=t0,
        index_raw=ch0 + np.arange(nx) * dch,
    )


def test_channel_axis_starts_at_ch0():
    d = make(nx=5, ch0=2000)
    np.testing.assert_array_equal(d.channels(), [2000, 2001, 2002, 2003, 2004])


def test_default_ch0_is_zero_so_existing_behaviour_is_unchanged():
    assert make().ch0 == 0
    np.testing.assert_array_equal(make(nx=3).channels(), [0, 1, 2])


def test_truncate_takes_channel_numbers_not_row_indices():
    """`ch_range` is in the numbers `channels()` reports, like `read`'s
    min_ch/max_ch and like `select_channels` — not positions in the array."""
    d = make(nx=100, ch0=2000).truncate(ch_range=(2010, 2040))
    assert d.ch0 == 2010 and d.nx == 30
    assert d.channels()[0] == 2010 and d.channels()[-1] == 2039


def test_truncate_raises_when_the_range_misses_the_array():
    """Row indices used to be silently clipped to an empty result; the units
    now agree, so a range that selects nothing is a mistake worth naming."""
    with pytest.raises(ValueError, match="no channel selected"):
        make(nx=100, ch0=2000).truncate(ch_range=(10, 40))


def test_truncate_clips_a_partly_overlapping_range():
    d = make(nx=100, ch0=2000).truncate(ch_range=(2090, 9999))
    assert d.nx == 10 and d.channels()[0] == 2090


def test_truncate_on_time_only_leaves_ch0_alone():
    assert make(nx=10, ch0=2000).truncate(t_range=(0.0, 0.1)).ch0 == 2000


def test_skip_ch_keeps_the_first_channel_as_the_anchor():
    d = make(nx=20, ch0=2000).skip_ch(4)
    assert d.ch0 == 2000
    np.testing.assert_array_equal(d.channels()[:3], [2000, 2004, 2008])


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
    assert out.channels()[0] == 1


# --- two names for one row: raw (index_raw) and taptest ------------------
#
# `ch0`/`dch` describe the reader's raw array, which shifts when fiber is added
# or removed. `index_taptest` is frozen at survey time and is what dasinfo
# joins on. A window that has been cut down to located channels needs both.

def _dasinfo(tmp_path, n=12, unlocated=(2, 5, 9)):
    """A catalog where a few raw channels never got a taptest geolocation."""
    import pandas as pd
    from dasio import DASinfo

    rows = [
        {
            'index': i,
            'status': 0 if i in unlocated else 1,
            'lat': 37.0 + i * 1e-4,
            'lon': -118.0,
        }
        for i in range(n)
    ]
    path = tmp_path / 'info.csv'
    pd.DataFrame(rows).to_csv(path, index=False)
    return DASinfo.from_csv(path)


def test_select_channels_with_a_dasinfo_records_both_labels(tmp_path):
    info = _dasinfo(tmp_path)
    d = make(nx=12, ch0=0).select_taptest(info)

    assert d.nx == 9                                     # 3 unlocated dropped
    np.testing.assert_array_equal(
        d.index_raw, [0, 1, 3, 4, 6, 7, 8, 10, 11])
    np.testing.assert_array_equal(d.channels(type='taptest'), np.arange(9))
    # raw has gaps where the survey did not reach; taptest does not
    assert not np.all(np.diff(d.index_raw) == 1)
    assert np.all(np.diff(d.channels(type='taptest')) == 1)


def test_a_catalog_is_intersected_not_demanded(tmp_path):
    """A list of numbers is a request and a missing one is a mistake; a catalog
    describes a whole deployment, of which a window holds a part."""
    info = _dasinfo(tmp_path, n=12)
    d = make(nx=6, ch0=0).select_taptest(info)   # catalog runs past nx
    assert d.nx == 4                                        # 2 and 5 are unlocated
    np.testing.assert_array_equal(d.index_raw, [0, 1, 3, 4])


def test_channels_reports_the_active_type_or_the_one_asked_for(tmp_path):
    info = _dasinfo(tmp_path)
    d = make(nx=12, ch0=0).select_taptest(info)

    assert d.channel_type == 'taptest'                 # select_taptest set it
    np.testing.assert_array_equal(d.channels(), np.arange(9))
    np.testing.assert_array_equal(d.channels(type='raw'), [0, 1, 3, 4, 6, 7, 8, 10, 11])
    d.channel_type = 'raw'
    np.testing.assert_array_equal(d.channels(), [0, 1, 3, 4, 6, 7, 8, 10, 11])


def test_channels_hands_back_a_copy():
    d = make(nx=5)
    idx = d.channels()
    idx[:] = -1
    np.testing.assert_array_equal(d.channels(), np.arange(5))


def test_channels_raises_rather_than_falling_back(tmp_path):
    d = make(nx=5)
    with pytest.raises(ValueError, match="no 'taptest' channel numbers"):
        d.channels(type='taptest')


def test_slicing_keeps_every_label_aligned(tmp_path):
    info = _dasinfo(tmp_path)
    d = make(nx=12, ch0=0).select_taptest(info)

    sub = d.truncate(ch_range=(2, 8)).skip_ch(2)
    np.testing.assert_array_equal(sub.channels(type='taptest'), [2, 4, 6])
    np.testing.assert_array_equal(sub.index_raw, [3, 6, 8])
    assert sub.channel_type == 'taptest'               # carried through the slice
    np.testing.assert_array_equal(sub.channels(), [2, 4, 6])


def test_select_channels_selects_in_the_active_axis(tmp_path):
    """Whatever `channels()` reports is what you select by."""
    info = _dasinfo(tmp_path)
    d = make(nx=12, ch0=0).select_taptest(info)

    by_taptest = d.select(ch_index=[2, 5])           # taptest is active
    d.channel_type = 'raw'
    by_raw = d.select(ch_index=[3, 7])           # the same two rows
    np.testing.assert_array_equal(
        by_raw.index_raw,
        by_taptest.index_raw)


def test_truncate_follows_channel_type(tmp_path):
    """The whole point of the unification: one rule, and `ch_range` obeys
    whichever axis is active."""
    info = _dasinfo(tmp_path)
    d = make(nx=12, ch0=0).select_taptest(info)
    # raw 0,1,3,4,6,7,8,10,11  <->  taptest 0..8

    by_taptest = d.truncate(ch_range=(2, 6))         # taptest is active
    np.testing.assert_array_equal(
        by_taptest.index_raw, [3, 4, 6, 7])
    np.testing.assert_array_equal(by_taptest.channels(), [2, 3, 4, 5])

    d.channel_type = 'raw'
    by_raw = d.truncate(ch_range=(3, 8))
    np.testing.assert_array_equal(
        by_raw.index_raw, [3, 4, 6, 7])


def test_ch0_tracks_the_raw_labels_through_a_slice(tmp_path):
    """`ch0 + c0*dch` is the row's raw number only while the rows are a
    ramp. After a gappy selection it drifts, and the viewer and picker read
    `ch0` directly."""
    info = _dasinfo(tmp_path)
    d = make(nx=12, ch0=0).select_taptest(info)
    for sub in (d.truncate(ch_range=(3, 8)),
                d.truncate(ch_range=(2, 7)),
                d.skip_ch(2)):
        assert sub.ch0 == sub.index_raw[0]
