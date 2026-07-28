"""`DASdb.plot_timeline` — layout choice and render safety.

The original version handed Timestamp/Timedelta objects straight to `barh` and
then called `autofmt_xdate()` on a 1.6-inch-tall figure. Under some rcParams the
axes collapsed far enough that a `bbox_inches='tight'` save asked for a canvas of
millions of pixels and matplotlib raised. These tests pin the geometry down.
"""
from datetime import datetime, timedelta, timezone

import matplotlib
import pandas as pd
import pytest

matplotlib.use('Agg')

from dasio.dasdb import DASdb


def _db(n_days, start=datetime(2023, 11, 21, tzinfo=timezone.utc), gap_after=None):
    """Catalog of `n_days` one-day files, optionally with a one-day gap."""
    rows = []
    t = start
    for i in range(n_days):
        if gap_after is not None and i == gap_after:
            t += timedelta(days=1)                 # skip a day -> a real gap
        rows.append(dict(
            file=f'f{i}.h5', begin_time=t,
            end_time=t + timedelta(days=1) - timedelta(seconds=1),
            fs=1.0, nt=86400, nx=100, first_sample=0, dx=1.0,
            gauge_length_m=None,
        ))
        t += timedelta(days=1)
    return DASdb(pd.DataFrame(rows), 'Proc')


def test_single_month_catalog_uses_one_continuous_axis():
    ax = _db(5).plot_timeline()
    assert len(ax.get_yticks()) == 0           # no per-month rows
    matplotlib.pyplot.close(ax.figure)


def test_multi_month_catalog_splits_into_one_row_per_month():
    ax = _db(70).plot_timeline()               # Nov -> Jan, three months
    assert [t.get_text() for t in ax.get_yticklabels()] == \
        ['2023-11', '2023-12', '2024-01']
    assert ax.get_xlabel() == 'day of month'
    matplotlib.pyplot.close(ax.figure)


def test_by_month_can_be_forced_either_way():
    assert len(_db(70, ).plot_timeline(by_month=False).get_yticks()) == 0
    assert len(_db(5).plot_timeline(by_month=True).get_yticklabels()) == 1
    matplotlib.pyplot.close('all')


@pytest.mark.parametrize('n_days', [5, 70, 400])
def test_tight_bbox_render_stays_a_sane_size(n_days):
    """The reported failure: a tight-bbox save demanding millions of pixels."""
    from io import BytesIO

    ax = _db(n_days).plot_timeline()
    buf = BytesIO()
    ax.figure.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    assert buf.tell() < 2_000_000              # a normal PNG, not a monster
    w, h = ax.figure.get_size_inches()
    assert w < 40 and h < 40
    matplotlib.pyplot.close(ax.figure)


def test_gaps_are_drawn_as_their_own_bars():
    without = _db(10)
    with_gap = _db(10, gap_after=5)
    assert with_gap.n_segments == 2 and without.n_segments == 1
    ax = with_gap.plot_timeline(by_month=False)
    assert len(ax.patches) == 3                # two segments plus one gap
    matplotlib.pyplot.close(ax.figure)
