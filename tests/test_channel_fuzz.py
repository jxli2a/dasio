"""Randomised check that channel labels never drift from the rows they describe.

Every row's payload *is* its raw channel number, so any misalignment
between `channels` and `data` shows up as a payload mismatch rather than as a
plausible-looking wrong number.
"""
import random
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from dasio import DASinfo
from dasio.dasdata import DASdata

NX, NT = 40, 120
T0 = datetime(2023, 1, 1, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def info(tmp_path_factory):
    path = tmp_path_factory.mktemp("info") / "i.csv"
    pd.DataFrame([
        {"index": i, "status": 0 if i % 7 == 3 else 1, "lat": 37.0, "lon": -118.0}
        for i in range(2000, 2000 + NX)
    ]).to_csv(path, index=False)
    return DASinfo.from_csv(path)


def fresh():
    opt = 2000 + np.arange(NX)
    return DASdata(
        data=np.repeat(opt[:, None].astype(np.float32), NT, axis=1),
        fs=100.0, dt=0.01, nt=NT, nx=NX, dx=2.0, begin_time=T0, end_time=T0,
        units="strain/s", index_raw=opt,
    )


def check(d, where):
    opt = d.index_raw
    assert len(opt) == d.nx, f"{where}: {len(opt)} labels for {d.nx} rows"
    assert len(d.channels()) == d.nx, f"{where}: active axis length"
    if d.nx:
        np.testing.assert_array_equal(
            d.data[:, 0], opt.astype(np.float32),
            err_msg=f"{where}: labels do not describe the rows")
        assert d.ch0 == opt[0], f"{where}: ch0 {d.ch0} vs raw[0] {opt[0]}"


@pytest.mark.parametrize("seed", range(60))
def test_labels_follow_the_rows_through_random_chains(seed, info):
    r = random.Random(seed)

    def rng_range(d, via_truncate):
        ax = d.channels()
        lo = int(r.choice(ax.tolist()))
        hi = lo + r.randint(1, max(2, len(ax) // 2))
        return (d.truncate(ch_range=(lo, hi)) if via_truncate
                else d.select(ch_range=(lo, hi)))

    def rng_index(d):
        ax = d.channels().tolist()
        picks = r.sample(ax, r.randint(1, min(len(ax), 8)))
        if r.random() < 0.3:
            r.shuffle(picks)
        return d.select(ch_index=[int(x) for x in picks])

    def rng_mask(d):
        m = np.zeros(d.nx, dtype=bool)
        m[r.sample(range(d.nx), max(1, d.nx // 3))] = True
        return d.select(ch_index=m)

    ops = [
        ("select_taptest", lambda d: d.select_taptest(info)),
        ("set raw", lambda d: replace(d, channel_type="raw")),
        ("set taptest", lambda d: (
            replace(d, channel_type="taptest") if d.dasinfo is not None else d)),
        ("ch_range", lambda d: rng_range(d, False)),
        ("truncate", lambda d: rng_range(d, True)),
        ("ch_index", rng_index),
        ("mask", rng_mask),
        ("skip_ch", lambda d: d.skip_ch(r.choice([2, 3, 5]))),
        ("skip_t", lambda d: d.skip_t(r.choice([2, 4]))),
        ("t_range", lambda d: d.select(t_range=(0.0, r.uniform(0.3, 1.0)))),
    ]

    d, trail = fresh(), ["fresh"]
    check(d, "fresh")
    for _ in range(r.randint(1, 6)):
        if d.nx == 0:
            break
        name, op = r.choice(ops)
        try:
            d = op(d)
        except ValueError as e:                     # a legitimate refusal
            assert "no channel selected" in str(e) or "not in this DASdata" in str(e)
            break
        trail.append(name)
        check(d, " -> ".join(trail))

