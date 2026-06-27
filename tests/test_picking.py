"""Tests for dasio.picking — exercises dasio-owned logic with the phasenet seams mocked
(no weight downloads, no GPU, no real phasenet inference)."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from dasio.dasdata import DASdata
import dasio.picking as pk


def _d(nx=8, nt=400):
    x = np.random.default_rng(0).standard_normal((nx, nt)).astype(np.float32)
    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    return DASdata(
        data=x,
        fs=100.0,
        dt=0.01,
        nt=nt,
        nx=nx,
        dx=1.0,
        begin_time=t0,
        end_time=t0,
        units="strain/s",
    )


def _fake_detect(scores, channel_ids, event_id, begin, dt_s, vmin, device):
    return [
        {
            "channel_index": "2",
            "phase_index": 100,
            "phase_time": "2023-01-01T00:00:01",
            "phase_score": 0.9,
            "phase_type": "P",
        },
        {
            "channel_index": "5",
            "phase_index": 250,
            "phase_time": "2023-01-01T00:00:02.5",
            "phase_score": 0.8,
            "phase_type": "S",
        },
    ]


@pytest.fixture
def mocked(monkeypatch):
    monkeypatch.setattr(
        pk, "predict_2d", lambda *a, **k: np.zeros((3, 8, 400), np.float32)
    )
    monkeypatch.setattr(pk, "detect_picks", _fake_detect)
    monkeypatch.setattr(pk, "_load_model", lambda model, device: object())


def test_pick_phases_returns_picks(mocked):
    res = pk.pick_phases(_d(), device="cpu")
    assert isinstance(res, pk.Picks)
    assert list(res.df.columns) == [
        "channel_index", "phase_index", "phase_type", "phase_time", "phase_score",
    ]
    assert res.df["channel_index"].tolist() == [2, 5]
    assert set(res.df["phase_type"]) == {"P", "S"}
    assert res.df["phase_index"].dtype.kind == "i" and res.df["phase_score"].dtype.kind == "f"
    assert res.scores is None  # default: no heatmap
    assert "1 P" in repr(res) and "1 S" in repr(res) and "phasenet-das+" in repr(res)


def test_return_scores_keeps_heatmap(mocked):
    res = pk.pick_phases(_d(), device="cpu", return_scores=True)
    assert res.scores.shape == (3, 8, 400)


def test_bad_model_raises():
    with pytest.raises(ValueError, match="model"):
        pk.pick_phases(_d(), model="nope", device="cpu")


def test_missing_phasenet_raises(monkeypatch):
    monkeypatch.setattr(pk, "predict_2d", None)
    monkeypatch.setattr(pk, "detect_picks", None)
    with pytest.raises(ImportError, match="phasenet"):
        pk.pick_phases(_d(), device="cpu")


def test_lazy_exports_resolve():
    import dasio

    assert callable(dasio.pick_phases)
    assert dasio.Picks.__name__ == "Picks"


def test_picks_plot_scatters_p_and_s(mocked):
    import matplotlib

    matplotlib.use("Agg")
    res = pk.pick_phases(_d(), device="cpu")
    ax = res.plot()
    assert len(ax.collections) == 2  # P and S scatter layers
    assert ax.get_xlabel() == "channel" and ax.get_ylabel() == "time (s)"
    assert ax.yaxis_inverted()  # time downward


def test_picks_plot_applies_t0_offset():
    import matplotlib

    matplotlib.use("Agg")
    df = pd.DataFrame(
        {"channel_index": [2], "phase_type": ["P"], "phase_index": [100], "phase_score": [0.9]}
    )
    p = pk.Picks(df=df, fs=100.0, model="m", begin_time=None, nx=8, nt=400, t0_sec=-30.0)
    ax = p.plot()  # y = t0_sec + sample/fs = -30 + 1.0, matching DASdata.time_axis
    assert ax.collections[0].get_offsets()[0][1] == pytest.approx(-29.0)
