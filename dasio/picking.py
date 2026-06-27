"""PhaseNet-DAS P/S phase picking. `phasenet` (AI4EPS) and torch are optional and
imported lazily: `pip install 'dasio[pick]'`."""

import importlib
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

try:
    from phasenet.utils.inference import predict_2d, detect_picks
except ImportError:  # picking stays importable without phasenet
    predict_2d = detect_picks = None

# name -> (phasenet builder module, GitHub release tag, checkpoint filename)
_MODELS = {
    "phasenet-das": ("phasenet_das", "PhaseNet-DAS-v1", "PhaseNet-DAS-v1.pth"),
    "phasenet-das+": (
        "phasenet_das_plus",
        "PhaseNet-DAS-Plus-Arcata-v1",
        "phasenet_das_plus_arcata_v1.pth",
    ),
}
_CACHE = {}  # (model, device) -> loaded model


def _download(release, filename):
    path = os.path.expanduser(f"~/.cache/phasenet/{release}/{filename}")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        url = f"https://github.com/AI4EPS/models/releases/download/{release}/{filename}"
        urllib.request.urlretrieve(url, path)
    return path


def _load_model(model, device):
    if (model, device) not in _CACHE:
        import torch

        builder, release, filename = _MODELS[model]
        ckpt = torch.load(
            _download(release, filename), map_location="cpu", weights_only=False
        )
        build = importlib.import_module(f"phasenet.models.{builder}").build_model
        net = build(**ckpt.get("model_config", {}))
        net.load_state_dict(ckpt.get("model_ema_weights", ckpt["model"]))
        _CACHE[(model, device)] = net.to(device).eval()
    return _CACHE[(model, device)]


@dataclass
class Picks:
    df: object  # DataFrame: channel_index, phase_index, phase_type, phase_time, phase_score
    fs: float
    model: str
    begin_time: Optional[datetime]
    nx: int
    nt: int
    t0_sec: float = 0.0  # seconds-axis value at sample 0, matching DASdata.time_axis
    scores: Optional[np.ndarray] = None

    def __repr__(self):
        n_p = int((self.df["phase_type"] == "P").sum())
        n_s = int((self.df["phase_type"] == "S").sum())
        return f"Picks({n_p} P, {n_s} S over {self.nx} ch, {self.model})"

    def plot(self, ax=None, s=4, **kwargs):
        """Scatter P (red) / S (blue) picks — channel (x) vs time (y, downward)."""
        import matplotlib.pyplot as plt

        own = ax is None
        if own:
            _, ax = plt.subplots(**kwargs)
        for phase, color in (("P", "red"), ("S", "blue")):
            sub = self.df[self.df["phase_type"] == phase]
            ax.scatter(
                sub["channel_index"],
                self.t0_sec + sub["phase_index"] / self.fs,  # match DASdata.time_axis
                s=s,
                c=color,
                label=f"{phase} ({len(sub)})",
            )
        ax.set_xlabel("channel")
        ax.set_ylabel("time (s)")
        ax.legend()
        if own:
            ax.invert_yaxis()  # time downward, seismic convention
        return ax


def pick_phases(
    d,
    model="phasenet-das+",
    *,
    min_prob=0.3,
    device=None,
    return_scores=False,
    nx_win=2048,
    nt_win=4096,
    min_t_overlap=0.1,
    taper_nx=True,
    taper_nt=True,
):
    """Pick P/S phases with PhaseNet-DAS. Feeds d.data verbatim (model self-normalizes)."""
    import pandas as pd

    if model not in _MODELS:
        raise ValueError(f"model must be one of {list(_MODELS)}, got {model!r}")
    if predict_2d is None:
        raise ImportError("phase picking needs phasenet: pip install 'dasio[pick]'")
    if device is None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    net = _load_model(model, device)
    waveform = d.data.astype(np.float32)[np.newaxis]  # (1, nx, nt)
    out = predict_2d(
        net,
        waveform,
        nx_win,
        nt_win,
        min_t_overlap=min_t_overlap,
        device=device,
        taper_nx=taper_nx,
        taper_nt=taper_nt,
    )
    scores = out[0] if isinstance(out, tuple) else out  # (3, nx, nt) [noise, P, S]

    begin = d.begin_time.isoformat() if d.begin_time is not None else None
    raw = detect_picks(
        scores,
        [str(i) for i in range(d.nx)],
        "",
        begin,
        dt_s=d.dt,
        vmin=min_prob,
        device=device,
    )
    cols = ["channel_index", "phase_index", "phase_type", "phase_time", "phase_score"]
    if raw:
        r = pd.DataFrame(raw)
        df = pd.DataFrame(
            {
                "channel_index": r["channel_index"].astype(int),
                "phase_index": r["phase_index"].astype(int),
                "phase_type": r["phase_type"],
                "phase_time": pd.to_datetime(r["phase_time"], format="ISO8601"),
                "phase_score": r["phase_score"].astype(float),
            }
        )
    else:
        df = pd.DataFrame(columns=cols)
    return Picks(
        df=df,
        fs=d.fs,
        model=model,
        begin_time=d.begin_time,
        nx=d.nx,
        nt=d.nt,
        t0_sec=getattr(d, "t0_sec", 0.0),
        scores=scores if return_scores else None,
    )
