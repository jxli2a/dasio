"""Regression: OptaSense metadata must not split a gap-free file into
thousands of one-sample chunks due to float cancellation.

`RawDataTime` is int64 microseconds since epoch (~1.67e15). Scaling to
seconds (~1.67e9) before `np.diff` loses precision: the true 1/fs stride
comes out with ~1.6e-7 s noise. At fs=1000 (dt=1e-3) a too-tight rtol on
`np.isclose` drops the tolerance into that noise floor and every-other
sample reads as an acquisition gap, exploding the catalog. See
DAS-utilities DAS_db.py (rtol=1e-3) for the original that never hit this.
"""
import numpy as np
import h5py
import pytest

from dasio.readers.optasense import read_optasense_metadata

# Epoch-scale start (matches real captures) so the float cancellation is
# actually exercised; fs=1000 puts the stride at 1e-3 s where the noise
# is comparable to a tight isclose tolerance.
T0_US = 1_673_591_029_958_700  # microseconds since epoch (real South Pole value)
FS = 1000.0
NX, NT = 8, 60_000


def _write_regular_optasense(path, t0_us=T0_US, fs=FS, nx=NX, nt=NT):
    with h5py.File(path, "w") as f:
        acq = f.create_group("Acquisition")
        acq.attrs["GaugeLength"] = 10.0
        acq.attrs["SpatialSamplingInterval"] = 1.0
        raw = acq.create_group("Raw[0]")
        raw.attrs["OutputDataRate"] = fs
        raw.create_dataset(
            "RawData", data=np.zeros((nx, nt), dtype=np.int32))
        # exactly-regular integer-microsecond timestamps: no real gaps
        step_us = int(round(1e6 / fs))
        raw.create_dataset(
            "RawDataTime",
            data=(t0_us + np.arange(nt, dtype=np.int64) * step_us))
        custom = acq.create_group("Custom")
        custom.attrs["Num Output Channels"] = nx
    return path


def test_regular_stride_epoch_scale_yields_single_row(tmp_path):
    """A gap-free 1 kHz file at epoch scale must be one contiguous chunk."""
    p = _write_regular_optasense(tmp_path / "regular.h5")
    metas = read_optasense_metadata(p)
    assert len(metas) == 1, (
        f"expected 1 contiguous chunk, got {len(metas)} — float "
        "cancellation in stride detection is splitting a gap-free file"
    )
    m = metas[0]
    assert m["nt"] == NT
    assert m["first_sample"] == 0
    assert m["fs"] == FS


def test_genuine_intrafile_gap_still_splits(tmp_path):
    """A real gap (missing samples mid-file) must still produce two rows."""
    p = tmp_path / "gapped.h5"
    step_us = int(round(1e6 / FS))
    t = T0_US + np.arange(NT, dtype=np.int64) * step_us
    # insert a 500-sample gap at the midpoint
    t[NT // 2:] += 500 * step_us
    with h5py.File(p, "w") as f:
        acq = f.create_group("Acquisition")
        raw = acq.create_group("Raw[0]")
        raw.attrs["OutputDataRate"] = FS
        raw.create_dataset("RawData", data=np.zeros((NX, NT), dtype=np.int32))
        raw.create_dataset("RawDataTime", data=t)
        custom = acq.create_group("Custom")
        custom.attrs["Num Output Channels"] = NX
    metas = read_optasense_metadata(p)
    assert len(metas) == 2, f"expected 2 chunks across the gap, got {len(metas)}"
    assert metas[0]["nt"] + metas[1]["nt"] == NT
