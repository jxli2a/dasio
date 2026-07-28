"""OptaSense `RawData` ships in two axis orders; the reader must honour both.

Files carry a `Dimensions` attribute on `RawData` saying which they are:

    mammoth_south  (9775, 30000)   Dimensions=[b'locus', b'time']   channel-major
    iceland_quantx (48000, 10200)  Dimensions=[b'time', b'locus']   time-major

Assuming channel-major silently transposes a time-major file: the returned
DASdata has nx and nt swapped, and every value is read from the wrong axis, so
the numbers are garbage rather than merely mis-shaped.
"""
import h5py
import numpy as np
import pytest

from dasio.readers.optasense import read_optasense_metadata, read_optasense_raw

NX, NT, FS = 6, 40, 100.0
T0_US = 1_700_000_000_000_000
# Distinct per (channel, sample) so a transpose cannot go unnoticed.
PAYLOAD = (np.arange(NX)[:, None] * 1000 + np.arange(NT)[None, :]).astype(np.int32)


def _write(path, data, dims):
    with h5py.File(path, "w") as f:
        acq = f.create_group("Acquisition")
        acq.attrs["GaugeLength"] = 10.0
        acq.attrs["SpatialSamplingInterval"] = 1.0
        raw = acq.create_group("Raw[0]")
        raw.attrs["OutputDataRate"] = FS
        ds = raw.create_dataset("RawData", data=data)
        ds.attrs["Dimensions"] = np.array(dims, dtype="S6")
        raw.create_dataset(
            "RawDataTime",
            data=T0_US + (np.arange(NT) * (1e6 / FS)).astype(np.int64),
        )
        custom = acq.create_group("Custom")
        custom.attrs["Fibre Refractive Index"] = 1.468
        custom.attrs["Laser Wavelength (nm)"] = 1550.0
        custom.attrs["Num Output Channels"] = NX
    return path


@pytest.fixture
def locus_major(tmp_path):
    return _write(tmp_path / "locus.h5", PAYLOAD, [b"locus", b"time"])


@pytest.fixture
def time_major(tmp_path):
    return _write(tmp_path / "time.h5", PAYLOAD.T.copy(), [b"time", b"locus"])


def test_time_major_file_reads_with_the_same_shape_as_channel_major(time_major):
    d = read_optasense_raw(time_major)
    assert d.data.shape == (NX, NT)
    assert (d.nx, d.nt) == (NX, NT)


def test_time_major_file_reads_the_same_values(locus_major, time_major):
    """Both files hold identical data, only stored along different axes."""
    np.testing.assert_array_equal(
        read_optasense_raw(time_major).data, read_optasense_raw(locus_major).data
    )


def test_channel_and_sample_selection_apply_to_the_right_axes(time_major):
    d = read_optasense_raw(time_major, min_ch=2, max_ch=5, first_sample=10, n_samples=7)
    assert d.data.shape == (3, 7)
    np.testing.assert_array_equal(d.data, PAYLOAD[2:5, 10:17].astype(d.data.dtype))


def test_metadata_reports_the_true_channel_count_without_the_custom_attr(tmp_path):
    """`nx` falls back to RawData.shape[0], which is the time axis on these files."""
    p = _write(tmp_path / "nocustom.h5", PAYLOAD.T.copy(), [b"time", b"locus"])
    with h5py.File(p, "r+") as f:
        del f["Acquisition/Custom"].attrs["Num Output Channels"]
    (meta,) = read_optasense_metadata(p)
    assert meta["nx"] == NX
    assert meta["nt"] == NT
