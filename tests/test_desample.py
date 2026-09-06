"""desample_window has no test file; drive it on a real Proc file."""
import numpy as np, h5py, pandas as pd, pytest
from dasio.dasdb import DASdb
from dasio.readers.proc import read_metadata_proc
import dasio.desample as ds


@pytest.fixture
def procs(tmp_path):
    NX, NT, FS = 16, 256, 100.0
    paths = []
    for k in range(2):
        p = tmp_path / f"proc{k}.hdf5"
        with h5py.File(p, "w") as f:
            d = f.create_dataset(
                "Data",
                data=np.random.default_rng(k).standard_normal((NX, NT)).astype(np.float32))
            d.attrs.update(nCh=NX, nt=NT, dt=1.0 / FS, fs=FS, dCh=2.0, GaugeLength=10.0)
            t0 = 1699999999.0 + k * NT / FS
            d.attrs["startTime"] = pd.Timestamp(t0, unit="s", tz="UTC").isoformat()
            d.attrs["endTime"] = pd.Timestamp(
                t0 + (NT - 1) / FS, unit="s", tz="UTC").isoformat()
            f.create_dataset("Acquisition_origin", data=np.float32(0.0))
        paths.append(p)
    return paths


def test_desample_window_preserves_the_channel_axis(procs):
    db = DASdb(pd.DataFrame([read_metadata_proc(p) for p in procs]), "Proc")
    rows = db.df.sort_values("begin_time")
    from dasio.schema import RawWindow
    rw = RawWindow(rows=rows,
                   begin_time=rows.begin_time.min(),
                   end_time=rows.end_time.max(),
                   has_pad_before=False, has_pad_after=False)
    out = ds.desample_window(rw, format="Proc", min_ch=4, max_ch=12)

    assert out.nx == 8
    np.testing.assert_array_equal(out.channels(), np.arange(4, 12))
    assert out.ch0 == 4 and out.dch == 1
    # the labels must be exactly one per row, or every downstream lookup slides
    assert len(out.index_raw) == out.nx


def test_proc_read_records_min_ch(procs):
    """`readers/proc.py` sliced `dset[min_ch:max_ch]` but never recorded the
    offset, so a Proc read reported channels 0..nx-1 whatever was asked for —
    and `desample_window`, which reads files directly rather than through
    `DASdb.read`, inherited it."""
    from dasio.dasfile import DASFile
    d = DASFile(procs[0], format="Proc").read(min_ch=4, max_ch=12)
    assert d.nx == 8
    np.testing.assert_array_equal(d.channels(), np.arange(4, 12))


def test_channel_axis_survives_chunked_reads(procs):
    """`desample_window` reads in `nchbuffer`-wide channel chunks and stacks
    them. Labelling the output from the first chunk gives an axis shorter than
    the data whenever more than one chunk is needed."""
    db = DASdb(pd.DataFrame([read_metadata_proc(p) for p in procs]), "Proc")
    rows = db.df.sort_values("begin_time")
    from dasio.schema import RawWindow
    rw = RawWindow(rows=rows, begin_time=rows.begin_time.min(),
                   end_time=rows.end_time.max(),
                   has_pad_before=False, has_pad_after=False)

    out = ds.desample_window(rw, format="Proc", min_ch=2, max_ch=14, nchbuffer=3)
    assert out.nx == 12                                  # 4 chunks of 3
    assert len(out.index_raw) == out.nx        # one label per row
    np.testing.assert_array_equal(out.channels(), np.arange(2, 14))
