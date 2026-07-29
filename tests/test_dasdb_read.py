"""DASdb.read should carry the per-file `units` into the concatenated DASdata."""
from datetime import timedelta

import pandas as pd
import pytest

from dasio.dasdb import DASdb
from dasio.dasfile import DASFile
from dasio.readers.optasense import read_optasense_metadata
from dasio.readers.proc import read_metadata_proc


def test_dasdb_read_propagates_units(proc_file):
    meta = read_metadata_proc(proc_file)
    db = DASdb(pd.DataFrame([meta]), "Proc")

    # what a direct single-file read yields
    expected = DASFile(proc_file, format="Proc").read().units
    assert expected != "unknown"

    out = db.read(meta["begin_time"], meta["begin_time"] + timedelta(seconds=1.0))
    assert out.units == expected          # no longer dropped to "unknown"


def test_dasdb_read_attaches_the_factor_by_default(optasense_file):
    """Data stays raw counts — unwrap has to run on the concatenated array
    first — but the factor rides along so `.to_physical()` needs no re-read."""
    metas = read_optasense_metadata(optasense_file)
    db = DASdb(pd.DataFrame(metas), "OptaSense")
    m0 = metas[0]
    out = db.read(m0["begin_time"], m0["begin_time"] + timedelta(seconds=1.0))
    assert out.units == "count"
    assert out.physical_factor != 1.0


def test_dasdb_read_with_factor_attaches_conversion(optasense_file):
    """with_factor=True attaches DASFile.factor() so to_physical() works."""
    metas = read_optasense_metadata(optasense_file)
    db = DASdb(pd.DataFrame(metas), "OptaSense")
    m0 = metas[0]
    fac = DASFile(optasense_file, format="OptaSense").factor()
    assert fac != 1.0

    out = db.read(
        m0["begin_time"], m0["begin_time"] + timedelta(seconds=1.0),
        with_factor=True,
    )
    # factor is attached but not yet applied (units still raw counts)
    assert out.units == "count"
    assert out.physical_factor == fac
    # and to_physical() now succeeds, taking counts all the way to microstrain
    phys = out.to_physical()
    assert phys.units == "microstrain"
    assert phys.physical_factor == 1.0


def test_from_dir_detects_the_asn_day_channel_layout(tmp_path, asn_file):
    """ASN writes <YYYYMMDD>/<channel>/<HHMMSS>.hdf5 — three levels down. The
    auto-detect probe only globbed two, so a real ASN tree raised 'no .h5/.hdf5
    files ... to detect format from' unless format= was passed."""
    import shutil
    from dasio.dasdb import DASdb

    nested = tmp_path / "20250716" / "dphi"
    nested.mkdir(parents=True)
    shutil.copy(asn_file, nested / "001114.hdf5")

    assert DASdb.from_dir(tmp_path, progress=False).format == "ASN"


@pytest.mark.parametrize("depth", ["root", "day", "leaf"])
def test_asn_scan_accepts_being_pointed_below_the_day_level(tmp_path, asn_file, depth):
    """ASN's layout is <root>/<YYYYMMDD>/<ch>/*.hdf5, and the scanner looked
    only for day folders inside what it was given — so cataloguing a single
    day, or the channel folder itself, silently returned zero files."""
    import shutil
    from dasio.dasdb import list_das_files

    leaf = tmp_path / "20250716" / "dphi"
    leaf.mkdir(parents=True)
    shutil.copy(asn_file, leaf / "000004.hdf5")

    target = {"root": tmp_path, "day": leaf.parent, "leaf": leaf}[depth]
    assert len(list_das_files(target, "ASN")) == 1


@pytest.fixture
def gapped_db(tmp_path):
    """Five 10 s Proc files with a 15 s acquisition gap after the third.

    Each file is filled with its own constant so a misplaced write is visible
    as the wrong value, not just the wrong length.
    """
    import h5py
    import numpy as np
    from datetime import datetime, timezone

    t0 = datetime(2023, 1, 1, tzinfo=timezone.utc)
    rows = []
    for k, start in enumerate([0, 10, 20, 45, 55]):
        p = tmp_path / f"f{k}.h5"
        with h5py.File(p, "w") as f:
            ds = f.create_dataset("Data", data=np.full((6, 1000), float(k + 1), np.float32))
            for a, v in (("nCh", 6), ("nt", 1000), ("dt", 0.01), ("fs", 100.0), ("dCh", 1.0)):
                ds.attrs[a] = v
            b = t0 + timedelta(seconds=start)
            ds.attrs["startTime"] = b.strftime("%Y-%m-%dT%H:%M:%S.%f+00:00")
            ds.attrs["endTime"] = (b + timedelta(seconds=9.99)).strftime(
                "%Y-%m-%dT%H:%M:%S.%f+00:00")
            f.create_dataset("Acquisition_origin", data=np.float32(0.0))
        rows.append(dict(file=str(p), begin_time=b, end_time=b + timedelta(seconds=9.99),
                         fs=100.0, nt=1000, nx=6, dx=1.0, gauge_length_m=None,
                         first_sample=0))
    return DASdb(pd.DataFrame(rows), "Proc"), t0


def test_fill_gap_zero_fills_and_keeps_the_time_axis_uniform(gapped_db):
    import numpy as np

    db, t0 = gapped_db
    d = db.read(t0, t0 + timedelta(seconds=70))
    assert d.nt == 6500                                  # 30 s + 15 s gap + 20 s
    np.testing.assert_array_equal(d.data[0, :1000], 1.0)
    np.testing.assert_array_equal(d.data[0, 2000:3000], 3.0)
    np.testing.assert_array_equal(d.data[0, 3000:4500], 0.0)   # the gap
    np.testing.assert_array_equal(d.data[0, 4500:5500], 4.0)


def test_fill_gap_false_closes_the_gap(gapped_db):
    import numpy as np

    db, t0 = gapped_db
    d = db.read(t0, t0 + timedelta(seconds=70), fill_gap=False)
    assert d.nt == 5000                                  # the 15 s gap is dropped
    assert not (d.data == 0).any()
    np.testing.assert_array_equal(d.data[0, 2000:3000], 3.0)
    np.testing.assert_array_equal(d.data[0, 3000:4000], 4.0)   # abuts, no zeros


def test_read_output_is_c_contiguous(gapped_db):
    """The C++ bandpass reads the raw buffer; a concatenation of transposed
    reader output could come back F-contiguous."""
    db, t0 = gapped_db
    assert db.read(t0, t0 + timedelta(seconds=70)).data.flags["C_CONTIGUOUS"]


def test_cli_format_is_optional_for_both_create_and_update(tmp_path, proc_file, capsys):
    """`to_csv` writes no format column, so the update path cannot recover it
    from the catalog — the CLI resolves it from the raw directory instead."""
    import shutil
    from dasio.dasdb import main

    raw = tmp_path / "raw"; raw.mkdir()
    shutil.copy(proc_file, raw / "a.h5")
    cat = tmp_path / "cat.csv"

    main(["--from", str(raw), "--dasdb", str(cat)])
    assert DASdb.from_file(cat, format="Proc").n_files == 1
    main(["--from", str(raw), "--dasdb", str(cat)])   # update
    assert DASdb.from_file(cat, format="Proc").n_files == 1


def test_scan_warns_when_no_file_matches_the_format(tmp_path, proc_file, capsys):
    """Readers return [] for a file failing their signature check so a mixed
    directory scans cleanly; every file failing means the format is wrong, and
    the only symptom used to be an empty catalog."""
    import shutil
    from dasio.dasdb import scan_metadata

    raw = tmp_path / "raw"; raw.mkdir()
    shutil.copy(proc_file, raw / "a.h5")

    assert scan_metadata(raw, "OptaSense", progress=False).empty
    assert "none matched that format" in capsys.readouterr().err


@pytest.mark.parametrize("layout,fmt", [
    ("flat_h5", "OptaSense"), ("flat_hdf5", "APSensing"), ("nested", "OptaSense"),
])
def test_file_discovery_covers_the_layouts_in_the_archive(tmp_path, layout, fmt):
    """Real folders that scanned to an empty catalog: arcata_usgs nests under a
    dashed date (`2024-12-04/sensor_*.h5`) and south_korea_urban writes `.hdf5`,
    while the OptaSense/APSensing branch globbed only flat `*.h5`."""
    from dasio.dasdb import list_das_files

    target = {"flat_h5": tmp_path / "a.h5",
              "flat_hdf5": tmp_path / "a.hdf5",
              "nested": tmp_path / "2024-12-04" / "sensor_a.h5"}[layout]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.touch()
    assert list_das_files(tmp_path, fmt) == [target]
