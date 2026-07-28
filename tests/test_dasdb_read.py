"""DASdb.read should carry the per-file `units` into the concatenated DASdata."""
from datetime import timedelta

import pandas as pd

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
