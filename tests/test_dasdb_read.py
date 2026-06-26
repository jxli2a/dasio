"""DASdb.read should carry the per-file `units` into the concatenated DASdata."""
from datetime import timedelta

import pandas as pd

from dasio.dasdb import DASdb
from dasio.dasfile import DASFile
from dasio.readers.proc import read_metadata_proc


def test_dasdb_read_propagates_units(proc_file):
    meta = read_metadata_proc(proc_file)
    db = DASdb(pd.DataFrame([meta]), "Proc")

    # what a direct single-file read yields (convert=True default)
    expected = DASFile(proc_file, system="Proc").read().units
    assert expected != "unknown"

    out = db.read(meta["begin_time"], meta["begin_time"] + timedelta(seconds=1.0))
    assert out.units == expected          # no longer dropped to "unknown"
