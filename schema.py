"""Shared types + column conventions for the dasio subpackage.

`RawWindow` is the read plan produced by `DASdb.select_window` — kept
here so `dasdb.py` and `desample.py` can depend on one shared schema
module instead of each other. The per-file metadata shape lives in
`dasdata.py` as `DASmeta` (one TypedDict, used as a plain dict).

Time convention: `begin_time` / `end_time` are last-sample-inclusive
timestamps, matching the legacy DAS-utilities `DAS_db.py` contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd


# DataFrame column names used by DASdb — mirror DASmeta field names
# 1:1 so a list of DASmeta dicts is a valid pandas input. Continuity
# isn't persisted; DASdb.segments recomputes it from
# begin_time / end_time / fs / nx on every query.
_DF_COLUMNS = [
    'file', 'begin_time', 'end_time', 'fs', 'nt',
    'nx', 'dx', 'gauge_length_m', 'first_sample',
]

# Legacy DAS_db CSV ↔ DataFrame column translation. Applied only at
# the to_csv / from_csv boundary; the in-memory DataFrame always uses
# the snake_case names above.
_CSV_TO_DF = {
    'begTime':     'begin_time',
    'startTime':   'begin_time',      # older legacy alias
    'endTime':     'end_time',
    'nChannels':   'nx',
    'dCh':         'dx',
    'GaugeLen':    'gauge_length_m',
    'firstSample': 'first_sample',
}
_DF_TO_CSV = {
    'begin_time':     'begTime',
    'end_time':       'endTime',
    'nx':             'nChannels',
    'dx':             'dCh',
    'gauge_length_m': 'GaugeLen',
    'first_sample':   'firstSample',
}


@dataclass
class RawWindow:
    """Concrete read plan for one desample window.

    Returned by `DASdb.select_window`. Consumed by
    `desample.desample_window`. `rows` is a slice of the parent
    `DASdb.df` (so every column — `file`, `first_sample`, `nt`,
    `fs`, `dx`, `gauge_length_m`, `begin_time`, `end_time` — is
    accessible to the reader without a second trip through the
    catalog). For OptaSense, `first_sample` / `nt` skip past
    intra-file RawDataTime gaps; for ASN / Proc they're 0 / full
    file length.

    `has_pad_{before,after}` tell the pipeline whether the first /
    last row is a pad file (vs. the window's first / last target
    file), used by `desample_window`'s filter-edge trim logic.
    """
    rows:           pd.DataFrame
    begin_time:     datetime
    end_time:       datetime
    has_pad_before: bool
    has_pad_after:  bool

    @property
    def files(self) -> List[Path]:
        """Convenience for callers that only want the files."""
        return [Path(f) for f in self.rows['file']]
