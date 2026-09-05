"""ISO-8601 round-trip, including stamps finer than a microsecond."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from dasio.utils import iso_timestamp, parse_iso


@pytest.mark.parametrize("text,micro", [
    ("2023-11-14T22:13:21.549999952+00:00", 549999),   # pandas, nanoseconds
    ("2023-11-14T22:13:21.549999+00:00", 549999),
    ("2023-11-14T22:13:21.549+00:00", 549000),
    ("2023-11-14T22:13:21+00:00", 0),
])
def test_parse_iso_takes_any_fractional_precision(text, micro):
    """`datetime` stops at microseconds and `fromisoformat` before 3.11 rejects
    any other digit count, so a `pandas.Timestamp.isoformat` stamp made a Proc
    file readable on 3.12 and unreadable on 3.10. Only the 3.10 CI job can fail
    this one; 3.11+ parses all four either way."""
    t = parse_iso(text)
    assert t.microsecond == micro
    assert t.tzinfo is not None


def test_iso_timestamp_round_trips():
    t = datetime(2023, 11, 14, 22, 13, 21, 549999, tzinfo=timezone.utc)
    assert parse_iso(iso_timestamp(t)) == t


def test_a_pandas_stamp_survives_the_round_trip():
    """What the Proc fixtures write: a float epoch through pandas, which lands
    on nanoseconds."""
    text = pd.Timestamp(1699999999.0 + 1.55, unit="s", tz="UTC").isoformat()
    assert parse_iso(text).microsecond == 549999
