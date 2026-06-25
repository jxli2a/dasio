"""Small HDF5 / ISO-8601 / file-discovery helpers shared across dasio."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Union


def atomic_write(file: Union[str, Path], write_fn) -> None:
    """Run write_fn(tmp); on success rename to file, else delete tmp.

    Tmp file lives in the same directory as the target so the rename
    is a true atomic op (cross-fs renames silently fall back to
    copy+remove and break atomicity). On any exception the tmp is
    cleaned up and the destination is left untouched.
    """
    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_name(file.name + '.tmp')
    try:
        write_fn(tmp)
        tmp.replace(file)
    except BaseException:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def utcdatetime(*args, **kwargs) -> datetime:
    """Convenience constructor — always returns a UTC-aware `datetime`.

    A thin obspy.utcdatetime-flavored helper that bypasses the obspy
    dep. Returns a stdlib `datetime` (UTC tzinfo set), so it slots
    into anything in the codebase that already speaks `datetime`.

    Forms:
        utcdatetime()                 - now in UTC
        utcdatetime(1745776503.2)     - from POSIX timestamp (int|float)
        utcdatetime('2026-04-20T...') - ISO-8601 string; naive is taken as UTC
        utcdatetime(datetime(...))    - aware → converted; naive → UTC-stamped
        utcdatetime(2026, 4, 20, ...) - field-style, args go to `datetime(...)`,
                                        tzinfo pinned to UTC

    Any other tz-aware input is converted to UTC via astimezone, not
    relabeled — `dasio` insists on UTC anchors so a non-UTC value
    leaking through would silently shift downstream time axes.
    """
    if not args and not kwargs:
        return datetime.now(timezone.utc)
    if len(args) == 1 and not kwargs:
        x = args[0]
        if isinstance(x, datetime):
            return (x.astimezone(timezone.utc) if x.tzinfo
                    else x.replace(tzinfo=timezone.utc))
        if isinstance(x, (int, float)):
            return datetime.fromtimestamp(x, timezone.utc)
        if isinstance(x, str):
            dt = datetime.fromisoformat(x)
            return (dt.astimezone(timezone.utc) if dt.tzinfo
                    else dt.replace(tzinfo=timezone.utc))
        raise TypeError(
            f"utcdatetime: can't build from {type(x).__name__}"
        )
    # Field-style: datetime(year, month, day, ...) — pin tzinfo to UTC.
    # If the caller smuggled their own tzinfo in kwargs, convert it.
    dt = datetime(*args, **kwargs)
    return (dt.astimezone(timezone.utc) if dt.tzinfo
            else dt.replace(tzinfo=timezone.utc))


def default_nthreads() -> int:
    """Physical cores available to this process.

    Logical CPUs (hyperthreads) don't help an FP-bound OMP filter and
    can hurt via cache contention, so we count physical cores via
    psutil. Capped by `sched_getaffinity` so taskset / cgroup /
    container limits are respected (cron under a quota'd unit etc.).
    """
    import psutil
    n_phys = psutil.cpu_count(logical=False) or 1
    try:
        n_aff = len(os.sched_getaffinity(0))
    except AttributeError:
        n_aff = psutil.cpu_count(logical=True) or n_phys
    return min(n_phys, n_aff)


def list_data_files(root: Union[str, Path],
                    pattern: Union[str, Iterable[str]] = '*') -> List[Path]:
    """Sorted list of files under root matching one or more globs.

    Thin wrapper around pathlib.Path.glob so dasdb.list_das_files and
    any external caller share the same file-selection convention —
    analogous to how legacy DAS-utilities took a DAS_dir glob straight
    into glob.glob.

    pattern may be a single glob ('*.h5', 'DASProcTemp-*.h5') or an
    iterable of globs (['*.h5', '*.hdf5']). Subdirectory patterns
    ('*/*.hdf5') and recursive patterns ('**/*.h5') work too.
    """
    root = Path(root)
    if isinstance(pattern, (str, bytes)):
        return sorted(root.glob(pattern))
    out: List[Path] = []
    seen: set = set()
    for p in pattern:
        for f in root.glob(p):
            if f not in seen:
                seen.add(f)
                out.append(f)
    out.sort()
    return out


def iso_timestamp(dt: datetime) -> str:
    """Serialize a timezone-aware datetime as `YYYY-MM-DDTHH:MM:SS.ffffff+00:00`.

    Matches legacy Desample_DAS.py output exactly so adjacent files can
    be diff-compared byte-for-byte — %f always emits microseconds.
    """
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f+00:00')


def parse_iso(s) -> datetime:
    """Parse an ISO-8601 timestamp back into an aware datetime."""
    return datetime.fromisoformat(str(s))


