"""Reader for PASSCAL SEG-Y DAS files (e.g. legacy Ridgecrest AWS archives).

SEG-Y is a non-HDF5 binary format, so it does not go through the h5py
``detect_data_kind`` path the other readers use; ``DASFile`` routes the
``.segy`` suffix to ``read_passcal_segy``.

Layout: a 3600-byte file header (3200-byte textual + 400-byte binary), then
``ntrace`` traces, each a 240-byte trace header followed by ``ns`` samples.

PASSCAL DAS SEG-Y in the wild often ships an **empty** binary file header
(sample interval / sample count / format code all zero) and unreliable
trace-header timestamps — the original Ridgecrest reader hard-coded
1250 traces x 900000 samples x float32 for exactly this reason. This reader
uses the binary header when it is valid and otherwise recovers the geometry
from the file size plus one caller-supplied dimension (``n_samples`` or
``n_traces``) and the sample interval. The instrument phase->strain factor is
a parameter (``conversion_factor``), not baked in.
"""
from __future__ import annotations

import gzip
import struct
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..dasdata import DASdata, DASmeta, normalize_unit

# SEG-Y data sample format code -> (numpy base dtype char, bytes per sample).
# The endianness is prepended at read time: SEG-Y is nominally big-endian, but
# native-written PASSCAL DAS files are frequently little-endian, so it is
# auto-detected (or overridden). Code 1 (IBM float) is read as raw u4 and
# converted; 2 int32, 3 int16, 5 IEEE float32, 8 int8 are read directly.
_SAMPLE_BASE = {1: ("u4", 4), 2: ("i4", 4), 3: ("i2", 2), 5: ("f4", 4), 8: ("i1", 1)}

_FILE_HEADER_BYTES = 3600
_TRACE_HEADER_BYTES = 240


def _sane_fraction(sample: bytes, dtype: str) -> float:
    """Fraction of values that decode to finite, non-denormal, non-absurd
    magnitudes — the signature of the *correct* float endianness. A wrong
    byte order turns normal floats into NaN/Inf, tiny denormals, or ~1e38
    giants, all of which fall outside the [1e-30, 1e30] band."""
    with np.errstate(invalid="ignore", over="ignore"):       # wrong endian -> NaN/Inf probes
        a = np.frombuffer(sample, dtype=dtype).astype(np.float64)
        mag = np.abs(a)
        sane = np.isfinite(a) & ((a == 0) | ((mag >= 1e-30) & (mag <= 1e30)))
    return float(sane.mean()) if a.size else 0.0


def _detect_float_endian(sample: bytes, base: str) -> str:
    """Pick '>' or '<' for a float sample by which decodes to saner magnitudes."""
    return ">" if _sane_fraction(sample, ">" + base) >= _sane_fraction(sample, "<" + base) else "<"


def _first_trace_sample(file, buf, ns, bps, max_samp=20000) -> bytes:
    """First trace's data bytes (capped) for endianness sniffing."""
    nbytes = min(ns, max_samp) * bps
    off = _FILE_HEADER_BYTES + _TRACE_HEADER_BYTES
    if buf is None:
        with open(file, "rb") as fid:
            fid.seek(off)
            return fid.read(nbytes)
    return bytes(buf[off:off + nbytes])


def _ibm2ieee(ibm: np.ndarray) -> np.ndarray:
    """Vectorized IBM hexadecimal float (format code 1) -> IEEE float64."""
    ibm = ibm.astype(">u4").astype(np.uint32)
    sign = (ibm >> 31) & 0x01
    exponent = (ibm >> 24) & 0x7F
    mantissa = (ibm & 0x00FFFFFF) / float(1 << 24)
    return (1.0 - 2.0 * sign) * mantissa * 16.0 ** (exponent.astype(np.int32) - 64)


def _parse_binary_header(header: bytes):
    """Return (ns, sample_interval_us, format_code) from the 400-byte binary
    header (bytes 3200-3600). Any field may be 0 / invalid for PASSCAL DAS."""
    binh = header[3200:3600]
    sample_interval_us = struct.unpack(">h", binh[16:18])[0]
    ns = struct.unpack(">h", binh[20:22])[0]
    format_code = struct.unpack(">h", binh[24:26])[0]
    return ns, sample_interval_us, format_code


def _trace_header_time(trace_header: bytes) -> Optional[datetime]:
    """Recover GMT start time from a trace header's year/day/hour/min/sec
    (bytes 157-166). Returns None when the fields are absent/garbage."""
    yr, doy, hh, mm, ss = struct.unpack(">5h", trace_header[156:166])
    if 1970 <= yr <= 2100 and 1 <= doy <= 366 and 0 <= hh < 24:
        return (datetime(yr, 1, 1, tzinfo=timezone.utc)
                + timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss))
    return None


def _resolve_geometry(header, total_bytes, n_traces, n_samples,
                      sample_interval, sample_format):
    """Pin down (n_traces, ns, dt, format_code) from header + file size + hints.

    Header values win when valid; otherwise the file size and one supplied
    dimension recover the other. Raises ValueError with a clear message when
    the geometry is underdetermined.
    """
    hdr_ns, hdr_dt_us, hdr_fmt = _parse_binary_header(header)

    fmt = sample_format or (hdr_fmt if hdr_fmt in _SAMPLE_BASE else 5)
    if fmt not in _SAMPLE_BASE:
        raise ValueError(f"unsupported SEG-Y sample format code {fmt!r}; "
                         f"known: {sorted(_SAMPLE_BASE)}")
    bps = _SAMPLE_BASE[fmt][1]
    data_bytes = total_bytes - _FILE_HEADER_BYTES

    ns = n_samples or (hdr_ns if hdr_ns > 0 else None)
    if ns is None and n_traces:
        ns = (data_bytes // n_traces - _TRACE_HEADER_BYTES) // bps
    if not ns or ns <= 0:
        raise ValueError(
            "cannot determine samples-per-trace from the SEG-Y header; pass "
            "n_samples= (or n_traces=, to derive it from the file size)."
        )

    if not n_traces:
        n_traces = data_bytes // (_TRACE_HEADER_BYTES + ns * bps)
    if n_traces <= 0:
        raise ValueError("cannot determine trace count; pass n_traces=.")

    dt = sample_interval or (hdr_dt_us / 1e6 if hdr_dt_us > 0 else None)
    if not dt or dt <= 0:
        raise ValueError(
            "cannot determine sample interval from the SEG-Y header; pass "
            "sample_interval= in seconds (e.g. 1/250)."
        )
    return int(n_traces), int(ns), float(dt), fmt


def _read_bytes(path: Path):
    """Return (header_bytes, full_buffer_or_None, total_bytes). For plain
    files the data is left on disk (full_buffer is None, read lazily via
    np.fromfile); for .gz the whole file is decompressed into memory."""
    suffixes = [s.lower() for s in path.suffixes]
    if suffixes and suffixes[-1] == ".gz":
        buf = gzip.open(path, "rb").read()
        return buf[:_FILE_HEADER_BYTES], buf, len(buf)
    with open(path, "rb") as fid:
        header = fid.read(_FILE_HEADER_BYTES)
    return header, None, path.stat().st_size


def read_passcal_segy(
        file: Union[str, Path],
        *,
        n_traces: Optional[int] = None,
        n_samples: Optional[int] = None,
        sample_interval: Optional[float] = None,
        sample_format: Optional[int] = None,
        endian: str = "auto",
        conversion_factor: float = 1.0,
        begin_time: Optional[datetime] = None,
        dx: Optional[float] = None,
        gauge_length_m: Optional[float] = None,
        units: str = "unknown",
    ) -> DASdata:
    """Read a PASSCAL SEG-Y DAS file into a `DASdata` (shape ``(n_traces, ns)``).

    Parameters
    ----------
    file : path to a ``.segy`` file (a ``.gz``-compressed one is decompressed
        in memory; only the ``.segy`` suffix is auto-routed by ``DASFile``).
    n_traces, n_samples : geometry overrides used when the binary header is
        empty (common for PASSCAL DAS). Supply at least one; the other is
        derived from the file size. Header values are used when valid.
    sample_interval : sample spacing in **seconds** (e.g. ``1/250``). Required
        when the header's interval is 0/invalid.
    sample_format : SEG-Y data format code (1 IBM float, 2 int32, 3 int16,
        5 IEEE float32, 8 int8). Defaults to the header's, else 5.
    endian : ``'auto'`` (default), ``'big'``/``'>'`` or ``'little'``/``'<'``.
        SEG-Y is nominally big-endian, but natively-written PASSCAL DAS files
        are often little-endian; ``'auto'`` sniffs the first trace and picks
        the byte order that decodes to sane magnitudes. IBM floats (code 1)
        are always big-endian.
    conversion_factor : multiply samples by this to reach physical units
        (instrument-specific phase->strain; not baked in). Default 1.0.
    begin_time : absolute UTC start. Falls back to the trace-header GMT, then
        to the Unix epoch when neither is available.
    dx, gauge_length_m, units : not stored in SEG-Y; pass through if known.
    """
    file = Path(file)
    header, buf, total_bytes = _read_bytes(file)
    nx, ns, dt, fmt = _resolve_geometry(
        header, total_bytes, n_traces, n_samples, sample_interval, sample_format,
    )

    base = _SAMPLE_BASE[fmt][0]
    if fmt == 1:                                             # IBM float: big-endian words
        ec = ">"
    elif endian in ("big", ">"):
        ec = ">"
    elif endian in ("little", "<"):
        ec = "<"
    elif endian == "auto":
        ec = (_detect_float_endian(_first_trace_sample(file, buf, ns, _SAMPLE_BASE[fmt][1]), base)
              if base.startswith("f") else ">")           # ints: default big-endian
    else:
        raise ValueError(f"endian must be 'auto', 'big'/'>' or 'little'/'<', got {endian!r}")

    samp_dt = ec + base
    trace_dt = np.dtype([("hdr", "V%d" % _TRACE_HEADER_BYTES), ("data", samp_dt, ns)])
    if buf is None:                                          # plain file: read from disk
        arr = np.fromfile(str(file), dtype=trace_dt, count=nx, offset=_FILE_HEADER_BYTES)
    else:                                                    # .gz: already in memory
        arr = np.frombuffer(buf, dtype=trace_dt, count=nx, offset=_FILE_HEADER_BYTES)

    raw = arr["data"]
    data = _ibm2ieee(raw) if fmt == 1 else raw
    data = np.ascontiguousarray(data, dtype=np.float32)
    if conversion_factor != 1.0:
        data *= np.float32(conversion_factor)

    if begin_time is None:
        begin_time = _trace_header_time(bytes(arr["hdr"][0])) if nx else None
    if begin_time is None:
        begin_time = datetime(1970, 1, 1, tzinfo=timezone.utc)
    end_time = begin_time + timedelta(seconds=(ns - 1) * dt) if ns else begin_time

    return DASdata(
        data=data, fs=1.0 / dt, dt=dt, nt=ns, nx=nx, dx=dx,
        begin_time=begin_time, end_time=end_time,
        gauge_length_m=gauge_length_m, system="PASSCAL_SEGY",
        raw_meta={"segy_format_code": fmt, "endian": ec,
                  "conversion_factor": conversion_factor},
        units=normalize_unit(units),
    )


def read_passcal_segy_metadata(
        file: Union[str, Path],
        *,
        n_traces: Optional[int] = None,
        n_samples: Optional[int] = None,
        sample_interval: Optional[float] = None,
        sample_format: Optional[int] = None,
        begin_time: Optional[datetime] = None,
        dx: Optional[float] = None,
        gauge_length_m: Optional[float] = None,
    ) -> DASmeta:
    """Read a PASSCAL SEG-Y file's `DASmeta` (geometry + timing, no payload).

    Same geometry resolution as `read_passcal_segy`; reads only the file and
    first-trace headers. For `.gz` inputs the whole file is decompressed to
    learn its size, so pass `n_samples`/`n_traces` to keep it cheap.
    """
    file = Path(file)
    header, buf, total_bytes = _read_bytes(file)
    nx, ns, dt, _fmt = _resolve_geometry(
        header, total_bytes, n_traces, n_samples, sample_interval, sample_format,
    )
    if begin_time is None:
        if buf is None:
            with open(file, "rb") as fid:
                fid.seek(_FILE_HEADER_BYTES)
                th0 = fid.read(_TRACE_HEADER_BYTES)
        else:
            th0 = buf[_FILE_HEADER_BYTES:_FILE_HEADER_BYTES + _TRACE_HEADER_BYTES]
        begin_time = _trace_header_time(th0)
    if begin_time is None:
        begin_time = datetime(1970, 1, 1, tzinfo=timezone.utc)
    end_time = begin_time + timedelta(seconds=(ns - 1) * dt) if ns else begin_time

    return DASmeta(
        file=str(file), begin_time=begin_time, end_time=end_time,
        fs=1.0 / dt, nt=ns, nx=nx, first_sample=0,
        dx=dx, gauge_length_m=gauge_length_m,
    )
