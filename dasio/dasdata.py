"""Universal DAS data container used across all readers and the desample pipeline."""
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Optional, Tuple, TypedDict, Union

import numpy as np


VALID_UNITS = frozenset({
    "count", "radian", "radian/s", "strain", "strain/s",
    "microstrain", "microstrain/s",
})

# units after applying physical_factor (count->strain, radian/s->strain/s, ...).
_PHYSICAL_UNIT = {
    "count": "strain", "radian": "strain", "radian/s": "strain/s",
}

# Substring -> canonical, longest patterns first so "microstrain/s" wins over "strain/s".
_UNIT_PATTERNS = (
    ("microstrain/s", "microstrain/s"),
    ("radian/s", "radian/s"),
    ("strain/s", "strain/s"),
    ("microstrain", "microstrain"),
    ("radian", "radian"),
    ("strain", "strain"),
    ("count", "count"),
)


def normalize_unit(s) -> str:
    """Map a free-form unit string onto the controlled vocabulary.

    Returns 'unknown' for anything unrecognized. Case-insensitive;
    matches the first (longest) known substring.
    """
    if not s:
        return "unknown"
    t = str(s).strip().lower().replace("/sec", "/s")
    for pat, canon in _UNIT_PATTERNS:
        if pat in t:
            return canon
    return "unknown"


class DASmeta(TypedDict):
    """One row of per-file catalog metadata (what DASdb holds).

    Dict-shaped so vendor scanners can emit it directly and pandas
    turns a list of them straight into a DataFrame. For ASN / Proc
    each file yields exactly one DASmeta; OptaSense yields one per
    contiguous RawDataTime chunk (same path, different first_sample
    and nt).

    begin_time / end_time are last-sample-inclusive, matching the
    legacy DAS_db contract.
    """
    file:           str
    begin_time:     datetime
    end_time:       datetime
    fs:             float
    nt:             int
    nx:             int
    first_sample:   int
    dx:             Optional[float]
    gauge_length_m: Optional[float]


@dataclass
class DASdata:
    data:            np.ndarray
    fs:              float
    dt:              float
    nt:              int
    nx:              int
    dx:              float
    begin_time:      datetime
    end_time:        datetime
    gauge_length_m:  Optional[float] = None
    system:          str = 'unknown'
    raw_meta:        Optional[dict] = None
    # `t0_sec` is the seconds-axis value at sample 0. Default 0 so a
    # bare `time_axis()` reads "0, dt, 2·dt, …". Event-data readers
    # set it negative (e.g. -30.00) so sample 0 lands at t = −30.00 s
    # and the event origin lands at t = 0 — making
    # `truncate(t_range=(-2, 10))` mean "2 s before to 10 s after the
    # event." `begin_time` stays the absolute anchor, `t0_sec` is the
    # seconds-frame anchor; the two together pin both views.
    t0_sec:          float = 0.0
    # Physical unit of `data`, from the controlled VALID_UNITS vocabulary
    # ("unknown" = not tagged). Set by the readers; NOT auto-updated by
    # processing (integrate/differentiate change the quantity but not this tag).
    units:           str = "unknown"
    # Multiply `data` by this to reach physical units: strain (OptaSense
    # count->strain) or strain/s (AP Sensing radian/s->strain/s; ASN is
    # already strain/s so the factor is 1.0). Populated by
    # DASFile.read(with_factor=True); 1.0 otherwise.
    physical_factor: float = 1.0

    # ---- Read-only accessors ----------------------------------------------

    @property
    def shape(self):
        return (self.nx, self.nt)

    @property
    def info(self) -> dict:
        """Snapshot of scalar metadata as a plain dict (no `data`/`raw_meta`).

        Useful for logging, JSON serialization, and stuffing into
        downstream metadata records — anything that needs the
        identifying fields of a DASdata without dragging the array.
        """
        keys = (
            'fs', 'dt', 'nt', 'nx', 'dx', 'begin_time', 'end_time',
            't0_sec', 'gauge_length_m', 'system', 'units',
        )
        return {k: getattr(self, k) for k in keys}

    @property
    def time_axis(self) -> np.ndarray:
        """Time axis in seconds (`t0_sec` at sample 0; advances by `dt`)."""
        return self.t0_sec + np.arange(self.nt) * self.dt

    @property
    def datetime_axis(self) -> np.ndarray:
        """Time axis as numpy `datetime64[ns]` (absolute, naive UTC).

        `numpy.datetime64` is timezone-naive; we strip `begin_time`'s
        tzinfo before conversion to avoid the noisy "no explicit
        representation of timezones" warning. All DASdata timestamps
        are UTC by convention so the strip is a label change only.
        """
        step = np.timedelta64(int(round(self.dt * 1e9)), 'ns')
        anchor = np.datetime64(self.begin_time.replace(tzinfo=None))
        return anchor + np.arange(self.nt) * step

    @property
    def plot(self):
        """Plot accessor: `d.plot()` (≡ `d.plot.imshow()`),
        `d.plot.imshow(...)`, `d.plot.wiggle(...)`.

        Implementation lives in `dasio.plot`; lazy-imported so a
        bare `import dasdata` doesn't pull in matplotlib.
        """
        from .plot import _PlotAccessor
        return _PlotAccessor(self)

    # ---- Window selection -------------------------------------------------

    def truncate(
            self,
            ch_range: Optional[Tuple[int, int]] = None,
            t_range:  Optional[Tuple[Union[datetime, float], Union[datetime, float]]] = None,
        ) -> 'DASdata':
        """Slice a contiguous channel range and / or time window, returning a fresh DASdata.

        For an arbitrary (non-contiguous) set of channels, use
        `select_channels`.

        Parameters
        ----------
        ch_range : (min_ch, max_ch), optional
            Contiguous channel-index range, `max_ch` exclusive.
            Out-of-bounds values are clipped to `[0, self.nx]`. `None`
            keeps all.
        t_range : (begin, end), optional
            Time range. The two ends must be the same type, either:
            `datetime` — absolute timestamps, clipped to overlap with
            `[self.begin_time, self.end_time]`. `int` or `float` —
            seconds in the DASdata's own frame, where `self.t0_sec`
            is the value at sample 0; `t_range=(-2, 10)` on event
            data with `t0_sec=-30` selects 2 s before to 10 s after
            the event. `None` keeps the full window.

        Returns a new DASdata with `data`, `nx`, `nt`, `begin_time`,
        `end_time`, and `t0_sec` updated. `dt`, `fs`, `dx` are
        unchanged (no decimation). The `data` array is C-contiguous
        (a copy when the slice was strided, a view otherwise) so
        downstream `bandpass()` etc. don't hit the silent-stride bug.
        """
        # Channel range
        if ch_range is None:
            c0, c1 = 0, self.nx
        else:
            c0, c1 = ch_range
            c0 = max(0, int(c0))
            c1 = min(self.nx, int(c1))

        # Time range → sample-index bounds, in self's seconds frame
        if t_range is None:
            t0_idx, t1_idx = 0, self.nt
        else:
            t0, t1 = t_range
            if isinstance(t0, datetime):
                t0_sec_in = (t0 - self.begin_time).total_seconds() + self.t0_sec
                t1_sec_in = (t1 - self.begin_time).total_seconds() + self.t0_sec
            else:
                t0_sec_in, t1_sec_in = float(t0), float(t1)
            t0_idx = max(0, int(round((t0_sec_in - self.t0_sec) / self.dt)))
            t1_idx = min(self.nt, int(round((t1_sec_in - self.t0_sec) / self.dt)))

        new_data = np.ascontiguousarray(self.data[c0:c1, t0_idx:t1_idx])
        new_nt = new_data.shape[1]
        new_nx = new_data.shape[0]
        new_begin = self.begin_time + timedelta(seconds=t0_idx * self.dt)
        new_end = (
            new_begin + timedelta(seconds=(new_nt - 1) * self.dt)
            if new_nt else new_begin
        )
        new_t0 = self.t0_sec + t0_idx * self.dt
        return replace(
            self, data=new_data, nx=new_nx, nt=new_nt,
            begin_time=new_begin, end_time=new_end, t0_sec=new_t0,
        )

    def select_channels(self, channels) -> 'DASdata':
        """Select an arbitrary set of channels, returning a fresh DASdata.

        Parameters
        ----------
        channels : array-like
            Integer index array or a length-`nx` boolean mask (e.g. a
            list of good channels). Channels are returned in the given
            order. For a contiguous channel range or a time window,
            use `truncate`.

        Only `data` and `nx` change; the time axis is untouched. Note
        that an arbitrary selection generally leaves the channel axis
        non-uniformly spaced, so `dx` no longer describes the true
        inter-channel spacing. The returned `data` is C-contiguous.
        """
        new_data = np.ascontiguousarray(self.data[np.asarray(channels)])
        return replace(self, data=new_data, nx=new_data.shape[0])

    def to_physical(self) -> "DASdata":
        """Return a copy with `physical_factor` applied to `data`.

        `data` is multiplied by `physical_factor`, the factor is reset to
        1.0, and `units` is advanced to its physical counterpart
        (count->strain, radian/s->strain/s). When `physical_factor` is
        already 1.0 and `units` is already physical (e.g. strain/s,
        microstrain), returns an unchanged copy. When `physical_factor` is
        1.0 but `units` still requires conversion (count/radian/radian/s),
        raises `ValueError` — call `DASFile.read(with_factor=True)` first
        so that the conversion factor is attached.

        Raises
        ------
        ValueError
            If called on non-physical units (count/radian/radian/s) without
            an attached conversion factor (i.e. `physical_factor == 1.0`).
        """
        if self.physical_factor == 1.0:
            if self.units in _PHYSICAL_UNIT:
                raise ValueError(
                    f"to_physical(): units={self.units!r} require a conversion "
                    f"factor, but physical_factor is 1.0. "
                    f"Read the file with DASFile.read(with_factor=True) first."
                )
            # Units already physical (strain/s, microstrain, etc.) — genuine no-op.
            return replace(self)
        new_data = (self.data * np.float32(self.physical_factor)).astype(np.float32)
        return replace(
            self, data=new_data, physical_factor=1.0,
            units=_PHYSICAL_UNIT.get(self.units, self.units),
        )

    # ---- OOP-style processing entry points ---------------------------------
    # Thin shims over `dasio.processing.*`; the functional
    # form remains the source of truth (and the canonical test target).
    # Lazy import here avoids dragging numba + the cpp filter extension
    # into every module that just wants to construct a DASdata — a fresh
    # `import dasdata` stays free until a processing method is called.

    def bandpass(
            self,
            fmin: float, fmax: float,
            order: int = 14, zerophase: bool = True, copy: bool = True,
        ) -> 'DASdata':
        """Butterworth bandpass along the time axis. See `processing.bandpass`."""
        from .processing import bandpass as _bp
        return _bp(
            self, fmin, fmax,
            order=order, zerophase=zerophase, copy=copy,
        )

    def detrend(self, copy: bool = True) -> 'DASdata':
        """Per-channel linear detrend along time. See `processing.detrend`."""
        from .processing import detrend as _det
        return _det(self, copy=copy)

    def taper(self, alpha: float = 0.4, copy: bool = True) -> 'DASdata':
        """Tukey edge taper along time. See `processing.taper`."""
        from .processing import taper as _t
        return _t(self, alpha=alpha, copy=copy)

    def differentiate(self, copy: bool = True) -> 'DASdata':
        """Time-axis derivative. See `processing.differentiate`."""
        from .processing import differentiate as _diff
        return _diff(self, copy=copy)

    def integrate(self, copy: bool = True) -> 'DASdata':
        """Time-axis cumulative integral. See `processing.integrate`."""
        from .processing import integrate as _int
        return _int(self, copy=copy)

    def subtract_common_mode(
            self, ch_min: int = 0, ch_max: Optional[int] = None,
            copy: bool = True,
        ) -> 'DASdata':
        """Common-mode noise rejection. See `processing.subtract_common_mode`."""
        from .processing import subtract_common_mode as _scm
        return _scm(self, ch_min=ch_min, ch_max=ch_max, copy=copy)
    
    def unwrap(self, factor: int = 1, copy: bool = True) -> 'DASdata':
        """OptaSense int32 phase-wrap correction. See `processing.unwrap`."""
        from .processing import unwrap as _uw
        return _uw(self, factor=factor, copy=copy)
