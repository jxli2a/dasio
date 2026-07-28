"""Universal DAS data container used across all readers and the desample pipeline."""
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Optional, Tuple, TypedDict, Union

import numpy as np


# What `to_physical` turns each unit into. Everything lands on microstrain, so
# a window's unit never depends on which instrument recorded it, and seismic
# amplitudes read as ~1e2 rather than the ~1e-4 strain would give. Units absent
# here are already microstrain, or untagged, and pass through untouched.
_TO_MICROSTRAIN = {
    "count":    "microstrain",       # OptaSense phase counts
    "radian":   "microstrain",
    "radian/s": "microstrain/s",     # AP Sensing
    "strain":   "microstrain",       # ASN: physical already, owes only the 1e6
    "strain/s": "microstrain/s",
}

# The instrument units above — meaningless until a vendor `physical_factor` is
# applied, where strain merely needs scaling. `dasfile.read` attaches the
# factor for exactly these.
_NEEDS_FACTOR = ("count", "radian", "radian/s")

# Everything `normalize_unit` accepts: the convertible units plus the two they
# land on. Longest first, because it takes the first substring match and
# "strain" sits inside "microstrain" — derived rather than hand-ordered so the
# invariant cannot rot as the table grows.
VALID_UNITS = tuple(sorted(
    _TO_MICROSTRAIN.keys() | set(_TO_MICROSTRAIN.values()),
    key=lambda u: (-len(u), u),
))


def normalize_unit(s) -> str:
    """Map a free-form unit string onto `VALID_UNITS`, else 'unknown'.

    Case-insensitive substring match, so a vendor's
    "Strain rate (microstrain/sec)" comes back as "microstrain/s".
    """
    t = str(s or "").strip().lower().replace("/sec", "/s")
    return next((u for u in VALID_UNITS if u in t), "unknown")


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
    # Channel index of row 0 — the channel-axis counterpart to `t0_sec`.
    # A read with `min_ch=2000` returns rows that are really fiber channels
    # 2000.., and without this anchor that offset is lost: plots and picks come
    # out shifted by a constant with nothing to reveal it. `select_channels`
    # leaves it meaningless (arbitrary subset), exactly as it does `dx`.
    ch0:             int = 0
    # Channel stride: 1 normally, `step` after `skip_ch`. Without it
    # `channel_axis` would report consecutive numbers for a decimated view and
    # be quietly wrong by a growing amount.
    dch:             int = 1
    # Physical unit of `data`, from the controlled VALID_UNITS vocabulary
    # ("unknown" = not tagged). Set by the readers; `differentiate`/`integrate`
    # propagate the rate (strain <-> strain/s).
    units:           str = "unknown"
    # The vendor's raw->strain constant (OptaSense count->strain, AP Sensing
    # radian/s->strain/s; 1.0 for ASN, which already stores strain). Attached
    # by `DASFile.read(with_factor=True)`, the default; `to_physical` applies
    # it together with the strain->microstrain 1e6 and resets it to 1.0.
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

    def skip_ch(self, step: int) -> 'DASdata':
        """Keep every `step`-th channel (uniform decimation), updating `dx`.

        A coarse, faster channel view — `d.skip_ch(5)` keeps 1 channel in 5.
        Because the stride is uniform, `dx` scales by `step` so the channel
        axis stays physically correct (unlike `select_channels`, whose
        arbitrary picks leave `dx` meaningless). No spatial anti-alias
        filter is applied, so this is for display/preview, not analysis.
        `step <= 1` returns an unchanged copy. `data` is C-contiguous.
        """
        step = max(1, int(step))
        if step == 1:
            return replace(self)
        new_data = np.ascontiguousarray(self.data[::step])
        new_dx = self.dx * step if self.dx is not None else None
        return replace(self, data=new_data, nx=new_data.shape[0], dx=new_dx)

    def skip_t(self, step: int) -> 'DASdata':
        """Keep every `step`-th time sample (uniform decimation), updating `dt`/`fs`.

        A coarse, faster time view — `d.skip_t(5)` keeps 1 sample in 5.
        `dt` scales by `step` and `fs` divides by it, so the time axis stays
        correct; `begin_time`/`t0_sec` are unchanged (sample 0 is kept) and
        `end_time` snaps to the last kept sample. No temporal anti-alias
        filter is applied, so this is for display/preview, not analysis
        (use a proper resample/decimate for that). `step <= 1` returns an
        unchanged copy. `data` is C-contiguous.
        """
        step = max(1, int(step))
        if step == 1:
            return replace(self)
        new_data = np.ascontiguousarray(self.data[:, ::step])
        new_nt = new_data.shape[1]
        new_dt = self.dt * step
        new_end = (
            self.begin_time + timedelta(seconds=(new_nt - 1) * new_dt)
            if new_nt else self.begin_time
        )
        return replace(self, data=new_data, nt=new_nt, dt=new_dt,
                       fs=self.fs / step, end_time=new_end)

    def to_physical(self) -> "DASdata":
        """Return a copy in microstrain (or microstrain/s), whatever the vendor.

        Applies `physical_factor` and the strain -> microstrain 1e6 in one
        float32 pass, resets the factor to 1.0 and updates `units`. Counts,
        radians and strain all land on the same unit, so downstream code and
        colorbars never have to ask which vendor a window came from.

        A metadata-only copy when the data is already microstrain — sharing
        `data` rather than duplicating it — so calling this unconditionally is
        both safe and idempotent.

        Raises
        ------
        ValueError
            On count / radian / radian/s with no factor attached
            (`physical_factor == 1.0`), rather than silently relabeling raw
            counts as microstrain. Read with `with_factor=True` (the default).
        """
        if self.units in _NEEDS_FACTOR and self.physical_factor == 1.0:
            raise ValueError(
                f"to_physical(): units={self.units!r} require a conversion "
                f"factor, but physical_factor is 1.0. "
                f"Read the file with DASFile.read(with_factor=True) first."
            )
        target = _TO_MICROSTRAIN.get(self.units)
        if target is None:                      # already microstrain, or untagged
            return replace(self)
        # otherwise return a fresh array with the factor applied, and reset the factor to 1.0
        return replace(
            self, data=self.data * (self.physical_factor * 1e6),
            physical_factor=1.0, units=target
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
            nthreads: Optional[int] = None,
        ) -> 'DASdata':
        """Butterworth bandpass along the time axis. See `processing.bandpass`."""
        from .processing import bandpass as _bp
        return _bp(
            self, fmin, fmax,
            order=order, zerophase=zerophase, copy=copy, nthreads=nthreads,
        )

    def detrend(self, copy: bool = True) -> 'DASdata':
        """Per-channel linear detrend along time. See `processing.detrend`."""
        from .processing import detrend as _det
        return _det(self, copy=copy)

    def taper(self, alpha: float = 0.4, copy: bool = True) -> 'DASdata':
        """Tukey edge taper along time. See `processing.taper`."""
        from .processing import taper as _t
        return _t(self, alpha=alpha, copy=copy)

    def differentiate(self, copy: bool = True, method: str = "central") -> 'DASdata':
        """Time-axis derivative. See `processing.differentiate`."""
        from .processing import differentiate as _diff
        return _diff(self, copy=copy, method=method)

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

    def downsample(
            self, factor: int, anti_alias: bool = True,
            order: int = 8, zerophase: bool = True, copy: bool = True,
            nthreads: Optional[int] = None,
        ) -> 'DASdata':
        """Integer-factor time downsample: anti-alias low-pass + stride.

        Analysis-grade decimation — unlike `skip_t` (a bare stride that
        aliases), this low-passes below the new Nyquist first. See
        `processing.downsample`.
        """
        from .processing import downsample as _ds
        return _ds(self, factor, anti_alias=anti_alias, order=order,
                   zerophase=zerophase, copy=copy, nthreads=nthreads)
