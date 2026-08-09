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
    # `format` is the on-disk format that picked the reader; `origin` the
    # interrogator the samples came off (/Acquisition_origin for Proc), and the
    # only one of the two that survives desampling. A Proc capture from a
    # QuantX reads format='Proc', origin='OptaSense'. Both mirror `DASFile`.
    format:          str = 'unknown'
    origin:          str = 'unknown'
    raw_meta:        Optional[dict] = None
    # Seconds-axis value at sample 0; `begin_time` is the absolute anchor and
    # this the seconds-frame one. Event readers set it negative (e.g. -30) so
    # the event origin lands at t = 0, making `truncate(t_range=(-2, 10))` mean
    # 2 s before to 10 s after the event.
    t0_sec:          float = 0.0
    # Physical unit of `data`, from VALID_UNITS ("unknown" = not tagged). Set by
    # the readers; `differentiate`/`integrate` propagate the rate.
    units:           str = "unknown"
    # The vendor's raw->strain constant (OptaSense count->strain, AP Sensing
    # radian/s->strain/s; 1.0 for ASN, already strain). Attached by
    # `DASFile.read(with_factor=True)`; `to_physical` applies it with the 1e6.
    physical_factor: float = 1.0
    # Channel number of every row, `{name: array}`. 'raw' is `index_raw` and is
    # always present; `select_taptest` adds 'taptest'. `ch0`/`dch` derive from it.
    channels:        Optional[dict] = None
    channel_axis_name: str = 'raw'   # which label `channel_axis` reports

    def __post_init__(self):
        # So no caller ever has to ask whether the axis was filled in.
        if self.channels is None:
            self.channels = {'raw': np.arange(self.nx)}

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
            't0_sec', 'gauge_length_m', 'format', 'origin', 'units',
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
    def channel_axis(self) -> np.ndarray:
        """Channel number of every row, in the axis `channel_axis_name` selects.
        This is what `select`'s `ch_range` and `ch_index` match against.
        """
        return self.channels[self.channel_axis_name]

    def set_channel_axis(self, name: str) -> 'DASdata':
        """Switch `channel_axis` to the `name` labels. Mutates, returns self.
        Args:
            name: 'raw' (the reader's index, always present) or a label
                `select_taptest` attached, i.e. 'taptest'.
        Returns:
            self.
        Raises:
            ValueError: if `name` was never attached.
        Unlike `truncate` or `skip_ch` this changes no samples, so it mutates
        rather than returning a copy — `d.set_channel_axis('taptest')` as a bare
        statement has to work.
        """
        if name not in self.channels:
            raise ValueError(
                f'no {name!r} channel labels on this DASdata '
                f'(have {sorted(self.channels)}). '
                f'`select_taptest(dasinfo)` is what attaches them.'
            )
        self.channel_axis_name = name
        return self

    @property
    def ch0(self) -> int:
        """Optical channel of row 0, read off `channels['raw']`."""
        axis = self.channels['raw']
        return int(axis[0]) if axis.size else 0

    @property
    def dch(self) -> int:
        """Mean raw channel step; exact while the rows are a ramp.

        Kept because a texture and an `imshow` extent can only be placed on a
        uniform grid, so they need one spacing rather than a list.
        """
        axis = self.channels['raw']
        if axis.size < 2:
            return 1
        return max(1, int(round(float(np.diff(axis).mean()))))

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

    def select(self, ch_range=None, ch_index=None, t_range=None) -> 'DASdata':
        """Cut a time window and / or pick channels, in one call.
        Args:
            ch_range: `(min_ch, max_ch)`, max exclusive, in the channel numbers
                `channel_axis` reports. Ends outside the array are clipped.
            ch_index: explicit channel numbers, in the order wanted, or a
                length-`nx` boolean mask (which selects by row position). A
                number the array does not hold raises.
            t_range: `(begin, end)`, both `datetime`, or both seconds in this
                DASdata's own frame where `t0_sec` is the value at sample 0.
        Returns:
            A fresh DASdata. `dt`, `fs` and `dx` are unchanged — this selects,
            it does not decimate — and `data` is C-contiguous.
        Raises:
            ValueError: if the two channel arguments together select nothing.

        The kept channels are the intersection: `ch_range` filters, `ch_index`
        requests, and either may be `None` for no constraint. A channel named
        by `ch_index` but excluded by `ch_range` is dropped rather than raising
        — that is what the pair is for.
        """
        axis = self.channel_axis
        rows = np.arange(self.nx)
        if ch_index is not None:
            req = np.asarray(ch_index)
            if req.dtype == bool:
                if req.shape != (self.nx,):
                    raise ValueError(
                        f"a boolean mask needs one entry per channel "
                        f"({self.nx}), got {req.size}"
                    )
                rows = np.flatnonzero(req)
            else:
                # `sorter=`, since an earlier pick may have left the axis
                # unsorted — searchsorted would then return wrong rows.
                req = req.astype(np.int64, copy=False).ravel()
                order = np.argsort(axis, kind="stable")
                hit = np.searchsorted(axis, req, sorter=order)
                rows = order[np.clip(hit, 0, self.nx - 1)]
                miss = axis[rows] != req
                if miss.any():
                    bad = np.unique(req[miss]).tolist()
                    raise ValueError(
                        f"{len(bad)} channel(s) not in this DASdata: {bad[:8]}"
                        f"{'...' if len(bad) > 8 else ''} "
                        f"(channel_axis spans {axis.min()}..{axis.max()})"
                    )
        if ch_range is not None:
            lo, hi = ch_range
            rows = rows[(axis[rows] >= lo) & (axis[rows] < hi)]
        if self.nx and rows.size == 0:
            raise ValueError(
                f'no channel selected (ch_range={ch_range}); {self.channel_axis_name} '
                f'channel_axis spans {axis.min()}..{axis.max()}'
            )
        # Time range -> sample-index bounds, in self's seconds frame
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

        new_data = np.ascontiguousarray(self.data[rows][:, t0_idx:t1_idx])
        new_nt = new_data.shape[1]
        new_begin = self.begin_time + timedelta(seconds=t0_idx * self.dt)
        new_end = (
            new_begin + timedelta(seconds=(new_nt - 1) * self.dt)
            if new_nt else new_begin
        )
        return replace(
            self, data=new_data, nx=new_data.shape[0], nt=new_nt,
            begin_time=new_begin, end_time=new_end,
            t0_sec=self.t0_sec + t0_idx * self.dt,
            channels={k: v[rows] for k, v in self.channels.items()},
        )

    def truncate(self, ch_range=None, t_range=None) -> 'DASdata':
        """A contiguous window — `select(ch_range=..., t_range=...)`."""
        return self.select(ch_range=ch_range, t_range=t_range)

    def select_taptest(self, dasinfo) -> 'DASdata':
        """Keep the surveyed channels a `DASinfo` lists, and report their index.

        Args:
            dasinfo: the catalog. Pass `dasinfo.active()` to drop bad-quality
                channels too; `located()` on that subset is a no-op.
        Returns:
            A fresh DASdata carrying both 'raw' and 'taptest' labels, with
            `channel_axis` already switched to 'taptest'.

        Intersected, not demanded: a catalog covers the whole deployment and
        this window holds a part of it.
        """
        info = dasinfo.located()
        raw = self.channels['raw']
        want = np.asarray(info.index_raw, dtype=np.int64)
        rows = np.flatnonzero(np.isin(raw, want))
        # `index_taptest` is in the catalog's row order; reindex onto ours
        tap = info.df['index_taptest'].to_numpy()
        at = {int(r): i for i, r in enumerate(want)}
        labels = {k: v[rows] for k, v in self.channels.items()}
        labels['taptest'] = np.array([tap[at[int(c)]] for c in raw[rows]])
        out = replace(
            self, data=np.ascontiguousarray(self.data[rows]),
            nx=rows.size, channels=labels
        )
        return out.set_channel_axis('taptest')

    def skip_ch(self, step: int) -> 'DASdata':
        """Keep every `step`-th channel (uniform decimation), updating `dx`.

        A coarse, faster channel view — `d.skip_ch(5)` keeps 1 channel in 5.
        Because the stride is uniform, `dx` scales by `step` so the channel
        axis stays physically correct (unlike an arbitrary `select`, which
        leaves `dx` meaningless). No spatial anti-alias
        filter is applied, so this is for display/preview, not analysis.
        `step <= 1` returns an unchanged copy. `data` is C-contiguous.
        """
        step = max(1, int(step))
        if step == 1:
            return replace(self)
        new_data = np.ascontiguousarray(self.data[::step])
        new_dx = self.dx * step if self.dx is not None else None
        return replace(
            self, data=new_data, nx=new_data.shape[0], dx=new_dx,
            channels={k: v[::step] for k, v in self.channels.items()},
        )

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
        return replace(
            self, data=new_data, nt=new_nt, dt=new_dt,
            fs=self.fs / step, end_time=new_end
        )

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
    
    def unwrap_int32(self, factor: int = 1, copy: bool = True) -> 'DASdata':
        """OptaSense int32 phase-wrap correction. See `processing.unwrap`."""
        from .processing import unwrap_int32 as _uw
        return _uw(self, factor=factor, copy=copy)

    def median_filter_1d(
            self, kernel_size: int, axis: str = 't', copy: bool = True
        ) -> 'DASdata':
        """Running median along time or channels. See `processing.median_filter`."""
        from .processing import median_filter_1d as _mf
        return _mf(self, kernel_size, axis=axis, copy=copy)

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
        return _ds(
            self, factor, anti_alias=anti_alias, order=order,
            zerophase=zerophase, copy=copy, nthreads=nthreads
        )
