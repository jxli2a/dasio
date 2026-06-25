"""Channel-metadata catalog for one snapshot of an instrument configuration.

`DASinfo` wraps a pandas DataFrame indexed by `index_raw` (the row position
in the reader's raw `(nx_raw, nt)` array). Each row has:

    fiber           'n' / 's' / 'c' / future 'e' / 'w'. 'u' marks an unknown
                    fiber (NaN rows in the CSV are mapped to 'u'). Single-
                    fiber rigs default to 'n' when the input CSV lacks a
                    fiber column.
    taptest         1 if a taptest survey produced a geolocation, else 0.
                    Defaults to 1 when the CSV lacks the column.
    quality         1 if the located channel is currently good for downstream
                    processing, 0 if known noisy / muted. Only meaningful
                    when taptest == 1; forced to 0 when taptest == 0.
                    Defaults to 1 when the CSV lacks the column.
    index_taptest   Sequential integer per fiber within taptest==1 rows.
                    NA when taptest == 0. Preserved verbatim if the CSV
                    already supplies the column; otherwise computed.
    lat, lon        WGS84 from the taptest survey. NaN when taptest == 0.

Subset operations (`by_fiber`, `located`, `active`) preserve `index_raw` and
`index_taptest` — they shrink the row count without renumbering.

For cross-instrument-config durability, the join key on located channels is
the composite `(fiber, index_taptest)` — frozen at survey time. `index_raw`
shifts when fiber is added or removed on the instrument side.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

# CSV column renames applied on load. The source column is dropped after
# the rename. `status` -> `taptest` is included; if the renamed column
# arrives with string dtype (i.e. 'good' / 'bad' values), it is converted
# to 1/0 ints inside `_set_taptest_quality`.
#
# `spiky` is NOT a pure rename — it's a polarity-flipped derivation of
# `quality` (spiky=0 means good, quality=1 means good). That conversion
# happens in `_set_taptest_quality` rather than in `_apply_rename_aliases`.
_RENAME_ALIASES = {
    "status": "taptest",
    "index": "index_raw",
    "ichan_before_taptest": "index_raw",
    "ichan_after_taptest": "index_taptest",
    "taptest_channel_index": "index_taptest",
    "longitude": "lon",
    "latitude": "lat",
    "elevation": "ele",
    "azimuth": "azi",
    "dipping angle": "dip",
}


@dataclass(frozen=True)
class DASinfo:
    """Immutable channel catalog. Construct via `from_csv` in production;
    tests construct directly with a pre-normalized DataFrame."""

    df: pd.DataFrame

    # ---- factory --------------------------------------------------------

    @classmethod
    def from_csv(cls, path: Union[str, Path]) -> "DASinfo":
        """Read a CSV in the canonical DASinfo schema.

        Required columns: `lat` (or `latitude`), `lon` (or `longitude`).
        Optional columns (defaulted when absent):

            fiber           defaults to 'n'.
            taptest         defaults to 1.
            quality         defaults to 1. Forced to 0 wherever taptest==0.
            index_raw       defaults to row position (0..N-1).
            index_taptest   computed per-fiber starting at 0 for
                            taptest==1 rows when missing.

        Rename aliases (source column dropped after rename):

            index                   -> index_raw
            ichan_before_taptest    -> index_raw
            ichan_after_taptest     -> index_taptest
            taptest_channel_index   -> index_taptest
            status                  -> taptest (string 'good' / 'bad'
                                    converted to 1 / 0)
            spiky                   -> quality (polarity-flipped:
                                    spiky=0 -> quality=1; spiky=1 -> 0)
            latitude                -> lat
            longitude               -> lon
            elevation               -> ele
            azimuth                 -> azi
            dipping angle           -> dip

        Production LF / EQ deployment CSVs (Iceland + Santorini) load
        directly through these aliases — no CSV mutation needed before
        consumers migrate to `DASinfo.from_csv`.
        """
        df = pd.read_csv(path)
        df = cls._apply_rename_aliases(df)
        df = cls._set_fiber(df)
        df = cls._set_taptest_quality(df)
        df = cls._set_index_taptest(df)
        df = cls._set_index_raw(df)
        df = cls._reorder_columns(df)
        return cls(df=df)

    # ---- subsetting -----------------------------------------------------

    def by_fiber(self, letter: str) -> "DASinfo":
        """Rows where `fiber == letter`. Empty result is allowed."""
        return DASinfo(df=self.df[self.df["fiber"] == letter])

    def located(self) -> "DASinfo":
        """Rows where `taptest == 1` (channel has a geolocation)."""
        return DASinfo(df=self.df[self.df["taptest"] == 1])

    def active(self) -> "DASinfo":
        """Rows where `taptest == 1 AND quality == 1`. The set fed to the
        picker / locator in production."""
        mask = (self.df["taptest"] == 1) & (self.df["quality"] == 1)
        return DASinfo(df=self.df[mask])

    # ---- bridge from index_raw-space to numpy-slicing -------------------

    @property
    def index_raw(self) -> np.ndarray:
        """`index_raw` values for the current rows, as a numpy int64 array.

        Use to slice the reader's `(nx_raw, nt)` output array down to this
        DASinfo's subset:

        >>> d_raw = reader(path)					# shape (nx_raw, nt)
        >>> d = d_raw[dasinfo.active().index_raw, :]
        >>> # row k of d corresponds to dasinfo.active().df.index[k]
        """
        return self.df.index.to_numpy(dtype=np.int64)

    # ---- coord helpers --------------------------------------------------

    def coord_lookup(self) -> pd.DataFrame:
        """`lat`/`lon` columns indexed by `index_raw`. Suitable for joining
        picks (which carry `index_raw`) to coordinates."""
        return self.df[["lat", "lon"]]

    @property
    def projection_origin(self) -> tuple[float, float]:
        """Mean (lat, lon) over located channels. Raises if none are
        located."""
        located = self.df[self.df["taptest"] == 1]
        if len(located) == 0:
            raise ValueError(
                "No located channels (taptest==1); cannot derive " "projection origin."
            )
        return (float(located["lat"].mean()), float(located["lon"].mean()))

    # ---- plotting -------------------------------------------------------

    def plot(
        self,
        *,
        every: int = 500,
        index: str = "taptest",
        ax=None,
        **kwargs,
    ):
        """Scatter-plot every Nth located channel on a (lon, lat) map.

        Parameters:

            every       stride. Default 500.
            index       'taptest' (default) strides `index_taptest` per
                        fiber; selected channels are located by
                        construction. 'raw' strides `index_raw` and skips
                        rows without a geolocation (taptest == 0).
            ax          matplotlib Axes to draw on; created if None.
            **kwargs    forwarded to `ax.scatter`; defaults to red marker
                        with black edge.

        Annotation labels are `<fiber><index_taptest>` for taptest mode
        and the bare `<index_raw>` integer for raw mode. Returns (fig, ax).
        """
        import matplotlib.pyplot as plt

        if index not in ("taptest", "raw"):
            raise ValueError(f"index must be 'taptest' or 'raw'; got {index!r}")

        located = self.df[self.df["taptest"] == 1]
        if index == "taptest":
            tap = located["index_taptest"].astype("int64")
            sel = located[tap % every == 0]
        else:
            sel = located[located.index.to_numpy() % every == 0]

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        # Backbone: one stroked polyline per fiber. Black under-stroke +
        # red over-stroke gives the line a red core with a black edge.
        # Channels are traversed in taptest order so the line is
        # geometrically continuous.
        for _, group in located.groupby("fiber", sort=False):
            ordered = group.sort_values("index_taptest")
            ax.plot(
                ordered["lon"],
                ordered["lat"],
                "-",
                color="black",
                linewidth=2.5,
                zorder=1,
            )
            ax.plot(
                ordered["lon"],
                ordered["lat"],
                "-",
                color="red",
                linewidth=1.5,
                zorder=2,
            )

        # Stride markers on top of the backbone.
        scatter_kwargs = dict(c="red", edgecolors="black", s=40, zorder=4)
        scatter_kwargs.update(kwargs)
        ax.scatter(sel["lon"], sel["lat"], **scatter_kwargs)

        for raw_idx, row in sel.iterrows():
            if index == "taptest":
                label = str(row['index_taptest'])
            else:
                label = str(int(raw_idx))
            ax.annotate(
                label,
                (row["lon"], row["lat"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        return fig, ax

    # ---- conveniences ---------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        fibers = sorted(set(self.df["fiber"]))
        n_loc = int((self.df["taptest"] == 1).sum())
        n_act = int(((self.df["taptest"] == 1) & (self.df["quality"] == 1)).sum())
        return (
            f"DASinfo(n={len(self.df)}, located={n_loc}, "
            f"active={n_act}, fibers={fibers})"
        )

    # ---- internals ------------------------------------------------------

    @staticmethod
    def _apply_rename_aliases(df: pd.DataFrame) -> pd.DataFrame:
        # Build the rename map in `_RENAME_ALIASES` insertion order.
        # Two precedence rules guard against duplicate-target collisions:
        #   1. The canonical target already exists in the CSV ->
        #      skip the rename; the source column is dropped as a stray.
        #   2. Another alias earlier in the dict already maps to this
        #      target (e.g. both `ichan_after_taptest` and
        #      `taptest_channel_index` mapping to `index_taptest`) ->
        #      first one wins; later ones become strays. Without this
        #      check `pd.rename` would silently produce two columns of
        #      the same name and `df['index_taptest']` would return a
        #      DataFrame instead of a Series.
        rename: dict[str, str] = {}
        targets_taken: set[str] = set(df.columns)
        for src, dst in _RENAME_ALIASES.items():
            if src not in df.columns:
                continue
            if dst in targets_taken:
                continue
            rename[src] = dst
            targets_taken.add(dst)
        if rename:
            df = df.rename(columns=rename)
        # Drop any stray source column whose rename was skipped (target
        # already existed, or an earlier alias claimed the same target).
        stray = [src for src in _RENAME_ALIASES if src in df.columns]
        return df.drop(columns=stray) if stray else df

    @staticmethod
    def _set_fiber(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "fiber" not in df.columns:
            df["fiber"] = "n"
        else:
            df["fiber"] = df["fiber"].fillna("u").astype(str)
        return df

    @staticmethod
    def _set_taptest_quality(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "taptest" in df.columns:
            # When renamed from `status`, the column arrives as strings
            # ('good' / 'bad' / ...). Convert to 1 / 0. Match BOTH
            # numpy object dtype AND pandas/pyarrow string dtypes —
            # pandas 2.x with the PyArrow backend stores post-read_csv
            # strings as `string[pyarrow]`, NOT `object`, so the
            # narrow `dtype == object` check silently skipped the
            # conversion and the later `.astype("int8")` raised
            # `invalid literal for int() with base 10: 'good'` on
            # the Santorini deployment.
            if (df["taptest"].dtype == object
                    or pd.api.types.is_string_dtype(df["taptest"])):
                df["taptest"] = (df["taptest"] == "good").astype("int8")
        else:
            df["taptest"] = np.int8(1)
        df["taptest"] = df["taptest"].astype("int8")

        if "quality" not in df.columns:
            if "spiky" in df.columns:
                # Polarity flip: spiky=0 (good) -> quality=1; spiky=1
                # (bad) -> quality=0. Missing / non-numeric values
                # default to "bad" (quality=0).
                spiky = pd.to_numeric(
                    df["spiky"], errors="coerce",
                ).fillna(1).astype("int8")
                df["quality"] = (spiky == 0).astype("int8")
            else:
                df["quality"] = np.int8(1)
        df["quality"] = df["quality"].astype("int8")
        # Drop the now-redundant source column (post-conversion).
        if "spiky" in df.columns:
            df = df.drop(columns=["spiky"])

        # Quality only meaningful when taptest==1.
        df.loc[df["taptest"] == 0, "quality"] = np.int8(0)
        return df

    @staticmethod
    def _set_index_taptest(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "index_taptest" in df.columns:
            df["index_taptest"] = pd.to_numeric(
                df["index_taptest"],
                errors="coerce",
            ).astype("Int64")
            df.loc[df["taptest"] == 0, "index_taptest"] = pd.NA
            return df
        # Compute per-fiber, starting at 0, only for taptest==1 rows.
        df["index_taptest"] = pd.array([pd.NA] * len(df), dtype="Int64")
        located_mask = df["taptest"] == 1
        for _, group_idx in (
            df[located_mask]
            .groupby(
                "fiber",
                sort=False,
            )
            .groups.items()
        ):
            ordered = list(group_idx)
            df.loc[ordered, "index_taptest"] = pd.array(
                np.arange(len(ordered), dtype="int64"),
                dtype="Int64",
            )
        return df

    @staticmethod
    def _set_index_raw(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "index_raw" not in df.columns:
            df = df.reset_index(drop=True)
            df["index_raw"] = df.index.astype("int64")
        df["index_raw"] = pd.to_numeric(
            df["index_raw"],
            errors="coerce",
        ).astype("int64")
        return df.set_index("index_raw", drop=True)

    # Column ordering for the canonical DataFrame: metadata columns first,
    # any deployment-specific extras in the middle, geometry columns last.
    _LEADING_COLUMNS = ("fiber", "taptest", "quality", "index_taptest")
    _TRAILING_COLUMNS = ("lat", "lon", "ele", "azi", "dip")

    @classmethod
    def _reorder_columns(cls, df: pd.DataFrame) -> pd.DataFrame:
        leading = [c for c in cls._LEADING_COLUMNS if c in df.columns]
        trailing = [c for c in cls._TRAILING_COLUMNS if c in df.columns]
        fixed = set(leading) | set(trailing)
        middle = [c for c in df.columns if c not in fixed]
        return df[leading + middle + trailing]
