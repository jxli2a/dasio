"""`DASFile`: per-file I/O facade.

Wraps one `.h5` / `.hdf5` path. `system` (on-disk data kind) and
`origin` (original capture vendor) are detected lazily — only on
attribute access — so construction is free. This matters for
directory-scan hot paths (`DASdb.scan_metadata`) that build
thousands of `DASFile`s in a loop: eager detection would cost one
extra h5py open per file on top of the `.metadata()` read itself.

Thin glue over the vendor-specific readers + the OptaSense count-to-
strain factor. This is the canonical entry point for working with a
single DAS file. The free-function dispatchers in
``dasio/__init__.py`` (``read_das_data``, ``read_das_metadata``,
``factor_raw2strain``) are thin wrappers around it — kept for
one-shot call sites.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import List, Optional, Union

import h5py

from .dasdata import DASdata, DASmeta
from .readers.apsensing import (
    apsensing_radians2strain_factor,
    read_apsensing_metadata,
    read_apsensing_raw,
)
from .readers.asn import read_asn_metadata, read_asn_raw
from .readers.event import read_event, read_event_metadata
from .readers.optasense import (
    optasense_count2strain_factor,
    read_optasense_metadata,
    read_optasense_raw,
)
from .readers.detector import detect_data_kind, detect_origin
from .readers.proc import read_data_proc, read_metadata_proc


# Reader dispatch is keyed by the on-disk format (`data_kind`), not the
# origin vendor: a Proc file with ASN origin still needs `read_data_proc`
# to parse its `/Data` group, not `read_asn_raw` which expects raw
# ASN HDF5 paths.
_DATA_READERS = {
    'ASN':       read_asn_raw,
    'OptaSense': read_optasense_raw,
    'APSensing': read_apsensing_raw,
    'Proc':      read_data_proc,
    'Event':     read_event,
}

_METADATA_READERS = {
    'ASN':       read_asn_metadata,
    'OptaSense': read_optasense_metadata,
    'APSensing': read_apsensing_metadata,
    'Proc':      read_metadata_proc,
    'Event':     read_event_metadata,
}

# OptaSense and APSensing have non-trivial raw→strain factors. Keyed
# by the origin vendor, so a Proc file with OptaSense / APSensing
# origin still gets the right conversion from the flattened
# /Acquisition_origin attrs.
_FACTOR_FNS = {
    'OptaSense': optasense_count2strain_factor,
    'APSensing': apsensing_radians2strain_factor,
}


class DASFile:
    """
    One DAS HDF5 file with lazy format + origin detection.

    `system` is the on-disk data kind (`'ASN'`, `'OptaSense'`,
    `'Proc'`, …) and decides which reader dispatches. `origin` is
    the vendor that originally captured the data — equal to `system`
    for raw files, recovered from `/Acquisition_origin` attrs for
    Proc files, and used only to pick the right raw→strain factor.

    Both attributes are detected on first access and cached. Passing
    them up-front skips the detection open entirely — `.metadata()`
    needs only `system`, `.factor()` needs only `origin`.

    Parameters
    ----------
    path :
        Filesystem path to the .h5 / .hdf5 file.
    system :
        Pre-known data kind. If omitted, detected via
        ``detect_data_kind`` on first access.
    origin :
        Pre-known origin vendor. If omitted, detected via
        ``detect_origin`` on first access.
    """

    def __init__(
            self, path: Union[str, Path],
            system: Optional[str] = None,
            origin: Optional[str] = None,
        ):
        self.path = Path(path)
        self._system = system
        self._origin = origin

    @property
    def system(self) -> str:
        if self._system is None:
            with h5py.File(self.path, 'r') as f:
                self._system = detect_data_kind(f)
        return self._system

    @property
    def origin(self) -> str:
        if self._origin is None:
            with h5py.File(self.path, 'r') as f:
                self._origin = detect_origin(f)
        return self._origin

    def __repr__(self) -> str:
        return (f'DASFile({self.path!s}, system={self._system!r}, '
                f'origin={self._origin!r})')

    # ---- read / metadata / factor -------------------------------------

    def read(self, *, with_factor: bool = False, **kwargs) -> DASdata:
        """Load the payload as a `DASdata`.

        Keyword arguments pass through to the vendor reader
        (``min_ch``, ``max_ch``, ``first_sample``, ``n_samples``,
        ``convert``); the Proc reader ignores the first_sample /
        n_samples pair.
        When with_factor=True, attaches DASFile.factor() as DASdata.physical_factor
        (one extra file open; default False keeps the desample/dasdb read path unchanged).
        """
        try:
            reader = _DATA_READERS[self.system]
        except KeyError:
            raise ValueError(
                f'No data reader for system {self.system!r}; '
                f'known: {sorted(_DATA_READERS)}'
            )
        d = reader(self.path, **kwargs)
        if with_factor:
            d = replace(d, physical_factor=self.factor())
        return d

    def metadata(self) -> Union[DASmeta, List[DASmeta]]:
        """Return a `DASmeta` dict (ASN / Proc) or list of them
        (OptaSense when the file holds multiple RawDataTime chunks)."""
        try:
            reader = _METADATA_READERS[self.system]
        except KeyError:
            raise ValueError(
                f'No metadata reader for system {self.system!r}; '
                f'known: {sorted(_METADATA_READERS)}'
            )
        return reader(self.path)

    def factor(self) -> float:
        """Scalar factor to convert the raw payload to strain.

        Keyed by `origin`, so a Proc file whose raw capture came from
        OptaSense still gets the count→strain conversion. Returns 1.0
        for vendors whose payload already is strain / strain-rate.
        """
        fn = _FACTOR_FNS.get(self.origin)
        if fn is None:
            return 1.0
        with h5py.File(self.path, 'r') as f:
            return fn(f)
