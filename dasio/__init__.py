"""Public API for the dasio subpackage.

`DASFile` is the canonical per-file entry point — it wraps a path,
detects the vendor system once, and exposes `read` / `metadata` /
`factor` methods. The free-function dispatchers below
(`read_das_data`, `read_das_metadata`, `factor_raw2strain`) are thin
wrappers around it for one-shot call sites that don't want to hold a
`DASFile` instance.

Symbols are exposed through a module-level ``__getattr__`` lazy
table when they (a) need to avoid the runpy "found in sys.modules
before execution" warning when run with ``python -m
dasio.<mod>`` (DASdb, desample_*) or (b) drag in
heavy deps that streaming / viewer code shouldn't pay for at module
load — `processing` / `signal` import numba (~300 ms) and the
pybind11 filter extension. After the lazy entry, ``import
dasio`` is ~50 ms instead of ~340 ms; ``from … import
DASdata`` works without numba in scope; first call to ``bandpass``
(or ``from … import bandpass``) is when numba loads.
"""
__version__ = "0.1.0"

__all__ = [
    "DASdata", "DASmeta", "DASFile", "DASinfo",
    "read_das_data", "read_das_metadata", "factor_raw2strain",
    "read_data_proc", "read_metadata_proc", "write_data_proc",
    "read_asn_raw", "read_asn_metadata",
    "read_optasense_raw", "read_optasense_metadata", "optasense_count2strain_factor",
    "read_apsensing_raw", "read_apsensing_metadata", "apsensing_radians2strain_factor",
    "read_event", "read_event_metadata",
    "read_passcal_segy", "read_passcal_segy_metadata",
    "detect_data_kind", "detect_origin",
    "RawWindow",
]

from pathlib import Path
from typing import Optional, Union

import h5py

from .dasdata import DASdata, DASmeta
from .dasfile import DASFile
from .dasinfo import DASinfo
from .readers.proc import read_data_proc, read_metadata_proc, write_data_proc
from .readers.apsensing import (
    apsensing_radians2strain_factor,
    read_apsensing_metadata,
    read_apsensing_raw,
)
from .readers.asn import read_asn_raw, read_asn_metadata
from .readers.event import read_event, read_event_metadata
from .readers.optasense import (
    optasense_count2strain_factor,
    read_optasense_metadata,
    read_optasense_raw,
)
from .readers.detector import detect_data_kind, detect_origin
from .readers.passcal_segy import read_passcal_segy, read_passcal_segy_metadata
from .schema import RawWindow


def read_das_data(
        file: Union[str, Path], system: str, **kwargs,
    ) -> DASdata:
    """Read one DAS file using the system-appropriate reader.

    Thin wrapper over ``DASFile(file, system).read(**kwargs)``. Kept
    for one-shot callers that don't need to retain the ``DASFile``
    instance.
    """
    return DASFile(file, system=system).read(**kwargs)


def read_das_metadata(file: Union[str, Path], system: str):
    """Read one DAS file's metadata as a `DASmeta` dict (ASN / Proc) or
    list of `DASmeta` (OptaSense when the file splits on RawDataTime).

    Thin wrapper over ``DASFile(file, system).metadata()``.
    """
    return DASFile(file, system=system).metadata()


def factor_raw2strain(
        source: Union[str, Path, h5py.File],
        origin: Optional[str] = None,
    ) -> float:
    """Scalar factor to convert raw payload values to strain.

    Mirrors legacy ``DASutils.parse_factor_raw2strain``: dispatches on
    the ORIGIN vendor, not on the on-disk format — a Proc file
    captured by OptaSense still needs the count→strain factor.
    Auto-detects via ``detect_origin`` when omitted.

    Only OptaSense needs a non-trivial factor; everything else stores
    strain or strain-rate directly and gets 1.0.

    Accepts a file path or an already-open ``h5py.File`` (the latter
    is why the parameter is named `source` rather than `file`).
    """
    if isinstance(source, h5py.File):
        if origin is None:
            origin = detect_origin(source)
        if origin == 'OptaSense':
            return optasense_count2strain_factor(source)
        return 1.0
    return DASFile(source, origin=origin).factor()


_LAZY = {
    'desample_window':           ('desample',   'desample_window'),
    'desample_and_write_window': ('desample',   'desample_and_write_window'),
    'DASdb':                     ('dasdb',      'DASdb'),
    # processing + signal land behind a lazy fence so numba and the
    # pybind11 filter extension don't load on a bare `import dasio`.
    'bandpass':                  ('processing', 'bandpass'),
    'detrend':                   ('processing', 'detrend'),
    'taper':                     ('processing', 'taper'),
    'differentiate':             ('processing', 'differentiate'),
    'integrate':                 ('processing', 'integrate'),
    'unwrap':                    ('processing', 'unwrap'),
    'subtract_common_mode':      ('processing', 'subtract_common_mode'),
    'downsample':                ('processing', 'downsample'),
    'imshow':                    ('plot',       'imshow'),
    'wiggle':                    ('plot',       'wiggle'),
    'plot_xcorr':                ('plot',       'plot_xcorr'),
    'bandpass2d':                ('signal',     'bandpass2d'),
    'detrend_time':              ('signal',     'detrend_time'),
    'taper_time':                ('signal',     'taper_time'),
    'diff_time':                 ('signal',     'diff_time'),
    'gradient_time':             ('signal',     'gradient_time'),
    'integrate_time':            ('signal',     'integrate_time'),
    'preprocess_unwrap':         ('signal',     'preprocess_unwrap'),
}


def __getattr__(name):
    if name in _LAZY:
        from importlib import import_module
        mod_name, attr = _LAZY[name]
        mod = import_module(f'.{mod_name}', __name__)
        return getattr(mod, attr)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
