"""Sniff an open HDF5 file to decide which reader handles it.

detect_data_kind answers "which on-disk format is this". The return
value selects the reader dispatched from dasio.read_data_raw /
DASFile.

detect_origin answers "which vendor originally captured the samples".
For raw files this equals the kind; for Proc files it is recovered
from the flattened /Acquisition_origin attrs that write_data_proc
preserves.

Both live in dasio/readers/ because they're the keys the readers
package uses to dispatch within itself; the public API is still
re-exported from realTimeMonitor.dasio.
"""
from __future__ import annotations

import h5py


def detect_data_kind(f: h5py.File) -> str:
    """Return one of: 'Proc', 'ASN', 'OptaSense', 'APSensing',
    'Event', or 'Unknown'.
    """
    if 'Data' in f:
        return 'Proc'
    if 'acqSpec' in f:
        return 'ASN'
    if 'Acquisition' in f:
        return 'OptaSense'
    if 'ProcessingServer' in f:
        return 'APSensing'
    if 'data' in f and 'event_id' in f['data'].attrs:
        return 'Event'
    return 'Unknown'


def detect_origin(f: h5py.File) -> str:
    """Return one of: 'ASN', 'OptaSense', 'APSensing', 'Sintela',
    or 'Unknown'. Mirrors the legacy DASutils._get_data_system 'system'
    component.
    """
    kind = detect_data_kind(f)
    if kind != 'Proc':
        return kind
    if 'Acquisition_origin' in f:
        attrs = f['Acquisition_origin'].attrs
        if 'AcquisitionId' in attrs:
            return 'OptaSense'
        if 'acqSpec.YvsXDelay' in attrs:
            return 'ASN'
        if 'ProcessingServer.ClassifierVersion' in attrs:
            return 'APSensing'
        if 'acquisition.num_channels' in attrs:
            return 'Sintela'
    return 'Unknown'
