"""Sniff an open HDF5 file to decide which reader handles it.

detect_format answers "which on-disk format is this". The return
value selects the reader dispatched from dasio.read_data_raw /
DASFile.

detect_origin answers "which vendor originally captured the samples".
For raw files this equals the kind; for Proc files it is recovered
from the flattened /Acquisition_origin attrs that write_data_proc
preserves.

Both live in dasio/readers/ because they're the keys the readers
package uses to dispatch within itself; the public API is still
re-exported from dasio.
"""
from __future__ import annotations

import h5py


def detect_format(f: h5py.File) -> str:
    """Return one of: 'Basic', 'Proc', 'ASN', 'OptaSense', 'APSensing',
    'Silixa', 'Event', or 'Unknown'.
    """
    if 'Data' in f:
        return 'Proc'
    if 'acqSpec' in f:
        return 'ASN'
    if 'Acquisition' in f:
        return 'OptaSense'
    if 'ProcessingServer' in f:
        return 'APSensing'
    if 'Acoustic' in f and 'SamplingFrequency[Hz]' in f['Acoustic'].attrs:
        return 'Silixa'
    # ASN's payload is also `/data`, but it is caught above; the two formats
    # that reach here name themselves in the attrs.
    if 'data' in f:
        attrs = f['data'].attrs
        if attrs.get('format') == 'Basic':
            return 'Basic'
        if 'event_id' in attrs:
            return 'Event'
    return 'Unknown'


def detect_origin(f: h5py.File) -> str:
    """Return one of: 'ASN', 'OptaSense', 'APSensing', 'Silixa', 'Sintela',
    or 'Unknown'. Mirrors the 'system' component of the legacy
    DASutils._get_data_system (external name, unchanged).
    """
    fmt = detect_format(f)
    if fmt != 'Proc':
        return fmt
    if 'Acquisition_origin' in f:
        attrs = f['Acquisition_origin'].attrs
        if 'AcquisitionId' in attrs:
            return 'OptaSense'
        if 'acqSpec.YvsXDelay' in attrs:
            return 'ASN'
        if 'ProcessingServer.ClassifierVersion' in attrs:
            return 'APSensing'
        if 'SamplingFrequency[Hz]' in attrs:
            return 'Silixa'
        if 'acquisition.num_channels' in attrs:
            return 'Sintela'
    return 'Unknown'
