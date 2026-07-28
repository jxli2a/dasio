"""The OptaSense count->strain constants and how the factor uses them.

They are the only vendor physics hardcoded in the package — everything else is
read off the file — so a silent change to either moves every converted
amplitude with nothing to reveal it.
"""
import h5py
import numpy as np

from dasio.readers.optasense import _COUNT2PHASE, _ETA, optasense_count2strain_factor


def test_constants_match_legacy_dasutils():
    assert _ETA == 0.78
    assert np.isclose(_COUNT2PHASE, np.pi / 2 ** 15)


def test_factor_composes_the_constants_with_the_file_attrs(optasense_file):
    """factor = polarity * count2phase * lambda / (4*pi*eta*n*G). Compared on
    magnitude; polarity is an FPGA-attr branch tested by its own read path."""
    with h5py.File(optasense_file) as f:
        acq = f['Acquisition']
        G = float(acq.attrs['GaugeLength'])
        n = float(acq['Custom'].attrs['Fibre Refractive Index'])
        lam = float(acq['Custom'].attrs['Laser Wavelength (nm)']) * 1e-9
        got = optasense_count2strain_factor(f)
    expected = _COUNT2PHASE * lam / (4.0 * np.pi * _ETA * n * G)
    assert np.isclose(abs(got), abs(expected), rtol=1e-12)
