"""Vendor-derived physical constants, isolated for clarity and easy gating.

OptaSense values copied verbatim from the legacy
DAS-utilities DASutils.py::_parse_raw2strain_factor_optasense.
"""
import numpy as np

# Photo-elastic scaling factor for longitudinal strain in isotropic material.
OPTASENSE_ETA = 0.78
# Raw count to radians.
OPTASENSE_COUNT2PHASE = np.pi / 2 ** 15
