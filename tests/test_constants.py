import numpy as np


def test_constants_module_exposes_values():
    from dasio.readers import constants
    assert constants.OPTASENSE_ETA == 0.78
    assert np.isclose(constants.OPTASENSE_COUNT2PHASE, np.pi / 2 ** 15)


def test_optasense_module_still_exposes_old_names():
    # Backward-compat: the underscore names must still resolve to the same values.
    from dasio.readers import optasense
    assert optasense._ETA == 0.78
    assert np.isclose(optasense._COUNT2PHASE, np.pi / 2 ** 15)
