from math import isclose

import numpy as np
from scipy.stats import norm
from dfss_training import ppk, _moving_range


def test_ppk():
    standard_norm = norm()
    spec_limits = (-6.0, 6.0)
    lppk, uppk, ppk_val = ppk(standard_norm, spec_limits)
    assert isclose(lppk, 2.0, rel_tol=1e-4)
    assert isclose(uppk, 2.0, rel_tol=1e-4)
    assert isclose(ppk_val, 2.0, rel_tol=1e-4)


def test_moving_range():
    test_array = np.arange(0.0, 100.0, 1.0)
    rolling_range = _moving_range(test_array)
    np.testing.assert_allclose(rolling_range, 1.0)
