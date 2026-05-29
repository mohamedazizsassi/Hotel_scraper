import numpy as np
from models.forecasting.base import enforce_monotone


def test_enforce_monotone_sorts_crossed_rows():
    # row 0 is already ordered; row 1 has q10>q50>q90 (fully crossed)
    q10 = np.array([1.0, 9.0])
    q50 = np.array([2.0, 5.0])
    q90 = np.array([3.0, 1.0])
    lo, mid, hi = enforce_monotone(q10, q50, q90)
    assert np.allclose(lo, [1.0, 1.0])
    assert np.allclose(mid, [2.0, 5.0])
    assert np.allclose(hi, [3.0, 9.0])
    # invariant: lo <= mid <= hi everywhere
    assert np.all(lo <= mid) and np.all(mid <= hi)
