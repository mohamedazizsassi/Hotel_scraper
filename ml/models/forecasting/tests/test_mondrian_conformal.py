import numpy as np

from models.forecasting.mondrian_conformal import MondrianConformalCalibrator


def test_per_group_c_differs_and_widens_correctly():
    # Group "tight": raw interval already covers y -> small c.
    # Group "wide":  y sits far outside raw interval -> large c.
    n = 2000
    q_lo = np.zeros(2 * n)
    q_hi = np.ones(2 * n)
    y = np.concatenate([
        np.full(n, 0.5),     # tight: inside [0,1]
        np.full(n, 3.0),     # wide: far above 1
    ])
    groups = np.array(["tight"] * n + ["wide"] * n)
    cal = MondrianConformalCalibrator(alpha=0.20, min_cal_per_group=100).fit(q_lo, q_hi, y, groups)
    assert cal.c_by_group_["wide"] > cal.c_by_group_["tight"]


def test_small_group_falls_back_to_global():
    n = 500
    q_lo = np.zeros(n + 5)
    q_hi = np.ones(n + 5)
    y = np.concatenate([np.full(n, 0.5), np.full(5, 0.5)])
    groups = np.array(["big"] * n + ["tiny"] * 5)
    cal = MondrianConformalCalibrator(alpha=0.20, min_cal_per_group=100).fit(q_lo, q_hi, y, groups)
    # "tiny" has < min_cal_per_group points -> uses the global c
    assert cal.c_by_group_["tiny"] == cal.c_global_


def test_apply_uses_group_c_and_unseen_group_gets_global():
    n = 400
    q_lo = np.zeros(n)
    q_hi = np.ones(n)
    y = np.full(n, 2.0)
    groups = np.array(["g"] * n)
    cal = MondrianConformalCalibrator(alpha=0.20, min_cal_per_group=10).fit(q_lo, q_hi, y, groups)
    lo, hi = cal.apply(np.zeros(2), np.ones(2), np.array(["g", "never_seen"]))
    assert np.isclose(lo[0], 0.0 - cal.c_by_group_["g"])
    assert np.isclose(lo[1], 0.0 - cal.c_global_)   # unseen -> global fallback
