"""Tests for forecaster metrics."""
from __future__ import annotations

import numpy as np
import pytest

from models.forecasting.metrics import (
    coverage,
    mae,
    mape,
    pinball_loss,
    wape,
)


def test_mae_identity_zero():
    y = np.array([100.0, 200.0, 300.0])
    assert mae(y, y) == 0.0


def test_mae_known_value():
    assert mae([100, 100], [110, 90]) == 10.0


def test_mape_known_value():
    # error 10 on truth 100 → 10%
    assert mape([100, 100], [110, 90]) == 10.0


def test_mape_rejects_nonpositive_truth():
    with pytest.raises(ValueError):
        mape([0, 100], [10, 110])


def test_wape_known_value():
    # |10| + |10| = 20; sum truth = 200; → 10%
    assert wape([100, 100], [110, 90]) == 10.0


def test_pinball_q50_equals_half_mae():
    y = np.array([100.0, 200.0, 150.0])
    p = np.array([110.0, 180.0, 160.0])
    assert pinball_loss(y, p, 0.5) == pytest.approx(0.5 * mae(y, p))


def test_pinball_zero_at_perfect_prediction():
    y = np.array([100.0, 200.0])
    for q in (0.1, 0.5, 0.9):
        assert pinball_loss(y, y, q) == 0.0


def test_pinball_asymmetry():
    # Under-prediction at q=0.9 should hurt more than over-prediction.
    y = np.array([100.0])
    under = pinball_loss(y, np.array([80.0]), 0.9)   # pred 20 below
    over = pinball_loss(y, np.array([120.0]), 0.9)   # pred 20 above
    assert under > over


def test_pinball_rejects_bad_quantile():
    with pytest.raises(ValueError):
        pinball_loss([1.0], [1.0], 0.0)
    with pytest.raises(ValueError):
        pinball_loss([1.0], [1.0], 1.0)


def test_coverage_full_inclusion():
    y = np.array([50.0, 100.0, 150.0])
    lo = np.array([40.0, 80.0, 120.0])
    hi = np.array([60.0, 120.0, 180.0])
    assert coverage(y, lo, hi) == 1.0


def test_coverage_half():
    y = np.array([50.0, 100.0])
    lo = np.array([40.0, 200.0])
    hi = np.array([60.0, 300.0])
    assert coverage(y, lo, hi) == 0.5


def test_coverage_tolerates_crossed_bounds():
    # Per-row min/max: 50 lies in [40, 60] either way. Crossing is reported
    # by the caller as a separate diagnostic; coverage itself stays defined.
    assert coverage([50.0], [60.0], [40.0]) == 1.0
    assert coverage([30.0], [60.0], [40.0]) == 0.0


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        mae([1, 2, 3], [1, 2])
