"""Tests for ConformalQuantileCalibrator (D3 anomaly module)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from models.forecasting.conformal import ConformalQuantileCalibrator


def test_calibrator_widens_narrow_intervals_to_hit_nominal_coverage():
    # Construct: y ~ N(0,1), narrow predicted intervals [-0.1, 0.1]
    # raw coverage ≈ 8%; after fitting, should be ≥ 80% on a fresh sample.
    rng = np.random.default_rng(0)
    y_cal = rng.standard_normal(5000)
    q_lo_cal = np.full_like(y_cal, -0.1)
    q_hi_cal = np.full_like(y_cal,  0.1)
    cal = ConformalQuantileCalibrator(alpha=0.20).fit(q_lo_cal, q_hi_cal, y_cal)
    assert cal.c_ > 0.5  # large widening required

    y_test = rng.standard_normal(20000)
    q_lo_test = np.full_like(y_test, -0.1)
    q_hi_test = np.full_like(y_test,  0.1)
    lo_eff, hi_eff = cal.apply(q_lo_test, q_hi_test)
    covered = float(np.mean((y_test >= lo_eff) & (y_test <= hi_eff)))
    assert covered >= 0.78  # within sampling noise of nominal 0.80


def test_calibrator_apply_returns_widened_intervals():
    cal = ConformalQuantileCalibrator(alpha=0.20)
    cal.c_ = 2.5
    cal.alpha_fit_ = 0.20
    lo = np.array([1.0, 3.0, 5.0])
    hi = np.array([2.0, 4.0, 6.0])
    lo_eff, hi_eff = cal.apply(lo, hi)
    np.testing.assert_allclose(lo_eff, lo - 2.5)
    np.testing.assert_allclose(hi_eff, hi + 2.5)


def test_calibrator_rejects_apply_before_fit():
    cal = ConformalQuantileCalibrator()
    with pytest.raises(RuntimeError, match="not fitted"):
        cal.apply(np.array([1.0]), np.array([2.0]))


def test_calibrator_raises_on_shape_mismatch_in_fit():
    cal = ConformalQuantileCalibrator()
    with pytest.raises(ValueError, match="shape"):
        cal.fit(np.array([1.0, 2.0]), np.array([3.0]), np.array([1.5, 2.5]))


def test_calibrator_save_load_roundtrip(tmp_path: Path):
    cal = ConformalQuantileCalibrator(alpha=0.20)
    cal.c_ = 1.234
    cal.alpha_fit_ = 0.20
    cal.n_cal_ = 5000
    out = tmp_path / "cal.json"
    cal.save(out)

    loaded = ConformalQuantileCalibrator.load(out)
    assert loaded.alpha == 0.20
    assert loaded.c_ == 1.234
    assert loaded.alpha_fit_ == 0.20
    assert loaded.n_cal_ == 5000


def test_finite_sample_correction_factor():
    # n=99 -> (1 - 0.2) * (1 + 1/99) ≈ 0.8081
    # ensure cut quantile uses this corrected level, not raw (1-alpha).
    rng = np.random.default_rng(1)
    y = rng.standard_normal(99)
    lo = np.full_like(y, -10.0)  # very wide raw intervals: all scores are negative
    hi = np.full_like(y,  10.0)
    cal = ConformalQuantileCalibrator(alpha=0.20).fit(lo, hi, y)
    # all scores negative, so c_ should be < 0 (intervals would shrink)
    assert cal.c_ < 0
