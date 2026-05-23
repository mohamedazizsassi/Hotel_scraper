"""Tests for IntervalAnomalyDetector."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.anomaly.detector import IntervalAnomalyDetector


class _StubForecaster:
    """Returns hand-set q10/q50/q90 per row."""

    def __init__(self, q10, q50, q90):
        self._q = {"q10": np.asarray(q10), "q50": np.asarray(q50), "q90": np.asarray(q90)}

    def predict(self, X):
        return self._q


class _StubCalibrator:
    """Identity calibrator: c_ = 0."""

    c_ = 0.0
    alpha = 0.20

    def apply(self, q_lo, q_hi):
        return np.asarray(q_lo), np.asarray(q_hi)


def test_score_flags_row_above_q90_as_overpriced():
    f = _StubForecaster(q10=[1.0], q50=[2.0], q90=[3.0])
    det = IntervalAnomalyDetector(forecaster=f, calibrator=_StubCalibrator())
    X = pd.DataFrame({"x": [0]})
    y_log = np.array([4.0])  # above q90 by 1.0; width = 2.0
    out = det.score(X, y_log=y_log)
    assert bool(out["is_anomaly"].iloc[0]) is True
    assert out["anomaly_score"].iloc[0] == pytest.approx(0.5)


def test_score_flags_row_below_q10_as_underpriced():
    f = _StubForecaster(q10=[1.0], q50=[2.0], q90=[3.0])
    det = IntervalAnomalyDetector(forecaster=f, calibrator=_StubCalibrator())
    X = pd.DataFrame({"x": [0]})
    y_log = np.array([0.0])  # below q10 by 1.0
    out = det.score(X, y_log=y_log)
    assert bool(out["is_anomaly"].iloc[0]) is True
    assert out["anomaly_score"].iloc[0] == pytest.approx(-0.5)


def test_score_does_not_flag_row_inside_interval():
    f = _StubForecaster(q10=[1.0], q50=[2.0], q90=[3.0])
    det = IntervalAnomalyDetector(forecaster=f, calibrator=_StubCalibrator())
    X = pd.DataFrame({"x": [0]})
    y_log = np.array([2.5])
    out = det.score(X, y_log=y_log)
    assert bool(out["is_anomaly"].iloc[0]) is False
    assert out["anomaly_score"].iloc[0] == 0.0


def test_score_returns_all_required_columns():
    f = _StubForecaster(q10=[1.0, 1.0], q50=[2.0, 2.0], q90=[3.0, 3.0])
    det = IntervalAnomalyDetector(forecaster=f, calibrator=_StubCalibrator())
    X = pd.DataFrame({"x": [0, 1]})
    y_log = np.array([1.5, 4.0])
    out = det.score(X, y_log=y_log)
    expected = {
        "q10_log", "q50_log", "q90_log",
        "q10_cal_log", "q90_cal_log",
        "anomaly_score", "is_anomaly",
    }
    assert expected.issubset(set(out.columns))
    assert len(out) == 2


def test_score_handles_zero_width_interval_without_divide_by_zero():
    f = _StubForecaster(q10=[2.0], q50=[2.0], q90=[2.0])
    det = IntervalAnomalyDetector(forecaster=f, calibrator=_StubCalibrator())
    X = pd.DataFrame({"x": [0]})
    y_log = np.array([3.0])
    out = det.score(X, y_log=y_log)
    assert np.isfinite(out["anomaly_score"].iloc[0])
    assert bool(out["is_anomaly"].iloc[0]) is True
