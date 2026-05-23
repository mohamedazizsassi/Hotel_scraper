"""
Interval-based anomaly detector (D3).

A row is flagged anomalous when its log(price) falls outside the
conformal-calibrated [q10, q90] interval produced by the forecaster.
Anomaly score is the signed, width-normalized distance from the interval:
    score < 0   → observation below q10  (underpriced vs market)
    score == 0  → inside the interval
    score > 0   → observation above q90  (overpriced vs market)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class _ForecasterLike(Protocol):
    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]: ...


class _CalibratorLike(Protocol):
    def apply(
        self, q_lo: np.ndarray, q_hi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...


_EPS = 1e-9


@dataclass
class IntervalAnomalyDetector:
    """
    Composes a quantile forecaster + a conformal calibrator. Stateless at
    inference; both inputs are already fitted.
    """

    forecaster: _ForecasterLike
    calibrator: _CalibratorLike

    def score(self, X: pd.DataFrame, y_log: np.ndarray) -> pd.DataFrame:
        """
        Score `X` against observed log-prices `y_log`. Returns a frame aligned
        to X.index with prediction columns, calibrated interval, anomaly
        score, and boolean flag.
        """
        y_log = np.asarray(y_log, dtype=np.float64)
        if y_log.shape != (len(X),):
            raise ValueError(
                f"score: y_log length {y_log.shape} != X length {len(X)}"
            )

        preds = self.forecaster.predict(X)
        q10, q50, q90 = preds["q10"], preds["q50"], preds["q90"]
        q10_cal, q90_cal = self.calibrator.apply(q10, q90)

        width = np.maximum(q90_cal - q10_cal, _EPS)
        below = y_log < q10_cal
        above = y_log > q90_cal
        score = np.where(
            below, (y_log - q10_cal) / width,
            np.where(above, (y_log - q90_cal) / width, 0.0),
        )
        is_anomaly = below | above

        return pd.DataFrame(
            {
                "q10_log": q10,
                "q50_log": q50,
                "q90_log": q90,
                "q10_cal_log": q10_cal,
                "q90_cal_log": q90_cal,
                "anomaly_score": score,
                "is_anomaly": is_anomaly,
            },
            index=X.index,
        )
