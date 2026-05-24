"""
Recommender — composes a quantile forecaster + a conformal calibrator + a
rule library to produce per-row pricing recommendations on the TND scale.

Stateless at inference. Both forecaster and calibrator must be already fitted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from models.recommendation.rules import (
    DEFAULT_RULES,
    RowContext,
    apply_rules,
)

REQUIRED_CONTEXT_COLUMNS: tuple[str, ...] = (
    "price", "price_per_night",
    "hotel_name_normalized", "city_name", "stars_int", "macro_region",
    "stars_band", "scraped_at", "check_in", "nights", "adults",
    "boarding_canonical",
    "peer_medium_median", "peer_medium_count",
)


class _ForecasterLike(Protocol):
    feature_names_: list[str]

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]: ...


class _CalibratorLike(Protocol):
    def apply(
        self, q_lo: np.ndarray, q_hi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class Recommender:
    """
    Composition of forecaster + calibrator + rules. Produces a per-row
    DataFrame of recommendations on the TND scale.
    """

    forecaster: _ForecasterLike
    calibrator: _CalibratorLike
    rules: tuple = DEFAULT_RULES

    def score(
        self, df: pd.DataFrame, test_indices: np.ndarray,
    ) -> pd.DataFrame:
        """
        Predict q10/q50/q90 on `df.iloc[test_indices]`, apply the calibrator,
        evaluate the rule library row-by-row, and return a per-row frame.

        Output index runs 0..N-1 where N = len(test_indices).
        """
        missing = [c for c in REQUIRED_CONTEXT_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"score: input df missing required columns: {missing}"
            )

        feat_cols = list(self.forecaster.feature_names_)
        if not feat_cols:
            raise RuntimeError(
                "score: forecaster.feature_names_ is empty; was the forecaster fitted?"
            )

        sub = df.iloc[test_indices].reset_index(drop=True)
        X = sub[feat_cols]
        preds = self.forecaster.predict(X)
        q10_log = preds["q10"]
        q50_log = preds["q50"]
        q90_log = preds["q90"]
        q10_cal_log, q90_cal_log = self.calibrator.apply(q10_log, q90_log)

        q10_cal_tnd = np.exp(q10_cal_log)
        q50_tnd     = np.exp(q50_log)
        q90_cal_tnd = np.exp(q90_cal_log)
        current_tnd          = sub["price"].to_numpy(dtype=float)
        current_per_night    = sub["price_per_night"].to_numpy(dtype=float)

        # Vectorised interval_status before the rule loop (also feeds reasons).
        below = current_tnd < q10_cal_tnd
        above = current_tnd > q90_cal_tnd
        interval_status = np.where(
            below, "below_band",
            np.where(above, "above_band", "in_band"),
        )

        peer_median = sub["peer_medium_median"].to_numpy(dtype=float)
        peer_count  = sub["peer_medium_count"].fillna(0).astype(int).to_numpy()

        directions: list[str] = []
        recommended: list[float] = []
        deltas: list[float] = []
        reasons_col: list[list[str]] = []

        for i in range(len(sub)):
            pm = float(peer_median[i]) if not np.isnan(peer_median[i]) else None
            ctx = RowContext(
                current_price_tnd=float(current_tnd[i]),
                current_price_per_night_tnd=float(current_per_night[i]),
                q10_cal_tnd=float(q10_cal_tnd[i]),
                q50_tnd=float(q50_tnd[i]),
                q90_cal_tnd=float(q90_cal_tnd[i]),
                peer_medium_median_per_night_tnd=pm,
                peer_medium_count=int(peer_count[i]),
            )
            rec = apply_rules(ctx, rules=self.rules)
            directions.append(rec.direction)
            recommended.append(float(rec.recommended_price_tnd))
            delta = (
                (rec.recommended_price_tnd - current_tnd[i]) / current_tnd[i] * 100.0
                if current_tnd[i] > 0 else 0.0
            )
            deltas.append(float(delta))
            reasons_col.append(list(rec.reasons))

        return pd.DataFrame({
            "hotel_name_normalized": sub["hotel_name_normalized"].to_numpy(),
            "city_name":              sub["city_name"].to_numpy(),
            "stars_int":              sub["stars_int"].to_numpy(),
            "macro_region":           sub["macro_region"].to_numpy(),
            "stars_band":             sub["stars_band"].to_numpy(),
            "scraped_at":             sub["scraped_at"].to_numpy(),
            "check_in":               sub["check_in"].to_numpy(),
            "nights":                 sub["nights"].to_numpy(),
            "adults":                 sub["adults"].to_numpy(),
            "boarding_canonical":     sub["boarding_canonical"].to_numpy(),
            "current_price_tnd":      current_tnd,
            "q10_cal_tnd":            q10_cal_tnd,
            "q50_tnd":                q50_tnd,
            "q90_cal_tnd":            q90_cal_tnd,
            "interval_status":        interval_status,
            "direction":              directions,
            "recommended_price_tnd":  recommended,
            "delta_pct_vs_current":   deltas,
            "peer_medium_median":     peer_median,
            "peer_medium_count":      peer_count,
            "reasons":                reasons_col,
        })
