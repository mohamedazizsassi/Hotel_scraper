"""
Shared contract for all quantile forecasters in this package.

Every forecaster predicts the SAME three quantiles in LOG-price space and
exposes the SAME fit/predict/save/load surface, so eval (metrics.py),
calibration (conformal.py, mondrian_conformal.py), and the recommender are
model-agnostic.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd


@runtime_checkable
class QuantileForecaster(Protocol):
    """Structural type the bake-off harness depends on. LGBMQuantileForecaster,
    CatBoostMultiQuantileForecaster and XGBoostQuantileForecaster all satisfy it."""

    feature_names_: list[str]

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        categorical_features: Sequence[str] | None = None,
    ) -> "QuantileForecaster": ...

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Return {"q10","q50","q90"} as float64 arrays in LOG-price space."""
        ...

    def save(self, out_dir: str | Path) -> None: ...

    @classmethod
    def load(cls, model_dir: str | Path) -> "QuantileForecaster": ...


def enforce_monotone(
    q10: np.ndarray, q50: np.ndarray, q90: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row sort so q10 <= q50 <= q90, fixing quantile crossing.

    Independent quantile regressors can cross for some rows. Sorting the three
    values per row is the standard, distribution-free repair (Chernozhukov et
    al. 2010, "Quantile and Probability Curves Without Crossing"). Returns the
    sorted (lo, mid, hi) arrays; inputs are not mutated.
    """
    stacked = np.vstack([np.asarray(q10, dtype=np.float64),
                         np.asarray(q50, dtype=np.float64),
                         np.asarray(q90, dtype=np.float64)])
    stacked.sort(axis=0)
    return stacked[0], stacked[1], stacked[2]
