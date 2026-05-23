"""
LightGBM multi-quantile forecaster — trains three separate boosters (q10, q50, q90)
on the same features. Target is log(price); callers np.exp() before reporting on TND.

LightGBM does not support joint multi-quantile loss; the standard approach is
one model per quantile with identical hyperparams and seed. Quantile crossing
(q10 > q50 or q50 > q90 for some rows) is possible — we report the rate but
do not post-hoc enforce monotonicity at this stage (would mask model issues).

Per D2 (locked 2026-05-20): probabilistic from iteration 1, no point-only step.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Default hyperparams — minimal first pass, deliberately untuned.
# Tuning happens only if the q50 hotel-wise WAPE clears the 37.6% gate;
# blind tuning before that hides architectural issues.
DEFAULT_PARAMS: dict = {
    "objective": "quantile",          # alpha is set per-quantile in fit()
    "metric": "quantile",
    "num_leaves": 127,
    "learning_rate": 0.05,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "force_col_wise": True,           # avoids the row/col auto-detect warning
}

DEFAULT_NUM_BOOST_ROUND = 5000
DEFAULT_EARLY_STOPPING_ROUNDS = 200
QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)

# Per-quantile overrides on top of DEFAULT_PARAMS. Tails (q10/q90) need
# finer leaves because the signal is sparse out at the wings — leaving
# min_data_in_leaf=200 there produces narrow, over-confident intervals
# (the 2026-05-22 calibration failure: coverage80 ≈ 0.63 vs target 0.80).
DEFAULT_QUANTILE_OVERRIDES: dict[float, dict] = {
    0.10: {"min_data_in_leaf": 50},
    0.90: {"min_data_in_leaf": 50},
}


@dataclass
class LGBMQuantileForecaster:
    """
    Trains one LightGBM regressor per quantile in {0.10, 0.50, 0.90}.

    All three share the same hyperparams, same training data, same categorical
    feature list, same seed. Only `objective.alpha` differs.

    Attributes
    ----------
    params:
        Base LightGBM params (alpha is injected per-quantile by fit()).
    num_boost_round, early_stopping_rounds:
        Standard LightGBM training controls.
    seed:
        Random seed. Same seed + same data → byte-identical boosters.
    """

    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    quantile_overrides: dict = field(
        default_factory=lambda: {q: dict(o) for q, o in DEFAULT_QUANTILE_OVERRIDES.items()}
    )
    num_boost_round: int = DEFAULT_NUM_BOOST_ROUND
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS
    seed: int = 42

    boosters_: dict[float, lgb.Booster] = field(default_factory=dict)
    feature_names_: list[str] = field(default_factory=list)
    categorical_features_: list[str] = field(default_factory=list)
    best_iterations_: dict[float, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        categorical_features: Sequence[str] | None = None,
    ) -> "LGBMQuantileForecaster":
        """
        Fit one booster per quantile. `y_*` is log(price). Uses val set for
        early stopping on per-quantile pinball loss.
        """
        if list(X_train.columns) != list(X_val.columns):
            raise ValueError("fit: X_train and X_val columns must match")

        self.feature_names_ = list(X_train.columns)
        self.categorical_features_ = list(categorical_features or [])
        cat_idx = [self.feature_names_.index(c) for c in self.categorical_features_
                   if c in self.feature_names_]

        d_train = lgb.Dataset(
            X_train, label=y_train,
            categorical_feature=cat_idx if cat_idx else "auto",
            free_raw_data=False,
        )
        d_val = lgb.Dataset(
            X_val, label=y_val,
            reference=d_train,
            categorical_feature=cat_idx if cat_idx else "auto",
            free_raw_data=False,
        )

        for q in QUANTILES:
            params = dict(self.params)
            params["objective"] = "quantile"
            params["alpha"] = q
            params["seed"] = self.seed
            params.update(self.quantile_overrides.get(q, {}))
            logger.info(
                "training quantile=%.2f  min_data_in_leaf=%s",
                q, params.get("min_data_in_leaf"),
            )
            booster = lgb.train(
                params,
                d_train,
                num_boost_round=self.num_boost_round,
                valid_sets=[d_val],
                valid_names=["val"],
                callbacks=[
                    lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=100),
                ],
            )
            self.boosters_[q] = booster
            self.best_iterations_[q] = int(booster.best_iteration)
            logger.info(
                "quantile=%.2f best_iter=%d  best_score=%.5f",
                q, booster.best_iteration,
                booster.best_score["val"]["quantile"],
            )
        return self

    # -------------------------------------------------------------- predict
    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Predict the three quantiles in log-price space.

        Returns
        -------
        {"q10": ..., "q50": ..., "q90": ...} — np.float64 arrays of length len(X).
        """
        if not self.boosters_:
            raise RuntimeError("predict: fit() not called yet")
        if list(X.columns) != self.feature_names_:
            raise ValueError(
                f"predict: column mismatch.\n"
                f"  expected: {self.feature_names_[:5]}...\n"
                f"  got:      {list(X.columns)[:5]}..."
            )
        return {
            f"q{int(q * 100):02d}": self.boosters_[q].predict(
                X, num_iteration=self.best_iterations_[q]
            )
            for q in QUANTILES
        }

    # ------------------------------------------------------------------ io
    def save(self, out_dir: str | Path) -> None:
        """Persist three booster .txt files + metadata.json to `out_dir`."""
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        for q, booster in self.boosters_.items():
            qname = f"q{int(q * 100):02d}"
            booster.save_model(str(path / f"{qname}.txt"),
                               num_iteration=self.best_iterations_[q])
        meta = {
            "quantiles": list(QUANTILES),
            "params": self.params,
            "quantile_overrides": {f"q{int(q*100):02d}": ov
                                   for q, ov in self.quantile_overrides.items()},
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
            "seed": self.seed,
            "feature_names": self.feature_names_,
            "categorical_features": self.categorical_features_,
            "best_iterations": {f"q{int(q*100):02d}": self.best_iterations_[q]
                                for q in QUANTILES},
        }
        (path / "metadata.json").write_text(json.dumps(meta, indent=2),
                                            encoding="utf-8")
        logger.info("saved boosters + metadata to %s", path)
