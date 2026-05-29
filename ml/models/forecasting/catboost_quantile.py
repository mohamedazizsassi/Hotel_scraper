"""
CatBoost MultiQuantile forecaster — ONE model predicts q10/q50/q90 jointly,
so the three quantiles share a tree structure and cannot cross (unlike the
3-booster LightGBM). Target is log(price). GPU via task_type="GPU".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from models.forecasting.base import enforce_monotone

QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)
_LOSS = "MultiQuantile:alpha=0.1,0.5,0.9"

DEFAULT_PARAMS: dict = {
    "depth": 8,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
}


@dataclass
class CatBoostMultiQuantileForecaster:
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    num_boost_round: int = 5000
    early_stopping_rounds: int = 200
    seed: int = 42
    task_type: str = "CPU"          # "GPU" on Kaggle
    monotone: bool = True           # post-hoc per-row sort

    model_: CatBoostRegressor | None = None
    feature_names_: list[str] = field(default_factory=list)
    categorical_features_: list[str] = field(default_factory=list)
    best_iteration_: int = 0

    def _to_pool(self, X: pd.DataFrame, y=None) -> Pool:
        # CatBoost wants categoricals as str (no NaN). Cast the declared cats.
        Xc = X.copy()
        for c in self.categorical_features_:
            if c in Xc.columns:
                Xc[c] = Xc[c].astype(str).fillna("nan")
        cat_idx = [Xc.columns.get_loc(c) for c in self.categorical_features_ if c in Xc.columns]
        return Pool(Xc, label=y, cat_features=cat_idx)

    def fit(self, X_train, y_train, X_val, y_val, categorical_features=None):
        self.feature_names_ = list(X_train.columns)
        self.categorical_features_ = list(categorical_features or [])
        model = CatBoostRegressor(
            loss_function=_LOSS,
            iterations=self.num_boost_round,
            random_seed=self.seed,
            task_type=self.task_type,
            verbose=False,
            **self.params,
        )
        model.fit(
            self._to_pool(X_train, y_train),
            eval_set=self._to_pool(X_val, y_val),
            use_best_model=True,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self.model_ = model
        self.best_iteration_ = int(model.get_best_iteration() or model.tree_count_)
        return self

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if self.model_ is None:
            raise RuntimeError("predict: fit() not called yet")
        if list(X.columns) != self.feature_names_:
            raise ValueError("predict: column mismatch vs feature_names_")
        out = np.asarray(self.model_.predict(self._to_pool(X)), dtype=np.float64)
        q10, q50, q90 = out[:, 0], out[:, 1], out[:, 2]
        if self.monotone:
            q10, q50, q90 = enforce_monotone(q10, q50, q90)
        return {"q10": q10, "q50": q50, "q90": q90}

    def save(self, out_dir: str | Path) -> None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.model_.save_model(str(path / "model.cbm"))
        (path / "metadata.json").write_text(json.dumps({
            "model_type": "catboost_multiquantile",
            "quantiles": list(QUANTILES),
            "params": self.params,
            "num_boost_round": self.num_boost_round,
            "early_stopping_rounds": self.early_stopping_rounds,
            "seed": self.seed,
            "monotone": self.monotone,
            "feature_names": self.feature_names_,
            "categorical_features": self.categorical_features_,
            "best_iteration": self.best_iteration_,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: str | Path) -> "CatBoostMultiQuantileForecaster":
        path = Path(model_dir)
        meta = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        inst = cls(
            params=meta["params"],
            num_boost_round=meta["num_boost_round"],
            early_stopping_rounds=meta["early_stopping_rounds"],
            seed=meta["seed"],
            monotone=meta.get("monotone", True),
        )
        inst.feature_names_ = list(meta["feature_names"])
        inst.categorical_features_ = list(meta["categorical_features"])
        inst.best_iteration_ = int(meta["best_iteration"])
        model = CatBoostRegressor()
        model.load_model(str(path / "model.cbm"))
        inst.model_ = model
        return inst
