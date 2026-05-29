"""
XGBoost quantile forecaster — ONE model, multi-quantile via the 2.0+
`reg:quantileerror` objective with a vector `quantile_alpha`. Target is
log(price). GPU via device="cuda". Categoricals via native enable_categorical.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from models.forecasting.base import enforce_monotone

QUANTILES: tuple[float, ...] = (0.10, 0.50, 0.90)

DEFAULT_PARAMS: dict = {
    "max_depth": 8,
    "learning_rate": 0.05,
    "min_child_weight": 5.0,
    "subsample": 0.8,
    "colsample_bytree": 0.9,
    "reg_lambda": 1.0,
}


@dataclass
class XGBoostQuantileForecaster:
    params: dict = field(default_factory=lambda: dict(DEFAULT_PARAMS))
    num_boost_round: int = 5000
    early_stopping_rounds: int = 200
    seed: int = 42
    device: str = "cpu"             # "cuda" on Kaggle GPU
    monotone: bool = True

    booster_: xgb.Booster | None = None
    feature_names_: list[str] = field(default_factory=list)
    categorical_features_: list[str] = field(default_factory=list)
    best_iteration_: int = 0

    def _dmatrix(self, X: pd.DataFrame, y=None) -> xgb.DMatrix:
        # Declared categoricals must be pandas category dtype for enable_categorical.
        Xc = X.copy()
        for c in self.categorical_features_:
            if c in Xc.columns:
                Xc[c] = Xc[c].astype("category")
        return xgb.DMatrix(Xc, label=y, enable_categorical=True)

    def fit(self, X_train, y_train, X_val, y_val, categorical_features=None):
        self.feature_names_ = list(X_train.columns)
        self.categorical_features_ = list(categorical_features or [])
        params = {
            "objective": "reg:quantileerror",
            "quantile_alpha": np.array(QUANTILES, dtype=np.float64),
            "tree_method": "hist",
            "device": self.device,
            "seed": self.seed,
            **self.params,
        }
        dtrain = self._dmatrix(X_train, y_train)
        dval = self._dmatrix(X_val, y_val)
        self.booster_ = xgb.train(
            params, dtrain,
            num_boost_round=self.num_boost_round,
            evals=[(dval, "val")],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=False,
        )
        self.best_iteration_ = int(self.booster_.best_iteration)
        return self

    def predict(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        if self.booster_ is None:
            raise RuntimeError("predict: fit() not called yet")
        if list(X.columns) != self.feature_names_:
            raise ValueError("predict: column mismatch vs feature_names_")
        out = self.booster_.predict(
            self._dmatrix(X), iteration_range=(0, self.best_iteration_ + 1)
        )
        out = np.asarray(out, dtype=np.float64).reshape(len(X), len(QUANTILES))
        q10, q50, q90 = out[:, 0], out[:, 1], out[:, 2]
        if self.monotone:
            q10, q50, q90 = enforce_monotone(q10, q50, q90)
        return {"q10": q10, "q50": q50, "q90": q90}

    def save(self, out_dir: str | Path) -> None:
        path = Path(out_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.booster_.save_model(str(path / "model.json"))
        (path / "metadata.json").write_text(json.dumps({
            "model_type": "xgboost_quantile",
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
    def load(cls, model_dir: str | Path) -> "XGBoostQuantileForecaster":
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
        booster = xgb.Booster()
        booster.load_model(str(path / "model.json"))
        inst.booster_ = booster
        return inst
