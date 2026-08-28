"""
Optuna HPO for the quantile forecaster bake-off.

Searches hyperparameters on the (cheap) 5M sample, minimizing summed pinball
loss across q10/q50/q90 on the hotel-wise val split (log space - the loss the
models optimize). Studies are persisted to SQLite and resumable across Kaggle
sessions via `load_if_exists`.
"""
from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd

from models.forecasting.catboost_quantile import CatBoostMultiQuantileForecaster
from models.forecasting.lgbm_quantile import LGBMQuantileForecaster
from models.forecasting.metrics import pinball_loss
from models.forecasting.xgboost_quantile import XGBoostQuantileForecaster

log = logging.getLogger("optuna_study")
QUANTILES = (0.10, 0.50, 0.90)


def make_forecaster(name: str, params: dict, seed: int = 42,
                    num_boost_round: int = 5000,
                    early_stopping_rounds: int = 200,
                    device: str = "cpu", task_type: str = "CPU"):
    """Build an unfitted forecaster of the named family with `params`."""
    if name == "lightgbm":
        base = dict(LGBMQuantileForecaster().params)
        base.update(params)
        return LGBMQuantileForecaster(params=base, seed=seed,
                                      num_boost_round=num_boost_round,
                                      early_stopping_rounds=early_stopping_rounds)
    if name == "catboost":
        return CatBoostMultiQuantileForecaster(params=params, seed=seed,
                                               num_boost_round=num_boost_round,
                                               early_stopping_rounds=early_stopping_rounds,
                                               task_type=task_type)
    if name == "xgboost":
        return XGBoostQuantileForecaster(params=params, seed=seed,
                                         num_boost_round=num_boost_round,
                                         early_stopping_rounds=early_stopping_rounds,
                                         device=device)
    raise ValueError(f"unknown model: {name}")


def suggest_params(trial: optuna.Trial, name: str) -> dict:
    if name == "lightgbm":
        return {
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        }
    if name == "catboost":
        return {
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }
    if name == "xgboost":
        return {
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.1, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 50.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
    raise ValueError(f"unknown model: {name}")


def _val_pinball(model, X_val, y_val) -> float:
    preds = model.predict(X_val)
    return sum(pinball_loss(y_val, preds[f"q{int(q*100):02d}"], q) for q in QUANTILES)


def objective(trial, name, X, y, idx, cats, num_boost_round, device, task_type) -> float:
    params = suggest_params(trial, name)
    model = make_forecaster(name, params, num_boost_round=num_boost_round,
                            device=device, task_type=task_type)
    tr, va = idx["train"], idx["val"]
    model.fit(X.iloc[tr], y[tr], X.iloc[va], y[va], categorical_features=cats)
    return _val_pinball(model, X.iloc[va], y[va])


def run_study(name, X, y, idx, cats, n_trials, storage=None, study_name=None,
              num_boost_round=5000, device="cpu", task_type="CPU") -> optuna.Study:
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name or f"bakeoff_{name}",
        storage=storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        lambda t: objective(t, name, X, y, idx, cats, num_boost_round, device, task_type),
        n_trials=n_trials,
    )
    log.info("%s best summed-pinball=%.5f params=%s", name, study.best_value, study.best_params)
    return study
