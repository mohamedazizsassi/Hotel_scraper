"""Smoke test for the LightGBM multi-quantile forecaster on a tiny synthetic frame."""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.forecasting.lgbm_quantile import (
    DEFAULT_PARAMS,
    LGBMQuantileForecaster,
    QUANTILES,
)


def _toy_xy(n: int = 600, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Synthetic price ~= linear in stars + nights + boarding_effect + noise."""
    rng = np.random.default_rng(seed)
    boarding = rng.choice(["BB", "HDP", "AI"], size=n)
    boarding_effect = pd.Series(boarding).map({"BB": 0.0, "HDP": 0.3, "AI": 0.6}).to_numpy()
    stars = rng.integers(3, 6, size=n).astype(np.float64)
    nights = rng.integers(1, 8, size=n).astype(np.float64)
    days = rng.integers(0, 90, size=n).astype(np.float64)
    noise = rng.normal(0.0, 0.15, size=n)
    log_price = 4.0 + 0.5 * stars + 0.1 * nights + boarding_effect + noise

    X = pd.DataFrame({
        "boarding_canonical": pd.Categorical(boarding),
        "stars_int": stars,
        "nights": nights,
        "days_until_checkin": days,
    })
    return X, log_price


def _tiny_params() -> dict:
    """Relax min_data_in_leaf so LightGBM can split a 400-row dataset."""
    p = dict(DEFAULT_PARAMS)
    p["min_data_in_leaf"] = 5
    p["num_leaves"] = 15
    return p


def test_fit_predict_shapes() -> None:
    X, y = _toy_xy(n=600)
    X_train, X_val = X.iloc[:400], X.iloc[400:500]
    y_train, y_val = y[:400], y[400:500]
    X_test = X.iloc[500:]

    model = LGBMQuantileForecaster(
        params=_tiny_params(),
        num_boost_round=50,
        early_stopping_rounds=10,
    ).fit(X_train, y_train, X_val, y_val, categorical_features=["boarding_canonical"])

    preds = model.predict(X_test)
    assert set(preds) == {"q10", "q50", "q90"}
    for k, arr in preds.items():
        assert arr.shape == (len(X_test),), f"{k} shape={arr.shape}"
        assert np.isfinite(arr).all(), f"{k} has non-finite predictions"


def test_quantiles_ordered_on_average() -> None:
    """Per-row crossing is allowed, but the means should be ordered q10 < q50 < q90."""
    X, y = _toy_xy(n=600)
    X_train, X_val = X.iloc[:400], X.iloc[400:500]
    y_train, y_val = y[:400], y[400:500]
    X_test = X.iloc[500:]

    model = LGBMQuantileForecaster(
        params=_tiny_params(),
        num_boost_round=80,
        early_stopping_rounds=10,
    ).fit(X_train, y_train, X_val, y_val, categorical_features=["boarding_canonical"])

    preds = model.predict(X_test)
    assert preds["q10"].mean() < preds["q50"].mean() < preds["q90"].mean()


def test_save_and_metadata(tmp_path) -> None:
    X, y = _toy_xy(n=400)
    X_train, X_val = X.iloc[:300], X.iloc[300:]
    y_train, y_val = y[:300], y[300:]

    model = LGBMQuantileForecaster(
        params=_tiny_params(),
        num_boost_round=20,
        early_stopping_rounds=5,
    ).fit(X_train, y_train, X_val, y_val, categorical_features=["boarding_canonical"])

    model.save(tmp_path)
    for q in QUANTILES:
        qname = f"q{int(q * 100):02d}"
        assert (tmp_path / f"{qname}.txt").exists()
    assert (tmp_path / "metadata.json").exists()


def test_forecaster_save_load_roundtrip(tmp_path):
    """Train a tiny model, save, reload, ensure predictions match exactly."""
    X, y = _toy_xy(n=500)
    X_train, X_val = X.iloc[:400], X.iloc[400:]
    y_train, y_val = y[:400], y[400:]

    m = LGBMQuantileForecaster(
        params=_tiny_params(),
        num_boost_round=20,
        early_stopping_rounds=5,
        seed=0,
    ).fit(X_train, y_train, X_val, y_val, categorical_features=["boarding_canonical"])
    m.save(tmp_path)

    m2 = LGBMQuantileForecaster.load(tmp_path)
    preds_1 = m.predict(X_val)
    preds_2 = m2.predict(X_val)
    for q in ("q10", "q50", "q90"):
        np.testing.assert_allclose(preds_1[q], preds_2[q])
