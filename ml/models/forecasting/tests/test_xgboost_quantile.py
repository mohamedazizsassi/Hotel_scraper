import numpy as np
import pandas as pd

from models.forecasting.xgboost_quantile import XGBoostQuantileForecaster


def _toy_xy(n=400, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.integers(0, 5, size=n)
    cat = pd.Categorical(rng.choice(["a", "b", "c"], size=n))
    y = 5.0 + 0.5 * x1 + 0.1 * x2 + rng.normal(scale=0.2, size=n)
    X = pd.DataFrame({"x1": x1, "x2": x2, "cat": cat})
    return X, y


def test_fit_predict_shape_and_keys():
    X, y = _toy_xy()
    m = XGBoostQuantileForecaster(num_boost_round=40, seed=42)
    m.fit(X.iloc[:300], y[:300], X.iloc[300:], y[300:], categorical_features=["cat"])
    preds = m.predict(X.iloc[300:])
    assert set(preds) == {"q10", "q50", "q90"}
    assert all(len(preds[k]) == 100 for k in preds)
    assert all(np.isfinite(preds[k]).all() for k in preds)


def test_save_load_round_trip(tmp_path):
    X, y = _toy_xy()
    m = XGBoostQuantileForecaster(num_boost_round=40, seed=42)
    m.fit(X.iloc[:300], y[:300], X.iloc[300:], y[300:], categorical_features=["cat"])
    before = m.predict(X.iloc[300:])
    m.save(tmp_path / "xgb")
    m2 = XGBoostQuantileForecaster.load(tmp_path / "xgb")
    after = m2.predict(X.iloc[300:])
    assert m2.feature_names_ == ["x1", "x2", "cat"]
    for k in before:
        assert np.allclose(before[k], after[k], atol=1e-5)
