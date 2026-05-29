import numpy as np
import pandas as pd

from models.forecasting.optuna_study import make_forecaster, run_study


def _toy_split_frame(n=600, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "x1": rng.normal(size=n),
        "x2": rng.integers(0, 5, size=n),
        "cat": pd.Categorical(rng.choice(["a", "b", "c"], size=n)),
    })
    y = 5.0 + 0.5 * X["x1"].to_numpy() + rng.normal(scale=0.2, size=n)
    idx = {"train": np.arange(0, 400), "val": np.arange(400, n)}
    return X, y, idx


def test_make_forecaster_each_model():
    for name in ("lightgbm", "catboost", "xgboost"):
        m = make_forecaster(name, {}, seed=42, num_boost_round=10)
        assert hasattr(m, "fit") and hasattr(m, "predict")


def test_run_study_resumes(tmp_path):
    X, y, idx = _toy_split_frame()
    storage = f"sqlite:///{tmp_path / 'study.db'}"
    s1 = run_study("xgboost", X, y, idx, cats=["cat"], n_trials=1,
                   storage=storage, study_name="t", num_boost_round=10)
    assert np.isfinite(s1.best_value)
    s2 = run_study("xgboost", X, y, idx, cats=["cat"], n_trials=1,
                   storage=storage, study_name="t", num_boost_round=10)
    assert len(s2.trials) >= 2   # resumed, not restarted


def test_bakeoff_smoke_runs(tmp_path):
    from models.forecasting.run_bakeoff import run_bakeoff_smoke
    report = run_bakeoff_smoke(out_dir=tmp_path)
    # one entry per model, each with a finite hotel-wise WAPE
    assert set(report) == {"lightgbm", "catboost", "xgboost"}
    for name in report:
        assert np.isfinite(report[name]["point_metrics_q50_tnd"]["wape_pct"])
