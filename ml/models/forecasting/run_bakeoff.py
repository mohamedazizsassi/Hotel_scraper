"""
CLI - run the 3-model quantile forecaster bake-off.

Flow per model: (1) Optuna HPO on the 5M sample, (2) refit best config on the
full snapshot, (3) global + Mondrian conformal on the cal split, (4) eval on
test, (5) write JSON report. Run as a module:

    python -m models.forecasting.run_bakeoff --sample <5M.parquet> --full <29M.parquet> \
        --models lightgbm catboost xgboost --n-trials 50 --device cuda --task-type GPU

--smoke runs the whole flow on synthetic data (no parquet/GPU) for CI.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from models.forecasting.conformal import ConformalQuantileCalibrator
from models.forecasting.data import categorical_feature_names, prepare_xy
from models.forecasting.metrics import coverage, mae, mape, wape
from models.forecasting.mondrian_conformal import MondrianConformalCalibrator
from models.forecasting.optuna_study import make_forecaster, run_study
from models.forecasting.splits import hotel_wise_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_bakeoff")
QUANTILES = (0.10, 0.50, 0.90)


def _eval_one(model, X, y, idx, df, group_col, n_trials_done):
    """Fit conformal on cal, evaluate on test, return a metrics dict."""
    te, ca = idx["test"], idx["cal"]
    preds = model.predict(X.iloc[te])
    q10, q50, q90 = preds["q10"], preds["q50"], preds["q90"]
    yt = y[te]
    q50_tnd, yt_tnd = np.exp(q50), np.exp(yt)

    preds_cal = model.predict(X.iloc[ca])
    glob = ConformalQuantileCalibrator(alpha=0.20).fit(preds_cal["q10"], preds_cal["q90"], y[ca])
    q10g, q90g = glob.apply(q10, q90)

    groups_cal = df.iloc[ca][group_col].astype(str).to_numpy()
    groups_te = df.iloc[te][group_col].astype(str).to_numpy()
    mond = MondrianConformalCalibrator(alpha=0.20).fit(
        preds_cal["q10"], preds_cal["q90"], y[ca], groups_cal)
    q10m, q90m = mond.apply(q10, q90, groups_te)

    return {
        "point_metrics_q50_tnd": {
            "mae_tnd": round(mae(yt_tnd, q50_tnd), 2),
            "mape_pct": round(mape(yt_tnd, q50_tnd), 2),
            "wape_pct": round(wape(yt_tnd, q50_tnd), 2),
        },
        "coverage80_raw": round(float(coverage(yt, q10, q90)), 4),
        "coverage80_global_cqr": round(float(coverage(yt, q10g, q90g)), 4),
        "coverage80_mondrian": round(float(coverage(yt, q10m, q90m)), 4),
        "median_width_global": round(float(np.median(np.exp(q90g) - np.exp(q10g))), 2),
        "median_width_mondrian": round(float(np.median(np.exp(q90m) - np.exp(q10m))), 2),
        "crossing_q10_q50": round(float(np.mean(preds["q10"] > preds["q50"])), 4),
        "n_trials": n_trials_done,
    }


def run_bakeoff(sample_path, full_path, models, n_trials, out_dir,
                device="cpu", task_type="CPU", storage_dir=None):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df_sample = pq.read_table(sample_path).to_pandas()
    Xs, ys, _ = prepare_xy(df_sample)
    cats = categorical_feature_names()
    idx_s = hotel_wise_split(df_sample["hotel_name_normalized"], seed=42)

    df_full = pq.read_table(full_path).to_pandas()
    Xf, yf, _ = prepare_xy(df_full)
    idx_f = hotel_wise_split(df_full["hotel_name_normalized"], seed=42)

    report = {}
    for name in models:
        storage = f"sqlite:///{Path(storage_dir or out) / f'study_{name}.db'}"
        study = run_study(name, Xs, ys, idx_s, cats, n_trials=n_trials,
                          storage=storage, study_name=f"bakeoff_{name}",
                          device=device, task_type=task_type)
        best = make_forecaster(name, study.best_params, device=device, task_type=task_type)
        best.fit(Xf.iloc[idx_f["train"]], yf[idx_f["train"]],
                 Xf.iloc[idx_f["val"]], yf[idx_f["val"]], categorical_features=cats)
        best.save(out / f"model_{name}")
        report[name] = _eval_one(best, Xf, yf, idx_f, df_full, "macro_region", len(study.trials))
        report[name]["best_params"] = study.best_params

    (out / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("wrote %s", out / "comparison.json")
    return report


def run_bakeoff_smoke(out_dir):
    """Tiny end-to-end run on synthetic data - no parquet, no GPU."""
    rng = np.random.default_rng(0)
    n = 800
    hotels = rng.choice([f"h{i}" for i in range(100)], size=n)
    df = pd.DataFrame({
        "hotel_name_normalized": hotels,
        "macro_region": rng.choice(["sahel", "cap_bon"], size=n),
        "price": np.exp(5 + 0.4 * rng.normal(size=n)),
    })
    X = pd.DataFrame({"x1": rng.normal(size=n),
                      "cat": pd.Categorical(rng.choice(["a", "b"], size=n))})
    y = np.log(df["price"].to_numpy())
    idx = hotel_wise_split(df["hotel_name_normalized"], seed=42)
    report = {}
    for name in ("lightgbm", "catboost", "xgboost"):
        m = make_forecaster(name, {}, num_boost_round=15)
        m.fit(X.iloc[idx["train"]], y[idx["train"]],
              X.iloc[idx["val"]], y[idx["val"]], categorical_features=["cat"])
        report[name] = _eval_one(m, X, y, idx, df, "macro_region", 0)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", default="artifacts/cache/sample_5M_seed42.parquet")
    ap.add_argument("--full", default="artifacts/features_2026-05-18.parquet")
    ap.add_argument("--models", nargs="+", default=["lightgbm", "catboost", "xgboost"])
    ap.add_argument("--n-trials", type=int, default=50)
    ap.add_argument("--out-dir", default="artifacts/reports/bakeoff")
    ap.add_argument("--device", default="cpu")          # "cuda" on Kaggle
    ap.add_argument("--task-type", default="CPU")       # "GPU" on Kaggle (CatBoost)
    ap.add_argument("--storage-dir", default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        print(json.dumps(run_bakeoff_smoke(args.out_dir), indent=2))
        return 0
    run_bakeoff(args.sample, args.full, args.models, args.n_trials, args.out_dir,
                device=args.device, task_type=args.task_type, storage_dir=args.storage_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
