"""
CLI — train the LightGBM multi-quantile forecaster on the cached training sample.

Outputs:
  - models/forecasting/lgbm_quantile_<date>/{q10,q50,q90}.txt + metadata.json
  - artifacts/reports/lgbm_quantile_<date>_<split>.json

Gate (from baselines_2026-05-21):
  - hotel-wise WAPE must beat 37.6%  (primary, linear_hedonic)
  - time-wise  WAPE should beat 29.5% (secondary)
If q50 hotel-wise WAPE does not clear 37.6%, stop and diagnose — do not tune blindly.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_engineering.model_feature_sets import FORECASTER_FEATURES, FORECASTER_TARGET
from models.forecasting.conformal import ConformalQuantileCalibrator
from models.forecasting.data import categorical_feature_names, prepare_xy
from models.forecasting.lgbm_quantile import LGBMQuantileForecaster
from models.forecasting.metrics import coverage, mae, mape, pinball_loss, wape
from models.forecasting.splits import (
    SplitIndices,
    hotel_wise_split,
    time_wise_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_lgbm")

GATE_WAPE = {"hotel_wise": 37.6, "time_wise": 29.5}


def _per_segment_wape(
    df_eval: pd.DataFrame, y_true_tnd: np.ndarray, y_pred_tnd: np.ndarray, by: str
) -> dict[str, dict[str, float]]:
    """WAPE broken down by a categorical column."""
    levels = df_eval[by].to_numpy()
    out: dict[str, dict[str, float]] = {}
    for level in pd.unique(levels):
        mask = levels == level if level is not None else pd.isna(levels)
        if mask.sum() < 100:
            continue
        out[str(level)] = {
            "n": int(mask.sum()),
            "wape_pct": round(wape(y_true_tnd[mask], y_pred_tnd[mask]), 2),
        }
    return out


def _eval_split(
    split_name: str,
    df: pd.DataFrame,
    idx: SplitIndices,
    out_models_dir: Path,
    out_reports_dir: Path,
    seed: int,
    num_boost_round: int | None,
    early_stopping_rounds: int | None,
) -> dict[str, Any]:
    train_i, val_i, test_i = idx["train"], idx["val"], idx["test"]
    log.info("%s: train=%d val=%d test=%d", split_name, len(train_i), len(val_i), len(test_i))

    # prepare_xy once on the full frame, then slice — categorical dtype is
    # propagated correctly to every slice and the categories union is shared
    # between train/val/test (LightGBM needs this).
    X, y, _ = prepare_xy(df)
    cats = categorical_feature_names()

    X_train, y_train = X.iloc[train_i], y[train_i]
    X_val, y_val = X.iloc[val_i], y[val_i]
    X_test, y_test = X.iloc[test_i], y[test_i]

    # None → fall through to dataclass defaults (single source of truth).
    forecaster_kwargs: dict[str, Any] = {"seed": seed}
    if num_boost_round is not None:
        forecaster_kwargs["num_boost_round"] = num_boost_round
    if early_stopping_rounds is not None:
        forecaster_kwargs["early_stopping_rounds"] = early_stopping_rounds
    model = LGBMQuantileForecaster(**forecaster_kwargs)
    log.info(
        "%s: num_boost_round=%d  early_stopping=%d",
        split_name, model.num_boost_round, model.early_stopping_rounds,
    )

    t0 = time.time()
    model.fit(X_train, y_train, X_val, y_val, categorical_features=cats)
    fit_s = time.time() - t0
    log.info("%s: fit_seconds=%.1f", split_name, fit_s)

    # Persist boosters BEFORE eval so a metric crash doesn't destroy the fit.
    model_dir = out_models_dir / f"{split_name}"
    model.save(model_dir)

    # Fit conformal calibrator on the cal split (CQR, Romano et al 2019).
    # Calibrator widens raw q10/q90 so empirical coverage hits the nominal 0.80.
    cal_i = idx["cal"]
    X_cal, y_cal = X.iloc[cal_i], y[cal_i]
    log.info("%s: fitting conformal calibrator on n_cal=%d", split_name, len(cal_i))
    t0 = time.time()
    preds_cal_log = model.predict(X_cal)
    calibrator = ConformalQuantileCalibrator(alpha=0.20).fit(
        preds_cal_log["q10"], preds_cal_log["q90"], y_cal,
    )
    cal_fit_s = time.time() - t0
    log.info(
        "%s: calibrator c_=%.5f (log-scale), n_cal=%d, fit %.1fs",
        split_name, calibrator.c_, calibrator.n_cal_, cal_fit_s,
    )
    calibrator.save(model_dir / "conformal.json")

    t0 = time.time()
    preds_log = model.predict(X_test)
    pred_s = time.time() - t0

    q10_log, q50_log, q90_log = preds_log["q10"], preds_log["q50"], preds_log["q90"]
    q10_tnd, q50_tnd, q90_tnd = np.exp(q10_log), np.exp(q50_log), np.exp(q90_log)
    y_true_tnd = np.exp(y_test)

    # Point metrics on q50, TND scale (matches baselines)
    point = {
        "mae_tnd":  round(mae(y_true_tnd,  q50_tnd), 2),
        "mape_pct": round(mape(y_true_tnd, q50_tnd), 2),
        "wape_pct": round(wape(y_true_tnd, q50_tnd), 2),
    }

    # Pinball loss on log-price scale (the loss the model actually optimised)
    # and on TND scale (interpretable). The two ranks should match; we report both.
    pinball_log = {
        "q10": round(pinball_loss(y_test, q10_log, 0.10), 5),
        "q50": round(pinball_loss(y_test, q50_log, 0.50), 5),
        "q90": round(pinball_loss(y_test, q90_log, 0.90), 5),
    }
    pinball_tnd = {
        "q10": round(pinball_loss(y_true_tnd, q10_tnd, 0.10), 2),
        "q50": round(pinball_loss(y_true_tnd, q50_tnd, 0.50), 2),
        "q90": round(pinball_loss(y_true_tnd, q90_tnd, 0.90), 2),
    }

    # Interval calibration — target coverage ≈ 0.80 for [q10, q90]
    cov_80 = float(coverage(y_test, q10_log, q90_log))  # log-space is monotonic
    q10_cal_log, q90_cal_log = calibrator.apply(q10_log, q90_log)
    cov_80_calibrated = float(coverage(y_test, q10_cal_log, q90_cal_log))
    log.info(
        "%s: coverage80 raw=%.3f → calibrated=%.3f (target 0.80)",
        split_name, cov_80, cov_80_calibrated,
    )
    # Quantile crossing diagnostic
    crossing_q10_q50 = float(np.mean(q10_log > q50_log))
    crossing_q50_q90 = float(np.mean(q50_log > q90_log))

    df_test_view = df.iloc[test_i]
    per_segment_q50 = {
        "by_macro_region": _per_segment_wape(df_test_view, y_true_tnd, q50_tnd, "macro_region"),
        "by_stars_band":   _per_segment_wape(df_test_view, y_true_tnd, q50_tnd, "stars_band"),
    }

    gate = GATE_WAPE[split_name]
    gate_status = "PASS" if point["wape_pct"] < gate else "FAIL"
    log.info(
        "%s  WAPE q50 = %.2f%%  vs gate %.1f%%  → %s   (coverage80=%.3f, fit %.1fs)",
        split_name, point["wape_pct"], gate, gate_status, cov_80, fit_s,
    )

    return {
        "split": split_name,
        "n_train": int(len(train_i)),
        "n_val":   int(len(val_i)),
        "n_test":  int(len(test_i)),
        "fit_seconds":     round(fit_s, 2),
        "predict_seconds": round(pred_s, 2),
        "best_iterations": model.best_iterations_,
        "point_metrics_q50_tnd": point,
        "pinball_loss_log": pinball_log,
        "pinball_loss_tnd": pinball_tnd,
        "interval_coverage_80": round(cov_80, 4),
        "interval_coverage_80_calibrated": round(cov_80_calibrated, 4),
        "conformal_c_log": round(calibrator.c_, 5),
        "conformal_n_cal": calibrator.n_cal_,
        "quantile_crossing_rate": {
            "q10_above_q50": round(crossing_q10_q50, 4),
            "q50_above_q90": round(crossing_q50_q90, 4),
        },
        "gate": {
            "linear_hedonic_wape_pct": gate,
            "lgbm_q50_wape_pct": point["wape_pct"],
            "status": gate_status,
        },
        "per_segment_wape_q50": per_segment_q50,
        "model_dir": str(model_dir),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="artifacts/cache/sample_5M_seed42.parquet")
    ap.add_argument("--models-dir", default=None,
                    help="Default: models/forecasting/lgbm_quantile_<date>/")
    ap.add_argument("--out-dir", default="artifacts/reports")
    ap.add_argument("--split", choices=("hotel_wise", "time_wise", "both"), default="both")
    ap.add_argument("--seed", type=int, default=42)
    # CLI defaults intentionally fall through to the dataclass defaults in
    # lgbm_quantile.py (DEFAULT_NUM_BOOST_ROUND, DEFAULT_EARLY_STOPPING_ROUNDS).
    # Hard-coding defaults here once shadowed a dataclass change and silently
    # disabled a calibration experiment (2026-05-22). Single source of truth:
    # the dataclass.
    ap.add_argument("--num-boost-round", type=int, default=None)
    ap.add_argument("--early-stopping", type=int, default=None)
    args = ap.parse_args()

    today = date.today().isoformat()
    parquet = Path(args.parquet)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir) if args.models_dir else \
        Path("models") / "forecasting" / f"lgbm_quantile_{today}"
    models_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading %s", parquet)
    df = pq.read_table(parquet).to_pandas()
    log.info("loaded: %d rows, %d cols", len(df), df.shape[1])

    splits: list[tuple[str, SplitIndices]] = []
    if args.split in ("hotel_wise", "both"):
        log.info("computing hotel_wise split")
        splits.append(("hotel_wise",
                       hotel_wise_split(df["hotel_name_normalized"], seed=args.seed)))
    if args.split in ("time_wise", "both"):
        log.info("computing time_wise split")
        splits.append(("time_wise", time_wise_split(df["scraped_at"])))

    for split_name, idx in splits:
        report = _eval_split(
            split_name, df, idx,
            out_models_dir=models_dir,
            out_reports_dir=out_dir,
            seed=args.seed,
            num_boost_round=args.num_boost_round,
            early_stopping_rounds=args.early_stopping,
        )
        report["date"] = today
        report["parquet"] = str(parquet)
        report["target"] = FORECASTER_TARGET
        report["target_transform"] = "log"
        report["n_features"] = len(FORECASTER_FEATURES)

        out_path = out_dir / f"lgbm_quantile_{today}_{split_name}.json"
        out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        log.info("wrote %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
