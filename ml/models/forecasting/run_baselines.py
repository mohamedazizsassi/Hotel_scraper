"""
CLI — fit/evaluate the four baselines on the cached training sample.

Outputs one JSON report per split to ml/artifacts/reports/baselines_<date>_<split>.json.

Numbers from this run are the gate: any trained GBT must beat all four on
the hotel-wise hold-out, or it does not ship.
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

from feature_engineering.model_feature_sets import FORECASTER_TARGET
from models.forecasting.baselines import all_baselines
from models.forecasting.metrics import mae, mape, wape
from models.forecasting.splits import (
    SplitIndices,
    hotel_wise_split,
    time_wise_split,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_baselines")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _eval_one(
    name: str,
    baseline,
    df_train: pd.DataFrame, y_train: np.ndarray,
    df_eval: pd.DataFrame, y_eval: np.ndarray,
) -> dict[str, Any]:
    t0 = time.time()
    baseline.fit(df_train, y_train)
    fit_s = time.time() - t0

    t0 = time.time()
    y_pred_log = baseline.predict(df_eval)
    pred_s = time.time() - t0

    y_true_tnd = np.exp(y_eval)
    y_pred_tnd = np.exp(y_pred_log)

    return {
        "name": name,
        "fit_seconds": round(fit_s, 2),
        "predict_seconds": round(pred_s, 2),
        "mae_tnd": round(mae(y_true_tnd, y_pred_tnd), 2),
        "mape_pct": round(mape(y_true_tnd, y_pred_tnd), 2),
        "wape_pct": round(wape(y_true_tnd, y_pred_tnd), 2),
    }


def _per_segment_wape(
    df_eval: pd.DataFrame, y_true_tnd: np.ndarray, y_pred_tnd: np.ndarray, by: str
) -> dict[str, dict[str, float]]:
    """WAPE broken down by a categorical column. y_* arrays must be positional."""
    levels = df_eval[by].to_numpy()
    out: dict[str, dict[str, float]] = {}
    for level in pd.unique(levels):
        mask = levels == level if level is not None else pd.isna(levels)
        if mask.sum() < 100:  # skip tiny segments
            continue
        out[str(level)] = {
            "n": int(mask.sum()),
            "wape_pct": round(wape(y_true_tnd[mask], y_pred_tnd[mask]), 2),
        }
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--parquet",
        default="artifacts/cache/sample_5M_seed42.parquet",
        help="Cached training sample.",
    )
    ap.add_argument(
        "--out-dir", default="artifacts/reports",
        help="Where to drop the JSON reports.",
    )
    ap.add_argument(
        "--split", choices=("hotel_wise", "time_wise", "both"), default="both",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    parquet = Path(args.parquet)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading %s", parquet)
    df = pq.read_table(parquet).to_pandas()
    log.info("loaded: %d rows, %d cols", len(df), df.shape[1])

    y_log = np.log(df[FORECASTER_TARGET].to_numpy(dtype=np.float64))

    splits_to_run: list[tuple[str, SplitIndices]] = []
    if args.split in ("hotel_wise", "both"):
        log.info("computing hotel_wise split")
        splits_to_run.append(
            ("hotel_wise",
             hotel_wise_split(df["hotel_name_normalized"], seed=args.seed))
        )
    if args.split in ("time_wise", "both"):
        log.info("computing time_wise split")
        splits_to_run.append(("time_wise", time_wise_split(df["scraped_at"])))

    today = date.today().isoformat()

    for split_name, idx in splits_to_run:
        train_i, val_i, test_i = idx["train"], idx["val"], idx["test"]
        # Use test as the hold-out (val is reserved for hyperparam tuning later).
        df_train, df_eval = df.iloc[train_i], df.iloc[test_i]
        y_train, y_eval = y_log[train_i], y_log[test_i]

        log.info(
            "%s: train=%d val=%d test=%d (eval on test)",
            split_name, len(train_i), len(val_i), len(test_i),
        )

        baseline_results: list[dict[str, Any]] = []
        per_segment: dict[str, dict[str, dict[str, float]]] = {}
        for name, baseline in all_baselines().items():
            log.info("  → %s", name)
            res = _eval_one(name, baseline, df_train, y_train, df_eval, y_eval)
            baseline_results.append(res)
            log.info(
                "     mae=%.0f TND  mape=%.2f%%  wape=%.2f%%",
                res["mae_tnd"], res["mape_pct"], res["wape_pct"],
            )
            # Segment breakdown: only for the best baseline (lowest WAPE so far).
            y_pred_tnd = np.exp(baseline.predict(df_eval))
            y_true_tnd = np.exp(y_eval)
            per_segment[name] = {
                "by_macro_region": _per_segment_wape(df_eval, y_true_tnd, y_pred_tnd, "macro_region"),
                "by_stars_band": _per_segment_wape(df_eval, y_true_tnd, y_pred_tnd, "stars_band"),
            }

        report = {
            "date": today,
            "parquet": str(parquet),
            "split": split_name,
            "seed": args.seed,
            "n_train": int(len(train_i)),
            "n_val": int(len(val_i)),
            "n_test": int(len(test_i)),
            "target": FORECASTER_TARGET,
            "target_transform": "log",
            "metric_scale": "TND (raw price after exp)",
            "baselines": baseline_results,
            "per_segment_wape": per_segment,
        }

        out_path = out_dir / f"baselines_{today}_{split_name}.json"
        out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        log.info("wrote %s", out_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
