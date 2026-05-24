"""
CLI — score a parquet against trained quantile boosters + conformal calibrator
and the rule library, emit per-segment recommendation report.

Inputs:
  --parquet           training/test cache (defaults to the same 5M sample)
  --models-dir        directory containing q10.txt, q50.txt, q90.txt, conformal.json
  --split             which split's test rows to score (hotel_wise | time_wise)
  --out-report        JSON report path
  --sample-per-direction  rows per direction (raise/hold/lower) in sample_recommendations
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from models.forecasting.conformal import ConformalQuantileCalibrator
from models.forecasting.lgbm_quantile import LGBMQuantileForecaster
from models.forecasting.splits import hotel_wise_split, time_wise_split
from models.recommendation.recommender import Recommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_recommender")


def _per_segment_directions(
    df: pd.DataFrame, by: str,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for level, sub in df.groupby(by, dropna=False):
        if len(sub) < 100:
            continue
        n = len(sub)
        out[str(level)] = {
            "n": int(n),
            "raise_pct": round(float((sub["direction"] == "raise").mean()) * 100, 2),
            "hold_pct":  round(float((sub["direction"] == "hold").mean())  * 100, 2),
            "lower_pct": round(float((sub["direction"] == "lower").mean()) * 100, 2),
        }
    return out


def _stratified_sample(
    df: pd.DataFrame, k_per_direction: int, seed: int,
) -> list[dict]:
    parts = []
    rng = np.random.default_rng(seed)
    for direction in ("raise", "hold", "lower"):
        sub = df[df["direction"] == direction]
        take = min(k_per_direction, len(sub))
        if take == 0:
            continue
        idx = rng.choice(len(sub), size=take, replace=False)
        parts.append(sub.iloc[idx])
    if not parts:
        return []
    sampled = pd.concat(parts, ignore_index=True)
    return sampled.to_dict(orient="records")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="artifacts/cache/sample_5M_seed42.parquet")
    ap.add_argument("--models-dir", required=True,
                    help="e.g. models/forecasting/lgbm_quantile_2026-05-23/hotel_wise")
    ap.add_argument("--split", choices=("hotel_wise", "time_wise"), required=True)
    ap.add_argument("--out-report", default=None,
                    help="Default: artifacts/reports/recommender_<date>_<split>.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sample-per-direction", type=int, default=20)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    today = date.today().isoformat()
    out_path = Path(args.out_report) if args.out_report else \
        Path("artifacts/reports") / f"recommender_{today}_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("loading boosters + calibrator from %s", models_dir)
    forecaster = LGBMQuantileForecaster.load(models_dir)
    calibrator = ConformalQuantileCalibrator.load(models_dir / "conformal.json")

    log.info("loading parquet %s", args.parquet)
    df = pq.read_table(args.parquet).to_pandas()

    # Add derived columns required by Recommender
    if "price_per_night" not in df.columns:
        df["price_per_night"] = df["price"] / df["nights"]
    if "check_in" not in df.columns:
        # Derive check_in from scraped_at and days_until_checkin
        df["scraped_at"] = pd.to_datetime(df["scraped_at"])
        df["check_in"] = df["scraped_at"] + pd.to_timedelta(df["days_until_checkin"], unit="D")

    # Convert categorical features to category dtype for forecaster
    categorical_cols = [
        "boarding_canonical", "room_base", "room_view", "room_tier", "room_occupancy",
        "best_peer_granularity_used", "macro_region", "stars_band", "market_segment_id"
    ]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    if args.split == "hotel_wise":
        idx = hotel_wise_split(df["hotel_name_normalized"], seed=args.seed)
    else:
        idx = time_wise_split(df["scraped_at"])
    test_i = idx["test"]
    log.info("scoring %d test rows", len(test_i))

    rec = Recommender(forecaster=forecaster, calibrator=calibrator)
    scored = rec.score(df, test_indices=test_i)

    n = len(scored)
    direction_counts = scored["direction"].value_counts().to_dict()
    direction_pct = {
        k: round(direction_counts.get(k, 0) / n * 100, 2)
        for k in ("raise", "hold", "lower")
    }

    by_status: dict[str, dict[str, float]] = {}
    for status, sub in scored.groupby("interval_status"):
        by_status[str(status)] = {
            "n": int(len(sub)),
            "pct": round(len(sub) / n * 100, 2),
        }

    # delta-pct stats on action rows only (raise|lower)
    action_rows = scored[scored["direction"].isin(("raise", "lower"))]
    if len(action_rows) > 0:
        deltas = action_rows["delta_pct_vs_current"].to_numpy()
        delta_q = {
            "p25": round(float(np.percentile(deltas, 25)), 2),
            "p50": round(float(np.percentile(deltas, 50)), 2),
            "p75": round(float(np.percentile(deltas, 75)), 2),
        }
    else:
        delta_q = {"p25": 0.0, "p50": 0.0, "p75": 0.0}

    log.info(
        "%s: direction_pct raise=%.2f  hold=%.2f  lower=%.2f",
        args.split, direction_pct["raise"], direction_pct["hold"], direction_pct["lower"],
    )

    report: dict[str, Any] = {
        "date": today,
        "split": args.split,
        "models_dir": str(models_dir),
        "parquet": args.parquet,
        "n_test": int(n),
        "calibrator": {
            "alpha": calibrator.alpha,
            "c_log_scale": calibrator.c_,
            "n_cal_fit": calibrator.n_cal_,
        },
        "summary": {
            "direction_counts": {k: int(direction_counts.get(k, 0))
                                 for k in ("raise", "hold", "lower")},
            "direction_pct": direction_pct,
            "by_status": by_status,
            "delta_pct_quantiles_on_action_rows": delta_q,
        },
        "per_segment": {
            "by_macro_region": _per_segment_directions(scored, "macro_region")
                if "macro_region" in scored.columns else {},
            "by_stars_band":   _per_segment_directions(scored, "stars_band")
                if "stars_band" in scored.columns else {},
        },
        "sample_recommendations": _stratified_sample(
            scored, args.sample_per_direction, args.seed,
        ),
    }
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
