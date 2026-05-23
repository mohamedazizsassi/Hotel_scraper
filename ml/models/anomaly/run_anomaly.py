"""
CLI — score a parquet against trained quantile boosters + conformal calibrator,
emit per-segment anomaly report.

Inputs:
  --parquet           training/test cache (defaults to the same 5M sample)
  --models-dir        directory containing q10.txt, q50.txt, q90.txt, conformal.json
  --split             which split's test rows to score (hotel_wise | time_wise)
  --out-report        JSON report path

Outputs:
  - <out-report>: anomaly rate (overall + per-segment), top-K extreme rows
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

from models.anomaly.detector import IntervalAnomalyDetector
from models.forecasting.conformal import ConformalQuantileCalibrator
from models.forecasting.data import prepare_xy
from models.forecasting.lgbm_quantile import LGBMQuantileForecaster
from models.forecasting.splits import hotel_wise_split, time_wise_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_anomaly")


def _per_segment_flag_rate(df: pd.DataFrame, by: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for level, sub in df.groupby(by, dropna=False):
        if len(sub) < 100:
            continue
        out[str(level)] = {
            "n": int(len(sub)),
            "flag_rate_pct":    round(float(sub["is_anomaly"].mean()) * 100, 2),
            "overpriced_pct":   round(float((sub["anomaly_score"] > 0).mean()) * 100, 2),
            "underpriced_pct":  round(float((sub["anomaly_score"] < 0).mean()) * 100, 2),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="artifacts/cache/sample_5M_seed42.parquet")
    ap.add_argument("--models-dir", required=True,
                    help="e.g. models/forecasting/lgbm_quantile_2026-05-22/hotel_wise")
    ap.add_argument("--split", choices=("hotel_wise", "time_wise"), required=True)
    ap.add_argument("--out-report", default=None,
                    help="Default: artifacts/reports/anomaly_<date>_<split>.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    today = date.today().isoformat()
    out_path = Path(args.out_report) if args.out_report else \
        Path("artifacts/reports") / f"anomaly_{today}_{args.split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("loading boosters + calibrator from %s", models_dir)
    forecaster = LGBMQuantileForecaster.load(models_dir)
    calibrator = ConformalQuantileCalibrator.load(models_dir / "conformal.json")

    log.info("loading parquet %s", args.parquet)
    df = pq.read_table(args.parquet).to_pandas()

    if args.split == "hotel_wise":
        idx = hotel_wise_split(df["hotel_name_normalized"], seed=args.seed)
    else:
        idx = time_wise_split(df["scraped_at"])
    test_i = idx["test"]
    log.info("scoring %d test rows", len(test_i))

    X, y_log, _ = prepare_xy(df)
    X_test, y_test_log = X.iloc[test_i], y_log[test_i]

    det = IntervalAnomalyDetector(forecaster=forecaster, calibrator=calibrator)
    scored = det.score(X_test, y_log=y_test_log)
    df_test_view = df.iloc[test_i].reset_index(drop=True)
    scored = scored.reset_index(drop=True)

    # report
    flag_rate = float(scored["is_anomaly"].mean())
    overpriced = float((scored["anomaly_score"] > 0).mean())
    underpriced = float((scored["anomaly_score"] < 0).mean())
    log.info(
        "%s: flag_rate=%.3f  (over=%.3f  under=%.3f)  expected ≈ 0.20",
        args.split, flag_rate, overpriced, underpriced,
    )

    # join identifying columns for the top-K view
    joined = df_test_view.assign(
        anomaly_score=scored["anomaly_score"],
        is_anomaly=scored["is_anomaly"],
        q10_cal_log=scored["q10_cal_log"],
        q90_cal_log=scored["q90_cal_log"],
    )
    top_cols = [
        "hotel_name_normalized", "city_name", "stars_int", "macro_region",
        "stars_band", "scraped_at", "check_in", "nights", "price",
        "boarding_canonical", "anomaly_score", "is_anomaly",
    ]
    top_cols = [c for c in top_cols if c in joined.columns]
    top_k = (joined.iloc[scored["anomaly_score"].abs()
                          .nlargest(args.top_k).index][top_cols]
             .to_dict(orient="records"))

    report: dict[str, Any] = {
        "date": today,
        "split": args.split,
        "models_dir": str(models_dir),
        "parquet": args.parquet,
        "n_test": int(len(test_i)),
        "calibrator": {
            "alpha": calibrator.alpha,
            "c_log_scale": calibrator.c_,
            "n_cal_fit": calibrator.n_cal_,
        },
        "overall": {
            "flag_rate_pct":   round(flag_rate * 100, 2),
            "overpriced_pct":  round(overpriced * 100, 2),
            "underpriced_pct": round(underpriced * 100, 2),
            "expected_pct":    round((1 - calibrator.alpha) * 100, 2),  # informational
        },
        "per_segment": {
            "by_macro_region": _per_segment_flag_rate(joined, "macro_region")
                if "macro_region" in joined.columns else {},
            "by_stars_band":   _per_segment_flag_rate(joined, "stars_band")
                if "stars_band" in joined.columns else {},
        },
        "top_extreme": top_k,
    }
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
