"""
Post-pipeline feature validation.

The pipeline stages each enforce their own contract on entry, but a
fully-assembled feature frame can still hide regressions: a leakage bug
in stage 7 that the unit tests miss, a range drift after a scraper
schema change, an unmapped boarding category creeping back in. This
module is the catch-all gate run **before** any artifact is written.

``validate_features(df)`` returns a ``ValidationReport`` and also writes
it to ``ml/artifacts/reports/validation_{timestamp}.json``. The CLI in
stage 11 exits non-zero on a failed report and aborts before parquet /
postgres writes — silent corruption is never written downstream.

Checks (in order):

1. **Leakage audit** — sample up to ``sample_size`` rows and, for each
   peer aggregate, recompute the slice median with the row's own hotel
   excluded; compare against the stored value. For each sur_demande
   rate, recompute the same way. Any mismatch is a failure.

2. **Range checks** — every row must satisfy
   ``days_until_checkin ∈ [0, 365]`` and
   ``price_per_night ∈ [30, 20000]``. ``|delta_vs_peer_*_median_pct| >
   500`` rows are flagged as warnings (not failures — extreme spreads
   are possible in this market).

3. **Coverage** — non-null rate per column logged. Columns below
   ``COVERAGE_WARN_RATE`` produce warnings.
   ``boarding_canonical == "UNKNOWN"`` above
   ``UNKNOWN_BOARDING_FAIL_RATE`` is a failure.

4. **Distributions** — log percentiles of the `delta_vs_peer_*` columns
   and (boarding_canonical, stars_int) cell counts; flag cells with
   fewer than ``MIN_CELL_ROWS`` as warnings (sparse modelling slices).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .competitive_features import GRANULARITIES, PEER_GROUP_KEYS
from .config import (
    DAYS_UNTIL_CHECKIN_MAX,
    DAYS_UNTIL_CHECKIN_MIN,
    PRICE_PER_NIGHT_MAX,
    PRICE_PER_NIGHT_MIN,
    UNKNOWN_BOARDING_FAIL_RATE,
)
from .demand_features import SUR_DEMANDE_SLICES

logger = logging.getLogger(__name__)


COVERAGE_WARN_RATE: float = 0.90
DELTA_PCT_FLAG_ABS: float = 500.0
MIN_CELL_ROWS: int = 100
LEAKAGE_TOLERANCE: float = 1e-6
LEAKAGE_DEFAULT_SAMPLE: int = 1_000
DEFAULT_REPORTS_DIR: Path = Path("ml/artifacts/reports")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    timestamp: str
    n_rows: int
    n_columns: int
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    coverage: dict[str, float] = field(default_factory=dict)
    cell_counts_below_min: dict[str, int] = field(default_factory=dict)
    leakage_sample_size: int = 0
    leakage_mismatches: int = 0

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["passed"] = self.passed
        return d

    def to_json_file(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_features(
    df: pd.DataFrame,
    *,
    sample_size: int = LEAKAGE_DEFAULT_SAMPLE,
    reports_dir: Path | None = DEFAULT_REPORTS_DIR,
    random_state: int = 42,
) -> ValidationReport:
    """
    Run the full validation suite and return a ``ValidationReport``.

    Parameters
    ----------
    df:
        Fully-assembled feature frame (post-stage-8).
    sample_size:
        Cap on rows used for the leakage audit (it is O(rows × slices)
        per granularity, so capping is essential at production scale).
    reports_dir:
        If non-None, the report is written to
        ``{reports_dir}/validation_{timestamp}.json``.
    random_state:
        Seed for the leakage-audit row sample.

    Returns
    -------
    ValidationReport
        ``report.passed`` is False if any check produced a failure.
        Warnings do not affect ``passed``.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report = ValidationReport(
        timestamp=ts,
        n_rows=len(df),
        n_columns=df.shape[1],
    )

    if df.empty:
        report.failures.append("validate_features: empty DataFrame")
        _persist(report, reports_dir)
        return report

    _check_ranges(df, report)
    _check_coverage(df, report)
    _check_unknown_boarding(df, report)
    _check_delta_pct_outliers(df, report)
    _check_cell_counts(df, report)
    _check_leakage(df, report, sample_size=sample_size, random_state=random_state)

    _persist(report, reports_dir)
    _log_report(report)
    return report


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_ranges(df: pd.DataFrame, report: ValidationReport) -> None:
    if "days_until_checkin" in df.columns:
        d = pd.to_numeric(df["days_until_checkin"], errors="coerce")
        bad = d.dropna().pipe(
            lambda s: ((s < DAYS_UNTIL_CHECKIN_MIN) | (s > DAYS_UNTIL_CHECKIN_MAX)).sum()
        )
        if bad:
            report.failures.append(
                f"days_until_checkin: {int(bad)} rows outside "
                f"[{DAYS_UNTIL_CHECKIN_MIN}, {DAYS_UNTIL_CHECKIN_MAX}]"
            )

    if "price_per_night" in df.columns:
        p = pd.to_numeric(df["price_per_night"], errors="coerce")
        bad = p.dropna().pipe(
            lambda s: ((s < PRICE_PER_NIGHT_MIN) | (s > PRICE_PER_NIGHT_MAX)).sum()
        )
        if bad:
            report.failures.append(
                f"price_per_night: {int(bad)} rows outside "
                f"[{PRICE_PER_NIGHT_MIN}, {PRICE_PER_NIGHT_MAX}]"
            )


def _check_coverage(df: pd.DataFrame, report: ValidationReport) -> None:
    for col in df.columns:
        rate = float(df[col].notna().mean())
        report.coverage[col] = rate
        if rate < COVERAGE_WARN_RATE:
            report.warnings.append(
                f"coverage[{col}]: non_null_rate={rate:.4f} < {COVERAGE_WARN_RATE}"
            )


def _check_unknown_boarding(df: pd.DataFrame, report: ValidationReport) -> None:
    if "boarding_canonical" not in df.columns:
        return
    rate = float((df["boarding_canonical"] == "UNKNOWN").mean())
    if rate > UNKNOWN_BOARDING_FAIL_RATE:
        report.failures.append(
            f"boarding_canonical UNKNOWN rate {rate:.4f} > "
            f"{UNKNOWN_BOARDING_FAIL_RATE}"
        )


def _check_delta_pct_outliers(df: pd.DataFrame, report: ValidationReport) -> None:
    for g in GRANULARITIES:
        col = f"delta_vs_peer_{g}_median_pct"
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        flagged = int((s.abs() > DELTA_PCT_FLAG_ABS).sum())
        if flagged:
            report.warnings.append(
                f"{col}: {flagged} rows with |value| > {DELTA_PCT_FLAG_ABS}"
            )


def _check_cell_counts(df: pd.DataFrame, report: ValidationReport) -> None:
    if not {"boarding_canonical", "stars_int"}.issubset(df.columns):
        return
    cells = df.groupby(
        ["boarding_canonical", "stars_int"], dropna=False, observed=True
    ).size()
    sparse = cells[cells < MIN_CELL_ROWS]
    if not sparse.empty:
        report.cell_counts_below_min = {
            f"{b}|{s}": int(n) for (b, s), n in sparse.items()
        }
        report.warnings.append(
            f"{len(sparse)} (boarding, stars) cells below {MIN_CELL_ROWS} rows"
        )


def _check_leakage(
    df: pd.DataFrame,
    report: ValidationReport,
    *,
    sample_size: int,
    random_state: int,
) -> None:
    """
    Recompute peer aggregates for a sample of rows and compare against
    the stored values. Any deviation beyond ``LEAKAGE_TOLERANCE`` is a
    failure: it means the row's own hotel was not excluded.
    """
    n = min(sample_size, len(df))
    if n == 0:
        return
    sample_idx = df.sample(n=n, random_state=random_state).index
    report.leakage_sample_size = n

    mismatches = 0
    examples: list[str] = []

    for granularity in GRANULARITIES:
        median_col = f"peer_{granularity}_median"
        if median_col not in df.columns:
            continue
        keys = list(PEER_GROUP_KEYS[granularity])

        for idx in sample_idx:
            row = df.loc[idx]
            stored = row[median_col]
            expected = _brute_force_peer_median(df, row, keys)
            if pd.isna(stored) and pd.isna(expected):
                continue
            if pd.isna(stored) or pd.isna(expected):
                mismatches += 1
                if len(examples) < 5:
                    examples.append(
                        f"{median_col}@{idx}: stored={stored} expected={expected}"
                    )
                continue
            if abs(float(stored) - float(expected)) > LEAKAGE_TOLERANCE:
                mismatches += 1
                if len(examples) < 5:
                    examples.append(
                        f"{median_col}@{idx}: stored={stored} expected={expected}"
                    )

    for col, keys in SUR_DEMANDE_SLICES.items():
        if col not in df.columns or "sur_demande" not in df.columns:
            continue
        for idx in sample_idx:
            row = df.loc[idx]
            stored = row[col]
            expected = _brute_force_sur_demande_rate(df, row, list(keys))
            if pd.isna(stored) and pd.isna(expected):
                continue
            if pd.isna(stored) or pd.isna(expected):
                mismatches += 1
                if len(examples) < 5:
                    examples.append(
                        f"{col}@{idx}: stored={stored} expected={expected}"
                    )
                continue
            if abs(float(stored) - float(expected)) > LEAKAGE_TOLERANCE:
                mismatches += 1
                if len(examples) < 5:
                    examples.append(
                        f"{col}@{idx}: stored={stored} expected={expected}"
                    )

    report.leakage_mismatches = mismatches
    if mismatches:
        report.failures.append(
            f"leakage: {mismatches} aggregate(s) disagree with brute-force "
            f"recompute (sample={n}). Examples: {examples}"
        )


# ---------------------------------------------------------------------------
# Brute-force recomputation helpers
# ---------------------------------------------------------------------------

def _slice_mask(df: pd.DataFrame, row: pd.Series, keys: list[str]) -> pd.Series:
    mask = pd.Series(True, index=df.index)
    for k in keys:
        ref = row[k]
        col = df[k]
        if pd.isna(ref):
            mask &= col.isna()
        else:
            mask &= col.eq(ref)
    return mask


def _brute_force_peer_median(
    df: pd.DataFrame, row: pd.Series, keys: list[str],
) -> float:
    mask = _slice_mask(df, row, keys)
    mask &= df["hotel_name_normalized"] != row["hotel_name_normalized"]
    peers = pd.to_numeric(df.loc[mask, "price_per_night"], errors="coerce").dropna()
    return float(np.median(peers)) if len(peers) else float("nan")


def _brute_force_sur_demande_rate(
    df: pd.DataFrame, row: pd.Series, keys: list[str],
) -> float:
    mask = _slice_mask(df, row, keys)
    mask &= df["hotel_name_normalized"] != row["hotel_name_normalized"]
    sd = df.loc[mask, "sur_demande"]
    sd = pd.to_numeric(sd.astype("Float64"), errors="coerce").dropna()
    return float(sd.mean()) if len(sd) else float("nan")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _persist(report: ValidationReport, reports_dir: Path | None) -> None:
    if reports_dir is None:
        return
    out = reports_dir / f"validation_{report.timestamp}.json"
    report.to_json_file(out)
    logger.info("validation report written to %s", out)


def _log_report(report: ValidationReport) -> None:
    if report.passed:
        logger.info(
            "validation PASSED  rows=%d  warnings=%d",
            report.n_rows, len(report.warnings),
        )
    else:
        logger.error(
            "validation FAILED  rows=%d  failures=%d  warnings=%d",
            report.n_rows, len(report.failures), len(report.warnings),
        )
    for f in report.failures:
        logger.error("FAIL: %s", f)
    for w in report.warnings:
        logger.warning("WARN: %s", w)
