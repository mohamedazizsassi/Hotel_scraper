"""
Unit tests for ``feature_engineering.validators``.

Build a small post-stage-8 frame, run each individual check, and verify
the report contents.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from feature_engineering.competitive_features import add_competitive_features
from feature_engineering.demand_features import add_demand_features
from feature_engineering.validators import (
    DELTA_PCT_FLAG_ABS,
    MIN_CELL_ROWS,
    ValidationReport,
    validate_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _row(*, hotel: str, price: float, sur: bool = False, **overrides) -> dict:
    base = {
        "hotel_name_normalized": hotel,
        "city_name": "hammamet",
        "stars_int": 4,
        "boarding_canonical": "HDP",
        "room_base": "chambre",
        "room_view": "mer",
        "nights": 3,
        "adults": 2,
        "check_in": pd.Timestamp("2026-07-01"),
        # scrape_date is part of PEER_GROUP_KEYS and SUR_DEMANDE_SLICES.
        "scrape_date": pd.Timestamp("2026-05-14"),
        "price_per_night": float(price),
        "days_until_checkin": 30,
        "sur_demande": sur,
    }
    base.update(overrides)
    return base


def _pipeline(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["sur_demande"] = df["sur_demande"].astype("boolean")
    df = add_competitive_features(df)
    df = add_demand_features(df)
    return df


def _good_frame() -> pd.DataFrame:
    return _pipeline([
        _row(hotel="A", price=100.0, sur=False),
        _row(hotel="B", price=150.0, sur=True),
        _row(hotel="C", price=200.0, sur=False),
        _row(hotel="D", price=180.0, sur=False),
        _row(hotel="E", price=220.0, sur=True),
    ])


# ---------------------------------------------------------------------------
# Empty / structural
# ---------------------------------------------------------------------------

def test_empty_frame_fails() -> None:
    report = validate_features(pd.DataFrame(), reports_dir=None)
    assert not report.passed
    assert any("empty" in f for f in report.failures)


def test_good_frame_passes_no_failures(tmp_path: Path) -> None:
    df = _good_frame()
    report = validate_features(df, reports_dir=None)
    assert report.passed, f"unexpected failures: {report.failures}"
    assert report.leakage_mismatches == 0


# ---------------------------------------------------------------------------
# Range checks
# ---------------------------------------------------------------------------

def test_out_of_range_days_until_checkin_fails() -> None:
    df = _good_frame()
    df.loc[0, "days_until_checkin"] = 999  # > 365
    report = validate_features(df, reports_dir=None)
    assert not report.passed
    assert any("days_until_checkin" in f for f in report.failures)


def test_out_of_range_price_per_night_fails() -> None:
    df = _good_frame()
    df.loc[0, "price_per_night"] = 1.0  # below 30
    report = validate_features(df, reports_dir=None)
    assert not report.passed
    assert any("price_per_night" in f for f in report.failures)


# ---------------------------------------------------------------------------
# Unknown boarding
# ---------------------------------------------------------------------------

def test_unknown_boarding_above_threshold_fails() -> None:
    df = _good_frame()
    # Set 50 % to UNKNOWN.
    df.loc[df.index[: len(df) // 2], "boarding_canonical"] = "UNKNOWN"
    report = validate_features(df, reports_dir=None)
    assert not report.passed
    assert any("UNKNOWN rate" in f for f in report.failures)


# ---------------------------------------------------------------------------
# Delta-pct outliers (warnings, not failures)
# ---------------------------------------------------------------------------

def test_delta_pct_outliers_emit_warnings() -> None:
    df = _good_frame()
    df.loc[0, "delta_vs_peer_medium_median_pct"] = DELTA_PCT_FLAG_ABS + 1.0
    report = validate_features(df, reports_dir=None)
    assert report.passed, "outlier delta should warn, not fail"
    assert any("delta_vs_peer_medium_median_pct" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------

def test_low_coverage_emits_warning() -> None:
    df = _good_frame()
    df.loc[df.index[:4], "room_view"] = pd.NA  # 80 % null
    report = validate_features(df, reports_dir=None)
    assert any("coverage[room_view]" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Sparse cells
# ---------------------------------------------------------------------------

def test_sparse_cells_flagged_as_warning() -> None:
    # Each (boarding, stars) cell here has ~5 rows, well under MIN_CELL_ROWS.
    df = _good_frame()
    report = validate_features(df, reports_dir=None)
    assert report.cell_counts_below_min  # at least one cell flagged
    assert any(f"below {MIN_CELL_ROWS} rows" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------

def test_leakage_audit_catches_corrupted_aggregate() -> None:
    """
    Manually overwrite a peer aggregate with a value that includes the
    row's own price; the brute-force recompute should disagree and
    surface a failure.
    """
    df = _good_frame()
    # Force row 0's peer_medium_median to the with-self median (includes own).
    keys = ["city_name", "stars_int", "boarding_canonical",
            "nights", "adults", "check_in"]
    own_row = df.loc[0]
    same_slice = df
    for k in keys:
        same_slice = same_slice[same_slice[k] == own_row[k]]
    with_self = float(same_slice["price_per_night"].median())
    df.loc[0, "peer_medium_median"] = with_self

    report = validate_features(df, sample_size=len(df), reports_dir=None)
    # If the with-self median differs from the without-self median for
    # row 0 (which is our setup since prices vary), this should fail.
    assert not report.passed
    assert any("leakage" in f for f in report.failures)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def test_report_written_to_disk(tmp_path: Path) -> None:
    df = _good_frame()
    report = validate_features(df, reports_dir=tmp_path)
    files = list(tmp_path.glob("validation_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["passed"] is True
    assert payload["n_rows"] == len(df)


def test_report_passed_property() -> None:
    r = ValidationReport(timestamp="x", n_rows=1, n_columns=1)
    assert r.passed is True
    r.failures.append("oops")
    assert r.passed is False
