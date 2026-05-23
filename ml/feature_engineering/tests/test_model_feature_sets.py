"""
Tests for model_feature_sets.py — feature partition, per-model selectors,
storage projections, and leakage guard.
"""
from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering import model_feature_sets as mfs
from feature_engineering.assemble import assemble_features


# ---------------------------------------------------------------------------
# Toy raw frame suitable for assemble_features
# ---------------------------------------------------------------------------

def _toy_raw_frame() -> pd.DataFrame:
    """Minimal raw frame that passes all assemble_features stages."""
    check_in = pd.Timestamp("2026-07-01")
    scraped_at = pd.Timestamp("2026-05-14T10:00:00+00:00")
    rows = []
    # 3 groups × 3 hotels = 9 rows — enough for non-empty peer slices
    for city, stars in [("hammamet", 3), ("hammamet", 4), ("sousse", 3)]:
        for i in range(3):
            rows.append({
                "source": "tunisiepromo",
                "scraped_at": scraped_at,
                "scrape_run_id": "run_test",
                "check_in": check_in,
                "check_out": check_in + pd.Timedelta(days=3),
                "nights": 3,
                "days_until_checkin": 48,
                "city_id": 1,
                "city_name": city,
                "adults": 2,
                "children": 0,
                "hotel_name": f"Hotel {city} {i}",
                "hotel_name_normalized": f"{city}_{stars}_h{i}",
                "stars": str(stars),
                "boarding_name": "demi pension",
                "room_name": "Chambre Double",
                "price": float(100 + i * 10) * 3,
                "price_per_night": float(100 + i * 10),
                "sur_demande": False,
                "supplements": [],
            })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def assembled_df() -> pd.DataFrame:
    """Run assemble_features once per test module."""
    return assemble_features(_toy_raw_frame())


# ---------------------------------------------------------------------------
# Test 1: four categories are pairwise disjoint
# ---------------------------------------------------------------------------

def test_categories_pairwise_disjoint():
    categories = {
        "IDENTIFIER_COLUMNS":      set(mfs.IDENTIFIER_COLUMNS),
        "TARGETS":                  set(mfs.TARGETS),
        "LEAKY_OBSERVED_FEATURES":  set(mfs.LEAKY_OBSERVED_FEATURES),
        "PREDICTIVE_FEATURES":      set(mfs.PREDICTIVE_FEATURES),
    }
    names = list(categories)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlap = categories[a] & categories[b]
            assert not overlap, (
                f"{a} and {b} share columns: {sorted(overlap)}"
            )


# ---------------------------------------------------------------------------
# Test 2: union of four categories equals assembled frame columns
# ---------------------------------------------------------------------------

def test_categories_union_covers_assembled_columns(assembled_df):
    all_defined = (
        set(mfs.IDENTIFIER_COLUMNS)
        | set(mfs.TARGETS)
        | set(mfs.LEAKY_OBSERVED_FEATURES)
        | set(mfs.PREDICTIVE_FEATURES)
    )
    actual = set(assembled_df.columns)
    missing_from_manifest = actual - all_defined
    missing_from_frame = all_defined - actual
    assert not missing_from_manifest, (
        f"Columns in assembled frame not in manifest: {sorted(missing_from_manifest)}"
    )
    assert not missing_from_frame, (
        f"Columns in manifest not in assembled frame: {sorted(missing_from_frame)}"
    )


# ---------------------------------------------------------------------------
# Test 3: FORECASTER_FEATURES is disjoint from TARGETS ∪ LEAKY_OBSERVED_FEATURES
# ---------------------------------------------------------------------------

def test_forecaster_features_disjoint_from_targets_and_leaky():
    forbidden = set(mfs.TARGETS) | set(mfs.LEAKY_OBSERVED_FEATURES)
    overlap = set(mfs.FORECASTER_FEATURES) & forbidden
    assert not overlap, (
        f"FORECASTER_FEATURES contains leaky/target columns: {sorted(overlap)}"
    )


# ---------------------------------------------------------------------------
# Test 4: assert_no_leakage raises for forecaster on bad cols, no-op elsewhere
# ---------------------------------------------------------------------------

def test_assert_no_leakage_raises_for_forecaster_on_leaky_col():
    leaky_col = mfs.LEAKY_OBSERVED_FEATURES[0]
    with pytest.raises(ValueError, match="leaky"):
        mfs.assert_no_leakage([leaky_col], model="forecaster")


def test_assert_no_leakage_raises_for_forecaster_on_target_col():
    with pytest.raises(ValueError, match="leaky"):
        mfs.assert_no_leakage(["price_per_night"], model="forecaster")


def test_assert_no_leakage_noop_for_anomaly():
    leaky_col = mfs.LEAKY_OBSERVED_FEATURES[0]
    mfs.assert_no_leakage([leaky_col], model="anomaly")  # must not raise


def test_assert_no_leakage_noop_for_recommender():
    mfs.assert_no_leakage(list(mfs.ANOMALY_FEATURES), model="recommender")  # must not raise


def test_assert_no_leakage_silent_on_clean_forecaster_cols():
    mfs.assert_no_leakage(list(mfs.FORECASTER_FEATURES), model="forecaster")  # must not raise


# ---------------------------------------------------------------------------
# Forecaster target constants reflect the locked decision (2026-05-20):
# log(price) with `nights` as a feature, NOT log1p(price_per_night).
# ---------------------------------------------------------------------------

def test_forecaster_target_is_price():
    assert mfs.FORECASTER_TARGET == "price"
    assert mfs.FORECASTER_TARGET_TRANSFORM == "log"


def test_nights_is_a_forecaster_feature():
    assert "nights" in mfs.FORECASTER_FEATURES


# ---------------------------------------------------------------------------
# Test 5: POSTGRES_SERVING_COLUMNS ⊆ assembled frame columns
# ---------------------------------------------------------------------------

def test_postgres_serving_columns_subset_of_assembled(assembled_df):
    actual = set(assembled_df.columns)
    missing = set(mfs.POSTGRES_SERVING_COLUMNS) - actual
    assert not missing, (
        f"POSTGRES_SERVING_COLUMNS references columns absent from assembled frame: "
        f"{sorted(missing)}"
    )


# ---------------------------------------------------------------------------
# Test 6: storage coverage — POSTGRES_SERVING ∪ CALENDAR_DIM ∪ SEGMENT_DIM
#          covers every PREDICTIVE_FEATURE
# ---------------------------------------------------------------------------

def test_storage_covers_all_predictive_features():
    covered = (
        set(mfs.POSTGRES_SERVING_COLUMNS)
        | set(mfs.CALENDAR_DIM_COLUMNS)
        | set(mfs.SEGMENT_DIM_COLUMNS)
    )
    missing = set(mfs.PREDICTIVE_FEATURES) - covered
    assert not missing, (
        f"PREDICTIVE_FEATURES not covered by any storage partition: {sorted(missing)}"
    )
