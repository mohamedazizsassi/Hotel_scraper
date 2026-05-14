"""
Unit tests for ``feature_engineering.demand_features``.

Same self-exclusion contract as ``competitive_features``: a row's own
hotel is removed from every slice before computing rates and counts.
"""
from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering.demand_features import add_demand_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _row(*, hotel: str, sur: bool, **overrides) -> dict:
    base = {
        "hotel_name_normalized": hotel,
        "city_name": "hammamet",
        "stars_int": 4,
        "boarding_canonical": "HDP",
        "check_in": pd.Timestamp("2026-07-01"),
        # scrape_date is part of SUR_DEMANDE_SLICES and
        # ACTIVITY_COUNT_KEYS; default all rows to one day so the
        # existing single-slice assertions still hold.
        "scrape_date": pd.Timestamp("2026-05-14"),
        "sur_demande": sur,
    }
    base.update(overrides)
    return base


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["sur_demande"] = df["sur_demande"].astype("boolean")
    return df


# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------

def test_missing_required_columns_raises() -> None:
    with pytest.raises(RuntimeError, match="missing required columns"):
        add_demand_features(pd.DataFrame({"sur_demande": [True]}))


# ---------------------------------------------------------------------------
# Hand-computed rates with self-exclusion
# ---------------------------------------------------------------------------

def test_sur_demande_rate_excludes_self() -> None:
    df = _df([
        _row(hotel="A", sur=True),
        _row(hotel="B", sur=False),
        _row(hotel="C", sur=False),
        _row(hotel="D", sur=True),
    ])
    out = add_demand_features(df)
    # Row A's peers: B(F), C(F), D(T) → rate = 1/3.
    # Row B's peers: A(T), C(F), D(T) → rate = 2/3.
    assert out.loc[0, "sur_demande_rate_city_checkin"] == pytest.approx(1/3)
    assert out.loc[1, "sur_demande_rate_city_checkin"] == pytest.approx(2/3)


def test_changing_own_sur_demande_does_not_change_own_rate() -> None:
    """Leakage canary."""
    base = _df([
        _row(hotel="A", sur=False),
        _row(hotel="B", sur=False),
        _row(hotel="C", sur=True),
    ])
    rate_before = float(add_demand_features(base).loc[0, "sur_demande_rate_city_checkin"])

    perturbed = base.copy()
    perturbed.loc[0, "sur_demande"] = True
    perturbed["sur_demande"] = perturbed["sur_demande"].astype("boolean")
    rate_after = float(add_demand_features(perturbed).loc[0, "sur_demande_rate_city_checkin"])

    assert rate_before == rate_after


def test_self_exclusion_is_per_hotel_not_per_row() -> None:
    """
    Hotel A has two rows in the slice. When computing the rate for
    either A row, BOTH A rows are excluded from numerator and
    denominator.
    """
    df = _df([
        _row(hotel="A", sur=True),
        _row(hotel="A", sur=True),    # variant of A
        _row(hotel="B", sur=False),
        _row(hotel="C", sur=False),
    ])
    out = add_demand_features(df)
    # For both A rows: peers = {B(F), C(F)} → rate = 0.0, count = 2.
    a = out[out["hotel_name_normalized"] == "A"]
    assert a["sur_demande_rate_city_checkin"].tolist() == [0.0, 0.0]
    assert a["city_activity_count_checkin"].tolist() == [2, 2]


# ---------------------------------------------------------------------------
# Slice independence
# ---------------------------------------------------------------------------

def test_finer_slice_uses_smaller_peer_set() -> None:
    """
    Two different stars buckets in the same city/check_in. The
    coarser city+check_in rate aggregates across both; the finer
    city+stars+check_in rate stays inside one.
    """
    df = _df([
        _row(hotel="A", stars_int=3, sur=True),
        _row(hotel="B", stars_int=3, sur=False),
        _row(hotel="C", stars_int=4, sur=True),
        _row(hotel="D", stars_int=4, sur=True),
    ])
    out = add_demand_features(df)
    # Row A (3*): coarse peers = {B, C, D} → 2/3. Fine peers = {B} → 0/1.
    assert out.loc[0, "sur_demande_rate_city_checkin"] == pytest.approx(2/3)
    assert out.loc[0, "sur_demande_rate_city_stars_checkin"] == pytest.approx(0.0)


def test_no_peers_yields_nan_rate_and_zero_count() -> None:
    df = _df([_row(hotel="A", sur=True)])  # solo
    out = add_demand_features(df)
    assert pd.isna(out.loc[0, "sur_demande_rate_city_checkin"])
    assert int(out.loc[0, "city_activity_count_checkin"]) == 0


def test_different_check_in_dates_are_independent_slices() -> None:
    df = _df([
        _row(hotel="A", sur=True,  check_in=pd.Timestamp("2026-07-01")),
        _row(hotel="B", sur=False, check_in=pd.Timestamp("2026-08-01")),
    ])
    out = add_demand_features(df)
    # Each row alone in its slice → NaN rate, zero count.
    assert out["sur_demande_rate_city_checkin"].isna().all()
    assert out["city_activity_count_checkin"].tolist() == [0, 0]


# ---------------------------------------------------------------------------
# city_activity_count_checkin
# ---------------------------------------------------------------------------

def test_city_activity_count_is_other_hotel_row_count() -> None:
    df = _df([
        _row(hotel="A", sur=True),
        _row(hotel="A", sur=False),    # second A row
        _row(hotel="B", sur=False),
        _row(hotel="C", sur=True),
    ])
    out = add_demand_features(df)
    # Both A rows: count = rows in slice (4) - A's own row count (2) = 2.
    # B row: 4 - 1 = 3. C row: 4 - 1 = 3.
    assert out["city_activity_count_checkin"].astype(int).tolist() == [2, 2, 3, 3]


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------

def test_nan_sur_demande_dropped_from_rate_calculation() -> None:
    df = _df([
        _row(hotel="A", sur=True),
        _row(hotel="B", sur=True),
        _row(hotel="C", sur=False),
        _row(hotel="D", sur=False),
    ])
    df.loc[1, "sur_demande"] = pd.NA
    df["sur_demande"] = df["sur_demande"].astype("boolean")

    out = add_demand_features(df)
    # Row A's peers: B(NA dropped), C(F), D(F) → rate = 0/2 = 0.0
    assert out.loc[0, "sur_demande_rate_city_checkin"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Non-mutation
# ---------------------------------------------------------------------------

def test_does_not_mutate_input() -> None:
    df = _df([
        _row(hotel="A", sur=True),
        _row(hotel="B", sur=False),
    ])
    snapshot = df.copy(deep=True)
    add_demand_features(df)
    pd.testing.assert_frame_equal(df, snapshot)
