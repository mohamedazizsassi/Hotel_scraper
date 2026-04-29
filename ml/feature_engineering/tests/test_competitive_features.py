"""
Unit tests for ``feature_engineering.competitive_features``.

Toy DataFrames are constructed with a single peer slice so we can predict
each row's peer aggregate by hand. The leakage-critical assertion —
"changing my own price does not change my own peer_median" — is
exercised explicitly here and again from the cross-cutting
``test_leakage.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering.competitive_features import (
    GRANULARITIES,
    MIN_PEERS_FOR_BEST,
    add_competitive_features,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _row(*, hotel: str, price: float, **overrides) -> dict:
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
        "price_per_night": float(price),
    }
    base.update(overrides)
    return base


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------

def test_missing_required_columns_raises() -> None:
    with pytest.raises(RuntimeError, match="missing required columns"):
        add_competitive_features(pd.DataFrame({"price_per_night": [100.0]}))


# ---------------------------------------------------------------------------
# Three-hotel single-slice — hand-computed expected medians
# ---------------------------------------------------------------------------

def test_three_hotels_in_same_slice_peer_median_excludes_self() -> None:
    df = _df([
        _row(hotel="A", price=100.0),
        _row(hotel="B", price=150.0),
        _row(hotel="C", price=200.0),
    ])
    out = add_competitive_features(df)

    # For row A, peers = {B=150, C=200} → median 175.
    # For row B, peers = {A=100, C=200} → median 150.
    # For row C, peers = {A=100, B=150} → median 125.
    assert out["peer_tight_median"].tolist() == [175.0, 150.0, 125.0]
    assert out["peer_medium_median"].tolist() == [175.0, 150.0, 125.0]
    assert out["peer_loose_median"].tolist() == [175.0, 150.0, 125.0]


def test_peer_count_is_other_hotel_count() -> None:
    df = _df([
        _row(hotel="A", price=100.0),
        _row(hotel="B", price=150.0),
        _row(hotel="C", price=200.0),
    ])
    out = add_competitive_features(df)
    assert out["peer_tight_count"].tolist() == [2, 2, 2]


def test_changing_own_price_does_not_change_own_peer_median() -> None:
    """Leakage canary — own price must not contaminate own aggregate."""
    base = _df([
        _row(hotel="A", price=100.0),
        _row(hotel="B", price=150.0),
        _row(hotel="C", price=200.0),
    ])
    base_out = add_competitive_features(base)
    a_median_before = float(base_out.loc[base_out["hotel_name_normalized"] == "A",
                                         "peer_tight_median"].iloc[0])

    perturbed = base.copy()
    perturbed.loc[perturbed["hotel_name_normalized"] == "A", "price_per_night"] = 9_999.0
    perturbed_out = add_competitive_features(perturbed)
    a_median_after = float(perturbed_out.loc[perturbed_out["hotel_name_normalized"] == "A",
                                             "peer_tight_median"].iloc[0])

    assert a_median_before == a_median_after


# ---------------------------------------------------------------------------
# Multiple rows per hotel (e.g. base + variants from supplement_expansion)
# ---------------------------------------------------------------------------

def test_self_exclusion_is_per_hotel_not_per_row() -> None:
    """
    Hotel A has two rows in the slice (base + variant). When computing
    peer_median for either A row, BOTH A rows must be excluded — peers
    are OTHER hotels, not other rows of the same hotel.
    """
    df = _df([
        _row(hotel="A", price=100.0),
        _row(hotel="A", price=120.0),  # variant of A
        _row(hotel="B", price=200.0),
        _row(hotel="C", price=300.0),
    ])
    out = add_competitive_features(df)

    # Both A rows: peers = {B=200, C=300} → median 250, count 2.
    a_rows = out[out["hotel_name_normalized"] == "A"]
    assert a_rows["peer_tight_median"].tolist() == [250.0, 250.0]
    assert a_rows["peer_tight_count"].tolist() == [2, 2]


# ---------------------------------------------------------------------------
# Stat shapes
# ---------------------------------------------------------------------------

def test_peer_std_nan_when_only_one_peer() -> None:
    df = _df([
        _row(hotel="A", price=100.0),
        _row(hotel="B", price=200.0),
    ])
    out = add_competitive_features(df)
    # Each row has exactly 1 peer → std undefined.
    assert out["peer_tight_std"].isna().all()


def test_zero_peers_produce_nan_aggregates() -> None:
    """A solo hotel in its slice has count=0 and NaN aggregates."""
    df = _df([_row(hotel="A", price=100.0)])
    out = add_competitive_features(df)
    assert out["peer_tight_count"].iloc[0] == 0
    for col in ("peer_tight_median", "peer_tight_p25", "peer_tight_p75",
                "peer_tight_min", "peer_tight_max", "peer_tight_std",
                "delta_vs_peer_tight_median_pct", "rank_in_peer_tight"):
        assert pd.isna(out[col].iloc[0])


def test_delta_pct_sign_and_magnitude() -> None:
    df = _df([
        _row(hotel="A", price=100.0),
        _row(hotel="B", price=100.0),
        _row(hotel="C", price=200.0),
    ])
    out = add_competitive_features(df)
    # Row A: peers {100, 200} median 150 → delta = (100-150)/150*100 = -33.33.
    assert out.loc[0, "delta_vs_peer_tight_median_pct"] == pytest.approx(-100/3, abs=1e-6)


def test_rank_excludes_self() -> None:
    df = _df([
        _row(hotel="A", price=50.0),
        _row(hotel="B", price=100.0),
        _row(hotel="C", price=200.0),
    ])
    out = add_competitive_features(df)
    # Row A (50): peers {100, 200} → 0 below, 0 equal → rank 0.0
    # Row C (200): peers {50, 100} → 2 below → rank 1.0
    assert out.loc[0, "rank_in_peer_tight"] == pytest.approx(0.0)
    assert out.loc[2, "rank_in_peer_tight"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Granularity differences — tight vs medium vs loose
# ---------------------------------------------------------------------------

def test_tight_excludes_rooms_with_different_view_medium_keeps_them() -> None:
    """
    Hotels A/B/C share city/stars/boarding/nights/adults/check_in (so
    they're in the same MEDIUM peer group), but only A and B have
    room_view=mer. C has room_view=jardin → C is in a different TIGHT
    peer group.
    """
    df = _df([
        _row(hotel="A", price=100.0, room_view="mer"),
        _row(hotel="B", price=200.0, room_view="mer"),
        _row(hotel="C", price=300.0, room_view="jardin"),
    ])
    out = add_competitive_features(df)

    # Row A's tight peers = {B} (just B, with view=mer). Medium = {B, C}.
    assert out.loc[0, "peer_tight_count"] == 1
    assert out.loc[0, "peer_tight_median"] == 200.0
    assert out.loc[0, "peer_medium_count"] == 2
    assert out.loc[0, "peer_medium_median"] == 250.0


# ---------------------------------------------------------------------------
# best_peer_granularity_used selector
# ---------------------------------------------------------------------------

def test_best_granularity_falls_through_when_tight_too_sparse() -> None:
    """
    Tight slice has only 1 peer (< MIN_PEERS_FOR_BEST). Medium has
    enough peers. Selector should pick "medium".
    """
    rows = [_row(hotel="A", price=100.0, room_view="mer"),
            _row(hotel="B", price=110.0, room_view="mer")]
    # Add 6 more hotels with room_view=jardin (different tight slice but
    # same medium slice).
    for i, h in enumerate(["C", "D", "E", "F", "G", "H"]):
        rows.append(_row(hotel=h, price=120.0 + i, room_view="jardin"))
    out = add_competitive_features(_df(rows))

    a_row = out[out["hotel_name_normalized"] == "A"].iloc[0]
    assert int(a_row["peer_tight_count"]) == 1
    assert int(a_row["peer_medium_count"]) >= MIN_PEERS_FOR_BEST
    assert a_row["best_peer_granularity_used"] == "medium"


def test_best_granularity_picks_tight_when_populated() -> None:
    rows = [_row(hotel=h, price=100.0 + i, room_view="mer")
            for i, h in enumerate(["A", "B", "C", "D", "E", "F"])]
    out = add_competitive_features(_df(rows))
    # Each row has 5 peers in tight → tight qualifies.
    assert (out["best_peer_granularity_used"] == "tight").all()


def test_best_granularity_na_when_no_slice_qualifies() -> None:
    df = _df([_row(hotel="A", price=100.0)])  # solo
    out = add_competitive_features(df)
    assert pd.isna(out.iloc[0]["best_peer_granularity_used"])


# ---------------------------------------------------------------------------
# Different peer slices isolate from each other
# ---------------------------------------------------------------------------

def test_different_check_in_dates_are_independent_slices() -> None:
    df = _df([
        _row(hotel="A", price=100.0, check_in=pd.Timestamp("2026-07-01")),
        _row(hotel="B", price=999.0, check_in=pd.Timestamp("2026-08-01")),  # different date
    ])
    out = add_competitive_features(df)
    # Each row is alone in its slice → 0 peers.
    assert out["peer_tight_count"].tolist() == [0, 0]
    assert out["peer_medium_count"].tolist() == [0, 0]


# ---------------------------------------------------------------------------
# Non-mutation
# ---------------------------------------------------------------------------

def test_does_not_mutate_input() -> None:
    df = _df([
        _row(hotel="A", price=100.0),
        _row(hotel="B", price=200.0),
    ])
    snapshot = df.copy(deep=True)
    add_competitive_features(df)
    pd.testing.assert_frame_equal(df, snapshot)
