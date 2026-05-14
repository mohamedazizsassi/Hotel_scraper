"""
Unit tests for ``temporal_dedup.dedup_to_most_recent_per_business_key``.

The dedup stage is the cross-sectional contract enforcer: after it runs,
every (source, hotel_name_normalized, check_in, boarding_name,
room_name, nights, adults) tuple must appear exactly once and carry the
most recent observation.
"""
from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering.temporal_dedup import (
    BUSINESS_KEY,
    dedup_to_most_recent_per_business_key,
)


def _row(**overrides) -> dict:
    defaults = {
        "source": "promohotel",
        "hotel_name_normalized": "h1",
        "check_in": pd.Timestamp("2026-07-01"),
        "boarding_name": "Demi pension",
        "room_name": "Chambre Double",
        "nights": 3,
        "adults": 2,
        "scraped_at": pd.Timestamp("2026-05-01T08:00:00Z"),
        "price_per_night": 250.0,
    }
    defaults.update(overrides)
    return defaults


def test_unique_keys_passthrough() -> None:
    """One row per business key already — output equals input row count."""
    df = pd.DataFrame([
        _row(hotel_name_normalized="h1"),
        _row(hotel_name_normalized="h2"),
        _row(hotel_name_normalized="h3"),
    ])
    out = dedup_to_most_recent_per_business_key(df)
    assert len(out) == 3


def test_duplicate_key_keeps_most_recent_scrape() -> None:
    """Same business key on three scrape dates — keep the latest."""
    df = pd.DataFrame([
        _row(scraped_at=pd.Timestamp("2026-05-01T08:00:00Z"), price_per_night=200.0),
        _row(scraped_at=pd.Timestamp("2026-05-10T08:00:00Z"), price_per_night=250.0),
        _row(scraped_at=pd.Timestamp("2026-05-05T08:00:00Z"), price_per_night=225.0),
    ])
    out = dedup_to_most_recent_per_business_key(df)
    assert len(out) == 1
    assert out.iloc[0]["price_per_night"] == pytest.approx(250.0)
    assert out.iloc[0]["scraped_at"] == pd.Timestamp("2026-05-10T08:00:00Z")


def test_different_sources_are_separate_keys() -> None:
    """Source is part of the business key — same hotel/date in two
    sources survives as two rows."""
    df = pd.DataFrame([
        _row(source="promohotel"),
        _row(source="tunisiepromo"),
    ])
    out = dedup_to_most_recent_per_business_key(df)
    assert len(out) == 2
    assert set(out["source"]) == {"promohotel", "tunisiepromo"}


def test_different_rooms_are_separate_keys() -> None:
    """Same hotel/board/check_in/nights but different room — separate keys."""
    df = pd.DataFrame([
        _row(room_name="Chambre Double"),
        _row(room_name="Chambre Double Vue Mer"),
    ])
    out = dedup_to_most_recent_per_business_key(df)
    assert len(out) == 2


def test_business_key_set_matches_module_constant() -> None:
    """Lock the business-key definition. Any change to BUSINESS_KEY must
    be intentional — this test fails first, forcing a review."""
    assert BUSINESS_KEY == (
        "source",
        "hotel_name_normalized",
        "check_in",
        "boarding_name",
        "room_name",
        "nights",
        "adults",
    )


def test_missing_required_column_fails_loudly() -> None:
    df = pd.DataFrame([{"source": "promohotel"}])  # missing everything else
    with pytest.raises(RuntimeError, match="missing columns"):
        dedup_to_most_recent_per_business_key(df)


def test_tie_on_scraped_at_resolves_deterministically() -> None:
    """Two rows with identical scraped_at must produce a stable output
    across runs — exercises mergesort stability."""
    same_ts = pd.Timestamp("2026-05-10T08:00:00Z")
    df = pd.DataFrame([
        _row(scraped_at=same_ts, price_per_night=200.0),
        _row(scraped_at=same_ts, price_per_night=250.0),
    ])
    out1 = dedup_to_most_recent_per_business_key(df.copy())
    out2 = dedup_to_most_recent_per_business_key(df.copy())
    assert len(out1) == 1
    assert out1.iloc[0]["price_per_night"] == out2.iloc[0]["price_per_night"]


def test_empty_input_passthrough() -> None:
    df = pd.DataFrame(
        columns=list(BUSINESS_KEY) + ["scraped_at", "price_per_night"],
    )
    out = dedup_to_most_recent_per_business_key(df)
    assert len(out) == 0


def test_business_key_column_values_preserved_after_dedup() -> None:
    """Column dtypes and values for the surviving row are unchanged."""
    df = pd.DataFrame([
        _row(scraped_at=pd.Timestamp("2026-05-01T08:00:00Z"), price_per_night=200.0),
        _row(scraped_at=pd.Timestamp("2026-05-10T08:00:00Z"), price_per_night=250.0),
    ])
    out = dedup_to_most_recent_per_business_key(df)
    for col in BUSINESS_KEY:
        assert out.iloc[0][col] == df.iloc[1][col]
