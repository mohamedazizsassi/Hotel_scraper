"""
Sanity checks for the constants in ``feature_engineering.config``.

The config module is mostly declarative data, so these tests guard the
*shape* of the data — closed-set membership, no overlaps, ordered
buckets, etc. — rather than its values. A typo here would silently break
downstream stages, so an inexpensive shape-test is worth the cost.
"""
from __future__ import annotations

from datetime import date

import pytest

from feature_engineering import config


# ---------------------------------------------------------------------------
# Boarding taxonomy
# ---------------------------------------------------------------------------

def test_boarding_canonical_map_values_are_in_closed_set() -> None:
    canonical_codes = set(config.BOARDING_CANONICAL_VALUES) - {"UNKNOWN"}
    map_values = set(config.BOARDING_CANONICAL_MAP.values())
    assert map_values.issubset(canonical_codes), (
        f"Map produces values outside the closed set: {map_values - canonical_codes}"
    )


def test_boarding_canonical_keys_are_lowercased_and_stripped() -> None:
    for raw in config.BOARDING_CANONICAL_MAP:
        assert raw == raw.lower().strip(), (
            f"BOARDING_CANONICAL_MAP key {raw!r} must be lower+strip "
            "to match the lookup applied in taxonomy.py"
        )


def test_unknown_boarding_fail_rate_is_a_fraction() -> None:
    assert 0.0 < config.UNKNOWN_BOARDING_FAIL_RATE < 1.0


# ---------------------------------------------------------------------------
# Room patterns
# ---------------------------------------------------------------------------

def test_room_pattern_dicts_are_non_empty() -> None:
    for name in (
        "ROOM_BASE_PATTERNS",
        "ROOM_VIEW_PATTERNS",
        "ROOM_TIER_PATTERNS",
        "ROOM_OCCUPANCY_PATTERNS",
    ):
        d = getattr(config, name)
        assert isinstance(d, dict) and d, f"{name} is empty"


def test_room_base_coverage_warn_is_a_fraction() -> None:
    assert 0.0 < config.ROOM_BASE_COVERAGE_WARN <= 1.0


# ---------------------------------------------------------------------------
# Cleaning thresholds
# ---------------------------------------------------------------------------

def test_price_thresholds_are_ordered() -> None:
    assert config.PRICE_PER_NIGHT_MIN > 0
    assert config.PRICE_PER_NIGHT_MIN < config.PRICE_PER_NIGHT_MAX


def test_days_until_checkin_thresholds_are_ordered() -> None:
    assert 0 <= config.DAYS_UNTIL_CHECKIN_MIN < config.DAYS_UNTIL_CHECKIN_MAX


def test_nights_allowed_is_non_empty_subset_of_small_ints() -> None:
    assert config.NIGHTS_ALLOWED
    assert all(isinstance(n, int) and 1 <= n <= 30 for n in config.NIGHTS_ALLOWED)


# ---------------------------------------------------------------------------
# Booking-window buckets
# ---------------------------------------------------------------------------

def test_booking_window_buckets_are_contiguous() -> None:
    """Buckets should cover [0, ∞) with no gaps and no overlaps."""
    buckets = config.BOOKING_WINDOW_BUCKETS
    assert buckets[0][1] == 0, "first bucket must start at 0"
    for (_, _, hi_prev), (_, lo_next, _) in zip(buckets, buckets[1:]):
        assert lo_next == hi_prev + 1, (
            f"bucket gap/overlap between hi={hi_prev} and lo={lo_next}"
        )


def test_booking_window_bucket_labels_unique() -> None:
    labels = [b[0] for b in config.BOOKING_WINDOW_BUCKETS]
    assert len(labels) == len(set(labels))


# ---------------------------------------------------------------------------
# Calendar tables
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "RAMADAN_PERIODS",
    "TUNISIA_SCHOOL_HOLIDAYS",
    "FRANCE_SCHOOL_HOLIDAYS",
    "GERMANY_SCHOOL_HOLIDAYS",
    "UK_SCHOOL_HOLIDAYS",
])
def test_holiday_ranges_are_chronologically_ordered(name: str) -> None:
    table = getattr(config, name)
    for start, end in table:
        assert isinstance(start, date) and isinstance(end, date)
        assert start <= end, f"{name}: range {start}..{end} is reversed"


def test_tunisia_public_holidays_fixed_keys_are_valid_dates() -> None:
    for (m, d) in config.TUNISIA_PUBLIC_HOLIDAYS_FIXED:
        # Validate by constructing a date in any leap-friendly year.
        date(2026, m, d)


def test_tunisia_islamic_holidays_are_within_horizon() -> None:
    # Hardcoded for 2026–2027; if the list extends, update this test.
    for d in config.TUNISIA_ISLAMIC_HOLIDAYS:
        assert 2026 <= d.year <= 2027


def test_eu_school_holiday_tables_non_empty() -> None:
    for name in ("FRANCE_SCHOOL_HOLIDAYS", "GERMANY_SCHOOL_HOLIDAYS", "UK_SCHOOL_HOLIDAYS"):
        assert getattr(config, name), f"{name} is empty"
