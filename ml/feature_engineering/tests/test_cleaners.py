"""
Tests for ``feature_engineering.cleaners.clean``.

Covers:
    * Required-column assertion.
    * Date parsing — happy path, NaT handling, fail-loudly threshold.
    * ``days_until_checkin`` rebuild + range filter.
    * ``stars`` coercion across messy formats.
    * ``stars_int`` per-hotel modal imputation.
    * Price filters (out-of-range, null) and ``price_per_night`` recompute
      when stored vs implied diverge by > 1 TND.
    * ``nights`` whitelist filter.
    * Final dtype contract.
"""
from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering.cleaners import clean


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_row(**overrides) -> dict:
    row = {
        "source": "promohotel",
        "scraped_at": "2026-04-20T10:00:00Z",
        "check_in": "2026-05-01",
        "nights": 3,
        "stars": "4",
        "hotel_name_normalized": "hotel_x",
        "hotel_name": "Hotel X",
        "city_name": "Hammamet",
        "boarding_name": "Demi Pension",
        "room_name": "Chambre Double",
        "price": 300.0,
        "price_per_night": 100.0,
        "adults": 2,
        "children": 0,
    }
    row.update(overrides)
    return row


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Required columns
# ---------------------------------------------------------------------------

def test_missing_required_column_raises() -> None:
    df = pd.DataFrame({"source": ["promohotel"]})
    with pytest.raises(RuntimeError, match="missing required columns"):
        clean(df)


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def test_check_in_and_scraped_at_parsed() -> None:
    out = clean(_df([_base_row()]))
    assert pd.api.types.is_datetime64_any_dtype(out["check_in"])
    assert out["scraped_at"].dt.tz is not None


def test_unparseable_dates_below_threshold_dropped() -> None:
    rows = [_base_row() for _ in range(200)]
    rows.append(_base_row(check_in="not a date"))
    out = clean(_df(rows))
    assert len(out) == 200


def test_unparseable_dates_above_threshold_raises() -> None:
    rows = [_base_row() for _ in range(50)]
    rows += [_base_row(check_in="garbage") for _ in range(50)]
    with pytest.raises(RuntimeError, match="check_in parse failure rate"):
        clean(_df(rows))


# ---------------------------------------------------------------------------
# days_until_checkin
# ---------------------------------------------------------------------------

def test_days_until_checkin_rebuilt_from_dates() -> None:
    out = clean(_df([
        _base_row(scraped_at="2026-04-20T10:00:00Z", check_in="2026-04-25"),
    ]))
    assert int(out.iloc[0]["days_until_checkin"]) == 5


def test_days_until_checkin_outside_range_dropped() -> None:
    rows = [
        _base_row(check_in="2026-04-19"),    # -1 day, dropped
        _base_row(check_in="2026-04-25"),    # +5 days, kept
        _base_row(check_in="2030-01-01"),    # > 365 days, dropped
    ]
    out = clean(_df(rows))
    assert len(out) == 1
    assert int(out.iloc[0]["days_until_checkin"]) == 5


# ---------------------------------------------------------------------------
# stars coercion
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("4",          4),
    ("4.0",        4),
    ("4*",         4),
    ("4 étoiles",  4),
    ("  3   ",     3),
    ("5",          5),
])
def test_stars_int_extracts_first_digit_in_range(raw: str, expected: int) -> None:
    out = clean(_df([_base_row(stars=raw)]))
    assert int(out.iloc[0]["stars_int"]) == expected


def test_stars_int_imputed_from_hotel_mode() -> None:
    rows = [
        _base_row(hotel_name_normalized="hotel_a", stars="4"),
        _base_row(hotel_name_normalized="hotel_a", stars="4"),
        _base_row(hotel_name_normalized="hotel_a", stars=None),  # imputed → 4
    ]
    out = clean(_df(rows))
    assert len(out) == 3
    assert (out["stars_int"] == 4).all()


def test_stars_int_falls_back_to_city_mode() -> None:
    """
    A hotel with no observed stars anywhere is imputed from the city's
    modal star tier (D5 fix: tunisiepromo sparse-metadata hotels would
    otherwise be dropped).
    """
    rows = [
        _base_row(hotel_name_normalized="hotel_a", stars="4"),
        _base_row(hotel_name_normalized="hotel_b", stars="bogus"),  # city fallback -> 4
    ]
    out = clean(_df(rows))
    assert len(out) == 2
    assert (out["stars_int"] == 4).all()


def test_stars_int_unrecoverable_dropped() -> None:
    """
    A row whose hotel AND whose city have no observed stars is the
    truly-unrecoverable case and is dropped.
    """
    rows = [
        _base_row(hotel_name_normalized="hotel_a", city_name="Hammamet", stars="4"),
        _base_row(hotel_name_normalized="hotel_b", city_name="Bizerte", stars="bogus"),
    ]
    out = clean(_df(rows))
    assert len(out) == 1
    assert out.iloc[0]["hotel_name_normalized"] == "hotel_a"


# ---------------------------------------------------------------------------
# Price filters
# ---------------------------------------------------------------------------

def test_price_per_night_below_min_dropped() -> None:
    out = clean(_df([
        _base_row(price=60.0, price_per_night=20.0, nights=3),   # 20 < 30, dropped
        _base_row(price=300.0, price_per_night=100.0, nights=3),
    ]))
    assert len(out) == 1
    assert float(out.iloc[0]["price_per_night"]) == pytest.approx(100.0)


def test_price_per_night_above_max_dropped() -> None:
    out = clean(_df([
        _base_row(price=300.0, price_per_night=100.0, nights=3),
        _base_row(price=99_999.0, price_per_night=33_333.0, nights=3),  # > 20k, dropped
    ]))
    assert len(out) == 1


def test_price_per_night_recomputed_when_mismatched() -> None:
    # nights=3, price=300 → implied ppn=100, but stored=999. Mismatch >1 TND
    # → recompute to 100.
    out = clean(_df([_base_row(price=300.0, price_per_night=999.0, nights=3)]))
    assert float(out.iloc[0]["price_per_night"]) == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# nights whitelist
# ---------------------------------------------------------------------------

def test_nights_outside_whitelist_dropped() -> None:
    rows = [
        _base_row(nights=1, price=100.0, price_per_night=100.0),
        _base_row(nights=4, price=400.0, price_per_night=100.0),  # 4 not allowed
        _base_row(nights=7, price=700.0, price_per_night=100.0),
    ]
    out = clean(_df(rows))
    assert sorted(out["nights"].astype(int).tolist()) == [1, 7]


# ---------------------------------------------------------------------------
# Output contract
# ---------------------------------------------------------------------------

def test_output_dtypes() -> None:
    out = clean(_df([_base_row()]))
    assert str(out["nights"].dtype) == "Int16"
    assert str(out["stars_int"].dtype) == "Int8"
    assert str(out["days_until_checkin"].dtype) == "Int16"
    assert str(out["price"].dtype) == "float32"
    assert str(out["price_per_night"].dtype) == "float32"


def test_does_not_mutate_input() -> None:
    df = _df([_base_row()])
    snapshot = df.copy(deep=True)
    clean(df)
    pd.testing.assert_frame_equal(df, snapshot)
