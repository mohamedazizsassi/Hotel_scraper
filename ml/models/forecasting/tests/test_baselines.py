"""Smoke tests for the four baselines on a hand-built toy frame."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.forecasting.baselines import (
    CompetitorMedianBaseline,
    GlobalMedianBaseline,
    GroupMedianBaseline,
    LinearHedonicBaseline,
    all_baselines,
)


# ---------------------------------------------------------------------------
# Toy frame: 24 rows × all columns the baselines need
# ---------------------------------------------------------------------------

def _toy_frame(n: int = 24) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(0)
    cities = ["hammamet", "sousse", "djerba"]
    boardings = ["BB", "HDP", "AI"]
    rooms_base = ["chambre", "suite", "bungalow"]
    rows = []
    for i in range(n):
        city = cities[i % 3]
        stars = (i % 3) + 3   # 3..5
        boarding = boardings[i % 3]
        nights = (i % 3) + 1
        days = (i * 5) % 90
        price = float(50 + (stars - 3) * 80 + (i % 7) * 12)
        rows.append({
            "city_name": city,
            "stars_int": stars,
            "boarding_canonical": boarding,
            "nights": nights,
            "days_until_checkin": days,
            "adults": 2,
            "children": 0,
            "is_supplement_variant": False,
            "view_upgrade_count_offered": 0,
            "has_free_view_upgrade": False,
            "room_base": rooms_base[i % 3],
            "room_view": None,
            "room_tier": "standard",
            "room_occupancy": "double",
            "best_peer_granularity_used": "medium",
            "macro_region": "north",
            "stars_band": "mid",
            "market_segment_id": f"{city}_{stars}",
            "check_in_dow": i % 7,
            "check_in_month": (i % 12) + 1,
            "check_in_week_of_year": (i % 52) + 1,
            "check_in_day_of_month": (i % 28) + 1,
            "check_in_quarter": (i % 4) + 1,
            "is_weekend_checkin": (i % 7) >= 5,
            "is_ramadan": False,
            "is_tunisia_public_holiday": False,
            "is_tunisia_school_holiday": False,
            "is_school_holiday_france": False,
            "is_school_holiday_germany": False,
            "is_school_holiday_uk": False,
            "days_to_nearest_european_holiday": 30,
            "peer_medium_count": 10,
            "peer_medium_median": price * 1.02,
            "peer_medium_p25": price * 0.9,
            "peer_medium_p75": price * 1.1,
            "peer_medium_std": price * 0.05,
            "peer_loose_count": 30,
            "peer_loose_median": price * 1.0,
            "peer_tight_count": 3,
            "peer_tight_median": price * 1.05,
            "sur_demande_rate_city_checkin": 0.1,
            "sur_demande_rate_city_stars_checkin": 0.05,
            "sur_demande_rate_city_stars_boarding_checkin": 0.03,
            "city_activity_count_checkin": 100,
            "price": price,
        })
    df = pd.DataFrame(rows)
    y = np.log(df["price"].to_numpy())
    return df, y


# ---------------------------------------------------------------------------
# Per-baseline tests
# ---------------------------------------------------------------------------

def test_global_median_constant_output():
    df, y = _toy_frame()
    b = GlobalMedianBaseline().fit(df, y)
    p = b.predict(df)
    assert p.shape == (len(df),)
    assert np.allclose(p, np.median(y))


def test_group_median_returns_finite_predictions():
    df, y = _toy_frame()
    b = GroupMedianBaseline().fit(df, y)
    p = b.predict(df)
    assert p.shape == (len(df),)
    assert np.isfinite(p).all()


def test_group_median_unseen_group_falls_back():
    df_train, y_train = _toy_frame()
    b = GroupMedianBaseline().fit(df_train, y_train)

    # Test row with city/stars unseen at the narrowest level.
    df_test = df_train.head(1).copy()
    df_test["city_name"] = "totally_new_city"
    df_test["stars_int"] = 4  # known star count → falls back to ("stars_int",) level

    p = b.predict(df_test)
    assert np.isfinite(p).all()


def test_competitor_median_predicts_log_peer_median():
    df, y = _toy_frame()
    b = CompetitorMedianBaseline().fit(df, y)
    p = b.predict(df)
    expected = np.log(df["peer_medium_median"].to_numpy())
    np.testing.assert_allclose(p, expected, rtol=1e-9)


def test_competitor_median_falls_back_on_nan():
    df, y = _toy_frame()
    df.loc[0, "peer_medium_median"] = np.nan
    df.loc[0, "peer_loose_median"] = 200.0
    b = CompetitorMedianBaseline().fit(df, y)
    p = b.predict(df)
    assert np.isclose(p[0], np.log(200.0))


def test_competitor_median_global_fallback_when_all_peer_nan():
    df, y = _toy_frame()
    df.loc[0, "peer_medium_median"] = np.nan
    df.loc[0, "peer_loose_median"] = np.nan
    df.loc[0, "peer_tight_median"] = np.nan
    b = CompetitorMedianBaseline().fit(df, y)
    p = b.predict(df)
    assert np.isclose(p[0], np.median(y))


def test_linear_hedonic_fits_and_predicts():
    df, y = _toy_frame(n=60)
    b = LinearHedonicBaseline(max_fit_rows=None, alpha=1.0).fit(df, y)
    p = b.predict(df)
    assert p.shape == (len(df),)
    assert np.isfinite(p).all()


def test_all_baselines_registry():
    reg = all_baselines()
    assert set(reg) == {
        "global_median", "group_median", "competitor_median", "linear_hedonic"
    }
    for name, b in reg.items():
        assert hasattr(b, "fit") and hasattr(b, "predict"), name


def test_all_baselines_end_to_end():
    df, y = _toy_frame(n=60)
    for name, b in all_baselines().items():
        b.fit(df, y)
        p = b.predict(df)
        assert p.shape == (len(df),), name
        assert np.isfinite(p).all(), f"{name} produced non-finite predictions"
