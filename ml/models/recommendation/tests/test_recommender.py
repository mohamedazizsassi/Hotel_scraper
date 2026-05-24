"""Tests for the Recommender class."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.recommendation.recommender import Recommender


class _StubForecaster:
    """Returns hand-set q10/q50/q90 in log space, aligned to the input length."""

    feature_names_: list[str] = ["x0", "x1"]

    def __init__(self, q10_log, q50_log, q90_log):
        self._q = {
            "q10": np.asarray(q10_log, dtype=float),
            "q50": np.asarray(q50_log, dtype=float),
            "q90": np.asarray(q90_log, dtype=float),
        }

    def predict(self, X):
        n = len(X)
        return {k: v[:n] for k, v in self._q.items()}


class _StubCalibrator:
    """Identity calibrator: c_ = 0."""

    c_ = 0.0
    alpha = 0.20

    def apply(self, q_lo, q_hi):
        return np.asarray(q_lo), np.asarray(q_hi)


def _toy_df(n: int = 3) -> pd.DataFrame:
    # Toy frame uses nights=1 so price == price_per_night → tests can
    # reason about both rules without juggling scales.
    prices = [800.0, 1000.0, 1500.0][:n]
    return pd.DataFrame({
        "x0": [0.0] * n,
        "x1": [0.0] * n,
        "price": prices,
        "price_per_night": prices,
        "hotel_name_normalized": [f"hotel_{i}" for i in range(n)],
        "city_name": ["tunis"] * n,
        "stars_int": [5] * n,
        "macro_region": ["tunis_nord"] * n,
        "stars_band": ["5"] * n,
        "scraped_at": pd.to_datetime(["2026-05-01"] * n),
        "check_in": pd.to_datetime(["2026-06-01"] * n),
        "nights": [1] * n,
        "adults": [2] * n,
        "boarding_canonical": ["BB"] * n,
        "peer_medium_median": [1000.0] * n,
        "peer_medium_count": [10] * n,
    })


def test_score_returns_one_row_per_test_index():
    f = _StubForecaster(
        q10_log=np.log([900.0, 900.0, 900.0]),
        q50_log=np.log([1000.0, 1000.0, 1000.0]),
        q90_log=np.log([1100.0, 1100.0, 1100.0]),
    )
    rec = Recommender(forecaster=f, calibrator=_StubCalibrator())
    df = _toy_df(3)
    out = rec.score(df, test_indices=np.array([0, 2]))
    assert len(out) == 2


def test_score_directions_match_rules():
    # row 0 price=800 below q10=900 → raise
    # row 1 price=1000 in band → hold
    # row 2 price=1500 above q90=1100 → lower
    f = _StubForecaster(
        q10_log=np.log([900.0, 900.0, 900.0]),
        q50_log=np.log([1000.0, 1000.0, 1000.0]),
        q90_log=np.log([1100.0, 1100.0, 1100.0]),
    )
    rec = Recommender(forecaster=f, calibrator=_StubCalibrator())
    out = rec.score(_toy_df(3), test_indices=np.array([0, 1, 2]))
    assert list(out["direction"]) == ["raise", "hold", "lower"]
    assert list(out["interval_status"]) == ["below_band", "in_band", "above_band"]


def test_score_recommended_prices_are_positive_tnd():
    f = _StubForecaster(
        q10_log=np.log([900.0, 900.0, 900.0]),
        q50_log=np.log([1000.0, 1000.0, 1000.0]),
        q90_log=np.log([1100.0, 1100.0, 1100.0]),
    )
    rec = Recommender(forecaster=f, calibrator=_StubCalibrator())
    out = rec.score(_toy_df(3), test_indices=np.array([0, 1, 2]))
    assert (out["recommended_price_tnd"] > 0).all()
    assert (out["q10_cal_tnd"] > 0).all()
    assert (out["q50_tnd"] > 0).all()
    assert (out["q90_cal_tnd"] > 0).all()


def test_score_delta_pct_vs_current_is_zero_when_hold():
    f = _StubForecaster(
        q10_log=np.log([900.0, 900.0, 900.0]),
        q50_log=np.log([1000.0, 1000.0, 1000.0]),
        q90_log=np.log([1100.0, 1100.0, 1100.0]),
    )
    rec = Recommender(forecaster=f, calibrator=_StubCalibrator())
    out = rec.score(_toy_df(3), test_indices=np.array([1]))
    assert out["direction"].iloc[0] == "hold"
    assert out["delta_pct_vs_current"].iloc[0] == pytest.approx(0.0)


def test_score_delta_pct_vs_current_matches_recommendation():
    # row 0 price=800 → recommended q50=1000 → delta = +25%
    f = _StubForecaster(
        q10_log=np.log([900.0, 900.0, 900.0]),
        q50_log=np.log([1000.0, 1000.0, 1000.0]),
        q90_log=np.log([1100.0, 1100.0, 1100.0]),
    )
    rec = Recommender(forecaster=f, calibrator=_StubCalibrator())
    out = rec.score(_toy_df(3), test_indices=np.array([0]))
    assert out["delta_pct_vs_current"].iloc[0] == pytest.approx(25.0, abs=0.5)


def test_score_required_columns_present():
    f = _StubForecaster(
        q10_log=np.log([900.0, 900.0, 900.0]),
        q50_log=np.log([1000.0, 1000.0, 1000.0]),
        q90_log=np.log([1100.0, 1100.0, 1100.0]),
    )
    rec = Recommender(forecaster=f, calibrator=_StubCalibrator())
    out = rec.score(_toy_df(3), test_indices=np.array([0, 1, 2]))
    expected = {
        "hotel_name_normalized", "city_name", "stars_int", "macro_region",
        "stars_band", "scraped_at", "check_in", "nights", "adults",
        "boarding_canonical",
        "current_price_tnd",
        "q10_cal_tnd", "q50_tnd", "q90_cal_tnd",
        "interval_status", "direction", "recommended_price_tnd",
        "delta_pct_vs_current",
        "peer_medium_median", "peer_medium_count",
        "reasons",
    }
    assert expected.issubset(set(out.columns))


def test_score_reasons_is_list_of_strings_length_1_to_3():
    f = _StubForecaster(
        q10_log=np.log([900.0, 900.0, 900.0]),
        q50_log=np.log([1000.0, 1000.0, 1000.0]),
        q90_log=np.log([1100.0, 1100.0, 1100.0]),
    )
    rec = Recommender(forecaster=f, calibrator=_StubCalibrator())
    out = rec.score(_toy_df(3), test_indices=np.array([0, 1, 2]))
    for r in out["reasons"]:
        assert isinstance(r, list)
        assert 1 <= len(r) <= 3
        assert all(isinstance(s, str) for s in r)


def test_score_raises_on_missing_required_column():
    f = _StubForecaster(
        q10_log=np.log([900.0]), q50_log=np.log([1000.0]), q90_log=np.log([1100.0]),
    )
    rec = Recommender(forecaster=f, calibrator=_StubCalibrator())
    df = _toy_df(1).drop(columns=["peer_medium_median"])
    with pytest.raises(ValueError, match="peer_medium_median"):
        rec.score(df, test_indices=np.array([0]))
