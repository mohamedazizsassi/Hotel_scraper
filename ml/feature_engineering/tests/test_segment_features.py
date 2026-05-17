"""Tests for segment_features.add_segment_features."""

from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering.config import CITY_TO_MACRO_REGION, MACRO_REGIONS, STARS_BANDS
from feature_engineering.segment_features import add_segment_features


def _df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_mapping_values_are_valid_regions() -> None:
    """Every value in CITY_TO_MACRO_REGION is a declared macro region."""
    assert set(CITY_TO_MACRO_REGION.values()) <= set(MACRO_REGIONS)


def test_known_city_known_stars_yields_expected_segment() -> None:
    df = _df([
        {"city_name": "sousse",   "stars_int": 4},
        {"city_name": "djerba",   "stars_int": 5},
        {"city_name": "hammamet", "stars_int": 3},
        {"city_name": "tozeur",   "stars_int": 2},
    ])
    out = add_segment_features(df)
    assert out["macro_region"].tolist() == ["sahel", "djerba", "cap_bon", "sud"]
    assert out["stars_band"].tolist() == ["4", "5", "3", "low"]
    assert out["market_segment_id"].tolist() == [
        "sahel_4", "djerba_5", "cap_bon_3", "sud_low",
    ]


def test_unmapped_city_raises() -> None:
    df = _df([{"city_name": "narnia", "stars_int": 4}])
    with pytest.raises(RuntimeError, match="unmapped city_name"):
        add_segment_features(df)


def test_missing_required_column_raises() -> None:
    df = _df([{"stars_int": 4}])
    with pytest.raises(RuntimeError, match="missing required column 'city_name'"):
        add_segment_features(df)


def test_out_of_range_stars_yields_na_segment() -> None:
    df = _df([{"city_name": "sousse", "stars_int": 7}])
    out = add_segment_features(df)
    assert pd.isna(out["stars_band"].iloc[0])
    assert pd.isna(out["market_segment_id"].iloc[0])


def test_stars_bands_constants_complete() -> None:
    assert set(STARS_BANDS) == {"low", "3", "4", "5"}
