"""Tests for hotel_wise_split and time_wise_split."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.forecasting.splits import hotel_wise_split, time_wise_split


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _groups_frame(n_hotels: int = 100, rows_per_hotel: int = 30) -> pd.Series:
    return pd.Series(
        [f"hotel_{i:03d}" for i in range(n_hotels) for _ in range(rows_per_hotel)]
    )


def _scraped_at(n: int = 1000) -> pd.Series:
    base = pd.Timestamp("2026-04-30T10:00:00+00:00")
    return pd.Series(
        [base + pd.Timedelta(minutes=i) for i in range(n)]
    ).sample(frac=1.0, random_state=7).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Hotel-wise split
# ---------------------------------------------------------------------------

def test_hotel_wise_no_hotel_appears_in_two_buckets():
    groups = _groups_frame()
    s = hotel_wise_split(groups, test_frac=0.2, val_frac=0.1, seed=42)
    train_h = set(groups.iloc[s["train"]].unique())
    val_h = set(groups.iloc[s["val"]].unique())
    test_h = set(groups.iloc[s["test"]].unique())
    assert not (train_h & val_h)
    assert not (train_h & test_h)
    assert not (val_h & test_h)


def test_hotel_wise_covers_all_rows():
    groups = _groups_frame()
    s = hotel_wise_split(groups, test_frac=0.2, val_frac=0.1, seed=42)
    total = len(s["train"]) + len(s["val"]) + len(s["test"])
    assert total == len(groups)


def test_hotel_wise_deterministic():
    groups = _groups_frame()
    a = hotel_wise_split(groups, seed=42)
    b = hotel_wise_split(groups, seed=42)
    np.testing.assert_array_equal(a["train"], b["train"])
    np.testing.assert_array_equal(a["val"], b["val"])
    np.testing.assert_array_equal(a["test"], b["test"])


def test_hotel_wise_seed_changes_split():
    groups = _groups_frame()
    a = hotel_wise_split(groups, seed=42)
    b = hotel_wise_split(groups, seed=7)
    # at least one bucket must differ
    assert not (
        np.array_equal(a["train"], b["train"])
        and np.array_equal(a["val"], b["val"])
        and np.array_equal(a["test"], b["test"])
    )


def test_hotel_wise_proportions_within_tolerance():
    # 1000 hotels, evenly hashed → should be close to target fractions.
    groups = pd.Series([f"h_{i:04d}" for i in range(1000) for _ in range(5)])
    s = hotel_wise_split(groups, test_frac=0.2, val_frac=0.1, seed=42)
    n = len(groups)
    assert abs(len(s["test"]) / n - 0.20) < 0.05
    assert abs(len(s["val"]) / n - 0.10) < 0.05


def test_hotel_wise_rejects_bad_fractions():
    groups = _groups_frame()
    with pytest.raises(ValueError):
        hotel_wise_split(groups, test_frac=0.8, val_frac=0.3)
    with pytest.raises(ValueError):
        hotel_wise_split(groups, test_frac=0.0)


# ---------------------------------------------------------------------------
# Time-wise split
# ---------------------------------------------------------------------------

def test_time_wise_strict_ordering():
    s_at = _scraped_at(1000)
    s = time_wise_split(s_at, test_frac=0.2, val_frac=0.1)
    t_train = s_at.iloc[s["train"]]
    t_val = s_at.iloc[s["val"]]
    t_test = s_at.iloc[s["test"]]
    assert t_train.max() <= t_val.min()
    assert t_val.max() <= t_test.min()


def test_time_wise_covers_all_rows():
    s_at = _scraped_at(1000)
    s = time_wise_split(s_at, test_frac=0.2, val_frac=0.1)
    total = len(s["train"]) + len(s["val"]) + len(s["test"])
    assert total == len(s_at)


def test_time_wise_proportions():
    s_at = _scraped_at(1000)
    s = time_wise_split(s_at, test_frac=0.2, val_frac=0.1)
    assert abs(len(s["test"]) / 1000 - 0.20) < 0.01
    assert abs(len(s["val"]) / 1000 - 0.10) < 0.01
