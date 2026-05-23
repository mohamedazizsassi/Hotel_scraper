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
    total = len(s["train"]) + len(s["val"]) + len(s["cal"]) + len(s["test"])
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
    # val and cal each get half of val_frac (0.05 each).
    groups = pd.Series([f"h_{i:04d}" for i in range(1000) for _ in range(5)])
    s = hotel_wise_split(groups, test_frac=0.2, val_frac=0.1, seed=42)
    n = len(groups)
    assert abs(len(s["test"]) / n - 0.20) < 0.05
    assert abs((len(s["val"]) + len(s["cal"])) / n - 0.10) < 0.05
    assert abs(len(s["val"]) / n - 0.05) < 0.03
    assert abs(len(s["cal"]) / n - 0.05) < 0.03


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
    assert s_at.iloc[s["train"]].max() <= s_at.iloc[s["val"]].min()
    assert s_at.iloc[s["val"]].max()   <= s_at.iloc[s["cal"]].min()
    assert s_at.iloc[s["cal"]].max()   <= s_at.iloc[s["test"]].min()


def test_time_wise_covers_all_rows():
    s_at = _scraped_at(1000)
    s = time_wise_split(s_at, test_frac=0.2, val_frac=0.1)
    total = len(s["train"]) + len(s["val"]) + len(s["cal"]) + len(s["test"])
    assert total == len(s_at)


def test_time_wise_proportions():
    s_at = _scraped_at(1000)
    s = time_wise_split(s_at, test_frac=0.2, val_frac=0.1)
    assert abs(len(s["test"]) / 1000 - 0.20) < 0.01
    assert abs((len(s["val"]) + len(s["cal"])) / 1000 - 0.10) < 0.01


def test_hotel_wise_split_returns_four_buckets():
    groups = pd.Series([f"h{i:03d}" for i in range(500) for _ in range(10)])
    idx = hotel_wise_split(groups, seed=42)
    assert set(idx.keys()) == {"train", "val", "cal", "test"}
    total = sum(len(idx[k]) for k in idx)
    assert total == len(groups)
    # disjoint
    all_idx = np.concatenate([idx[k] for k in idx])
    assert len(np.unique(all_idx)) == total


def test_hotel_wise_split_val_and_cal_share_no_hotel():
    groups = pd.Series([f"h{i:03d}" for i in range(500) for _ in range(10)])
    idx = hotel_wise_split(groups, seed=42)
    val_hotels = set(groups.iloc[idx["val"]])
    cal_hotels = set(groups.iloc[idx["cal"]])
    test_hotels = set(groups.iloc[idx["test"]])
    train_hotels = set(groups.iloc[idx["train"]])
    assert val_hotels.isdisjoint(cal_hotels)
    assert val_hotels.isdisjoint(test_hotels)
    assert cal_hotels.isdisjoint(test_hotels)
    assert train_hotels.isdisjoint(val_hotels | cal_hotels | test_hotels)


def test_time_wise_split_cal_strictly_between_val_and_test():
    n = 10000
    t = pd.Series(pd.date_range("2026-01-01", periods=n, freq="h"))
    idx = time_wise_split(t)
    assert t.iloc[idx["val"]].max()  < t.iloc[idx["cal"]].min()
    assert t.iloc[idx["cal"]].max()  < t.iloc[idx["test"]].min()
    assert t.iloc[idx["train"]].max() < t.iloc[idx["val"]].min()
