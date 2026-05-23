"""
Train / validation / test splitters for the price forecaster.

Two splits, per locked decision (2026-05-20):
    - hotel-wise (primary): a hotel appears in exactly one of {train, val, test}.
      Deterministic hash of hotel_name_normalized — same seed → same split.
    - time-wise (secondary): cut by scraped_at quantiles. Train is strictly
      earlier in time than val, val strictly earlier than test.

Random row-wise splits are forbidden (within-hotel leakage).
"""
from __future__ import annotations

import hashlib
from typing import TypedDict

import numpy as np
import pandas as pd


class SplitIndices(TypedDict):
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def _hash_unit(s: str, seed: int) -> float:
    """Deterministic map of a string to [0, 1). Uses MD5 (Python hash() is salted)."""
    digest = hashlib.md5(f"{seed}:{s}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def hotel_wise_split(
    groups: pd.Series,
    test_frac: float = 0.20,
    val_frac: float = 0.10,
    seed: int = 42,
) -> SplitIndices:
    """
    Partition rows by group (hotel) so no hotel appears in more than one bucket.

    Parameters
    ----------
    groups:
        Per-row group label (typically hotel_name_normalized). Length = n_rows.
    test_frac, val_frac:
        Target fraction of HOTELS (not rows) assigned to test / val. Train gets
        the remainder. Realised row fractions will be approximate.
    seed:
        Determines the hash. Same seed + same groups → byte-identical split.

    Returns
    -------
    SplitIndices: integer-position arrays into `groups`.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    if test_frac + val_frac >= 1.0:
        raise ValueError("test_frac + val_frac must be < 1")

    unique_hotels = groups.dropna().unique()
    hotel_to_h = {h: _hash_unit(str(h), seed) for h in unique_hotels}
    # Bucket boundaries on the unit interval.
    test_hi = test_frac
    val_hi = test_frac + val_frac

    bucket = groups.map(lambda h: hotel_to_h.get(h, np.nan))
    train_mask = bucket >= val_hi
    val_mask = (bucket >= test_hi) & (bucket < val_hi)
    test_mask = bucket < test_hi

    # Rows with NaN group (shouldn't happen on the cleaned frame) go to train
    # silently — but warn via assertion that no NaN groups exist.
    assert groups.notna().all(), "hotel_wise_split: groups contains NaN"

    idx = np.arange(len(groups))
    return SplitIndices(
        train=idx[train_mask.to_numpy()],
        val=idx[val_mask.to_numpy()],
        test=idx[test_mask.to_numpy()],
    )


def time_wise_split(
    scraped_at: pd.Series,
    test_frac: float = 0.20,
    val_frac: float = 0.10,
) -> SplitIndices:
    """
    Partition rows by time: train < val < test on `scraped_at` quantiles.

    Parameters
    ----------
    scraped_at:
        Per-row timestamp. Tz-aware is fine.
    test_frac, val_frac:
        Tail fractions assigned to test (latest) and val (just before test).

    Returns
    -------
    SplitIndices.
    """
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")
    if not 0.0 <= val_frac < 1.0:
        raise ValueError(f"val_frac must be in [0, 1), got {val_frac}")
    if test_frac + val_frac >= 1.0:
        raise ValueError("test_frac + val_frac must be < 1")
    assert scraped_at.notna().all(), "time_wise_split: scraped_at contains NaT"

    n = len(scraped_at)
    order = np.argsort(scraped_at.to_numpy(), kind="stable")
    cut_val = int(round(n * (1.0 - test_frac - val_frac)))
    cut_test = int(round(n * (1.0 - test_frac)))
    return SplitIndices(
        train=np.sort(order[:cut_val]),
        val=np.sort(order[cut_val:cut_test]),
        test=np.sort(order[cut_test:]),
    )
