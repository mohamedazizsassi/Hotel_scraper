"""
Forecaster data layer — load the frozen training parquet, build (X, y, groups)
for splitters and models.

Target convention (locked 2026-05-20):
    y = log(price), with `nights` as a feature.
    NOT log1p(price_per_night) — see ml/CLAUDE.md "Locked decisions".
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from feature_engineering.model_feature_sets import (
    FORECASTER_FEATURES,
    FORECASTER_TARGET,
)

logger = logging.getLogger(__name__)

# String-typed columns inside FORECASTER_FEATURES that need categorical
# encoding (LightGBM/CatBoost native, get_dummies for OLS baseline).
# Booleans and small-int calendar codes are left as numeric — GBTs handle them.
_CATEGORICAL_FEATURES: tuple[str, ...] = (
    "boarding_canonical",
    "room_base",
    "room_view",
    "room_tier",
    "room_occupancy",
    "best_peer_granularity_used",
    "macro_region",
    "stars_band",
    "market_segment_id",
)

# Booking-window buckets from ml/CLAUDE.md domain vocabulary.
# Buckets are interpretability aid; raw `days_until_checkin` is still a feature.
_BOOKING_WINDOW_EDGES: tuple[tuple[int, int, str], ...] = (
    (0, 1, "D0-1"),
    (2, 7, "D2-7"),
    (8, 14, "D8-14"),
    (15, 30, "D15-30"),
    (31, 60, "D31-60"),
    (61, 90, "D61-90"),
    (91, 10_000, "D90+"),
)


def categorical_feature_names() -> list[str]:
    """Subset of FORECASTER_FEATURES that should be treated as categorical."""
    return list(_CATEGORICAL_FEATURES)


def booking_window_bucket(days: pd.Series) -> pd.Series:
    """
    Map `days_until_checkin` (int) to a coarse bucket label.

    Used for baseline group-medians and per-segment reporting. Not for the
    GBT (raw integer remains the model feature).
    """
    out = pd.Series(pd.NA, index=days.index, dtype="string")
    for lo, hi, label in _BOOKING_WINDOW_EDGES:
        mask = (days >= lo) & (days <= hi)
        out = out.mask(mask, label)
    return out


def load_training_frame(
    parquet_path: str | Path,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """
    Read the frozen training snapshot (full table).

    For a memory-bounded subset use ``sample_training_frame`` instead.

    Parameters
    ----------
    parquet_path:
        Path to ml/artifacts/features_YYYY-MM-DD.parquet.
    columns:
        Optional projection. None = all 77 cols.
    """
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Training parquet not found: {path}")

    cols = list(columns) if columns is not None else None
    df = pq.read_table(path, columns=cols).to_pandas()
    logger.info(
        "Loaded %s: %d rows, %d cols", path.name, len(df), df.shape[1]
    )
    return df


def sample_training_frame(
    parquet_path: str | Path,
    n_rows: int,
    columns: Sequence[str] | None = None,
    seed: int = 42,
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Reservoir-sample ``n_rows`` from the parquet without loading the full file.

    Streams from disk via DuckDB — peak memory ≈ result size, not source size.
    Use for smoke tests, EDA, and fast model iteration. Real training reads
    the full table via ``load_training_frame``.

    Parameters
    ----------
    cache_path:
        If set and the file exists, read it directly (seconds instead of
        minutes). If set and missing, sample then write the result. Caller
        is responsible for choosing distinct cache_paths for distinct
        (n_rows, seed, columns) combinations — no key validation is done.
    """
    if cache_path is not None:
        cp = Path(cache_path)
        if cp.exists():
            df = pq.read_table(cp).to_pandas()
            logger.info(
                "Loaded cached sample %s: %d rows, %d cols",
                cp.name, len(df), df.shape[1],
            )
            return df

    import duckdb  # local import to keep prod path dep-free

    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Training parquet not found: {path}")

    col_sql = "*" if columns is None else ", ".join(f'"{c}"' for c in columns)
    conn = duckdb.connect()
    try:
        conn.execute(f"SELECT setseed({seed / 2**31})")
        df = conn.execute(
            f"SELECT {col_sql} FROM read_parquet(?) "
            f"USING SAMPLE {int(n_rows)} ROWS (reservoir)",
            [str(path)],
        ).df()
    finally:
        conn.close()
    logger.info(
        "Sampled %s: %d rows, %d cols (target %d)",
        path.name, len(df), df.shape[1], n_rows,
    )

    if cache_path is not None:
        cp = Path(cache_path)
        cp.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cp, compression="snappy", index=False)
        logger.info("Cached sample to %s", cp)

    return df


def prepare_xy(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """
    Project to forecaster features, build log-price target, return groups.

    Returns
    -------
    X:
        DataFrame with exactly FORECASTER_FEATURES columns. Categorical
        string columns are cast to pandas "category" dtype. Other columns
        are left as-is.
    y:
        np.ndarray of log(price). Length == len(df).
    groups:
        pd.Series of hotel_name_normalized aligned to X.index, used by the
        hotel-wise splitter.
    """
    missing = [c for c in FORECASTER_FEATURES if c not in df.columns]
    if missing:
        raise RuntimeError(f"prepare_xy: missing forecaster features: {missing}")
    if FORECASTER_TARGET not in df.columns:
        raise RuntimeError(
            f"prepare_xy: target column {FORECASTER_TARGET!r} missing from frame"
        )
    if "hotel_name_normalized" not in df.columns:
        raise RuntimeError("prepare_xy: hotel_name_normalized missing from frame")

    price = df[FORECASTER_TARGET]
    if (price <= 0).any():
        n_bad = int((price <= 0).sum())
        raise ValueError(
            f"prepare_xy: {n_bad} rows have non-positive price; "
            "cannot take log. Cleaner should have removed these."
        )
    y = np.log(price.to_numpy(dtype=np.float64))

    X = df[list(FORECASTER_FEATURES)].copy()
    for col in _CATEGORICAL_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("category")

    groups = df["hotel_name_normalized"].reset_index(drop=True)
    X = X.reset_index(drop=True)
    return X, y, groups
