"""
MongoDB reader for the feature_engineering pipeline.

Reads raw scraped hotel-price records from the collection written by
the scraper (``hotel_scraper.hotel_prices``) and returns a pandas
DataFrame with an explicit, projected column set. Uses
``pymongoarrow.api.find_arrow_all`` for a zero-copy Arrow read path;
the plain ``list(collection.find())`` approach is explicitly avoided
because it materialises every BSON document into a Python dict before
pandas even sees it — unacceptable at 1.5M rows.

Read-only. This module never writes to MongoDB (the scraper is the sole
writer); it does not modify indexes, collections, or server state.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

import pandas as pd
from pymongo import MongoClient
from pymongoarrow.api import Schema, find_arrow_all
from pymongoarrow.monkey import patch_all
import pyarrow as pa

from .config import MONGO_COLLECTION, MONGO_DATABASE, MONGO_URI

logger = logging.getLogger(__name__)

patch_all()  # enables collection.find_arrow_all etc. (idempotent)


# ---------------------------------------------------------------------------
# Projection / schema
# ---------------------------------------------------------------------------
# Explicit projection keeps network bytes and Arrow memory bounded. Any
# field added to the scraper schema that ML wants must be added here —
# silent passthrough is a footgun.

PROJECTED_FIELDS: tuple[str, ...] = (
    "source",
    "scraped_at",
    "scrape_run_id",
    "check_in",
    "check_out",
    "nights",
    "days_until_checkin",
    "city_id",
    "city_name",
    "adults",
    "children",
    "hotel_name",
    "hotel_name_normalized",
    "stars",
    "boarding_name",
    "room_name",
    "price",
    "price_per_night",
    "sur_demande",
    "supplements",
)

# Fields whose non-null rate MUST be >= REQUIRED_NON_NULL_RATE. These are
# fields downstream logic can't recover from if missing.
REQUIRED_FIELDS: tuple[str, ...] = (
    "source",
    "scraped_at",
    "check_in",
    "nights",
    "city_name",
    "hotel_name_normalized",
    "boarding_name",
    "price",
)

REQUIRED_NON_NULL_RATE: float = 0.95


# Arrow schema. `supplements` is a variable-shape list<document>, so we
# read it as a bytes blob (`pa.binary()` is not ideal for dicts) — use
# `pa.list_(pa.struct([]))` with empty struct and fall back to pandas
# object dtype. Simpler and safer: declare it as string and parse later,
# OR let Arrow infer. We pick inference (schema=None for `supplements`)
# by reading a minimal schema explicitly, then merging.
#
# In practice: pymongoarrow supports list<struct<...>>. To avoid coupling
# to the exact supplement shape (free-form), we read `supplements` as a
# Python-object column via a best-effort cast after the Arrow read.

# NOTE on date fields: the scraper stores `scraped_at`, `check_in` and
# `check_out` as ISO-8601 *strings*, not BSON datetimes. We keep them as
# strings on the Arrow read path and parse to pandas datetime in
# cleaners.py (Stage 3). This matches the spec and avoids the strict
# pymongoarrow timestamp decoder rejecting string-typed date fields.
_ARROW_SCHEMA: Schema = Schema({
    "source":                 pa.string(),
    "scraped_at":             pa.string(),
    "scrape_run_id":          pa.string(),
    "check_in":               pa.string(),
    "check_out":              pa.string(),
    "nights":                 pa.int32(),
    "days_until_checkin":     pa.int32(),
    "city_id":                pa.int32(),     # scraper-local; never a join key
    "city_name":              pa.string(),
    "adults":                 pa.int32(),
    "children":               pa.int32(),
    "hotel_name":             pa.string(),
    "hotel_name_normalized":  pa.string(),
    "stars":                  pa.string(),   # raw — coerced in cleaners.py
    "boarding_name":          pa.string(),
    "room_name":              pa.string(),
    "price":                  pa.float64(),
    "price_per_night":        pa.float64(),
    "sur_demande":            pa.bool_(),
    "supplements":            pa.list_(pa.struct([
        ("name",  pa.string()),
        ("price", pa.float64()),
    ])),
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw_from_mongo(
    mongo_uri: str = MONGO_URI,
    database: str = MONGO_DATABASE,
    collection: str = MONGO_COLLECTION,
    scraped_after: datetime | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Load raw hotel-price observations from MongoDB into a pandas DataFrame.

    Parameters
    ----------
    mongo_uri:
        MongoDB connection URI. Defaults to config.MONGO_URI.
    database:
        Source database name. Defaults to config.MONGO_DATABASE.
    collection:
        Source collection name. Defaults to config.MONGO_COLLECTION.
    scraped_after:
        If set, only documents with ``scraped_at > scraped_after`` are
        read (for incremental pipeline runs). Naive datetimes are
        assumed UTC.
    limit:
        If set, hard-cap the number of rows read (useful for EDA /
        smoke tests). Must be > 0.

    Returns
    -------
    pd.DataFrame
        One row per scraped observation, columns restricted to
        PROJECTED_FIELDS. Sorted only insofar as the Mongo cursor order;
        downstream callers should not rely on ordering.

    Raises
    ------
    RuntimeError
        If the collection is empty, or if any field in REQUIRED_FIELDS
        has a non-null rate below REQUIRED_NON_NULL_RATE.
    ValueError
        If ``limit`` is provided and not positive.

    Notes
    -----
    Read path: pymongoarrow.find_arrow_all -> pyarrow.Table -> pandas.
    Avoid ``list(collection.find())`` — unacceptable memory cost at
    scale.
    """
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive, got {limit!r}")

    query: dict[str, Any] = {}
    if scraped_after is not None:
        query["scraped_at"] = {"$gt": scraped_after}

    logger.info(
        "Connecting to MongoDB database=%s collection=%s%s",
        database, collection,
        f" scraped_after={scraped_after.isoformat()}" if scraped_after else "",
    )

    t0 = time.perf_counter()
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10_000)
    try:
        coll = client[database][collection]
        table = find_arrow_all(
            coll,
            query,
            schema=_ARROW_SCHEMA,
            limit=limit or 0,
        )
    finally:
        client.close()

    df = table.to_pandas(types_mapper=pd.ArrowDtype)
    # Arrow-backed dtypes keep memory low but confuse some downstream
    # pandas idioms (groupby on Arrow strings is fine; fillna on Arrow
    # booleans has edge cases). Convert object-ish columns back to numpy
    # where it matters; keep timestamps / numerics as-is.
    for col in ("source", "city_name", "hotel_name",
                "hotel_name_normalized", "stars", "boarding_name",
                "room_name", "scrape_run_id"):
        if col in df.columns:
            df[col] = df[col].astype("string[python]")

    elapsed = time.perf_counter() - t0
    _log_load_summary(df, elapsed)
    _assert_load_contract(df)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_load_summary(df: pd.DataFrame, elapsed: float) -> None:
    if df.empty:
        logger.warning("Loaded 0 rows (elapsed=%.2fs)", elapsed)
        return
    ts = df["scraped_at"]
    logger.info(
        "Loaded %d rows in %.2fs  scraped_at=[%s .. %s]  unique_hotels=%d",
        len(df),
        elapsed,
        ts.min(),
        ts.max(),
        df["hotel_name_normalized"].nunique(),
    )


def _assert_load_contract(df: pd.DataFrame) -> None:
    if df.empty:
        raise RuntimeError(
            "MongoDB load returned zero rows. Expected at least a "
            "minimal scrape present in hotel_scraper.hotel_prices."
        )

    missing_cols = [c for c in PROJECTED_FIELDS if c not in df.columns]
    if missing_cols:
        raise RuntimeError(
            f"MongoDB load missing projected columns: {missing_cols}. "
            "Check the projection and the scraper's item schema."
        )

    failures: list[str] = []
    for col in REQUIRED_FIELDS:
        non_null_rate = df[col].notna().mean()
        if non_null_rate < REQUIRED_NON_NULL_RATE:
            failures.append(
                f"{col}: non_null_rate={non_null_rate:.3f} "
                f"< threshold={REQUIRED_NON_NULL_RATE}"
            )
    if failures:
        raise RuntimeError(
            "Required-field coverage check failed:\n  - "
            + "\n  - ".join(failures)
        )
