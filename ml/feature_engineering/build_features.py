"""
Reads raw scraper data (MongoDB by default, Parquet as fallback),
engineers features for the ML forecaster/anomaly/recommender models,
and returns a clean pandas DataFrame ready to be loaded into PostgreSQL.

Run directly:
    python ml/feature_engineering/build_features.py               # mongo
    python ml/feature_engineering/build_features.py --source parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ML_ROOT = REPO_ROOT / "ml"
DEFAULT_INPUT_DIR = REPO_ROOT / "scraper" / "output"

load_dotenv(ML_ROOT / ".env")

MONGO_PROJECTION = {
    "_id": 0,
    "source": 1, "scraped_at": 1, "scrape_run_id": 1,
    "check_in": 1, "check_out": 1, "nights": 1, "days_until_checkin": 1,
    "city_id": 1, "city_name": 1, "adults": 1, "children": 1,
    "hotel_name": 1, "hotel_name_normalized": 1, "stars": 1,
    "boarding_name": 1, "room_name": 1, "price": 1, "price_per_night": 1,
    "sur_demande": 1, "supplements": 1,
}

SEA_VIEW_KEYWORDS = ("mer", "sea")
GARDEN_VIEW_KEYWORDS = ("jardin", "garden")

# De-dup key: the business identity of a single scraped room-price observation.
OBSERVATION_KEYS = [
    "source",
    "hotel_name_normalized",
    "check_in",
    "nights",
    "adults",
    "children",
    "room_name",
    "boarding_name",
]

# Competitor scope: rooms comparable to each other on the same market.
# Source is included because promohotel vs tunisiepromo show different prices
# for the same physical hotel (different commission models).
COMPETITOR_KEYS = [
    "source",
    "city_name",
    "check_in",
    "nights",
    "boarding_name",
    "adults",
]

# Sea-view premium scope: same hotel, same stay, different view only.
VIEW_PREMIUM_KEYS = [
    "source",
    "hotel_name_normalized",
    "check_in",
    "nights",
    "adults",
    "boarding_name",
]


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required env var {name}. "
            f"Check ml/.env (template: ml/.env.example)."
        )
    return value


def _optional_int_env(name: str) -> int | None:
    raw = os.getenv(name)
    return int(raw) if raw else None


def _load_from_mongo(query: dict | None = None, limit: int = 0) -> pd.DataFrame:
    """
    Load raw hotel price records from MongoDB. Connection params come from
    ml/.env. Pass a Mongo `query` dict to filter (e.g. `{"scraped_at":
    {"$gte": ...}}`). Pass `limit` > 0 to cap rows (useful for EDA sampling).
    """
    from pymongo import MongoClient  # local import: parquet-only callers don't need it

    uri = _require_env("MONGO_URI")
    db_name = _require_env("MONGO_DB")
    coll_name = _require_env("MONGO_COLLECTION")

    client_kwargs = {
        "serverSelectionTimeoutMS": _optional_int_env("MONGO_SERVER_SELECTION_TIMEOUT_MS") or 30_000,
        "connectTimeoutMS": _optional_int_env("MONGO_CONNECT_TIMEOUT_MS") or 10_000,
        # 0 = no socket timeout — essential for large full-collection reads
        "socketTimeoutMS": _optional_int_env("MONGO_SOCKET_TIMEOUT_MS") or 0,
    }

    client = MongoClient(uri, **client_kwargs)
    try:
        coll = client[db_name][coll_name]
        cursor = coll.find(query or {}, projection=MONGO_PROJECTION).batch_size(5_000)
        if limit > 0:
            cursor = cursor.limit(limit)
        df = pd.DataFrame(cursor)
    finally:
        client.close()

    logger.info("Loaded %d raw rows from mongo %s.%s", len(df), db_name, coll_name)
    if df.empty:
        raise RuntimeError(f"Mongo collection {db_name}.{coll_name} returned zero rows")
    return df


def _load_from_parquet(input_dir: Path) -> pd.DataFrame:
    parquet_files = sorted(input_dir.glob("*.parquet"))
    csv_files = sorted(input_dir.glob("*.csv"))
    if not parquet_files and not csv_files:
        raise FileNotFoundError(f"No .parquet or .csv files in {input_dir}")

    frames: list[pd.DataFrame] = []
    skipped = 0
    for path in parquet_files:
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:
            logger.warning("Skipping unreadable parquet %s: %s", path.name, exc)
            skipped += 1
    for path in csv_files:
        try:
            frames.append(pd.read_csv(path))
        except Exception as exc:
            logger.warning("Skipping unreadable csv %s: %s", path.name, exc)
            skipped += 1

    if not frames:
        raise RuntimeError(f"All {skipped} file(s) in {input_dir} were unreadable")

    df = pd.concat(frames, ignore_index=True)
    logger.info(
        "Loaded %d raw rows from %d file(s); skipped %d",
        len(df), len(frames), skipped,
    )
    return df


def _parse_supplements(value) -> list[dict]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _supplement_features(supplements_raw: pd.Series) -> pd.DataFrame:
    parsed = supplements_raw.map(_parse_supplements)
    has_supp = parsed.map(lambda items: len(items) > 0)
    free_supp = parsed.map(
        lambda items: any((it.get("price") or 0) == 0 for it in items)
    )
    total = parsed.map(
        lambda items: float(sum((it.get("price") or 0) for it in items))
    )
    return pd.DataFrame({
        "has_supplement": has_supp.astype(bool),
        "free_supplement": free_supp.astype(bool),
        "total_supplement_price": total.astype("float32"),
    })


def _calendar_features(check_in: pd.Series) -> pd.DataFrame:
    dt = pd.to_datetime(check_in, errors="coerce")
    dow = dt.dt.dayofweek  # Mon=0 … Sun=6
    return pd.DataFrame({
        "check_in_date": dt,
        "day_of_week": dow.astype("Int8"),
        "month": dt.dt.month.astype("Int8"),
        # Tunisian hotel weekend = Fri/Sat nights.
        "weekend_flag": dow.isin([4, 5]).astype("int8"),
    })


def _view_flags(room_name: pd.Series) -> pd.DataFrame:
    rn = room_name.fillna("").str.lower()
    sea_pat = "|".join(SEA_VIEW_KEYWORDS)
    garden_pat = "|".join(GARDEN_VIEW_KEYWORDS)
    return pd.DataFrame({
        "is_sea_view": rn.str.contains(sea_pat, regex=True, na=False).astype("int8"),
        "is_garden_view": rn.str.contains(garden_pat, regex=True, na=False).astype("int8"),
    })


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stars_int"] = pd.to_numeric(df["stars"], errors="coerce").astype("Int8")
    df["price"] = pd.to_numeric(df["price"], errors="coerce").astype("float32")
    df["price_per_night"] = pd.to_numeric(df["price_per_night"], errors="coerce").astype("float32")
    df["days_until_checkin"] = pd.to_numeric(df["days_until_checkin"], errors="coerce").astype("Int16")
    return df


def _competitor_features(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(COMPETITOR_KEYS, dropna=False)["price_per_night"]
    group_sum = grp.transform("sum")
    group_cnt = grp.transform("count")

    competitor_sum = group_sum - df["price_per_night"]
    competitor_cnt = group_cnt - 1
    denom = competitor_cnt.where(competitor_cnt > 0)
    competitor_avg = (competitor_sum / denom).astype("float32")
    delta_pct = ((df["price_per_night"] - competitor_avg) / competitor_avg * 100).astype("float32")

    rank = grp.rank(method="min", ascending=True).astype("Int16")

    return pd.DataFrame({
        "competitor_avg_price": competitor_avg,
        "price_delta_pct": delta_pct,
        "price_rank_in_city": rank,
        "city_competitor_count": group_cnt.astype("Int16"),
    }, index=df.index)


def _sea_view_premium(df: pd.DataFrame) -> pd.Series:
    """
    Premium (TND/night) a hotel charges for sea-view vs garden-view
    on the same stay (same boarding, nights, adults, check_in).
    Broadcast to every row in that stay group. NaN when one of the
    two views is absent for that hotel/stay.
    """
    sea_avg = (
        df.loc[df["is_sea_view"] == 1]
          .groupby(VIEW_PREMIUM_KEYS)["price_per_night"].mean()
          .rename("_sea_avg")
    )
    garden_avg = (
        df.loc[df["is_garden_view"] == 1]
          .groupby(VIEW_PREMIUM_KEYS)["price_per_night"].mean()
          .rename("_garden_avg")
    )
    joined = (
        df[VIEW_PREMIUM_KEYS]
        .merge(sea_avg, on=VIEW_PREMIUM_KEYS, how="left")
        .merge(garden_avg, on=VIEW_PREMIUM_KEYS, how="left")
    )
    return (joined["_sea_avg"] - joined["_garden_avg"]).astype("float32").values


def build_features(
    source: str = "mongo",
    input_dir: Path = DEFAULT_INPUT_DIR,
    mongo_query: dict | None = None,
    limit: int = 0,
) -> pd.DataFrame:
    if source == "mongo":
        raw = _load_from_mongo(query=mongo_query, limit=limit)
    elif source == "parquet":
        raw = _load_from_parquet(input_dir)
    else:
        raise ValueError(f"Unknown source {source!r}; expected 'mongo' or 'parquet'")

    # Drop sur_demande and price-less rows; they poison aggregates.
    if "sur_demande" in raw.columns:
        raw = raw[raw["sur_demande"] != True]
    raw = raw[raw["price_per_night"].notna() & (raw["price_per_night"] > 0)]

    # Keep only the most recent scrape per observation.
    raw = (
        raw.sort_values("scraped_at", ascending=False)
           .drop_duplicates(subset=OBSERVATION_KEYS, keep="first")
           .reset_index(drop=True)
    )

    raw = _coerce_types(raw)

    supp = _supplement_features(raw["supplements"])
    cal = _calendar_features(raw["check_in"])
    views = _view_flags(raw["room_name"])

    df = pd.concat([raw, supp, cal, views], axis=1)
    df = pd.concat([df, _competitor_features(df)], axis=1)
    df["sea_view_premium"] = _sea_view_premium(df)

    return df


def _print_summary(df: pd.DataFrame) -> None:
    print("=" * 60)
    print(f"rows:              {len(df):,}")
    print(f"unique hotels:     {df['hotel_name_normalized'].nunique():,}")
    if df["check_in_date"].notna().any():
        lo = df["check_in_date"].min().date()
        hi = df["check_in_date"].max().date()
        print(f"check_in range:    {lo} -> {hi}")
    cities = sorted(df["city_name"].dropna().unique().tolist())
    sources = sorted(df["source"].dropna().unique().tolist())
    print(f"cities ({len(cities)}):       {cities}")
    print(f"sources:           {sources}")
    print(f"rows w/ comp_avg:  {df['competitor_avg_price'].notna().sum():,}")
    print(f"rows w/ sea prem:  {df['sea_view_premium'].notna().sum():,}")
    print("=" * 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=("mongo", "parquet"), default="mongo",
        help="Where to load raw scraper data from (default: mongo)",
    )
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
        help="Only used with --source parquet",
    )
    args = parser.parse_args()

    features = build_features(source=args.source, input_dir=args.input_dir)
    _print_summary(features)
