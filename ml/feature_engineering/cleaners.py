"""
Row-level cleaning for raw MongoDB observations.

The loader (``mongo_loader.py``) reads the scraper output verbatim with a
relaxed Arrow schema — dates arrive as ISO strings, stars as free text,
and the row set may contain implausible prices or unsupported night
counts. This module turns that raw frame into a canonical cross-sectional
observation table the rest of the pipeline can rely on.

Contract (what downstream modules can assume after ``clean`` returns):
    * ``check_in``   -> datetime64[ns] (naive, calendar day only)
    * ``scraped_at`` -> datetime64[ns, UTC] (timezone-aware)
    * ``days_until_checkin`` -> Int16 in [0, 365]
    * ``stars_int``  -> Int8 in [1, 7] (never null)
    * ``nights``     -> Int16 in {1, 2, 3, 5, 7}
    * ``price``, ``price_per_night`` -> float32, both non-null, positive
    * ``price_per_night`` equals ``price / nights`` within 1 TND
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from .config import (
    DAYS_UNTIL_CHECKIN_MAX,
    DAYS_UNTIL_CHECKIN_MIN,
    NIGHTS_ALLOWED,
    PRICE_PER_NIGHT_MAX,
    PRICE_PER_NIGHT_MIN,
)

logger = logging.getLogger(__name__)

NAT_FAIL_RATE: float = 0.01
# Tunisian hotels rated 1–5 stars (Office National du Tourisme Tunisien
# classification). Values outside this range are treated as unparseable.
STARS_RANGE: range = range(1, 6)

_STARS_DIGIT_RE: re.Pattern[str] = re.compile(r"(\d)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a cleaned, strictly-typed copy of the raw loader DataFrame.

    Parameters
    ----------
    df:
        Output of ``mongo_loader.load_raw_from_mongo``.

    Returns
    -------
    pd.DataFrame
        A new DataFrame (the input is not modified). Rows that fail any
        business-rule filter are dropped; dropped-row counts are logged.

    Raises
    ------
    RuntimeError
        If more than 1 % of ``check_in`` or ``scraped_at`` values fail
        to parse, or if a required column is missing.
    """
    _assert_required_columns(df)
    out = df.copy()

    out = _parse_dates(out)
    out = _rebuild_days_until_checkin(out)
    out = _coerce_stars(out)
    out = _impute_stars_per_hotel(out)
    out = _filter_price_per_night(out)
    out = _filter_nights(out)
    out = _reconcile_price_per_night(out)
    out = _finalise_dtypes(out)

    logger.info("clean() complete: %d rows remaining", len(out))
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step implementations
# ---------------------------------------------------------------------------

def _assert_required_columns(df: pd.DataFrame) -> None:
    required = (
        "source", "scraped_at", "check_in", "nights",
        "stars", "hotel_name_normalized",
        "price", "price_per_night",
    )
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"clean() missing required columns: {missing}")


def _parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    n = len(df)

    check_in = pd.to_datetime(
        df["check_in"].astype("string"), errors="coerce", utc=False,
    )
    scraped_at = pd.to_datetime(
        df["scraped_at"].astype("string"), errors="coerce", utc=True,
    )

    ci_fail = check_in.isna().mean()
    sa_fail = scraped_at.isna().mean()
    if ci_fail > NAT_FAIL_RATE:
        raise RuntimeError(
            f"check_in parse failure rate {ci_fail:.4f} > {NAT_FAIL_RATE} "
            f"(threshold). Inspect raw MongoDB values."
        )
    if sa_fail > NAT_FAIL_RATE:
        raise RuntimeError(
            f"scraped_at parse failure rate {sa_fail:.4f} > {NAT_FAIL_RATE}"
        )

    # Drop the rare unparseable rows.
    mask = check_in.notna() & scraped_at.notna()
    dropped = n - mask.sum()
    if dropped:
        logger.info("dropped %d rows with unparseable dates", dropped)

    df = df.loc[mask].copy()
    df["check_in"] = check_in.loc[mask]
    df["scraped_at"] = scraped_at.loc[mask]
    return df


def _rebuild_days_until_checkin(df: pd.DataFrame) -> pd.DataFrame:
    # scraped_at is tz-aware UTC; strip tz after flooring to midnight so
    # we can subtract a tz-naive check_in cleanly.
    scraped_day = df["scraped_at"].dt.tz_convert("UTC").dt.floor("D").dt.tz_localize(None)
    delta_days = (df["check_in"] - scraped_day).dt.days

    mask = (delta_days >= DAYS_UNTIL_CHECKIN_MIN) & (delta_days <= DAYS_UNTIL_CHECKIN_MAX)
    dropped = (~mask).sum()
    if dropped:
        logger.info(
            "dropped %d rows with days_until_checkin outside [%d, %d]",
            dropped, DAYS_UNTIL_CHECKIN_MIN, DAYS_UNTIL_CHECKIN_MAX,
        )
    df = df.loc[mask].copy()
    df["days_until_checkin"] = delta_days.loc[mask].astype("Int16")
    return df


def _coerce_stars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the first 1–7 digit from ``stars``. Handles "4", "4.0", "4*",
    "4 étoiles", "  3   ". Leaves ``stars`` column untouched; writes the
    integer into ``stars_int`` (nullable Int8).
    """
    def _extract(v: object) -> pd.NA | int:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return pd.NA
        s = str(v)
        m = _STARS_DIGIT_RE.search(s)
        if not m:
            return pd.NA
        n = int(m.group(1))
        return n if n in STARS_RANGE else pd.NA

    df["stars_int"] = df["stars"].map(_extract).astype("Int8")
    return df


def _impute_stars_per_hotel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing stars_int with the hotel's modal value. Drop rows where
    the hotel has no observed mode (i.e. every observation of that hotel
    is missing).
    """
    na_before = df["stars_int"].isna().sum()
    if na_before == 0:
        return df

    def _mode(s: pd.Series) -> pd.Series:
        m = s.dropna().mode()
        return pd.Series(m.iloc[0] if not m.empty else pd.NA, index=s.index, dtype="Int8")

    hotel_mode = df.groupby("hotel_name_normalized", dropna=False)["stars_int"].transform(_mode)
    df["stars_int"] = df["stars_int"].fillna(hotel_mode)

    na_after = df["stars_int"].isna().sum()
    imputed = na_before - na_after
    logger.info("stars_int imputed %d rows from hotel mode", imputed)

    if na_after:
        logger.info("dropped %d rows with no stars_int and no hotel mode", na_after)
        df = df.loc[df["stars_int"].notna()].copy()
    return df


def _filter_price_per_night(df: pd.DataFrame) -> pd.DataFrame:
    ppn = pd.to_numeric(df["price_per_night"], errors="coerce")
    too_low  = ppn < PRICE_PER_NIGHT_MIN
    too_high = ppn > PRICE_PER_NIGHT_MAX
    bad = too_low | too_high | ppn.isna()

    if bad.any():
        sample = df.loc[bad, ["hotel_name_normalized", "nights", "price", "price_per_night"]].head(5)
        logger.info(
            "dropping %d rows with price_per_night outside [%g, %g] or null; sample:\n%s",
            int(bad.sum()), PRICE_PER_NIGHT_MIN, PRICE_PER_NIGHT_MAX, sample.to_string(),
        )
    return df.loc[~bad].copy()


def _filter_nights(df: pd.DataFrame) -> pd.DataFrame:
    n_raw = pd.to_numeric(df["nights"], errors="coerce").astype("Int16")
    mask = n_raw.isin(list(NIGHTS_ALLOWED))
    dropped = (~mask).sum()
    if dropped:
        logger.info(
            "dropped %d rows with nights not in %s (top values: %s)",
            int(dropped), sorted(NIGHTS_ALLOWED),
            n_raw.loc[~mask].value_counts().head(5).to_dict(),
        )
    df = df.loc[mask].copy()
    df["nights"] = n_raw.loc[mask]
    return df


def _reconcile_price_per_night(df: pd.DataFrame) -> pd.DataFrame:
    price   = pd.to_numeric(df["price"], errors="coerce")
    ppn     = pd.to_numeric(df["price_per_night"], errors="coerce")
    implied = price / df["nights"].astype("float64")
    mismatch = (implied - ppn).abs() > 1.0

    if mismatch.any():
        logger.info(
            "recomputing price_per_night on %d rows (|implied - stored| > 1 TND)",
            int(mismatch.sum()),
        )
        ppn = ppn.where(~mismatch, implied)

    df["price"] = price.astype("float32")
    df["price_per_night"] = ppn.astype("float32")
    return df


def _finalise_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # `string[python]` keeps compatibility with taxonomy/downstream modules.
    for c in ("source", "city_name", "hotel_name", "hotel_name_normalized",
              "stars", "boarding_name", "room_name", "scrape_run_id"):
        if c in df.columns:
            df[c] = df[c].astype("string[python]")
    for c in ("adults", "children"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int16")
    return df
