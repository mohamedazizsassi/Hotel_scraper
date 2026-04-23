"""
Calendar feature engineering.

Adds deterministic calendar-derived features to a cleaned observation frame:
basic date parts (day-of-week, month, week, day-of-month, quarter, weekend
flag) and market-specific signals (Ramadan, Tunisia public + school holidays,
EU source-market school holidays, distance to nearest EU school-holiday day).

All holiday tables are imported from ``config``. No runtime external API
calls or internet access. Outputs are pure deterministic functions of
``check_in``.

Contract (what downstream modules can assume after ``add_calendar_features``):
    * check_in_dow                        -> Int8   in [0, 6]  (0 = Monday)
    * check_in_month                      -> Int8   in [1, 12]
    * check_in_week_of_year               -> Int8   in [1, 53]
    * check_in_day_of_month               -> Int8   in [1, 31]
    * check_in_quarter                    -> Int8   in [1, 4]
    * is_weekend_checkin                  -> boolean (Fri/Sat — TN hotel peak)
    * is_ramadan                          -> boolean
    * is_tunisia_public_holiday           -> boolean
    * is_tunisia_school_holiday           -> boolean
    * is_school_holiday_france            -> boolean
    * is_school_holiday_germany           -> boolean
    * is_school_holiday_uk                -> boolean
    * days_to_nearest_european_holiday    -> Int16  (>= 0, 0 if inside)

Returns a new DataFrame; the input is not modified.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import (
    FRANCE_SCHOOL_HOLIDAYS,
    GERMANY_SCHOOL_HOLIDAYS,
    RAMADAN_PERIODS,
    TUNISIA_ISLAMIC_HOLIDAYS,
    TUNISIA_PUBLIC_HOLIDAYS_FIXED,
    TUNISIA_SCHOOL_HOLIDAYS,
    UK_SCHOOL_HOLIDAYS,
)

logger = logging.getLogger(__name__)

# Tunisian hotel peak = Fri(4) + Sat(5) nights. Leisure weekend is Sat/Sun
# but the *hotel* signal tracks arrival nights, which cluster on Fri/Sat.
WEEKEND_DOW: frozenset[int] = frozenset({4, 5})

_NO_EU_HOLIDAY_SENTINEL: int = 32_767  # Int16 max — unused in practice (EU table always populated)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calendar features derived from ``check_in``.

    Parameters
    ----------
    df:
        Cleaned DataFrame from ``cleaners.clean``. Must contain ``check_in``
        as datetime-like (tz-naive calendar day, per the cleaner contract).

    Returns
    -------
    pd.DataFrame
        New DataFrame with the thirteen calendar columns added.

    Raises
    ------
    RuntimeError
        If ``check_in`` is missing or contains unparseable values (cleaner
        should have dropped these — this is a belt-and-braces check).
    """
    if "check_in" not in df.columns:
        raise RuntimeError("add_calendar_features: missing 'check_in'")

    out = df.copy()
    ci = pd.to_datetime(out["check_in"], errors="coerce")
    if ci.isna().any():
        raise RuntimeError(
            f"add_calendar_features: {int(ci.isna().sum())} unparseable "
            "check_in values. Upstream clean() should drop these first."
        )

    dow = ci.dt.dayofweek
    out["check_in_dow"]          = dow.astype("Int8")
    out["check_in_month"]        = ci.dt.month.astype("Int8")
    out["check_in_week_of_year"] = ci.dt.isocalendar().week.astype("Int8")
    out["check_in_day_of_month"] = ci.dt.day.astype("Int8")
    out["check_in_quarter"]      = ci.dt.quarter.astype("Int8")
    out["is_weekend_checkin"]    = dow.isin(list(WEEKEND_DOW)).astype("boolean")

    # Market flags are derived per calendar day; dedup to |unique days| ~ O(730)
    # for a 2-year horizon rather than computing over 3.9M rows.
    ci_day = ci.dt.normalize()
    unique_days = pd.DatetimeIndex(ci_day.unique())
    lookup = _build_date_lookup(unique_days)

    for col in (
        "is_ramadan",
        "is_tunisia_public_holiday",
        "is_tunisia_school_holiday",
        "is_school_holiday_france",
        "is_school_holiday_germany",
        "is_school_holiday_uk",
    ):
        out[col] = ci_day.map(lookup[col]).astype("boolean")
    out["days_to_nearest_european_holiday"] = (
        ci_day.map(lookup["days_to_nearest_european_holiday"]).astype("Int16")
    )

    _log_coverage(out)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expand_ranges(ranges: list[tuple[date, date]]) -> set[date]:
    """Expand inclusive (start, end) date ranges into a flat set of dates."""
    out: set[date] = set()
    for start, end in ranges:
        if end < start:
            raise RuntimeError(f"calendar config: end {end} before start {start}")
        d = start
        while d <= end:
            out.add(d)
            d += timedelta(days=1)
    return out


def _build_date_lookup(
    days: pd.DatetimeIndex,
) -> dict[str, dict[pd.Timestamp, object]]:
    """
    Pre-compute each flag for every unique check_in day. Returned as a
    dict-of-dicts keyed by column name then Timestamp so that pandas
    ``Series.map(dict)`` can broadcast back to the full frame.
    """
    ramadan           = _expand_ranges(RAMADAN_PERIODS)
    tn_public_fixed   = set(TUNISIA_PUBLIC_HOLIDAYS_FIXED.keys())  # (month, day)
    tn_public_islamic = set(TUNISIA_ISLAMIC_HOLIDAYS)
    tn_school         = _expand_ranges(TUNISIA_SCHOOL_HOLIDAYS)
    fr_school         = _expand_ranges(FRANCE_SCHOOL_HOLIDAYS)
    de_school         = _expand_ranges(GERMANY_SCHOOL_HOLIDAYS)
    uk_school         = _expand_ranges(UK_SCHOOL_HOLIDAYS)

    eu_union = fr_school | de_school | uk_school
    eu_sorted: np.ndarray = (
        np.array(sorted(eu_union), dtype="datetime64[D]")
        if eu_union else np.array([], dtype="datetime64[D]")
    )

    cols = (
        "is_ramadan",
        "is_tunisia_public_holiday",
        "is_tunisia_school_holiday",
        "is_school_holiday_france",
        "is_school_holiday_germany",
        "is_school_holiday_uk",
        "days_to_nearest_european_holiday",
    )
    lookup: dict[str, dict[pd.Timestamp, object]] = {c: {} for c in cols}

    for ts in days:
        if pd.isna(ts):
            continue
        d = ts.date()
        lookup["is_ramadan"][ts]                 = d in ramadan
        lookup["is_tunisia_public_holiday"][ts]  = (
            (d.month, d.day) in tn_public_fixed or d in tn_public_islamic
        )
        lookup["is_tunisia_school_holiday"][ts]  = d in tn_school
        lookup["is_school_holiday_france"][ts]   = d in fr_school
        lookup["is_school_holiday_germany"][ts]  = d in de_school
        lookup["is_school_holiday_uk"][ts]       = d in uk_school
        lookup["days_to_nearest_european_holiday"][ts] = _nearest_days(d, eu_sorted)

    return lookup


def _nearest_days(d: date, sorted_days: np.ndarray) -> int:
    """Min |d - h| over all h in sorted_days. 0 if d itself is in the set."""
    if sorted_days.size == 0:
        return _NO_EU_HOLIDAY_SENTINEL
    target = np.datetime64(d, "D")
    idx = int(np.searchsorted(sorted_days, target))
    best = _NO_EU_HOLIDAY_SENTINEL
    if idx < sorted_days.size:
        best = min(best, int(abs((sorted_days[idx] - target).astype(int))))
    if idx > 0:
        best = min(best, int(abs((sorted_days[idx - 1] - target).astype(int))))
    return best


def _log_coverage(df: pd.DataFrame) -> None:
    flags = (
        "is_weekend_checkin",
        "is_ramadan",
        "is_tunisia_public_holiday",
        "is_tunisia_school_holiday",
        "is_school_holiday_france",
        "is_school_holiday_germany",
        "is_school_holiday_uk",
    )
    for f in flags:
        rate = float(df[f].mean())
        logger.info("add_calendar_features: %-32s rate=%.4f", f, rate)
    d = df["days_to_nearest_european_holiday"]
    logger.info(
        "add_calendar_features: days_to_nearest_european_holiday "
        "min=%s median=%s max=%s",
        int(d.min()), int(d.median()), int(d.max()),
    )
