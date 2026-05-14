"""
Cross-sectional snapshot: dedup to most-recent observation per business key.

The raw MongoDB collection carries multiple ``scraped_at`` snapshots per
business key — daily scraping repeats the same (hotel, check_in, room,
boarding, nights, adults) probe day after day. The cross-sectional
contract in ``ml/CLAUDE.md`` §1 says the pipeline treats data as
cross-sectional, but the assembly stages prior to this module preserve
all historical rows, so competitive aggregates would silently mix
observations across time:

    row scraped 2026-05-01 has its peer_median computed against peer
    rows scraped on 2026-05-13 — i.e. a future quote leaks into the
    feature row's competitive context.

The leakage audit in ``validators.py`` only checks self-exclusion (own
hotel out of own aggregate), so it does NOT catch this. The fix is to
collapse the per-key time-series down to its latest observation before
stage 7 (competitive_features) runs.

Runs AFTER ``cleaners.clean`` (which gives us a parsed ``scraped_at``)
and BEFORE ``supplement_expansion.expand_supplements`` (so dedup applies
to raw scraped rows, not to synthetic supplement variants).

Contract:
    * Output row count <= input row count.
    * For every distinct value of the business key tuple
      ``(source, hotel_name_normalized, check_in, boarding_name,
        room_name, nights, adults)``,
      the output contains exactly one row — the one with the maximum
      ``scraped_at``.
    * Ties on ``scraped_at`` resolve deterministically (stable sort,
      original input order wins as a secondary key).
    * Column set, dtypes, and order are preserved.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


# Business-key columns. Order is significant only for log readability;
# pandas treats the subset as a set under drop_duplicates.
BUSINESS_KEY: tuple[str, ...] = (
    "source",
    "hotel_name_normalized",
    "check_in",
    "boarding_name",
    "room_name",
    "nights",
    "adults",
)


def dedup_to_most_recent_per_business_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the most recent observation per business key.

    Parameters
    ----------
    df:
        Cleaned DataFrame from ``cleaners.clean``. Must contain every
        column in ``BUSINESS_KEY`` plus ``scraped_at`` (tz-aware
        datetime, as produced by cleaners).

    Returns
    -------
    pd.DataFrame
        A new DataFrame with at most one row per business key, namely
        the row with the largest ``scraped_at``. Index is reset.

    Raises
    ------
    RuntimeError
        If a required column is missing.
    """
    required = list(BUSINESS_KEY) + ["scraped_at"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"dedup_to_most_recent_per_business_key: missing columns: {missing}"
        )

    n_before = len(df)
    if n_before == 0:
        logger.info("dedup_to_most_recent_per_business_key: empty input, passthrough")
        return df.reset_index(drop=True)

    # Stable mergesort on scraped_at ascending; drop_duplicates(keep="last")
    # then picks the largest scraped_at per business key. Stability ensures
    # ties resolve by original input order, so successive runs over
    # unchanged data are byte-identical.
    out = (
        df.sort_values("scraped_at", kind="mergesort")
        .drop_duplicates(subset=list(BUSINESS_KEY), keep="last")
        .reset_index(drop=True)
    )
    n_after = len(out)

    compression = n_before / n_after if n_after else float("nan")
    logger.info(
        "dedup_to_most_recent_per_business_key: %d -> %d rows (%.2fx compression)",
        n_before, n_after, compression,
    )
    return out
