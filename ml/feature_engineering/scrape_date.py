"""
Derive ``scrape_date`` from ``scraped_at``.

This column is the calendar UTC day on which a row was scraped. It
exists for one reason: to act as a key in peer-group and demand-slice
aggregates so that competitive features for a row are computed only
against other observations from the same scrape day. That kills the
temporal leakage documented in the 2026-05-14 audit (a row scraped on
2026-05-01 was previously seeing peer prices scraped on 2026-05-13)
WITHOUT discarding the per-offer time-series that ``days_until_checkin``
depends on.

The earlier dedup-to-most-recent approach (``temporal_dedup.py``) also
killed the leak, but at the cost of collapsing every offer to one row
and thereby flattening the booking-window trajectory — exactly the
non-linear effect ``CLAUDE.md`` flags as central to the Tunisian
market. Scrape-date keying preserves every (offer × scrape) observation
as an independent training row.

Contract:
    * Input must contain ``scraped_at`` as a tz-aware UTC datetime
      (the dtype produced by ``cleaners.clean``).
    * Output adds one column ``scrape_date`` of dtype
      ``datetime64[ns]`` (naive, midnight UTC), one per row.
    * No rows added or dropped. Column order is preserved (new column
      is appended last).
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


SCRAPE_DATE_COL: str = "scrape_date"


def add_scrape_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a ``scrape_date`` column derived from ``scraped_at``.

    Parameters
    ----------
    df:
        Cleaned DataFrame from ``cleaners.clean``. Must contain
        ``scraped_at`` as a tz-aware UTC datetime column.

    Returns
    -------
    pd.DataFrame
        New DataFrame with ``scrape_date`` appended. Input is not
        modified.

    Raises
    ------
    RuntimeError
        If ``scraped_at`` is missing.
    """
    if "scraped_at" not in df.columns:
        raise RuntimeError("add_scrape_date: missing 'scraped_at'")

    out = df.copy()
    # ``scraped_at`` is tz-aware UTC after cleaners. Floor to midnight
    # and drop tz so the resulting column has a stable ``datetime64[ns]``
    # dtype that groups cleanly with the other date-like keys in
    # PEER_GROUP_KEYS (e.g. ``check_in``).
    out[SCRAPE_DATE_COL] = (
        out["scraped_at"].dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None)
    )

    n_distinct = int(out[SCRAPE_DATE_COL].nunique())
    logger.info("add_scrape_date: %d distinct scrape days", n_distinct)
    return out
