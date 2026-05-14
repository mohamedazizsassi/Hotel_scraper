"""
Stage orchestration — raw frame → fully-engineered feature frame.

This module contains no I/O. It receives a cleaned-of-Mongo-internals
DataFrame (the output of ``mongo_loader.load_raw_from_mongo``) and runs
every feature-engineering stage in the canonical order defined in
``feature_engineering/CLAUDE.md``::

    clean
      → supplement_expansion        (cross-source alignment; before taxonomy)
      → canonicalize_boarding
      → parse_room
      → calendar
      → competitive  (leakage-critical, peer aggregates)
      → demand       (leakage-critical, sur_demande aggregates)

Validation, parquet, and postgres writes are deliberately outside this
function — they belong to the CLI in ``build_features.py``. Keeping
``assemble_features`` pure makes it usable from notebooks, tests, and
ad-hoc scripts without dragging filesystem or database side effects in.

Per-stage row counts and elapsed time are logged at INFO so a single
end-to-end run is easy to read in production logs.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

import pandas as pd

from .calendar_features import add_calendar_features
from .cleaners import clean
from .competitive_features import add_competitive_features
from .demand_features import add_demand_features
from .scrape_date import add_scrape_date
from .supplement_expansion import expand_supplements
from .taxonomy import canonicalize_boarding, parse_room

logger = logging.getLogger(__name__)


# Order matters.
# * add_scrape_date runs immediately after clean. The derived column is
#   then included in PEER_GROUP_KEYS (competitive_features) and in
#   SUR_DEMANDE_SLICES / ACTIVITY_COUNT_KEYS (demand_features), which
#   scopes every aggregate to within a single scrape day. This kills the
#   temporal leakage diagnosed on 2026-05-14 (rows from one scrape day
#   no longer pull peer prices from later scrape days) while preserving
#   every (offer × scrape_run) observation as an independent training
#   row — i.e. the within-offer booking-window trajectory is kept.
# * supplement_expansion MUST run before parse_room so synthetic variant
#   rows carry view information in `room_name` when the regex scan happens.
# * Competitive runs before demand by convention only; they are independent.
_STAGES: list[tuple[str, Callable[[pd.DataFrame], pd.DataFrame]]] = [
    ("clean", clean),
    ("add_scrape_date", add_scrape_date),
    ("supplement_expansion", expand_supplements),
    ("canonicalize_boarding", canonicalize_boarding),
    ("parse_room", parse_room),
    ("calendar", add_calendar_features),
    ("competitive", add_competitive_features),
    ("demand", add_demand_features),
]


def assemble_features(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Run every feature-engineering stage in canonical order.

    Parameters
    ----------
    raw:
        Output of ``mongo_loader.load_raw_from_mongo``: one row per
        scraped observation, columns restricted to the projected fields.

    Returns
    -------
    pd.DataFrame
        Fully-engineered feature frame, ready for validation and write.
        Row count differs from input because supplement expansion adds
        synthetic variant rows and cleaners drops out-of-range rows.

    Raises
    ------
    Whatever the underlying stages raise. ``assemble_features`` does not
    swallow stage errors — fail loudly is a non-negotiable.
    """
    if raw.empty:
        raise ValueError("assemble_features: input DataFrame is empty")

    logger.info("assemble: %d input rows, %d columns", len(raw), raw.shape[1])

    df = raw
    pipeline_start = time.perf_counter()
    for name, stage in _STAGES:
        rows_before = len(df)
        t0 = time.perf_counter()
        df = stage(df)
        elapsed = time.perf_counter() - t0
        logger.info(
            "stage=%s rows=%d (%+d) cols=%d elapsed=%.2fs",
            name, len(df), len(df) - rows_before, df.shape[1], elapsed,
        )

    total = time.perf_counter() - pipeline_start
    logger.info(
        "assemble done: %d rows, %d cols, %.2fs total",
        len(df), df.shape[1], total,
    )
    return df
