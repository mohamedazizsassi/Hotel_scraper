"""
Single source of truth for feature partitions, per-model selectors,
and storage projections.

Every column in the assembled feature frame belongs to exactly one of four
mutually-exclusive categories:

    IDENTIFIER_COLUMNS      — provenance / join keys; never enter a model split.
    TARGETS                 — the three prediction targets.
    LEAKY_OBSERVED_FEATURES — derived from the row's own price_per_night at
                              observation time; safe for anomaly/recommender
                              (price is observed), leaky for the forecaster
                              (it predicts price).
    PREDICTIVE_FEATURES     — everything else; safe inputs for all models.

Storage split (offline / online feature-store pattern):
    * Offline (Parquet) — full 77-col frame; training reads this.
    * Online (Postgres hotel_features) — POSTGRES_SERVING_COLUMNS (~52 cols).
    * Dim tables — calendar_dim (~730 rows) and segment_dim (~100 rows).
      Deterministic features that are pure functions of check_in or
      (city_name, stars_int) are joined at serve time instead of being
      duplicated across 24M+ observation rows.

Per-model selectors
-------------------
    FORECASTER_FEATURES  = PREDICTIVE_FEATURES        (price unknown at predict time)
    ANOMALY_FEATURES     = PREDICTIVE_FEATURES
                           + LEAKY_OBSERVED_FEATURES
                           + ("price_per_night",)     (price observed — detect outlier)
    RECOMMENDER_FEATURES = ANOMALY_FEATURES           (same input set)

Column names are imported from stage modules where those modules export the
relevant constants. For stages without exported output-column constants the
names are listed as string literals — they are stable contracts spelled out in
each stage module's docstring, and the union test in test_model_feature_sets
will catch any drift.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

from .competitive_features import GRANULARITIES
from .demand_features import ACTIVITY_COUNT_COL, SUR_DEMANDE_SLICES
from .scrape_date import SCRAPE_DATE_COL
from .segment_features import MACRO_REGION_COL, MARKET_SEGMENT_ID_COL, STARS_BAND_COL


# ---------------------------------------------------------------------------
# Identifier / provenance columns — never enter a model split
# ---------------------------------------------------------------------------

IDENTIFIER_COLUMNS: tuple[str, ...] = (
    # Raw Mongo fields kept for join-back and provenance only.
    "source",
    "scraped_at",
    "scrape_run_id",
    "check_in",       # join key for calendar_dim
    "check_out",
    "city_id",        # scraper-local int; NOT a cross-source join key
    "city_name",      # join key for segment_dim; also in segment_dim
    "hotel_name",
    "hotel_name_normalized",
    SCRAPE_DATE_COL,  # "scrape_date" — peer-group scoping key
    # Raw text fields superseded by canonical counterparts downstream.
    "stars",          # raw text (e.g. "4*"); canonical → stars_int
    "boarding_name",  # raw text; canonical → boarding_canonical
    "room_name",      # raw text; canonical → room_base / room_view / …
)


# ---------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------

TARGETS: tuple[str, ...] = (
    "price",            # primary forecaster target (log-transformed at training time)
    "price_per_night",
    "sur_demande",
)

# Locked 2026-05-20: forecaster trains on log(price) with `nights` as a feature.
# NOT log1p(price_per_night) — price/nights bakes in a 1/nights artefact because
# multi-night stays carry discounts. See ml/CLAUDE.md "Locked decisions".
FORECASTER_TARGET: str = "price"
FORECASTER_TARGET_TRANSFORM: str = "log"


# ---------------------------------------------------------------------------
# Leaky-observed features
# ---------------------------------------------------------------------------
# Derived from the row's *own* price_per_night in competitive_features:
#   observed_delta = (price_per_night − peer_median) / peer_median × 100
#   observed_rank  = percentile of own price among peers
#
# Leaky for the forecaster (which predicts price_per_night — circular).
# Safe for anomaly detection / recommender (price is already known).

LEAKY_OBSERVED_FEATURES: tuple[str, ...] = tuple(
    col
    for g in GRANULARITIES
    for col in (
        f"observed_delta_vs_peer_{g}_median_pct",
        f"observed_rank_in_peer_{g}",
    )
)  # 6 columns


# ---------------------------------------------------------------------------
# Predictive features
# ---------------------------------------------------------------------------

_PEER_AGG_STATS = ("count", "median", "p25", "p75", "min", "max", "std")

_PEER_COLUMNS: tuple[str, ...] = tuple(
    f"peer_{g}_{stat}"
    for g in GRANULARITIES
    for stat in _PEER_AGG_STATS
)

_DEMAND_COLUMNS: tuple[str, ...] = (
    *SUR_DEMANDE_SLICES.keys(),     # 3 sur_demande_rate_* cols
    ACTIVITY_COUNT_COL,             # city_activity_count_checkin
)

# Calendar features — deterministic functions of check_in only.
# Stored in calendar_dim, not in hotel_features.
CALENDAR_FEATURES: tuple[str, ...] = (
    "check_in_dow",
    "check_in_month",
    "check_in_week_of_year",
    "check_in_day_of_month",
    "check_in_quarter",
    "is_weekend_checkin",
    "is_ramadan",
    "is_tunisia_public_holiday",
    "is_tunisia_school_holiday",
    "is_school_holiday_france",
    "is_school_holiday_germany",
    "is_school_holiday_uk",
    "days_to_nearest_european_holiday",
)

# Segment features — deterministic functions of (city_name, stars_int).
# Stored in segment_dim, not in hotel_features.
SEGMENT_FEATURES: tuple[str, ...] = (
    MACRO_REGION_COL,       # "macro_region"
    STARS_BAND_COL,         # "stars_band"
    MARKET_SEGMENT_ID_COL,  # "market_segment_id"
)

PREDICTIVE_FEATURES: tuple[str, ...] = (
    # Raw quantitative inputs
    "nights",
    "days_until_checkin",
    "adults",
    "children",
    # Taxonomic canonicals (raw text counterparts are in IDENTIFIER_COLUMNS)
    "stars_int",
    "boarding_canonical",
    "room_base",
    "room_view",
    "room_tier",
    "room_occupancy",
    # Supplement-expansion flags
    "is_supplement_variant",
    "view_upgrade_count_offered",
    "has_free_view_upgrade",
    # Calendar (13 cols — dim-table partition, joined at serve time)
    *CALENDAR_FEATURES,
    # Segment (3 cols — dim-table partition, joined at serve time)
    *SEGMENT_FEATURES,
    # Peer-group aggregates (21 cols)
    *_PEER_COLUMNS,
    # Best-granularity selector
    "best_peer_granularity_used",
    # Demand-proxy aggregates (4 cols)
    *_DEMAND_COLUMNS,
)


# ---------------------------------------------------------------------------
# Per-model feature lists
# ---------------------------------------------------------------------------

FORECASTER_FEATURES: tuple[str, ...] = PREDICTIVE_FEATURES

ANOMALY_FEATURES: tuple[str, ...] = (
    *PREDICTIVE_FEATURES,
    *LEAKY_OBSERVED_FEATURES,
    "price_per_night",
)

RECOMMENDER_FEATURES: tuple[str, ...] = ANOMALY_FEATURES

FEATURES_FOR: dict[str, tuple[str, ...]] = {
    "forecaster":  FORECASTER_FEATURES,
    "anomaly":     ANOMALY_FEATURES,
    "recommender": RECOMMENDER_FEATURES,
}


# ---------------------------------------------------------------------------
# Storage projections
# ---------------------------------------------------------------------------

_EXCLUDED_FROM_POSTGRES: frozenset[str] = frozenset(
    # Calendar → calendar_dim (joined at serve time)
    list(CALENDAR_FEATURES)
    # Segment → segment_dim (joined at serve time)
    + list(SEGMENT_FEATURES)
    # Raw text superseded by canonicals (stars_int, boarding_canonical, room_*)
    + ["stars", "boarding_name", "room_name"]
    # Observed-delta / rank → derivable as a Postgres VIEW from
    # price_per_night and peer_{g}_median (both remain in hotel_features)
    + list(LEAKY_OBSERVED_FEATURES)
)

POSTGRES_SERVING_COLUMNS: tuple[str, ...] = tuple(
    col
    for col in (
        *IDENTIFIER_COLUMNS,
        *TARGETS,
        "nights",
        "days_until_checkin",
        "adults",
        "children",
        "stars_int",
        "boarding_canonical",
        "room_base",
        "room_view",
        "room_tier",
        "room_occupancy",
        "is_supplement_variant",
        "view_upgrade_count_offered",
        "has_free_view_upgrade",
        *_PEER_COLUMNS,
        "best_peer_granularity_used",
        *_DEMAND_COLUMNS,
    )
    if col not in _EXCLUDED_FROM_POSTGRES
)

# Dim-table column lists (join key + payload columns).
CALENDAR_DIM_COLUMNS: tuple[str, ...] = (
    "check_in",         # join key
    *CALENDAR_FEATURES,
)

SEGMENT_DIM_COLUMNS: tuple[str, ...] = (
    "city_name",        # join key (part 1)
    "stars_int",        # join key (part 2)
    *SEGMENT_FEATURES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def select_features(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Return a view of ``df`` with only the columns for ``model``.

    Parameters
    ----------
    df:
        Fully-assembled feature frame.
    model:
        One of ``"forecaster"``, ``"anomaly"``, ``"recommender"``.

    Returns
    -------
    pd.DataFrame
        Column subset (no copy — a pandas view where possible).

    Raises
    ------
    KeyError
        If ``model`` is not in FEATURES_FOR.
    RuntimeError
        If any required column is missing from ``df``.
    """
    if model not in FEATURES_FOR:
        raise KeyError(f"Unknown model {model!r}. Valid: {list(FEATURES_FOR)}")
    cols = list(FEATURES_FOR[model])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"select_features({model!r}): missing columns: {missing}"
        )
    return df[cols]


def assert_no_leakage(columns: Sequence[str], model: str) -> None:
    """
    Raise ``ValueError`` if ``columns`` contains a leaky column for the
    forecaster. No-op for anomaly and recommender (leaky cols are safe there).

    Parameters
    ----------
    columns:
        Iterable of column names to audit.
    model:
        Model name. Only ``"forecaster"`` is audited; others are skipped.

    Raises
    ------
    ValueError
        For the forecaster, if any column is in LEAKY_OBSERVED_FEATURES or
        TARGETS.
    """
    if model != "forecaster":
        return
    forbidden = frozenset(LEAKY_OBSERVED_FEATURES) | frozenset(TARGETS)
    bad = [c for c in columns if c in forbidden]
    if bad:
        raise ValueError(
            f"assert_no_leakage(forecaster): leaky/target columns found: {bad}. "
            "These columns are derived from the row's own price_per_night and "
            "must not enter the forecaster's feature matrix."
        )
