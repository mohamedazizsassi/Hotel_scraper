"""
Market-segment columns: macro_region, stars_band, market_segment_id.

Output of this stage drives the per-segment forecaster (one LightGBM
per market_segment_id) and the manager-facing segment label in the
recommender. The mapping is intentionally hand-curated and lives in
``config.py`` -- adding a new city is a small, reviewable edit, not a
silent data-derived clustering.

Contract (what downstream modules can assume after add_segment_features):
    * macro_region       -> str in MACRO_REGIONS
    * stars_band         -> str in STARS_BANDS
    * market_segment_id  -> f"{macro_region}_{stars_band}"

Failure mode: an unmapped city raises RuntimeError. This is intentional:
the scraper occasionally surfaces a new city, and silently bucketing it
into a default region would produce a bogus segment that trains a model
on the wrong peer group. Loud failure forces the operator to update the
mapping in config.py.
"""

from __future__ import annotations

import logging

import pandas as pd

from .config import CITY_TO_MACRO_REGION, STARS_BAND_BOUNDARIES

logger = logging.getLogger(__name__)

MACRO_REGION_COL: str = "macro_region"
STARS_BAND_COL: str = "stars_band"
MARKET_SEGMENT_ID_COL: str = "market_segment_id"


def add_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add macro_region, stars_band, and market_segment_id.

    Parameters
    ----------
    df:
        Frame after taxonomy/calendar (must contain ``city_name`` and
        ``stars_int``).

    Returns
    -------
    pd.DataFrame
        New DataFrame with three columns appended. Input is not modified.

    Raises
    ------
    RuntimeError
        If ``city_name`` or ``stars_int`` are missing, or if any
        non-null ``city_name`` value is not in
        ``CITY_TO_MACRO_REGION``.
    """
    for col in ("city_name", "stars_int"):
        if col not in df.columns:
            raise RuntimeError(f"add_segment_features: missing required column '{col}'")

    unknown = sorted(set(df["city_name"].dropna()) - set(CITY_TO_MACRO_REGION))
    if unknown:
        raise RuntimeError(
            f"add_segment_features: {len(unknown)} unmapped city_name value(s); "
            f"add them to CITY_TO_MACRO_REGION in config.py. Sample: {unknown[:10]}"
        )

    out = df.copy()
    out[MACRO_REGION_COL] = out["city_name"].map(CITY_TO_MACRO_REGION).astype("string[python]")
    out[STARS_BAND_COL] = (
        pd.to_numeric(out["stars_int"], errors="coerce")
          .map(STARS_BAND_BOUNDARIES)
          .astype("string[python]")
    )
    out[MARKET_SEGMENT_ID_COL] = (
        out[MACRO_REGION_COL].astype(str) + "_" + out[STARS_BAND_COL].astype(str)
    ).astype("string[python]")
    # Rows where stars_band is NA (stars_int outside 1-5) get a NA segment id.
    out.loc[out[STARS_BAND_COL].isna(), MARKET_SEGMENT_ID_COL] = pd.NA

    n_segments = int(out[MARKET_SEGMENT_ID_COL].nunique(dropna=True))
    logger.info(
        "add_segment_features: %d distinct segments, %d rows with NA segment",
        n_segments, int(out[MARKET_SEGMENT_ID_COL].isna().sum()),
    )
    return out
