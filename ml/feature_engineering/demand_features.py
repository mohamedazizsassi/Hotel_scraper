"""
Demand-proxy features derived from ``sur_demande``.

Bookings data does not exist in this pipeline, so demand pressure is
approximated from the only signal available: how often hotels in a given
slice are sold *on request* (``sur_demande=True``) rather than at a
quoted price. A higher rate inside a (city, check_in) slice is read as
tighter inventory, which downstream models can use as a market-pressure
feature.

Same self-exclusion discipline as ``competitive_features``:

* Aggregates exclude rows belonging to the row's own hotel (matched on
  ``hotel_name_normalized``). A hotel's own ``sur_demande`` flag must
  not enter the slice rate it sees.
* Empty slices yield NaN rate and zero count — never a silent zero.

Contract (what downstream modules can assume after ``add_demand_features``):
    * sur_demande_rate_city_checkin                  -> float64 in [0, 1] or NaN
    * sur_demande_rate_city_stars_checkin            -> float64 in [0, 1] or NaN
    * sur_demande_rate_city_stars_boarding_checkin   -> float64 in [0, 1] or NaN
    * city_activity_count_checkin                    -> Int32 (>= 0)

The activity count excludes the row's own hotel rows for consistency
with the rates: it answers "how many *other* observations exist in my
city on my check_in", which is the leakage-safe form of the
market-thickness signal.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Slice definitions — order does not matter here, each rate is independent.
# ---------------------------------------------------------------------------

SUR_DEMANDE_SLICES: dict[str, tuple[str, ...]] = {
    "sur_demande_rate_city_checkin":
        ("city_name", "check_in"),
    "sur_demande_rate_city_stars_checkin":
        ("city_name", "stars_int", "check_in"),
    "sur_demande_rate_city_stars_boarding_checkin":
        ("city_name", "stars_int", "boarding_canonical", "check_in"),
}

ACTIVITY_COUNT_KEYS: tuple[str, ...] = ("city_name", "check_in")
ACTIVITY_COUNT_COL: str = "city_activity_count_checkin"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_demand_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``sur_demande``-based demand-proxy aggregates and a slice
    activity count.

    Parameters
    ----------
    df:
        Frame produced by ``competitive_features`` (or by taxonomy if
        run independently). Required columns: ``sur_demande``,
        ``hotel_name_normalized``, plus every column referenced in
        ``SUR_DEMANDE_SLICES`` and ``ACTIVITY_COUNT_KEYS``.

    Returns
    -------
    pd.DataFrame
        New DataFrame with the four added columns. Input is not modified.

    Raises
    ------
    RuntimeError
        If a required column is missing.
    """
    _assert_required_columns(df)

    out = df.copy()
    for col_name, keys in SUR_DEMANDE_SLICES.items():
        out[col_name] = _self_excluding_rate(out, list(keys))
    out[ACTIVITY_COUNT_COL] = _self_excluding_count(out, list(ACTIVITY_COUNT_KEYS))

    _log_summary(out)
    return out


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------

def _self_excluding_rate(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """
    For each row, mean of ``sur_demande`` over rows in the same slice
    whose ``hotel_name_normalized`` differs. NaN ``sur_demande`` values
    are dropped from both numerator and denominator.

    Returns a float64 Series indexed by ``df.index``. NaN where the row
    has no other-hotel observations in its slice (or all such
    observations are NaN ``sur_demande``).
    """
    n = len(df)
    out = np.full(n, np.nan)

    pos_of = pd.Series(np.arange(n), index=df.index)
    hotels = df["hotel_name_normalized"].astype("string[python]").to_numpy()
    sd_raw = df["sur_demande"]
    # Convert to a (bool-or-NaN as float) array: True->1.0, False->0.0,
    # missing-> NaN. Pandas BooleanDtype with NA handles this cleanly via
    # the underlying mask.
    sd = pd.to_numeric(sd_raw.astype("Float64"), errors="coerce").to_numpy(dtype=np.float64)

    grouped = df.groupby(keys, dropna=False, sort=False, observed=True).groups
    for _, idx in grouped.items():
        positions = pos_of.loc[idx].to_numpy()
        h_grp = hotels[positions]
        s_grp = sd[positions]
        for pos, own_hotel in zip(positions, h_grp):
            mask = (h_grp != own_hotel) & (~np.isnan(s_grp))
            if not mask.any():
                continue
            out[pos] = float(s_grp[mask].mean())

    return pd.Series(out, index=df.index, dtype="float64")


def _self_excluding_count(df: pd.DataFrame, keys: list[str]) -> pd.Series:
    """
    For each row, count of rows in the same slice whose
    ``hotel_name_normalized`` differs.

    Implementation: total rows per slice minus rows of the same hotel
    inside that slice. Vectorised — no per-row Python loop.
    """
    slice_total = df.groupby(keys, dropna=False, sort=False, observed=True)[
        "hotel_name_normalized"
    ].transform("size")
    own_hotel_in_slice = df.groupby(
        keys + ["hotel_name_normalized"], dropna=False, sort=False, observed=True,
    )["hotel_name_normalized"].transform("size")
    count = (slice_total - own_hotel_in_slice).astype("int64")
    return count.astype("Int32")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_required_columns(df: pd.DataFrame) -> None:
    needed: set[str] = {"sur_demande", "hotel_name_normalized"}
    for keys in SUR_DEMANDE_SLICES.values():
        needed.update(keys)
    needed.update(ACTIVITY_COUNT_KEYS)
    missing = [c for c in sorted(needed) if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"add_demand_features: missing required columns: {missing}"
        )


def _log_summary(df: pd.DataFrame) -> None:
    for col in SUR_DEMANDE_SLICES:
        s = df[col]
        non_null = int(s.notna().sum())
        mean = float(s.mean()) if non_null else float("nan")
        logger.info(
            "%s: non_null=%d  mean=%.4f", col, non_null, mean,
        )
    cnt = df[ACTIVITY_COUNT_COL]
    logger.info(
        "%s: median=%.1f  max=%d  rows_with_zero=%d",
        ACTIVITY_COUNT_COL, float(cnt.median()), int(cnt.max()),
        int((cnt == 0).sum()),
    )
