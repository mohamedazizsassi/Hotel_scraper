"""
Leakage-critical peer-group aggregates.

For every observation, summarise the *other* hotels offering a comparable
product on the same date. Three granularities are produced so the model
can pick the tightest peer set actually populated for each row:

* ``tight``  — same city, stars, boarding, room_base, room_view, nights,
  adults, check_in. Closest-substitutes definition.
* ``medium`` — drops room_base / room_view. The primary modelling
  granularity; survives sparsity in room taxonomy.
* ``loose``  — only city, stars, nights, check_in. Always populated;
  used as a safety-net fallback.

Hard contract — non-negotiable: every aggregate excludes rows belonging
to the row's own hotel (matched on ``hotel_name_normalized``). Failing
this is a leakage bug that silently inflates model-quality numbers.
``test_leakage.py`` enforces it from outside this module too.

Contract (what downstream modules can assume after ``add_competitive_features``):

For each ``g`` in {tight, medium, loose}:
    * peer_{g}_count    -> Int32 (>= 0; 0 if no peers in slice)
    * peer_{g}_median   -> float64 or NaN if count == 0
    * peer_{g}_p25      -> float64 or NaN
    * peer_{g}_p75      -> float64 or NaN
    * peer_{g}_min      -> float64 or NaN
    * peer_{g}_max      -> float64 or NaN
    * peer_{g}_std      -> float64 or NaN (NaN when count < 2)
    * delta_vs_peer_{g}_median_pct -> float64 or NaN
    * rank_in_peer_{g}  -> float64 in [0, 1] or NaN  (own-price percentile)

Plus:
    * best_peer_granularity_used -> string in {tight, medium, loose} or
      <NA> if no granularity reaches MIN_PEERS_FOR_BEST.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Peer-group definitions — order is significant: from tightest to loosest.
# ``best_peer_granularity_used`` walks this list and stops at the first
# granularity with peer_count >= MIN_PEERS_FOR_BEST.
#
# ``scrape_date`` is the first key in every granularity. This is the
# temporal-leakage fix from the 2026-05-14 audit: peer aggregates pool
# only rows captured on the same scrape day, so a row scraped on day D
# cannot see peer prices from day D+1, D+2, etc. The fix preserves
# every (offer × scrape_run) observation as a separate training row —
# unlike a dedup-to-most-recent approach, the per-offer booking-window
# trajectory survives.
# ---------------------------------------------------------------------------

PEER_GROUP_KEYS: dict[str, tuple[str, ...]] = {
    "tight":  ("scrape_date", "city_name", "stars_int", "boarding_canonical",
               "room_base", "room_view", "nights", "adults", "check_in"),
    "medium": ("scrape_date", "city_name", "stars_int", "boarding_canonical",
               "nights", "adults", "check_in"),
    "loose":  ("scrape_date", "city_name", "stars_int", "nights", "check_in"),
}

GRANULARITIES: tuple[str, ...] = ("tight", "medium", "loose")

# A peer slice is "usable" once it carries this many other hotels.
# Five is the minimum that produces a non-degenerate p25/p75 spread.
MIN_PEERS_FOR_BEST: int = 5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_competitive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add peer-group aggregate columns (one block per granularity) plus a
    ``best_peer_granularity_used`` selector.

    Parameters
    ----------
    df:
        Frame produced by ``taxonomy`` (post-supplement-expansion +
        post-room-parsing). Required columns:
        ``hotel_name_normalized``, ``price_per_night``, plus every column
        referenced in ``PEER_GROUP_KEYS``.

    Returns
    -------
    pd.DataFrame
        New DataFrame with the added columns. Input is not modified.

    Raises
    ------
    RuntimeError
        If a required column is missing.
    """
    _assert_required_columns(df)

    out = df.copy()
    for g in GRANULARITIES:
        block = _peer_stats_for_granularity(out, g)
        out = pd.concat([out, block], axis=1)

    out["best_peer_granularity_used"] = _select_best_granularity(out)

    _log_summary(out)
    return out


# ---------------------------------------------------------------------------
# Per-granularity computation
# ---------------------------------------------------------------------------

def _peer_stats_for_granularity(df: pd.DataFrame, granularity: str) -> pd.DataFrame:
    """
    Compute peer aggregates for one granularity.

    Implementation: groupby the peer keys, then within each group iterate
    rows and compute aggregates over OTHER hotels (mask on
    ``hotel_name_normalized != own``). Numpy arrays are pre-allocated
    per row so the output aligns with ``df.index`` regardless of group
    iteration order.
    """
    keys = list(PEER_GROUP_KEYS[granularity])
    n = len(df)

    count   = np.zeros(n, dtype=np.int64)
    median  = np.full(n, np.nan)
    p25     = np.full(n, np.nan)
    p75     = np.full(n, np.nan)
    pmin    = np.full(n, np.nan)
    pmax    = np.full(n, np.nan)
    pstd    = np.full(n, np.nan)
    rank    = np.full(n, np.nan)

    hotels = df["hotel_name_normalized"].astype("string[python]").to_numpy()
    prices = pd.to_numeric(df["price_per_night"], errors="coerce").to_numpy(dtype=np.float64)

    # df.index → positional index 0..n-1 for the np arrays above.
    pos_of = pd.Series(np.arange(n), index=df.index)

    grouped = df.groupby(keys, dropna=False, sort=False, observed=True)

    for _, idx in grouped.groups.items():
        positions = pos_of.loc[idx].to_numpy()
        h_grp = hotels[positions]
        p_grp = prices[positions]

        for pos, own_hotel, own_price in zip(positions, h_grp, p_grp):
            mask = (h_grp != own_hotel) & (~np.isnan(p_grp))
            peers = p_grp[mask]
            cnt = peers.size
            count[pos] = cnt
            if cnt == 0:
                continue
            median[pos] = float(np.median(peers))
            p25[pos]    = float(np.percentile(peers, 25))
            p75[pos]    = float(np.percentile(peers, 75))
            pmin[pos]   = float(peers.min())
            pmax[pos]   = float(peers.max())
            pstd[pos]   = float(peers.std(ddof=0)) if cnt > 1 else np.nan
            if not np.isnan(own_price):
                below = float((peers < own_price).sum())
                equal = float((peers == own_price).sum())
                rank[pos] = (below + 0.5 * equal) / cnt

    with np.errstate(divide="ignore", invalid="ignore"):
        delta_pct = (prices - median) / median * 100.0

    block = pd.DataFrame(
        {
            f"peer_{granularity}_count":  pd.array(count, dtype="Int32"),
            f"peer_{granularity}_median": median,
            f"peer_{granularity}_p25":    p25,
            f"peer_{granularity}_p75":    p75,
            f"peer_{granularity}_min":    pmin,
            f"peer_{granularity}_max":    pmax,
            f"peer_{granularity}_std":    pstd,
            f"delta_vs_peer_{granularity}_median_pct": delta_pct,
            f"rank_in_peer_{granularity}": rank,
        },
        index=df.index,
    )
    return block


# ---------------------------------------------------------------------------
# Best-granularity selector
# ---------------------------------------------------------------------------

def _select_best_granularity(df: pd.DataFrame) -> pd.Series:
    """
    For each row pick the tightest granularity whose peer_count meets
    ``MIN_PEERS_FOR_BEST``. NA when no granularity qualifies.
    """
    n = len(df)
    out = pd.Series(pd.array([pd.NA] * n, dtype="string[python]"), index=df.index)
    for g in GRANULARITIES:  # tight first
        col = df[f"peer_{g}_count"]
        qualifies = col.fillna(0).astype("int64") >= MIN_PEERS_FOR_BEST
        unset = out.isna()
        target = qualifies & unset
        if target.any():
            out.loc[target] = g
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_required_columns(df: pd.DataFrame) -> None:
    needed: set[str] = {"hotel_name_normalized", "price_per_night"}
    for keys in PEER_GROUP_KEYS.values():
        needed.update(keys)
    missing: list[str] = [c for c in sorted(needed) if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"add_competitive_features: missing required columns: {missing}"
        )


def _log_summary(df: pd.DataFrame) -> None:
    for g in GRANULARITIES:
        cnt = df[f"peer_{g}_count"]
        zero = int((cnt == 0).sum())
        logger.info(
            "competitive[%s]: median_peer_count=%.1f  rows_with_zero_peers=%d",
            g, float(cnt.median()), zero,
        )
    sel = df["best_peer_granularity_used"]
    breakdown = sel.value_counts(dropna=False).to_dict()
    logger.info("best_peer_granularity_used distribution: %s", breakdown)


# Public re-export for tests / leakage checks that want the canonical key list.
def peer_keys(granularity: str) -> Sequence[str]:
    """Return the column tuple defining the peer slice for a granularity."""
    return PEER_GROUP_KEYS[granularity]
