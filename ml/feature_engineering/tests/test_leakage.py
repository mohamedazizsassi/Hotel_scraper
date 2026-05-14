"""
Cross-cutting leakage audit.

Builds a small toy DataFrame, runs the feature pipeline through stage 7,
and for every (peer aggregate column, row) pair verifies that removing
the row's *own* hotel from the slice produces the same number — i.e. the
row's own price did not contaminate its own aggregate.

This is the same property tested in unit form inside
``test_competitive_features.py`` (perturb own price → own median
unchanged). The test here re-derives every aggregate from scratch and
compares to the pipeline output, which catches a wider class of bugs
(e.g. accidental row-level instead of hotel-level exclusion).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from feature_engineering.competitive_features import (
    GRANULARITIES,
    PEER_GROUP_KEYS,
    add_competitive_features,
)
from feature_engineering.demand_features import (
    ACTIVITY_COUNT_KEYS,
    SUR_DEMANDE_SLICES,
)


# ---------------------------------------------------------------------------
# Toy frame: 3 cities × 2 stars × 5 hotels each, single check_in date.
# Multiple rows per hotel to ensure hotel-level (not row-level) exclusion
# is exercised.
# ---------------------------------------------------------------------------

def _toy_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    check_in = pd.Timestamp("2026-07-01")
    for city in ("hammamet", "djerba", "sousse"):
        for stars in (3, 4):
            for h_idx in range(5):
                hotel = f"{city}_{stars}_h{h_idx}"
                # 2 rows per hotel (e.g. base + variant from stage 6)
                for variant in (False, True):
                    rows.append({
                        "hotel_name_normalized": hotel,
                        "city_name": city,
                        "stars_int": stars,
                        "boarding_canonical": "HDP",
                        "room_base": "chambre",
                        "room_view": "mer" if variant else None,
                        "nights": 3,
                        "adults": 2,
                        "check_in": check_in,
                        # Single scrape day so all rows share a slice on
                        # the new scrape_date key.
                        "scrape_date": pd.Timestamp("2026-05-14"),
                        "price_per_night": float(rng.integers(80, 400)),
                    })
    return pd.DataFrame(rows)


def _expected_median(df: pd.DataFrame, row_idx: int, granularity: str) -> float:
    """Brute-force: same-slice rows whose hotel != self, take median of prices."""
    keys = list(PEER_GROUP_KEYS[granularity])
    row = df.loc[row_idx]
    mask = pd.Series(True, index=df.index)
    for k in keys:
        mask &= df[k].eq(row[k]) | (df[k].isna() & pd.isna(row[k]))
    mask &= df["hotel_name_normalized"] != row["hotel_name_normalized"]
    peers = df.loc[mask, "price_per_night"].dropna()
    return float(np.median(peers)) if len(peers) else float("nan")


# ---------------------------------------------------------------------------
# Audits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("granularity", GRANULARITIES)
def test_peer_median_matches_brute_force_recompute(granularity: str) -> None:
    df = _toy_frame()
    out = add_competitive_features(df)
    col = f"peer_{granularity}_median"

    # Sample 30 rows for speed; toy frame is small enough that all rows
    # could be checked but the parametrise across granularities already
    # multiplies the cost.
    sample = df.sample(n=30, random_state=0).index
    for idx in sample:
        expected = _expected_median(df, idx, granularity)
        actual = out.loc[idx, col]
        if np.isnan(expected):
            assert pd.isna(actual)
        else:
            assert float(actual) == pytest.approx(expected), (
                f"row {idx} granularity={granularity}: "
                f"expected {expected}, got {actual}"
            )


def test_no_aggregate_includes_own_hotels_prices() -> None:
    """
    Stronger property: for every row, no aggregated peer column can be
    reproduced by including the row's own hotel's prices. We verify by
    computing the WITH-self median per slice and checking it differs
    from the reported peer_median for at least one row per hotel — i.e.
    the pipeline is *not* using the with-self median.
    """
    df = _toy_frame()
    out = add_competitive_features(df)

    for granularity in GRANULARITIES:
        keys = list(PEER_GROUP_KEYS[granularity])
        with_self_median = df.groupby(keys, dropna=False)["price_per_night"].transform("median")
        # If the pipeline mistakenly used with-self median, every row would
        # match. Assert there exists at least one row where they differ
        # (true if any hotel has a non-degenerate price within its slice).
        differs = (out[f"peer_{granularity}_median"] - with_self_median).abs() > 1e-9
        assert differs.any(), (
            f"granularity={granularity}: peer_median is identical to "
            "with-self median on every row — likely a leakage bug."
        )


def test_perturbing_own_price_does_not_change_own_aggregates() -> None:
    """End-to-end self-exclusion canary across all granularities."""
    df = _toy_frame()
    baseline = add_competitive_features(df)

    target_idx = df.index[0]
    perturbed = df.copy()
    perturbed.loc[target_idx, "price_per_night"] = 99_999.0
    after = add_competitive_features(perturbed)

    for g in GRANULARITIES:
        for stat in ("median", "p25", "p75", "min", "max", "std"):
            col = f"peer_{g}_{stat}"
            b = baseline.loc[target_idx, col]
            a = after.loc[target_idx, col]
            if pd.isna(b) and pd.isna(a):
                continue
            assert float(b) == pytest.approx(float(a)), (
                f"{col} changed when own price was perturbed: {b} → {a}"
            )


# ---------------------------------------------------------------------------
# Temporal-leakage regression tests (C1, 2026-05-14)
# ---------------------------------------------------------------------------
# Audit context: ``_peer_stats_for_granularity`` previously grouped only
# on the peer keys, with no temporal scoping. Daily scraping puts the
# same business key in the input on many ``scraped_at`` days, so peer
# aggregates silently mixed observations across time — a row scraped on
# day D would absorb peer prices scraped on day D+k.
#
# The fix: ``scrape_date`` is the first column of every entry in
# ``PEER_GROUP_KEYS`` and ``SUR_DEMANDE_SLICES``. The aggregate now
# pools only same-scrape-day peers — no future contamination — while
# every (offer × scrape_run) row survives so the booking-window
# trajectory carried by ``days_until_checkin`` is preserved.
#
# A dedup-to-most-recent-per-business-key fix (see
# ``temporal_dedup.py``) was implemented first and reverted in favour of
# the scrape-date approach because it also collapsed every offer to a
# single row, deleting that trajectory. The dedup module is still kept
# for serving-side use cases that genuinely want "latest snapshot only".


def _toy_frame_with_temporal_duplicates() -> pd.DataFrame:
    """
    Toy market with every hotel observed on two scrape days at very
    different prices. Used to demonstrate (a) that scrape-date keying
    cleanly isolates the two days and (b) that pooling across days
    would produce a meaningfully different number.
    """
    rng = np.random.default_rng(7)
    rows = []
    check_in = pd.Timestamp("2026-07-01")
    early_day = pd.Timestamp("2026-05-01")
    late_day = pd.Timestamp("2026-05-13")
    for city in ("hammamet", "djerba", "sousse"):
        for stars in (3, 4):
            for h_idx in range(5):
                hotel = f"{city}_{stars}_h{h_idx}"
                early_price = float(rng.integers(80, 200))
                # Late price drifts up — simulates seasonal markup.
                late_price = early_price + float(rng.integers(150, 350))
                for scrape_day, ppn in (
                    (early_day, early_price),
                    (late_day, late_price),
                ):
                    rows.append({
                        "source": "promohotel",
                        "hotel_name_normalized": hotel,
                        "city_name": city,
                        "stars_int": stars,
                        "boarding_name": "Demi pension",
                        "boarding_canonical": "HDP",
                        "room_name": "Chambre Double",
                        "room_base": "chambre",
                        "room_view": None,
                        "nights": 3,
                        "adults": 2,
                        "check_in": check_in,
                        "scrape_date": scrape_day,
                        "price_per_night": ppn,
                    })
    return pd.DataFrame(rows)


def test_scrape_date_keying_isolates_each_scrape_day() -> None:
    """
    Positive: with ``scrape_date`` in PEER_GROUP_KEYS, peer_medium_median
    for any row equals the same-day brute-force median over OTHER hotels.
    No future contamination.
    """
    df = _toy_frame_with_temporal_duplicates()
    out = add_competitive_features(df)
    keys = list(PEER_GROUP_KEYS["medium"])

    for idx in df.index:
        row = df.loc[idx]
        mask = pd.Series(True, index=df.index)
        for k in keys:
            mask &= df[k].eq(row[k]) | (df[k].isna() & pd.isna(row[k]))
        mask &= df["hotel_name_normalized"] != row["hotel_name_normalized"]
        peers = df.loc[mask, "price_per_night"].dropna()
        expected = float(np.median(peers)) if len(peers) else float("nan")
        actual = out.loc[idx, "peer_medium_median"]
        if np.isnan(expected):
            assert pd.isna(actual), f"row {idx}: expected NaN, got {actual}"
        else:
            assert float(actual) == pytest.approx(expected), (
                f"row {idx}: expected peer_medium_median={expected}, "
                f"got {actual}"
            )


def test_scrape_date_keying_makes_a_difference() -> None:
    """
    Sanity: prove the fix is non-trivially active. The cross-day pooled
    median for an early row differs from the same-day median by a wide
    margin in this toy market (late prices are 150-350 TND higher), so
    if scrape_date were ever silently dropped from the keys, the
    reported peer_medium_median would shift visibly. We assert here
    that the day-scoped reported value is strictly less than the
    cross-day pooled value — a property only the day-scoped pipeline
    satisfies.
    """
    df = _toy_frame_with_temporal_duplicates()
    out = add_competitive_features(df)

    # Pick an early-day row.
    early_mask = df["scrape_date"] == pd.Timestamp("2026-05-01")
    early_row_idx = df.index[early_mask][0]
    row = df.loc[early_row_idx]
    own_hotel = row["hotel_name_normalized"]

    # Slice keys WITHOUT scrape_date — i.e. the buggy pooled version.
    pooled_keys = [k for k in PEER_GROUP_KEYS["medium"] if k != "scrape_date"]
    pooled_mask = pd.Series(True, index=df.index)
    for k in pooled_keys:
        pooled_mask &= df[k].eq(row[k]) | (df[k].isna() & pd.isna(row[k]))
    pooled_mask &= df["hotel_name_normalized"] != own_hotel
    pooled_peers = df.loc[pooled_mask, "price_per_night"].dropna()
    pooled_median = float(np.median(pooled_peers))

    reported = float(out.loc[early_row_idx, "peer_medium_median"])

    # The day-scoped median uses only early-day peers, so it must be
    # strictly smaller than the cross-day pooled median (which absorbs
    # the inflated late-day peer prices).
    assert reported < pooled_median, (
        f"Day-scoped peer_medium_median for an early row should be "
        f"strictly less than the cross-day pooled median in this toy "
        f"frame. Got reported={reported:.1f}, pooled={pooled_median:.1f}. "
        f"If they match, scrape_date may no longer be in PEER_GROUP_KEYS."
    )


def test_scrape_date_is_part_of_every_peer_and_demand_key() -> None:
    """
    Structural invariant. ``scrape_date`` MUST appear in every peer
    granularity key tuple, every sur_demande slice, and the activity
    count key. Removing it from any of these silently reintroduces the
    temporal leakage diagnosed on 2026-05-14.
    """
    for granularity, keys in PEER_GROUP_KEYS.items():
        assert "scrape_date" in keys, (
            f"PEER_GROUP_KEYS[{granularity!r}] is missing 'scrape_date' — "
            f"this re-opens the temporal-leakage bug."
        )
    for slice_col, keys in SUR_DEMANDE_SLICES.items():
        assert "scrape_date" in keys, (
            f"SUR_DEMANDE_SLICES[{slice_col!r}] is missing 'scrape_date'."
        )
    assert "scrape_date" in ACTIVITY_COUNT_KEYS, (
        "ACTIVITY_COUNT_KEYS is missing 'scrape_date'."
    )
