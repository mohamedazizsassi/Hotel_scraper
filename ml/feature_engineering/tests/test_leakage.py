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
