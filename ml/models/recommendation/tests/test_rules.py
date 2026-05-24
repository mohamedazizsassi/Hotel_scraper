"""Tests for the D4 rule library."""
from __future__ import annotations

import math

import pytest

from models.recommendation.rules import (
    DEFAULT_RULES,
    Recommendation,
    RowContext,
    apply_rules,
)


def _ctx(
    *,
    current_price=1000.0,
    current_per_night=None,           # default: current_price / 1 night
    q10_tnd=800.0,
    q50_tnd=1000.0,
    q90_tnd=1200.0,
    peer_medium_median=None,
    peer_medium_count=0,
):
    return RowContext(
        current_price_tnd=current_price,
        current_price_per_night_tnd=current_per_night if current_per_night is not None else current_price,
        q10_cal_tnd=q10_tnd,
        q50_tnd=q50_tnd,
        q90_cal_tnd=q90_tnd,
        peer_medium_median_per_night_tnd=peer_medium_median,
        peer_medium_count=peer_medium_count,
    )


# --- Rule 1: below band → raise ----------------------------------------------

def test_rule_below_band_raises_to_q50():
    rec = apply_rules(_ctx(current_price=700.0))  # below 800
    assert rec.direction == "raise"
    assert rec.recommended_price_tnd == pytest.approx(1000.0)
    assert any("below the calibrated lower bound" in r for r in rec.reasons)


# --- Rule 2: above band → lower ----------------------------------------------

def test_rule_above_band_lowers_to_q50():
    rec = apply_rules(_ctx(current_price=1500.0))  # above 1200
    assert rec.direction == "lower"
    assert rec.recommended_price_tnd == pytest.approx(1000.0)
    assert any("above the calibrated upper bound" in r for r in rec.reasons)


# --- Rule 3: inside band → hold ----------------------------------------------

def test_rule_in_band_holds():
    rec = apply_rules(_ctx(current_price=1000.0))
    assert rec.direction == "hold"
    assert rec.recommended_price_tnd == pytest.approx(1000.0)
    assert any("inside the calibrated" in r for r in rec.reasons)


def test_rule_in_band_at_lower_edge_still_holds():
    # current == q10_cal: predicate is strict <, so this is in-band.
    rec = apply_rules(_ctx(current_price=800.0))
    assert rec.direction == "hold"


def test_rule_in_band_at_upper_edge_still_holds():
    # current == q90_cal: predicate is strict >, so this is in-band.
    rec = apply_rules(_ctx(current_price=1200.0))
    assert rec.direction == "hold"


# --- Rules 4-5: peer notes only when sample size is sufficient ---------------
# Peer rules use per-night scale (current_per_night vs peer_medium_median).

def test_rule_peer_premium_note_appended_when_above_median_by_25pct():
    # in band on multi-night scale (current=1000, q10=800, q90=1200);
    # per-night premium 1000/1n=1000 vs peer 750 → +33% > 20%
    rec = apply_rules(_ctx(
        current_price=1000.0, current_per_night=1000.0,
        peer_medium_median=750.0, peer_medium_count=8,
    ))
    assert rec.direction == "hold"
    assert any("above the median" in r for r in rec.reasons)
    assert len(rec.reasons) == 2  # in_band + premium note


def test_rule_peer_discount_note_appended_when_below_median_by_25pct():
    rec = apply_rules(_ctx(
        current_price=1000.0, current_per_night=1000.0,
        peer_medium_median=1500.0, peer_medium_count=10,
    ))
    assert rec.direction == "hold"
    assert any("below the median" in r for r in rec.reasons)
    assert len(rec.reasons) == 2


def test_rule_peer_note_skipped_when_count_below_threshold():
    rec = apply_rules(_ctx(
        current_price=1000.0, current_per_night=1000.0,
        peer_medium_median=2000.0, peer_medium_count=3,
    ))
    # peer_medium_count=3 < 5 threshold → no peer note
    assert len(rec.reasons) == 1


def test_rule_peer_note_skipped_when_median_is_none():
    rec = apply_rules(_ctx(
        current_price=1000.0, current_per_night=1000.0,
        peer_medium_median=None, peer_medium_count=99,
    ))
    assert len(rec.reasons) == 1


def test_rule_peer_note_skipped_when_gap_below_20pct():
    # 10% premium on per-night scale → no note (threshold is >20%)
    rec = apply_rules(_ctx(
        current_price=1100.0, current_per_night=1100.0,
        peer_medium_median=1000.0, peer_medium_count=10,
    ))
    assert len(rec.reasons) == 1  # in_band only


def test_rule_peer_note_uses_per_night_scale_not_multi_night():
    # 3-night stay: total 1500, per-night 500. Peer per-night 600.
    # On multi-night scale, current 1500 vs peer 600 would falsely look like
    # +150% premium. On per-night (correct) scale: 500 vs 600 = −17%, below
    # the 20% threshold → no note expected.
    rec = apply_rules(_ctx(
        current_price=1500.0,           # multi-night TND
        current_per_night=500.0,        # per-night TND
        q10_tnd=1400.0, q50_tnd=1500.0, q90_tnd=1600.0,
        peer_medium_median=600.0,       # per-night TND
        peer_medium_count=10,
    ))
    assert rec.direction == "hold"
    assert len(rec.reasons) == 1  # in_band only; peer gap 17% is below threshold


# --- Combined: below band + peer discount note both fire ---------------------

def test_below_band_and_peer_discount_combine():
    rec = apply_rules(_ctx(
        current_price=500.0,            # multi-night, below q10=800
        current_per_night=500.0,        # per-night
        peer_medium_median=900.0,       # per-night; current is 44% below
        peer_medium_count=12,
    ))
    assert rec.direction == "raise"
    assert rec.recommended_price_tnd == pytest.approx(1000.0)
    # Two reasons: below_band primary + peer_discount note
    assert len(rec.reasons) == 2
    assert any("below the calibrated lower bound" in r for r in rec.reasons)
    assert any("below the median" in r for r in rec.reasons)


# --- Determinism --------------------------------------------------------------

def test_apply_rules_is_deterministic():
    ctx = _ctx(current_price=1000.0)
    a = apply_rules(ctx)
    b = apply_rules(ctx)
    assert a == b


# --- Rule list shape ----------------------------------------------------------

def test_default_rules_is_non_empty_and_ordered():
    assert len(DEFAULT_RULES) >= 3
    # First three rules must partition the row space (below / above / hold)
    # Defensive check: in any RowContext exactly one of {raise, lower, hold} fires.
    ctx = _ctx(current_price=1000.0)
    rec = apply_rules(ctx, rules=DEFAULT_RULES[:3])
    assert rec.direction in {"raise", "lower", "hold"}


# --- Recommendation dataclass invariants -------------------------------------

def test_recommendation_direction_must_be_valid():
    with pytest.raises(ValueError, match="direction"):
        Recommendation(direction="up", recommended_price_tnd=1.0, reasons=["x"])


def test_recommendation_recommended_price_must_be_positive():
    with pytest.raises(ValueError, match="recommended_price_tnd"):
        Recommendation(direction="hold", recommended_price_tnd=-1.0, reasons=["x"])


def test_recommendation_reasons_non_empty():
    with pytest.raises(ValueError, match="reasons"):
        Recommendation(direction="hold", recommended_price_tnd=1.0, reasons=[])
