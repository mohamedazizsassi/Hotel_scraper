"""
Rule library for the D4 recommender.

Each rule is a pure function from RowContext to (Recommendation | None).
Rules are evaluated in order by `apply_rules`. The first rule that returns
a Recommendation with a non-None direction sets the final direction and
recommended price. Subsequent rules that return a Recommendation with
direction=None may APPEND a reason (but cannot override the direction).

The rule library is versioned with code (no external config). Re-tuning
thresholds requires a code change; that is intentional at PFE scale and
defensible to the jury (reproducible, no config drift).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

Direction = Literal["raise", "hold", "lower"]
_VALID_DIRECTIONS: frozenset[str] = frozenset({"raise", "hold", "lower"})

# Peer-note rules fire only when the peer sample is large enough to trust
# and when the gap from peer median exceeds this fraction.
_PEER_NOTE_MIN_COUNT = 5
_PEER_NOTE_MIN_GAP_FRAC = 0.20


# ---------------------------------------------------------------------------
# Inputs / outputs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RowContext:
    """Read-only view of one observation + its calibrated forecast.

    Two price representations because the forecaster trains on log(price) —
    multi-night total — while peer aggregates are computed on price_per_night
    by the feature pipeline. The two scales must not be mixed:
      - band rules (R1/R2/R3) use the multi-night fields
      - peer rules (R4/R5) use the per-night fields
    """

    current_price_tnd: float                                # multi-night total, TND
    current_price_per_night_tnd: float                      # per-night TND
    q10_cal_tnd: float                                      # multi-night total, TND
    q50_tnd: float                                          # multi-night total, TND
    q90_cal_tnd: float                                      # multi-night total, TND
    peer_medium_median_per_night_tnd: Optional[float] = None  # per-night TND
    peer_medium_count: int = 0


@dataclass
class Recommendation:
    """Direction, recommended price, and 1-3 supporting reasons.

    Validates on construction so downstream JSON consumers never see
    out-of-vocabulary direction strings or negative prices.
    """

    direction: Direction
    recommended_price_tnd: float
    reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.direction not in _VALID_DIRECTIONS:
            raise ValueError(
                f"direction must be one of {sorted(_VALID_DIRECTIONS)}, got {self.direction!r}"
            )
        if not (self.recommended_price_tnd > 0):
            raise ValueError(
                f"recommended_price_tnd must be > 0, got {self.recommended_price_tnd}"
            )
        if not self.reasons:
            raise ValueError("reasons must contain at least one entry")


Rule = Callable[[RowContext], Optional["Recommendation"]]


# ---------------------------------------------------------------------------
# Primary rules (direction-setting). Exactly one fires per row.
# ---------------------------------------------------------------------------

def _rule_below_band_raise(ctx: RowContext) -> Optional[Recommendation]:
    if ctx.current_price_tnd < ctx.q10_cal_tnd:
        reason = (
            f"current {ctx.current_price_tnd:.0f} TND is below the calibrated "
            f"lower bound ({ctx.q10_cal_tnd:.0f}); model recommends raising "
            f"toward {ctx.q50_tnd:.0f}."
        )
        return Recommendation(direction="raise",
                              recommended_price_tnd=ctx.q50_tnd,
                              reasons=[reason])
    return None


def _rule_above_band_lower(ctx: RowContext) -> Optional[Recommendation]:
    if ctx.current_price_tnd > ctx.q90_cal_tnd:
        reason = (
            f"current {ctx.current_price_tnd:.0f} TND is above the calibrated "
            f"upper bound ({ctx.q90_cal_tnd:.0f}); model recommends lowering "
            f"toward {ctx.q50_tnd:.0f}."
        )
        return Recommendation(direction="lower",
                              recommended_price_tnd=ctx.q50_tnd,
                              reasons=[reason])
    return None


def _rule_in_band_hold(ctx: RowContext) -> Optional[Recommendation]:
    # Default rule: always fires when neither below_band nor above_band did.
    reason = (
        f"current {ctx.current_price_tnd:.0f} TND is inside the calibrated "
        f"80% interval [{ctx.q10_cal_tnd:.0f}, {ctx.q90_cal_tnd:.0f}]; no action."
    )
    return Recommendation(direction="hold",
                          recommended_price_tnd=ctx.current_price_tnd,
                          reasons=[reason])


# ---------------------------------------------------------------------------
# Secondary rules (reason-only). May fire alongside a primary rule.
# Per-night scale.
# ---------------------------------------------------------------------------

def _rule_peer_premium_note(ctx: RowContext) -> Optional[Recommendation]:
    if ctx.peer_medium_median_per_night_tnd is None or ctx.peer_medium_count < _PEER_NOTE_MIN_COUNT:
        return None
    peer = ctx.peer_medium_median_per_night_tnd
    if peer <= 0:
        return None
    gap = (ctx.current_price_per_night_tnd - peer) / peer
    if gap <= _PEER_NOTE_MIN_GAP_FRAC:
        return None
    reason = (
        f"note: charging {gap * 100:.0f}% above the median of "
        f"{ctx.peer_medium_count} comparable hotels ({peer:.0f} TND/night)."
    )
    # direction sentinel: secondary rules never set a direction.
    return Recommendation(direction="hold",  # placeholder; ignored by apply_rules
                          recommended_price_tnd=ctx.current_price_tnd,
                          reasons=[reason])


def _rule_peer_discount_note(ctx: RowContext) -> Optional[Recommendation]:
    if ctx.peer_medium_median_per_night_tnd is None or ctx.peer_medium_count < _PEER_NOTE_MIN_COUNT:
        return None
    peer = ctx.peer_medium_median_per_night_tnd
    if peer <= 0:
        return None
    gap = (peer - ctx.current_price_per_night_tnd) / peer
    if gap <= _PEER_NOTE_MIN_GAP_FRAC:
        return None
    reason = (
        f"note: charging {gap * 100:.0f}% below the median of "
        f"{ctx.peer_medium_count} comparable hotels ({peer:.0f} TND/night)."
    )
    return Recommendation(direction="hold",  # placeholder; ignored by apply_rules
                          recommended_price_tnd=ctx.current_price_tnd,
                          reasons=[reason])


# ---------------------------------------------------------------------------
# Rule list + driver
# ---------------------------------------------------------------------------

# Primary rules must come first and be ordered so exactly one fires per row.
# Secondary rules come after and only contribute reasons.
_PRIMARY_RULES: tuple[Rule, ...] = (
    _rule_below_band_raise,
    _rule_above_band_lower,
    _rule_in_band_hold,
)
_SECONDARY_RULES: tuple[Rule, ...] = (
    _rule_peer_premium_note,
    _rule_peer_discount_note,
)

DEFAULT_RULES: tuple[Rule, ...] = _PRIMARY_RULES + _SECONDARY_RULES


def apply_rules(
    ctx: RowContext, rules: tuple[Rule, ...] | list[Rule] = DEFAULT_RULES,
) -> Recommendation:
    """
    Evaluate `rules` in order against `ctx` and produce a single Recommendation.

    Rules are split into primary (direction-setting) and secondary (reason-only).
    The first primary rule to fire fixes the direction and recommended price.
    All secondary rules that fire contribute their reasons.

    When rules=DEFAULT_RULES, the split is at len(_PRIMARY_RULES).
    When rules is a custom list (e.g., DEFAULT_RULES[:3]), we evaluate all
    rules in order as primaries; the first to fire wins.

    Guarantees:
    - The returned Recommendation always has at least one reason.
    - The returned direction is in {raise, hold, lower}.
    - With DEFAULT_RULES, the first three primary rules partition the row space
      and the default in-band rule guarantees coverage.
    """
    primary: Recommendation | None = None
    extra_reasons: list[str] = []

    # Determine where primary rules end if we're using DEFAULT_RULES.
    num_primary = len(_PRIMARY_RULES) if rules is DEFAULT_RULES else len(rules)

    for i, rule in enumerate(rules):
        out = rule(ctx)
        if out is None:
            continue

        if i < num_primary:
            # Primary rule: sets direction and price.
            if primary is None:
                primary = out
            # Once primary is set, skip remaining primary rules.
        else:
            # Secondary rule: append reasons only.
            if primary is not None:
                extra_reasons.extend(out.reasons)

    if primary is None:
        raise RuntimeError(
            "apply_rules: no rule fired. The default in-band rule should "
            "always fire; check that DEFAULT_RULES includes _rule_in_band_hold."
        )

    return Recommendation(
        direction=primary.direction,
        recommended_price_tnd=primary.recommended_price_tnd,
        reasons=[*primary.reasons, *extra_reasons],
    )
