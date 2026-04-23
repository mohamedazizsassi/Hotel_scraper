"""
Boarding canonicalisation and room-name parsing.

Two transformations live here because they share a common shape —
regex / lookup-table mapping from free-text scraper columns onto a closed
set of modelling categories — and because they both need the same
"fail-loudly on low coverage" discipline.

Contract (what downstream modules can assume after ``canonicalize_boarding``
and ``parse_room`` have run):
    * ``boarding_canonical`` -> string in ``BOARDING_CANONICAL_VALUES``
    * ``room_base``          -> string or <NA>  (coverage target >= 95%)
    * ``room_view``          -> string or <NA>
    * ``room_tier``          -> string or <NA>
    * ``room_occupancy``     -> string or <NA>

Both functions return a new DataFrame; inputs are not modified.
"""

from __future__ import annotations

import logging
import re
import unicodedata

import pandas as pd

from .config import (
    BOARDING_CANONICAL_MAP,
    BOARDING_CANONICAL_VALUES,
    ROOM_BASE_COVERAGE_WARN,
    ROOM_BASE_PATTERNS,
    ROOM_OCCUPANCY_PATTERNS,
    ROOM_TIER_PATTERNS,
    ROOM_VIEW_PATTERNS,
    UNKNOWN_BOARDING_FAIL_RATE,
)

logger = logging.getLogger(__name__)

UNKNOWN_BOARDING: str = "UNKNOWN"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def canonicalize_boarding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``boarding_canonical`` derived from ``boarding_name``.

    The raw ``boarding_name`` is lowercased and stripped, then looked up
    in ``BOARDING_CANONICAL_MAP``. Values not in the map are assigned
    ``"UNKNOWN"``. The full list of unmapped distinct raw values is
    logged so future runs can extend the map.

    Parameters
    ----------
    df:
        Cleaned DataFrame from ``cleaners.clean``. Must contain
        ``boarding_name``.

    Returns
    -------
    pd.DataFrame
        New DataFrame with an added ``boarding_canonical`` column of
        dtype ``string[python]``.

    Raises
    ------
    RuntimeError
        If the ``UNKNOWN`` rate exceeds ``UNKNOWN_BOARDING_FAIL_RATE``
        (default 1%). Silent tolerance of a growing UNKNOWN bucket would
        quietly erode grouping quality across the pipeline.
    """
    if "boarding_name" not in df.columns:
        raise RuntimeError("canonicalize_boarding: missing 'boarding_name'")

    out = df.copy()
    raw = out["boarding_name"].astype("string[python]")
    key = raw.str.lower().str.strip()

    canonical = key.map(BOARDING_CANONICAL_MAP)
    unknown_mask = canonical.isna()
    canonical = canonical.where(~unknown_mask, UNKNOWN_BOARDING).astype("string[python]")

    unknown_rate = float(unknown_mask.mean()) if len(out) else 0.0
    unmapped_values = sorted(
        v for v in key.loc[unknown_mask].dropna().unique().tolist() if v
    )
    if unmapped_values:
        logger.info(
            "canonicalize_boarding: %d distinct unmapped boarding_name values "
            "(UNKNOWN rate=%.4f): %s",
            len(unmapped_values), unknown_rate, unmapped_values,
        )

    if unknown_rate > UNKNOWN_BOARDING_FAIL_RATE:
        raise RuntimeError(
            f"canonicalize_boarding: UNKNOWN rate {unknown_rate:.4f} "
            f"exceeds threshold {UNKNOWN_BOARDING_FAIL_RATE}. "
            f"Unmapped values: {unmapped_values}"
        )

    # Cross-check: every non-UNKNOWN value must be in the closed set.
    bad = set(canonical.dropna().unique()) - set(BOARDING_CANONICAL_VALUES)
    if bad:
        raise RuntimeError(
            f"canonicalize_boarding: produced values outside closed set: {bad}"
        )

    out["boarding_canonical"] = canonical
    return out


def parse_room(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ``room_base``, ``room_view``, ``room_tier``, ``room_occupancy``.

    Raw ``room_name`` is casefolded and Unicode-NFC-normalized, then
    scanned against each dimension's pattern dict in declaration order
    (first match wins). Unmatched rows get ``<NA>``.

    Parameters
    ----------
    df:
        Cleaned DataFrame from ``cleaners.clean``. Must contain
        ``room_name``.

    Returns
    -------
    pd.DataFrame
        New DataFrame with four additional string columns.

    Notes
    -----
    - Coverage is logged per column. If ``room_base`` coverage falls
      below ``ROOM_BASE_COVERAGE_WARN`` (default 95%), the top-20
      unmatched raw room names are logged for manual triage.
    - Unicode NFC is used rather than NFD so the regexes written with
      precomposed accents (``présidentielle``) match directly.
    """
    if "room_name" not in df.columns:
        raise RuntimeError("parse_room: missing 'room_name'")

    out = df.copy()
    raw = out["room_name"].astype("string[python]")
    normalised = raw.map(_normalise)

    base      = _first_match(normalised, ROOM_BASE_PATTERNS)
    view      = _first_match(normalised, ROOM_VIEW_PATTERNS)
    tier      = _first_match(normalised, ROOM_TIER_PATTERNS)
    occupancy = _first_match(normalised, ROOM_OCCUPANCY_PATTERNS)

    out["room_base"]      = base
    out["room_view"]      = view
    out["room_tier"]      = tier
    out["room_occupancy"] = occupancy

    n = len(out)
    if n:
        for label, col in (
            ("room_base",      base),
            ("room_view",      view),
            ("room_tier",      tier),
            ("room_occupancy", occupancy),
        ):
            coverage = float(col.notna().mean())
            logger.info("parse_room: %-15s coverage=%.3f", label, coverage)

        base_coverage = float(base.notna().mean())
        if base_coverage < ROOM_BASE_COVERAGE_WARN:
            unmatched = raw.loc[base.isna()].value_counts().head(20)
            logger.warning(
                "parse_room: room_base coverage %.3f < %.2f. Top-20 unmatched "
                "raw room_name values:\n%s",
                base_coverage, ROOM_BASE_COVERAGE_WARN, unmatched.to_string(),
            )

    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(value: object) -> str | None:
    """Unicode-NFC-normalize and casefold. Return None for NA/empty."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if value is pd.NA:
        return None
    s = str(value)
    if not s:
        return None
    s = unicodedata.normalize("NFC", s).casefold()
    return s


def _first_match(
    normalised: pd.Series,
    patterns: dict[str, str],
) -> pd.Series:
    """
    For each row, return the first label in ``patterns`` whose regex
    matches the normalised string. ``None`` if no pattern matches or
    the input is null. Returned as ``string[python]``.
    """
    compiled = [
        (label, re.compile(pat, flags=re.IGNORECASE | re.UNICODE))
        for label, pat in patterns.items()
    ]

    def _match(s: object) -> str | None:
        if s is None:
            return None
        text = s  # already normalised
        for label, rx in compiled:
            if rx.search(text):
                return label
        return None

    return normalised.map(_match).astype("string[python]")
