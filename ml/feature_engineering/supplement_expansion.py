"""
Cross-source alignment via supplement-row expansion.

The two scraper sources encode room variation differently:

* ``tunisiepromo`` ships one row per (hotel, stay, room variant) where the
  variant (view, tier) is already inlined in ``room_name``. Its
  ``supplements`` list is always empty.
* ``promohotel`` ships one base row plus a ``supplements`` list of
  **mutually exclusive** view-upgrade options (the user picks at most
  one). The base ``room_name`` does NOT mention the view; the supplement
  name does.

Stage 7 (competitive_features) joins rows on (..., room_view, ...) at the
*tight* peer-group granularity. If we hand promohotel rows to taxonomy
unchanged, ``parse_room`` cannot extract ``room_view`` from a base row
that has no view word in ``room_name`` — every promohotel row falls into
``room_view = NA`` and silently fails to align with tunisiepromo rows for
the same physical product.

This stage fixes that by expanding each promohotel base row with N
supplements into 1 base + N synthetic variant rows whose ``room_name``
carries the canonicalised view phrase. Taxonomy then sees the augmented
``room_name`` and the two sources land in one schema.

Runs BEFORE ``taxonomy.parse_room``.

Contract (what downstream modules can assume after ``expand_supplements``):
    * ``supplements`` column is removed.
    * Row count: ``len(out) == len(in) + sum(len(r.supplements) for r in in)``.
    * Every row carries:
        - ``is_supplement_variant``        : bool
        - ``view_upgrade_count_offered``   : Int8   (0 if none)
        - ``has_free_view_upgrade``        : bool
    * Variant rows have ``price`` = base.price + supplement.price,
      ``price_per_night`` recomputed, and ``room_name`` augmented with the
      canonicalised supplement label.

Failures are loud: an unknown-supplement-name rate above the 1 % threshold
raises ``RuntimeError`` rather than silently dropping the upgrade signal.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonicalisation tables
# ---------------------------------------------------------------------------
# Keys: raw supplement.name after .lower().strip(). Values: the view phrase
# appended to ``room_name`` on synthetic variant rows so taxonomy can read
# it. Free-upgrade entries fold to the same canonical phrase as the paid
# version because the room product is the same — only the price differs.

SUPPLEMENT_NAME_MAP: dict[str, str] = {
    "supplément vue mer":                            "vue mer",
    "supplément vue piscine":                        "vue piscine",
    "supplément vue mer latérale":                   "vue mer latérale",
    "supplément vue mer et piscine":                 "vue mer et piscine",
    "supplément coté mer":                           "coté mer",
    "supplément vue piscine offerte (stock limité)": "vue piscine",
    "supplément vue mer offerte (stock limité)":     "vue mer",
}

# Substring marker identifying a free upgrade (price=0 in the data, but
# the marker is more explicit than a price check).
FREE_UPGRADE_MARKER: str = "offerte (stock limité)"

# Hard fail threshold for unmapped supplement.name values.
UNKNOWN_SUPPLEMENT_FAIL_RATE: float = 0.01  # 1 %


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_supplements(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand promohotel supplement lists into synthetic variant rows.

    Parameters
    ----------
    df:
        Cleaned DataFrame from ``cleaners.clean``. Must contain
        ``supplements``, ``price``, ``price_per_night``, ``nights``, and
        ``room_name``. ``supplements`` is a list of ``{name, price}``
        dicts (or list-of-struct from pyarrow). Empty / null lists are
        treated as "no supplements".

    Returns
    -------
    pd.DataFrame
        New DataFrame, ``supplements`` column dropped. For each input row
        with ``len(supplements) == N`` the output contains N+1 rows:
        the original (``is_supplement_variant=False``) plus N synthetic
        variants (``is_supplement_variant=True``). Index is reset.

    Raises
    ------
    RuntimeError
        * If a required column is missing.
        * If the unknown-supplement-name rate exceeds
          ``UNKNOWN_SUPPLEMENT_FAIL_RATE`` (default 1 %).
    """
    _assert_required_columns(df)

    parsed = df["supplements"].map(_normalise_supplements)
    counts = parsed.map(len)
    has_free = parsed.map(_has_free_upgrade)

    _validate_supplement_names(parsed)

    base = df.drop(columns=["supplements"]).copy()
    base = base.reset_index(drop=True)
    base["is_supplement_variant"] = pd.array([False] * len(base), dtype="boolean")
    base["view_upgrade_count_offered"] = pd.array(counts.values, dtype="Int8")
    base["has_free_view_upgrade"] = pd.array(has_free.values, dtype="boolean")

    # Collect (positional source row index, supplement dict) pairs so we
    # can build variants by replicating the corresponding base rows and
    # then mutating price / room_name in vectorised fashion.
    src_positions: list[int] = []
    src_supps: list[dict[str, Any]] = []
    for pos, items in enumerate(parsed.values):
        for s in items:
            src_positions.append(pos)
            src_supps.append(s)

    if not src_positions:
        logger.info(
            "expand_supplements: no supplement rows to expand; "
            "passthrough %d rows", len(base),
        )
        return base

    variants = base.iloc[src_positions].reset_index(drop=True).copy()

    base_prices = base["price"].iloc[src_positions].astype("float64").values
    base_nights = base["nights"].iloc[src_positions].astype("float64").values
    base_room_names = base["room_name"].iloc[src_positions].values

    new_prices = []
    new_room_names = []
    for i, s in enumerate(src_supps):
        name_key = _name_key(s.get("name"))
        canon = SUPPLEMENT_NAME_MAP.get(name_key, name_key)
        var_price = float(base_prices[i]) + float(s.get("price") or 0.0)
        new_prices.append(var_price)
        new_room_names.append(_augment_room_name(base_room_names[i], canon))

    new_ppn = [p / n for p, n in zip(new_prices, base_nights)]

    variants["price"] = pd.array(new_prices, dtype=base["price"].dtype)
    variants["price_per_night"] = pd.array(new_ppn, dtype=base["price_per_night"].dtype)
    variants["room_name"] = pd.array(new_room_names, dtype=base["room_name"].dtype)
    variants["is_supplement_variant"] = pd.array([True] * len(variants), dtype="boolean")
    # view_upgrade_count_offered + has_free_view_upgrade already correct
    # because we copied them from the base rows of the same source row.

    out = pd.concat([base, variants], ignore_index=True)

    logger.info(
        "expand_supplements: %d base rows + %d variant rows = %d total "
        "(rows with supplements: %d, free upgrades present on %d rows)",
        len(base), len(variants), len(out),
        int((counts > 0).sum()), int(has_free.sum()),
    )
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "supplements", "price", "price_per_night", "nights", "room_name",
)


def _assert_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"expand_supplements: missing required columns: {missing}"
        )


def _normalise_supplements(value: object) -> list[dict[str, Any]]:
    """Coerce a raw supplements cell into a plain list of dicts."""
    if value is None:
        return []
    if isinstance(value, float) and pd.isna(value):
        return []
    if value is pd.NA:
        return []
    try:
        items = list(value)  # works for list, np.ndarray, pa.ListScalar
    except TypeError:
        return []
    out: list[dict[str, Any]] = []
    for it in items:
        if it is None:
            continue
        if isinstance(it, dict):
            out.append(it)
        elif hasattr(it, "as_py"):  # pa.StructScalar
            converted = it.as_py()
            if isinstance(converted, dict):
                out.append(converted)
        else:
            # Last-ditch: try attribute access for namedtuple-like rows
            try:
                out.append({"name": it["name"], "price": it["price"]})
            except (TypeError, KeyError, IndexError):
                continue
    return out


def _name_key(name: object) -> str:
    if name is None:
        return ""
    if isinstance(name, float) and pd.isna(name):
        return ""
    return str(name).strip().lower()


def _has_free_upgrade(items: Iterable[dict[str, Any]]) -> bool:
    for it in items:
        if FREE_UPGRADE_MARKER in _name_key(it.get("name")):
            return True
    return False


def _validate_supplement_names(parsed: pd.Series) -> None:
    """Raise if the unknown-name rate over all supplements exceeds 1 %."""
    total = 0
    unknown_counts: dict[str, int] = {}
    for items in parsed.values:
        for s in items:
            key = _name_key(s.get("name"))
            if not key:
                continue
            total += 1
            if key not in SUPPLEMENT_NAME_MAP:
                unknown_counts[key] = unknown_counts.get(key, 0) + 1

    if total == 0:
        return

    unknown_total = sum(unknown_counts.values())
    rate = unknown_total / total
    if unknown_counts:
        logger.info(
            "expand_supplements: %d distinct unmapped supplement names "
            "(unknown rate=%.4f over %d entries): %s",
            len(unknown_counts), rate, total, sorted(unknown_counts),
        )
    if rate > UNKNOWN_SUPPLEMENT_FAIL_RATE:
        raise RuntimeError(
            f"expand_supplements: unknown supplement-name rate {rate:.4f} "
            f"exceeds threshold {UNKNOWN_SUPPLEMENT_FAIL_RATE}. "
            f"Unmapped values (with counts): {sorted(unknown_counts.items())}"
        )


def _augment_room_name(base_name: object, canon: str) -> str:
    """Append the canonical supplement phrase to ``room_name``."""
    if base_name is None or (isinstance(base_name, float) and pd.isna(base_name)):
        return canon
    s = str(base_name).strip()
    if not s:
        return canon
    if not canon:
        return s
    return f"{s} {canon}"
