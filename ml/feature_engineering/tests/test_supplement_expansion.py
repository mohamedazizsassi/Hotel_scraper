"""
Tests for ``feature_engineering.supplement_expansion``.

Covers:
    * Row-count invariant: ``len(out) == len(in) + total_supplements_in``.
    * Variant-row construction: price = base + supp.price, ppn recomputed,
      room_name carries the canonicalised view phrase.
    * Base-row features (``view_upgrade_count_offered``,
      ``has_free_view_upgrade``, ``is_supplement_variant``) computed on
      the pre-expansion shape and broadcast to variants.
    * Free-upgrade detection via the "offerte (stock limité)" marker.
    * tunisiepromo passthrough — empty supplements means no expansion and
      all-zero/false features.
    * Unknown-supplement-name rate above 1 % → RuntimeError.
    * Unknown rate below threshold → tolerated.
    * ``supplements`` column dropped from the output.
"""
from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering.supplement_expansion import (
    SUPPLEMENT_NAME_MAP,
    expand_supplements,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _row(
    *,
    source: str = "promohotel",
    hotel: str = "hotel_x",
    room_name: str = "Chambre Double",
    nights: int = 3,
    price: float = 300.0,
    supplements: list[dict] | None = None,
) -> dict:
    return {
        "source": source,
        "hotel_name_normalized": hotel,
        "room_name": room_name,
        "nights": nights,
        "price": price,
        "price_per_night": price / nights,
        "supplements": supplements if supplements is not None else [],
    }


def _df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["price"] = df["price"].astype("float32")
    df["price_per_night"] = df["price_per_night"].astype("float32")
    df["nights"] = df["nights"].astype("Int16")
    df["room_name"] = df["room_name"].astype("string[python]")
    return df


# ---------------------------------------------------------------------------
# Structure / row-count invariants
# ---------------------------------------------------------------------------

def test_missing_required_columns_raises() -> None:
    with pytest.raises(RuntimeError, match="missing required columns"):
        expand_supplements(pd.DataFrame({"price": [10.0]}))


def test_supplements_column_dropped() -> None:
    df = _df([_row(supplements=[])])
    out = expand_supplements(df)
    assert "supplements" not in out.columns


def test_empty_supplements_no_expansion() -> None:
    df = _df([_row(source="tunisiepromo", supplements=[])])
    out = expand_supplements(df)
    assert len(out) == 1
    r = out.iloc[0]
    assert bool(r["is_supplement_variant"]) is False
    assert int(r["view_upgrade_count_offered"]) == 0
    assert bool(r["has_free_view_upgrade"]) is False


def test_row_count_invariant() -> None:
    df = _df([
        _row(supplements=[]),                                   # 1 → 1
        _row(supplements=[                                      # 1 → 1+2 = 3
            {"name": "Supplément Vue Mer", "price": 30.0},
            {"name": "Supplément Vue Piscine", "price": 20.0},
        ]),
        _row(source="tunisiepromo", supplements=[]),            # 1 → 1
    ])
    out = expand_supplements(df)
    assert len(out) == 1 + (1 + 2) + 1 == 5


# ---------------------------------------------------------------------------
# Variant-row content
# ---------------------------------------------------------------------------

def test_variant_price_and_room_name() -> None:
    df = _df([
        _row(
            room_name="Chambre Double",
            nights=3,
            price=300.0,
            supplements=[
                {"name": "Supplément Vue Mer", "price": 30.0},
                {"name": "Supplément Vue Piscine", "price": 15.0},
            ],
        ),
    ])
    out = expand_supplements(df)
    # 1 base + 2 variants
    base = out[~out["is_supplement_variant"].astype(bool)].iloc[0]
    variants = out[out["is_supplement_variant"].astype(bool)].sort_values("price").reset_index(drop=True)

    assert float(base["price"]) == pytest.approx(300.0)
    assert base["room_name"] == "Chambre Double"

    assert float(variants.iloc[0]["price"]) == pytest.approx(315.0)
    assert float(variants.iloc[0]["price_per_night"]) == pytest.approx(105.0)
    assert variants.iloc[0]["room_name"] == "Chambre Double vue piscine"

    assert float(variants.iloc[1]["price"]) == pytest.approx(330.0)
    assert float(variants.iloc[1]["price_per_night"]) == pytest.approx(110.0)
    assert variants.iloc[1]["room_name"] == "Chambre Double vue mer"


def test_free_upgrade_zero_price_variant() -> None:
    df = _df([
        _row(
            price=400.0,
            nights=2,
            supplements=[
                {"name": "Supplément Vue Mer Offerte (stock limité)", "price": 0.0},
            ],
        ),
    ])
    out = expand_supplements(df)
    variant = out[out["is_supplement_variant"].astype(bool)].iloc[0]
    assert float(variant["price"]) == pytest.approx(400.0)
    assert float(variant["price_per_night"]) == pytest.approx(200.0)
    assert variant["room_name"] == "Chambre Double vue mer"


def test_has_free_upgrade_broadcast_to_all_rows_for_observation() -> None:
    df = _df([
        _row(
            supplements=[
                {"name": "Supplément Vue Mer", "price": 30.0},
                {"name": "Supplément Vue Piscine Offerte (stock limité)", "price": 0.0},
            ],
        ),
    ])
    out = expand_supplements(df)
    # Every row from this observation (1 base + 2 variants) carries
    # has_free_view_upgrade=True and view_upgrade_count_offered=2.
    assert out["has_free_view_upgrade"].astype(bool).tolist() == [True, True, True]
    assert out["view_upgrade_count_offered"].astype(int).tolist() == [2, 2, 2]


def test_view_upgrade_count_zero_when_no_supplements() -> None:
    df = _df([_row(supplements=[])])
    out = expand_supplements(df)
    assert int(out.iloc[0]["view_upgrade_count_offered"]) == 0
    assert bool(out.iloc[0]["has_free_view_upgrade"]) is False


# ---------------------------------------------------------------------------
# Mixed-source case (tunisiepromo + promohotel side-by-side)
# ---------------------------------------------------------------------------

def test_mixed_sources_only_promohotel_expands() -> None:
    df = _df([
        _row(source="tunisiepromo", room_name="Chambre Double Vue Mer", supplements=[]),
        _row(source="promohotel",   room_name="Chambre Double",
             supplements=[{"name": "Supplément Vue Mer", "price": 25.0}]),
    ])
    out = expand_supplements(df)
    # tunisiepromo: 1 row, no variants. promohotel: 1 base + 1 variant.
    assert len(out) == 3
    assert (out["source"] == "tunisiepromo").sum() == 1
    assert (out["source"] == "promohotel").sum() == 2


# ---------------------------------------------------------------------------
# Validation — unknown supplement names
# ---------------------------------------------------------------------------

def test_unknown_supplement_name_above_threshold_raises() -> None:
    # 5 known + 5 unknown = 50 % unknown, well above 1 %.
    rows = []
    for _ in range(5):
        rows.append(_row(supplements=[{"name": "Supplément Vue Mer", "price": 10.0}]))
    for _ in range(5):
        rows.append(_row(supplements=[{"name": "Supplément Spa Privé", "price": 10.0}]))
    df = _df(rows)

    with pytest.raises(RuntimeError, match="unknown supplement-name rate"):
        expand_supplements(df)


def test_unknown_supplement_name_under_threshold_tolerated() -> None:
    # 200 known + 1 unknown = 0.5 %, below 1 %.
    rows = [_row(supplements=[{"name": "Supplément Vue Mer", "price": 10.0}]) for _ in range(200)]
    rows.append(_row(supplements=[{"name": "Supplément Mystère", "price": 10.0}]))
    df = _df(rows)

    out = expand_supplements(df)
    # 201 base + 201 variants
    assert len(out) == 402


# ---------------------------------------------------------------------------
# Map sanity — every map value should fold to the canonical phrase
# ---------------------------------------------------------------------------

def test_every_known_supplement_name_produces_canonical_room_name_suffix() -> None:
    rows = [
        _row(
            room_name="Chambre Double",
            price=100.0,
            nights=1,
            supplements=[{"name": raw, "price": 5.0}],
        )
        for raw in SUPPLEMENT_NAME_MAP
    ]
    df = _df(rows)
    out = expand_supplements(df)
    variants = out[out["is_supplement_variant"].astype(bool)]
    expected_phrases = {SUPPLEMENT_NAME_MAP[k] for k in SUPPLEMENT_NAME_MAP}
    actual_phrases = {
        rn.replace("Chambre Double ", "") for rn in variants["room_name"].tolist()
    }
    assert actual_phrases == expected_phrases


# ---------------------------------------------------------------------------
# Non-mutation guarantee
# ---------------------------------------------------------------------------

def test_does_not_mutate_input() -> None:
    df = _df([_row(supplements=[{"name": "Supplément Vue Mer", "price": 10.0}])])
    snapshot = df.copy(deep=True)
    expand_supplements(df)
    pd.testing.assert_frame_equal(df, snapshot)
