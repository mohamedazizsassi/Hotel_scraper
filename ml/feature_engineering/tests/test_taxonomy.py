"""
Tests for ``feature_engineering.taxonomy``.

Covers:
    * Every raw boarding string in ``BOARDING_CANONICAL_MAP`` maps to its
      expected canonical code, regardless of case / surrounding whitespace.
    * An unmapped boarding string below the 1 % fail threshold becomes
      ``"UNKNOWN"``; above the threshold the call raises.
    * A set of 20 hand-picked room strings parses to the expected
      (base, view, tier, occupancy) tuple.
"""
from __future__ import annotations

import pandas as pd
import pytest

from feature_engineering.config import BOARDING_CANONICAL_MAP
from feature_engineering.taxonomy import (
    UNKNOWN_BOARDING,
    canonicalize_boarding,
    parse_room,
)


# ---------------------------------------------------------------------------
# Boarding canonicalisation
# ---------------------------------------------------------------------------

def test_canonicalize_boarding_every_mapped_string_resolves() -> None:
    """Every key in the map produces its canonical code."""
    raw = list(BOARDING_CANONICAL_MAP.keys())
    expected = [BOARDING_CANONICAL_MAP[k] for k in raw]
    df = pd.DataFrame({"boarding_name": raw})

    out = canonicalize_boarding(df)

    assert out["boarding_canonical"].tolist() == expected


def test_canonicalize_boarding_is_case_and_whitespace_insensitive() -> None:
    df = pd.DataFrame({
        "boarding_name": [
            "  Demi Pension  ",
            "PETIT DÉJEUNER",
            "All Inclusive",
            "Pension Complète",
        ],
    })

    out = canonicalize_boarding(df)

    assert out["boarding_canonical"].tolist() == ["HDP", "BB", "AI", "PC"]


def test_canonicalize_boarding_unknown_under_threshold_is_tolerated() -> None:
    """One unmapped row out of >=100 is below the 1 % threshold."""
    raw = ["demi pension"] * 200 + ["chambre et gîte cosmique"]
    df = pd.DataFrame({"boarding_name": raw})

    out = canonicalize_boarding(df)

    assert (out["boarding_canonical"] == UNKNOWN_BOARDING).sum() == 1
    assert (out["boarding_canonical"] == "HDP").sum() == 200


def test_canonicalize_boarding_unknown_over_threshold_raises() -> None:
    raw = ["demi pension"] * 80 + ["mystery board"] * 20  # 20 % unknown
    df = pd.DataFrame({"boarding_name": raw})

    with pytest.raises(RuntimeError, match="UNKNOWN rate"):
        canonicalize_boarding(df)


def test_canonicalize_boarding_missing_column_raises() -> None:
    with pytest.raises(RuntimeError, match="boarding_name"):
        canonicalize_boarding(pd.DataFrame({"foo": [1, 2]}))


# ---------------------------------------------------------------------------
# Room parsing
# ---------------------------------------------------------------------------

# (raw_room_name, expected_base, expected_view, expected_tier, expected_occupancy)
ROOM_CASES: list[tuple[str, str | None, str | None, str | None, str | None]] = [
    ("Suite Présidentielle vue mer",        "suite",       "mer",      "presidentielle", None),
    ("Suite Junior vue jardin",             "suite",       "jardin",   "junior",         None),
    ("Chambre Double Standard",             "chambre",     None,       "standard",       "double"),
    ("Chambre Triple Supérieure",           "chambre",     None,       "superieure",     "triple"),
    ("Chambre Familiale vue piscine",       "chambre",     "piscine",  None,             "familiale"),
    ("Chambre Single Deluxe",               "chambre",     None,       "deluxe",         "single"),
    ("Chambre Double Sea View",             "chambre",     "mer",      None,             "double"),
    ("Studio Deluxe",                       "studio",      None,       "deluxe",         None),
    ("Appartement Familial",                "appartement", None,       None,             "familiale"),
    ("Bungalow Vue Jardin",                 "bungalow",    "jardin",   None,             None),
    ("Villa Exécutive vue mer",             "villa",       "mer",      "executive",      None),
    ("Chambre Individuelle",                "chambre",     None,       None,             "single"),
    ("Family Room Garden View",             "chambre",     "jardin",   None,             "familiale"),
    ("Pool View Double Room",               "chambre",     "piscine",  None,             "double"),
    ("Chambre Supérieure vue Montagne",     "chambre",     "montagne", "superieure",     None),
    ("Junior Suite Mountain View",          "suite",       "montagne", "junior",         None),
    ("Chambre Standard Double",             "chambre",     None,       "standard",       "double"),
    ("Suite Deluxe vue piscine",            "suite",       "piscine",  "deluxe",         None),
    ("Chambre Triple vue jardin",           "chambre",     "jardin",   None,             "triple"),
    ("Apart Familiale",                     "appartement", None,       None,             "familiale"),
]


@pytest.mark.parametrize(
    "raw,base,view,tier,occupancy",
    ROOM_CASES,
    ids=[c[0] for c in ROOM_CASES],
)
def test_parse_room_expected_values(
    raw: str,
    base: str | None,
    view: str | None,
    tier: str | None,
    occupancy: str | None,
) -> None:
    out = parse_room(pd.DataFrame({"room_name": [raw]}))
    row = out.iloc[0]

    assert (row["room_base"]      if pd.notna(row["room_base"])      else None) == base
    assert (row["room_view"]      if pd.notna(row["room_view"])      else None) == view
    assert (row["room_tier"]      if pd.notna(row["room_tier"])      else None) == tier
    assert (row["room_occupancy"] if pd.notna(row["room_occupancy"]) else None) == occupancy


def test_parse_room_null_and_empty_inputs_are_na() -> None:
    df = pd.DataFrame({"room_name": [None, "", pd.NA]})
    out = parse_room(df)
    for col in ("room_base", "room_view", "room_tier", "room_occupancy"):
        assert out[col].isna().all()


def test_parse_room_missing_column_raises() -> None:
    with pytest.raises(RuntimeError, match="room_name"):
        parse_room(pd.DataFrame({"foo": [1]}))


def test_parse_room_does_not_mutate_input() -> None:
    df = pd.DataFrame({"room_name": ["Suite Deluxe"]})
    before_cols = list(df.columns)
    _ = parse_room(df)
    assert list(df.columns) == before_cols
