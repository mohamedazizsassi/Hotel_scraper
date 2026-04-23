"""
Configuration for the ml/feature_engineering pipeline.

Central source of truth for:
    * taxonomies — boarding canonicalisation, room-name regex patterns
    * calendar reference data — Tunisia + EU source-market holidays
    * connection configuration — MongoDB and PostgreSQL URIs

Downstream modules in this package import from here. No other file
should redefine these constants.

Import-time side effects: loads `ml/.env` via python-dotenv so env vars
are visible to modules that import `config` directly (tests, notebooks,
one-off scripts) without each caller re-invoking `load_dotenv`.
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# .env loading
# ---------------------------------------------------------------------------

_REPO_ROOT: Path = Path(__file__).resolve().parents[2]
_ML_ROOT: Path = _REPO_ROOT / "ml"
load_dotenv(_ML_ROOT / ".env")


# ---------------------------------------------------------------------------
# Connection configuration
# ---------------------------------------------------------------------------
# Single-URI form per datastore. Both pymongo and SQLAlchemy/libpq accept
# URI strings natively, which keeps deployment portable across managed
# services (Atlas, RDS, Supabase, Neon) and local dev.

MONGO_URI: str = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DATABASE: str = os.environ.get("MONGO_DB", "hotel_scraper")
MONGO_COLLECTION: str = os.environ.get("MONGO_COLLECTION", "hotel_prices")

POSTGRES_URI: str = os.environ.get(
    "POSTGRES_URI"
    
)

POSTGRES_FEATURES_TABLE: str = "hotel_features"


# ---------------------------------------------------------------------------
# Boarding taxonomy
# ---------------------------------------------------------------------------
# Keys: raw `boarding_name` after .lower().strip() (accents preserved).
# Values: canonical code from the closed set below.
# Unmapped raw values are assigned "UNKNOWN". The pipeline fails loudly
# if the UNKNOWN rate exceeds UNKNOWN_BOARDING_FAIL_RATE.

BOARDING_CANONICAL_MAP: dict[str, str] = {
    "petit déjeuner":            "BB",
    "logement petit dejeuner":   "BB",

    "logement simple":           "LOG",
    "entrée simple":             "LOG",

    "demi pension":              "HDP",

    "demi pension plus":         "HDP_PLUS",
    "demi pension +":            "HDP_PLUS",

    "pension complete":          "PC",
    "pension complète":          "PC",

    "pension complete plus":     "PC_PLUS",
    "pension complète +":        "PC_PLUS",

    "all inclusive":             "AI",
    "all in hard":               "AI",
    "24h all inclusive":         "AI",

    "soft all inclusive":        "AI_SOFT",
    "all inclusive soft drink":  "AI_SOFT",
    "ultra all soft":            "AI_SOFT",
    "ultra soft all-inclusive":  "AI_SOFT",

    "ultra all inclusive":       "AI_ULTRA",
    "all inclusive gold":        "AI_ULTRA",
    "royal premium soft drinks": "AI_ULTRA",
}

BOARDING_CANONICAL_VALUES: frozenset[str] = frozenset({
    "BB", "LOG",
    "HDP", "HDP_PLUS",
    "PC", "PC_PLUS",
    "AI", "AI_SOFT", "AI_ULTRA",
    "UNKNOWN",
})

# Hard fail threshold for unmapped boarding_name values.
UNKNOWN_BOARDING_FAIL_RATE: float = 0.01  # 1 %


# ---------------------------------------------------------------------------
# Room parsing patterns
# ---------------------------------------------------------------------------
# Applied to `room_name` after casefold + Unicode NFC normalization.
# First match wins within each dimension. Unmatched rows → None.
# If `room_base` coverage falls below ROOM_BASE_COVERAGE_WARN, the
# pipeline logs the top-20 unmatched raw room names for triage.

ROOM_BASE_PATTERNS: dict[str, str] = {
    "suite":       r"\bsuite\b",
    "studio":      r"\bstudio\b",
    "appartement": r"\bappartement\b|\bapart\b",
    "bungalow":    r"\bbungalow\b",
    "villa":       r"\bvilla\b",
    # ``room`` is an English synonym for ``chambre`` and appears unprefixed
    # in scraped room names ("Superior Room", "Deluxe Room", "Swim Up Room").
    "chambre":     r"\bchambre\b|\broom\b",
}

ROOM_VIEW_PATTERNS: dict[str, str] = {
    "mer":      r"\bvue mer\b|\bsea view\b",
    "jardin":   r"\bvue jardin\b|\bgarden\b",
    "piscine":  r"\bvue piscine\b|\bpool\b",
    "montagne": r"\bmontagne\b|\bmountain\b",
}

ROOM_TIER_PATTERNS: dict[str, str] = {
    "presidentielle": r"\bprésidentielle\b|\bpresidentielle\b",
    "executive":      r"\bexécutive\b|\bexecutive\b",
    "deluxe":         r"\bdeluxe\b",
    "junior":         r"\bjunior\b",
    "superieure":     r"\bsupérieure\b|\bsuperieure\b|\bsupérieur\b",
    "standard":       r"\bstandard\b",
}

ROOM_OCCUPANCY_PATTERNS: dict[str, str] = {
    "familiale": r"\bfamili(?:ale|al)\b|\bfamily\b",
    "triple":    r"\btriple\b",
    "double":    r"\bdouble\b",
    "single":    r"\bsingle\b|\bindividuelle\b",
}

# Warn-only threshold. ``room_base`` feeds the *tight* peer-group key
# only; rows with ``<NA>`` base gracefully fall back to the *medium*
# granularity via ``best_peer_granularity_used``. 0.90 is the empirical
# floor after adding the English ``room`` synonym — remaining misses are
# tier+view strings with no type word (e.g. "Standard Vue Jardin") and
# mojibake artefacts from the scraper encoding bug.
ROOM_BASE_COVERAGE_WARN: float = 0.90


# ---------------------------------------------------------------------------
# Cleaning thresholds
# ---------------------------------------------------------------------------
# Referenced by cleaners.py. Documented here to keep magic numbers out
# of business-logic modules.

PRICE_PER_NIGHT_MIN: float = 30.0       # TND/night — below is implausible
PRICE_PER_NIGHT_MAX: float = 20000.0    # TND/night — above is implausible
NIGHTS_ALLOWED: frozenset[int] = frozenset({1, 2, 3, 5, 7})
DAYS_UNTIL_CHECKIN_MIN: int = 0
DAYS_UNTIL_CHECKIN_MAX: int = 365


# ---------------------------------------------------------------------------
# Booking-window buckets
# ---------------------------------------------------------------------------
# Interpretability feature; raw integer `days_until_checkin` is also kept
# as a modelling feature. Buckets are left-closed, right-closed: a value
# v falls in bucket (lo, hi] where lo >= 0. D0-1 is the first bucket so
# v=0 maps to "D0-1". Last bucket "D90+" is open on the right.

BOOKING_WINDOW_BUCKETS: list[tuple[str, int, int]] = [
    ("D0-1",    0,   1),
    ("D2-7",    2,   7),
    ("D8-14",   8,  14),
    ("D15-30", 15,  30),
    ("D31-60", 31,  60),
    ("D61-90", 61,  90),
    ("D90+",   91, 10_000),
]


# ---------------------------------------------------------------------------
# Calendar reference — Tunisia + EU source markets
# ---------------------------------------------------------------------------
# All data hardcoded. No external API calls at runtime.
# TODO: once the pipeline stabilises, replace the Islamic-holiday block
# below with a `hijri-converter` based computation for forward-dated
# years (2028+).

# --- Ramadan (moon-sighting approximation for Tunisia) ---------------------
# Cross-reference: https://www.timeanddate.com/holidays/tunisia/ramadan-begins
RAMADAN_PERIODS: list[tuple[date, date]] = [
    (date(2026, 2, 17), date(2026, 3, 18)),  # Ramadan 1447 AH
    (date(2027, 2,  7), date(2027, 3,  8)),  # Ramadan 1448 AH
]

# --- Tunisia fixed-date public holidays ------------------------------------
# Source: Journal Officiel de la République Tunisienne. Keyed (month, day).
TUNISIA_PUBLIC_HOLIDAYS_FIXED: dict[tuple[int, int], str] = {
    (1,  1): "New Year",
    (3, 20): "Independence Day",
    (4,  9): "Martyrs' Day",
    (5,  1): "Labour Day",
    (7, 25): "Republic Day",
    (8, 13): "Women's Day",
    (12, 17): "Revolution Day",
}

# --- Tunisia Islamic public holidays 2026–2027 -----------------------------
# Moon-sighting dependent; hardcoded until hijri-converter is integrated.
# Source: https://www.officeholidays.com/countries/tunisia/
TUNISIA_ISLAMIC_HOLIDAYS: list[date] = [
    # 2026 ----------------------------------------------------------------
    date(2026, 3, 20),  # Eid al-Fitr day 1
    date(2026, 3, 21),  # Eid al-Fitr day 2
    date(2026, 3, 22),  # Eid al-Fitr day 3
    date(2026, 5, 27),  # Eid al-Adha day 1
    date(2026, 5, 28),  # Eid al-Adha day 2
    date(2026, 6, 17),  # Islamic New Year (1 Muharram 1448)
    date(2026, 8, 26),  # Mawlid an-Nabi
    # 2027 ----------------------------------------------------------------
    date(2027, 3, 10),  # Eid al-Fitr day 1
    date(2027, 3, 11),  # Eid al-Fitr day 2
    date(2027, 3, 12),  # Eid al-Fitr day 3
    date(2027, 5, 17),  # Eid al-Adha day 1
    date(2027, 5, 18),  # Eid al-Adha day 2
    date(2027, 6,  6),  # Islamic New Year (1 Muharram 1449)
    date(2027, 8, 15),  # Mawlid an-Nabi
]

# --- Tunisia school holidays ----------------------------------------------
# Source: Ministère de l'Éducation — http://www.education.gov.tn/
# Approximate windows; exact dates vary year-to-year by a few days.
TUNISIA_SCHOOL_HOLIDAYS: list[tuple[date, date]] = [
    (date(2026,  1,  1), date(2026,  1,  5)),   # winter break tail 25/26
    (date(2026,  3, 20), date(2026,  3, 27)),   # spring break 2026
    (date(2026,  7,  1), date(2026,  9, 14)),   # summer 2026
    (date(2026, 12, 20), date(2027,  1,  5)),   # winter break 26/27
    (date(2027,  3, 20), date(2027,  3, 27)),   # spring break 2027
    (date(2027,  7,  1), date(2027,  9, 14)),   # summer 2027
]

# --- France school holidays 2026-2027 (union of Zones A / B / C) ----------
# Source: https://www.education.gouv.fr/calendrier-scolaire
# Union gives widest-coverage tourist-origin window (we don't know a
# French visitor's zone a priori).
FRANCE_SCHOOL_HOLIDAYS: list[tuple[date, date]] = [
    (date(2026,  2,  7), date(2026,  3,  9)),   # Hiver (zones A/B/C union)
    (date(2026,  4,  4), date(2026,  5,  4)),   # Printemps
    (date(2026,  7,  4), date(2026,  8, 31)),   # Été
    (date(2026, 10, 17), date(2026, 11,  2)),   # Toussaint
    (date(2026, 12, 19), date(2027,  1,  4)),   # Noël
    (date(2027,  2,  6), date(2027,  3,  8)),   # Hiver
    (date(2027,  4,  3), date(2027,  5,  3)),   # Printemps
    (date(2027,  7,  3), date(2027,  8, 30)),   # Été
    (date(2027, 10, 16), date(2027, 11,  1)),   # Toussaint
    (date(2027, 12, 18), date(2028,  1,  3)),   # Noël
]

# --- Germany school holidays 2026-2027 (union of 16 Länder) ---------------
# Source: https://www.kmk.org/service/ferien.html
GERMANY_SCHOOL_HOLIDAYS: list[tuple[date, date]] = [
    (date(2026,  2,  2), date(2026,  3, 14)),   # Winter / Fasching (union)
    (date(2026,  3, 23), date(2026,  4, 11)),   # Ostern
    (date(2026,  5, 15), date(2026,  5, 26)),   # Pfingsten
    (date(2026,  6, 22), date(2026,  9, 12)),   # Sommer
    (date(2026, 10,  5), date(2026, 11,  7)),   # Herbst
    (date(2026, 12, 19), date(2027,  1,  9)),   # Weihnachten
    (date(2027,  2,  1), date(2027,  3, 13)),
    (date(2027,  3, 22), date(2027,  4, 10)),
    (date(2027,  5, 14), date(2027,  5, 25)),
    (date(2027,  6, 21), date(2027,  9, 11)),
    (date(2027, 10,  4), date(2027, 11,  6)),
    (date(2027, 12, 18), date(2028,  1,  8)),
]

# --- UK school holidays 2026-2027 -----------------------------------------
# Source: https://www.gov.uk/school-term-holiday-dates
# Union across England / Wales / Scotland / NI. Local authority variation
# is a few days; acceptable for a demand-proxy signal.
UK_SCHOOL_HOLIDAYS: list[tuple[date, date]] = [
    (date(2026,  2, 14), date(2026,  2, 22)),   # February half-term
    (date(2026,  3, 28), date(2026,  4, 12)),   # Easter
    (date(2026,  5, 23), date(2026,  5, 31)),   # May half-term
    (date(2026,  7, 18), date(2026,  9,  1)),   # Summer
    (date(2026, 10, 24), date(2026, 11,  1)),   # October half-term
    (date(2026, 12, 19), date(2027,  1,  3)),   # Christmas
    (date(2027,  2, 13), date(2027,  2, 21)),
    (date(2027,  3, 27), date(2027,  4, 11)),
    (date(2027,  5, 22), date(2027,  5, 30)),
    (date(2027,  7, 17), date(2027,  8, 31)),
    (date(2027, 10, 23), date(2027, 10, 31)),
    (date(2027, 12, 18), date(2028,  1,  2)),
]
