"""
Enriches items with computed/normalized fields before storage.
"""

import re
import unicodedata

_STRIP_PREFIXES = re.compile(
    r"^(h[oô]tel|r[eé]sidence|club|dar|riad|maison\s+d['\u2019]h[oô]tes?|pension)\s+",
    re.IGNORECASE,
)
_MULTI_SPACE = re.compile(r"\s+")


def normalize_hotel_name(raw: str) -> str:
    """
    Normalize a hotel name for cross-source matching:
      1. Strip accents  (Hôtel → Hotel)
      2. Lowercase
      3. Remove common prefixes (Hotel, Résidence, Dar, …)
      4. Strip non-alphanumeric chars (keep spaces)
      5. Collapse whitespace and trim
    """
    nfkd = unicodedata.normalize("NFKD", raw)
    stripped = "".join(c for c in nfkd if not unicodedata.combining(c))
    lower = stripped.lower()
    lower = _STRIP_PREFIXES.sub("", lower)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lower)
    return _MULTI_SPACE.sub(" ", cleaned).strip()


class NormalizationPipeline:
    """
    Adds hotel_name_normalized and price_per_night to every item.
    Must run before DuplicateFilterPipeline and MongoDBPipeline.
    """

    def process_item(self, item, spider=None):
        raw = item.get("hotel_name") or ""
        item["hotel_name_normalized"] = normalize_hotel_name(raw)

        price = item.get("price")
        nights = item.get("nights")
        if price is not None and nights:
            item["price_per_night"] = round(price / nights, 2)

        stars = item.get("stars")
        item["stars"] = str(stars) if stars is not None else None

        return item
