"""
hotel_scraper/utils.py

Shared helpers used by all hotel price spiders.
"""

import base64
import json
from datetime import date, timedelta
from urllib.parse import quote


# ------------------------------------------------------------------ #
#  Date range generator                                                #
# ------------------------------------------------------------------ #

def date_range_1day(start: date, days: int = 30):
    """
    Yields (check_in, check_out) tuples where:
      - check_in  runs from `start` to `start + days - 1`
      - check_out = check_in + 1 day  (always 1-night stays)

    Usage:
        for ci, co in date_range_1day(date.today(), 30):
            ...
    """
    for offset in range(days):
        check_in  = start + timedelta(days=offset)
        check_out = check_in + timedelta(days=1)
        yield check_in.isoformat(), check_out.isoformat()


# ------------------------------------------------------------------ #
#  Payload builders                                                    #
# ------------------------------------------------------------------ #

def build_encoded_payload(
    check_in:  str,
    check_out: str,
    city_id:   int,
    adults:    int,
    children:  int = 0,
    extra_fields: dict | None = None,
) -> str:
    """
    Builds and base64-encodes the JSON search payload shared by
    promohotel.tn and tunisiepromo.tn.

    `extra_fields` are merged into the top-level SearchDetails dict
    (e.g. {"Product": "hotel"} for tunisiepromo).

    Returns the URL-safe base64 string ready to use as a query param.
    """
    search_details: dict = {
        "BookingDetails": {
            "CheckIn":  check_in,
            "CheckOut": check_out,
            "City":     str(city_id),
        },
        "Rooms": [
            {"Adult": str(adults), "children": int(children), "Child": []},
        ],
        "GroupingHotel":   True,
        "CombinationRooms": False,
        "BoardingByRooms":  False,
        "Filters":          {"Source": "all"},
    }

    if extra_fields:
        search_details.update(extra_fields)

    payload    = {"SearchDetails": search_details}
    json_bytes = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    b64        = base64.b64encode(json_bytes).decode("utf-8")
    return quote(b64, safe="")


# ------------------------------------------------------------------ #
#  Price parsing                                                       #
# ------------------------------------------------------------------ #

def to_float(value) -> float | None:
    """Safely converts price values (int / float / str) to float."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ".").strip())
        except ValueError:
            return None
    return None
