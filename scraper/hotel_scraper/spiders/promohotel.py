"""
Spider for promohotel.tn — hotel prices with room supplements.
"""

from hotel_scraper.constants import PROMOHOTEL_CITIES
from hotel_scraper.spiders.base import BaseHotelSpider
from hotel_scraper.utils import to_float


class PromoHotelSpider(BaseHotelSpider):

    name            = "promohotel"
    allowed_domains = ["promohotel.tn"]
    SOURCE          = "promohotel"
    BASE_URL        = "https://www.promohotel.tn/hotel/ajax-availability"
    CITY_MAP        = PROMOHOTEL_CITIES
    DEFAULT_CITY_ID = 3  # Sousse

    def _referer(self):
        return "https://www.promohotel.tn/"

    @staticmethod
    def _extract_rooms(boarding: dict) -> list[dict]:
        rooms = []
        for pax in boarding.get("Pax", []):
            for room in pax.get("Rooms", []):
                room_name  = room.get("Name") or "Unknown room"
                base_price = to_float(room.get("Price"))
                if base_price is None:
                    continue

                supplements = []
                for view in room.get("View", []):
                    view_name  = view.get("Name") or "Supplément"
                    view_price = to_float(view.get("Price"))
                    if view_price is None:
                        continue
                    supplements.append({"name": view_name, "price": view_price})

                rooms.append({
                    "name":        room_name,
                    "price":       base_price,
                    "sur_demande": bool(room.get("surDemande", False)),
                    "supplements": supplements,
                })
        return rooms
