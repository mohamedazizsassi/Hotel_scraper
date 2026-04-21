"""
Spider for tunisiepromo.tn — hotel prices (less room detail than promohotel).
"""

from datetime import date, datetime, timezone

import scrapy

from hotel_scraper.constants import TUNISIEPROMO_CITIES
from hotel_scraper.spiders.base import BaseHotelSpider
from hotel_scraper.utils import build_encoded_payload, date_range, to_float


class TunisiePromoSpider(BaseHotelSpider):

    name            = "tunisiepromo"
    allowed_domains = ["tunisiepromo.tn"]
    SOURCE          = "tunisiepromo"
    BASE_URL        = "https://www.tunisiepromo.tn/hotel/ajax-availability"
    CITY_MAP        = TUNISIEPROMO_CITIES
    DEFAULT_CITY_ID = 34

    # tunisiepromo's /hotel/ajax-availability requires a PHPSESSID cookie,
    # so we must enable cookies (disabled globally) and prime the session
    # by hitting the homepage before any availability request.
    custom_settings = {
        "COOKIES_ENABLED": True,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 1,
        "DOWNLOAD_DELAY": 2.5,
    }

    def _extra_fields(self):
        return {"Product": "hotel"}

    def _referer(self):
        return "https://www.tunisiepromo.tn/hotel"

    async def start(self):
        yield scrapy.Request(
            url="https://www.tunisiepromo.tn/",
            callback=self._after_session,
            dont_filter=True,
        )

    def _after_session(self, response):
        today = date.today()
        for check_in, check_out in date_range(today, self.days, nights=self.nights, start_day=self.start_day):
            encoded = build_encoded_payload(
                check_in     = check_in,
                check_out    = check_out,
                city_id      = self.city_id,
                adults       = self.adults,
                children     = self.children,
                extra_fields = self._extra_fields(),
            )
            url = f"{self.BASE_URL}?HotelSearch={encoded}"
            yield scrapy.Request(
                url      = url,
                callback = self.parse,
                headers  = {"Referer": self._referer()},
                meta     = {
                    "check_in":   check_in,
                    "check_out":  check_out,
                    "nights":     self.nights,
                    "city_id":    self.city_id,
                    "city_name":  self.city_name,
                    "adults":     self.adults,
                    "children":   self.children,
                    "dont_cache": True,
                },
                errback  = self.handle_error,
            )

    @staticmethod
    def _extract_rooms(boarding: dict) -> list[dict]:
        rooms = []
        try:
            raw_rooms = boarding["Pax"][0]["Rooms"]
        except (KeyError, IndexError, TypeError):
            return rooms

        for room in raw_rooms:
            room_name = room.get("Name") or "Unknown room"
            price = to_float(room.get("Price"))
            if price is None:
                continue
            rooms.append({"name": room_name, "price": price})

        return rooms
