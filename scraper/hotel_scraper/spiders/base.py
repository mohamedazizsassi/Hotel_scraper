"""
Base spider with shared logic for all hotel price scrapers.

Subclasses only need to define:
  - name, allowed_domains, SOURCE, BASE_URL
  - CITY_MAP            : dict mapping city_id → city_name
  - DEFAULT_CITY_ID     : int
  - _extra_fields()     : optional payload overrides
  - _extract_rooms()    : source-specific room parsing
  - _item_overrides()   : source-specific item field defaults
"""

import logging
import uuid
from datetime import date, datetime, timezone

import scrapy

from hotel_scraper.items import HotelPriceItem
from hotel_scraper.utils import build_encoded_payload, date_range, to_float

logger = logging.getLogger(__name__)


class BaseHotelSpider(scrapy.Spider):

    # Subclasses must set these
    SOURCE: str
    BASE_URL: str
    CITY_MAP: dict
    DEFAULT_CITY_ID: int

    def __init__(self, city_id=None, days=30, nights=1, adults=2, children=0, start_day=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.city_id   = int(city_id) if city_id is not None else self.DEFAULT_CITY_ID
        self.city_name = self.CITY_MAP.get(self.city_id, f"unknown-{self.city_id}")
        self.days      = int(days)
        self.nights    = int(nights)
        self.adults    = int(adults)
        self.children  = int(children)
        self.start_day = int(start_day)
        self.scrape_run_id = None

        if self.nights not in (1, 3, 5):
            logger.warning("Invalid nights=%s; defaulting to 1", self.nights)
            self.nights = 1

    @classmethod
    def from_crawler(cls, crawler, *args, **kwargs):
        spider = super().from_crawler(crawler, *args, **kwargs)
        run_id = crawler.settings.get("SCRAPE_RUN_ID")
        spider.scrape_run_id = run_id or uuid.uuid4().hex[:12]
        return spider

    # ------------------------------------------------------------------ #
    #  Request generation                                                  #
    # ------------------------------------------------------------------ #

    def _extra_fields(self) -> dict | None:
        """Override in subclass to add source-specific payload fields."""
        return None

    def _referer(self) -> str:
        """Override in subclass for source-specific Referer header."""
        return self.BASE_URL

    async def start(self):
        today = date.today()
        logger.info(
            "%s starting | city=%s (%d) | days=%d | nights=%d | start_day=%d",
            self.SOURCE, self.city_name, self.city_id,
            self.days, self.nights, self.start_day,
        )

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

    # ------------------------------------------------------------------ #
    #  Response parsing                                                    #
    # ------------------------------------------------------------------ #

    def parse(self, response):
        check_in  = response.meta["check_in"]
        check_out = response.meta["check_out"]
        nights    = response.meta["nights"]
        city_id   = response.meta["city_id"]
        city_name = response.meta["city_name"]
        adults    = response.meta["adults"]
        children  = response.meta["children"]
        scraped_at         = datetime.now(timezone.utc).isoformat()
        days_until_checkin = (date.fromisoformat(check_in) - date.today()).days

        try:
            data = response.json()
        except Exception as exc:
            logger.error("JSON parse error for %s → %s | body[:200]: %s", check_in, exc, response.text[:200])
            return

        hotels = data.get("HotelSearch", [])
        if not isinstance(hotels, list) or not hotels:
            logger.warning("No hotels returned for check_in=%s", check_in)
            return

        logger.info("check_in=%s → %d hotel(s) found", check_in, len(hotels))

        for hotel_entry in hotels:
            hotel_info = hotel_entry.get("Hotel", {})
            hotel_name = hotel_info.get("Name", "Unknown Hotel")
            stars      = hotel_info.get("Category", {}).get("Star", "N/A")

            boardings = hotel_entry.get("Price", {}).get("Boarding", [])
            if not boardings:
                continue

            for boarding in boardings:
                boarding_name = boarding.get("Name", "Unknown Boarding")
                rooms = self._extract_rooms(boarding)
                if not rooms:
                    continue

                for room in rooms:
                    item_data = dict(
                        source             = self.SOURCE,
                        scraped_at         = scraped_at,
                        scrape_run_id      = self.scrape_run_id,
                        check_in           = check_in,
                        check_out          = check_out,
                        nights             = nights,
                        days_until_checkin = days_until_checkin,
                        city_id            = city_id,
                        city_name          = city_name,
                        adults             = adults,
                        children           = children,
                        hotel_name         = hotel_name,
                        stars              = stars,
                        boarding_name      = boarding_name,
                        room_name          = room["name"],
                        price              = room["price"],
                        sur_demande        = room.get("sur_demande", False),
                        supplements        = room.get("supplements", []),
                    )
                    yield HotelPriceItem(**item_data)

    # ------------------------------------------------------------------ #
    #  Subclasses must implement                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_rooms(boarding: dict) -> list[dict]:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    #  Error handler                                                       #
    # ------------------------------------------------------------------ #

    def handle_error(self, failure):
        logger.error("Request failed: %s | %s", failure.request.url, failure.getErrorMessage())
