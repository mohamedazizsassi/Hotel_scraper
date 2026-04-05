"""
spiders/promohotel_spider.py

Scrapy spider for promohotel.tn
Scrapes hotel prices for every day from today → today+30 days.
"""

import logging
from datetime import date, datetime, timezone

import scrapy

from hotel_scraper.items import HotelPriceItem
from hotel_scraper.utils import build_encoded_payload, date_range_1day, to_float


logger = logging.getLogger(__name__)


class PromoHotelSpider(scrapy.Spider):
    """
    Spider: promohotel

    Scrapes availability & prices from promohotel.tn for a given city.

    Run:
        scrapy crawl promohotel
        scrapy crawl promohotel -a city_id=3 -a days=30

    Spider arguments:
        city_id  (int)  : city to scrape   (default: 3 = Sousse)
        days     (int)  : how many nights forward to scrape (default: 30)
        adults   (int)  : number of adults per room (default: 2)
    """

    name              = "promohotel"
    allowed_domains   = ["promohotel.tn"]
    SOURCE            = "promohotel"
    BASE_URL          = "https://www.promohotel.tn/hotel/ajax-availability"

    # ------------------------------------------------------------------ #
    #  Spider initialisation                                               #
    # ------------------------------------------------------------------ #

    def __init__(self, city_id=3, days=30, adults=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.city_id = int(city_id)
        self.days    = int(days)
        self.adults  = int(adults)

    # ------------------------------------------------------------------ #
    #  Request generation                                                  #
    # ------------------------------------------------------------------ #

    def start_requests(self):
        """
        Generates one Scrapy Request per day in the date window.
        Meta carries check_in / check_out so the parser knows which
        date window this response belongs to.
        """
        today = date.today()
        logger.info(
            "PromoHotel spider starting | city=%d | %d nights | from %s",
            self.city_id, self.days, today.isoformat(),
        )

        for check_in, check_out in date_range_1day(today, self.days):
            encoded = build_encoded_payload(
                check_in  = check_in,
                check_out = check_out,
                city_id   = self.city_id,
                adults    = self.adults,
            )
            url = f"{self.BASE_URL}?HotelSearch={encoded}"

            yield scrapy.Request(
                url      = url,
                callback = self.parse,
                headers  = {"Referer": "https://www.promohotel.tn/"},
                meta     = {
                    "check_in":  check_in,
                    "check_out": check_out,
                    "city_id":   self.city_id,
                    "adults":    self.adults,
                    # Don't cache — prices are live
                    "dont_cache": True,
                },
                errback  = self.handle_error,
            )

    # ------------------------------------------------------------------ #
    #  Response parsing                                                    #
    # ------------------------------------------------------------------ #

    def parse(self, response):
        """
        Parses the JSON response and yields one HotelPriceItem
        per room option found.
        """
        check_in  = response.meta["check_in"]
        check_out = response.meta["check_out"]
        city_id   = response.meta["city_id"]
        adults    = response.meta["adults"]
        scraped_at = datetime.now(timezone.utc).isoformat()

        try:
            data = response.json()
        except Exception as exc:
            logger.error(
                "JSON parse error for %s → %s | body[:200]: %s",
                check_in, exc, response.text[:200],
            )
            return

        hotels = data.get("HotelSearch", [])
        if not isinstance(hotels, list) or not hotels:
            logger.warning("No hotels returned for check_in=%s", check_in)
            return

        logger.info(
            "check_in=%s → %d hotel(s) found", check_in, len(hotels)
        )

        for hotel_entry in hotels:
            hotel_info = hotel_entry.get("Hotel", {})
            hotel_name = hotel_info.get("Name", "Unknown Hotel")
            stars      = hotel_info.get("Category", {}).get("Star", "N/A")

            boardings = hotel_entry.get("Price", {}).get("Boarding", [])
            if not boardings:
                logger.debug("No boarding info for %s on %s", hotel_name, check_in)
                continue

            for boarding in boardings:
                boarding_name = boarding.get("Name", "Unknown Boarding")
                rooms         = self._extract_rooms(boarding)

                if not rooms:
                    logger.debug(
                        "No rooms for %s / %s on %s",
                        hotel_name, boarding_name, check_in,
                    )
                    continue

                for room in rooms:
                    yield HotelPriceItem(
                        source        = self.SOURCE,
                        scraped_at    = scraped_at,
                        check_in      = check_in,
                        check_out     = check_out,
                        city_id       = city_id,
                        adults        = adults,
                        hotel_name    = hotel_name,
                        stars         = stars,
                        boarding_name = boarding_name,
                        room_name     = room["name"],
                        price         = room["price"],
                        sur_demande   = room["sur_demande"],
                        supplements   = room["supplements"],
                    )

    # ------------------------------------------------------------------ #
    #  Room detail extractor  (same logic as your original script)        #
    # ------------------------------------------------------------------ #

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

                rooms.append(
                    {
                        "name":        room_name,
                        "price":       base_price,
                        "sur_demande": bool(room.get("surDemande", False)),
                        "supplements": supplements,
                    }
                )
        return rooms

    # ------------------------------------------------------------------ #
    #  Error handler                                                       #
    # ------------------------------------------------------------------ #

    def handle_error(self, failure):
        logger.error(
            "Request failed: %s | %s",
            failure.request.url,
            failure.getErrorMessage(),
        )
