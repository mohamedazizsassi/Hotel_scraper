"""
spiders/tunisiepromo_spider.py

Scrapy spider for tunisiepromo.tn
Scrapes hotel prices for every day from today → today+30 days.
"""

import logging
from datetime import date, datetime, timezone

import scrapy

from hotel_scraper.items import HotelPriceItem
from hotel_scraper.utils import build_encoded_payload, date_range_1day, to_float


logger = logging.getLogger(__name__)


class TunisiePromoSpider(scrapy.Spider):
    """
    Spider: tunisiepromo

    Scrapes availability & prices from tunisiepromo.tn for a given city.

    Run:
        scrapy crawl tunisiepromo
        scrapy crawl tunisiepromo -a city_id=34 -a days=30

    Spider arguments:
        city_id  (int)  : city to scrape   (default: 34 = Hammamet)
        days     (int)  : how many nights forward to scrape (default: 30)
        adults   (int)  : number of adults per room (default: 2)
        children (int)  : number of children per room (default: 0)
    """

    name            = "tunisiepromo"
    allowed_domains = ["tunisiepromo.tn"]
    SOURCE          = "tunisiepromo"
    BASE_URL        = "https://www.tunisiepromo.tn/hotel/ajax-availability"

    # ------------------------------------------------------------------ #
    #  Spider initialisation                                               #
    # ------------------------------------------------------------------ #

    def __init__(self, city_id=34, days=30, adults=2, children=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.city_id = int(city_id)
        self.days    = int(days)
        self.adults  = int(adults)
        self.children = int(children)

    # ------------------------------------------------------------------ #
    #  Request generation                                                  #
    # ------------------------------------------------------------------ #

    async def start(self):
        """
        Generates one Scrapy Request per day in the date window.
        tunisiepromo requires an extra "Product": "hotel" field
        in the payload — handled via extra_fields.
        """
        today = date.today()
        logger.info(
            "TunisiePromo spider starting | city=%d | %d nights | from %s",
            self.city_id, self.days, today.isoformat(),
        )

        for check_in, check_out in date_range_1day(today, self.days):
            encoded = build_encoded_payload(
                check_in     = check_in,
                check_out    = check_out,
                city_id      = self.city_id,
                adults       = self.adults,
                children     = self.children,
                extra_fields = {"Product": "hotel"},   # tunisiepromo-specific
            )
            url = f"{self.BASE_URL}?HotelSearch={encoded}"

            yield scrapy.Request(
                url      = url,
                callback = self.parse,
                headers  = {"Referer": "https://www.tunisiepromo.tn/hotel"},
                meta     = {
                    "check_in":  check_in,
                    "check_out": check_out,
                    "city_id":   self.city_id,
                    "adults":    self.adults,
                    "children":  self.children,
                    "dont_cache": True,
                },
                errback  = self.handle_error,
            )

    # ------------------------------------------------------------------ #
    #  Response parsing                                                    #
    # ------------------------------------------------------------------ #

    def parse(self, response):
        """
        Parses the JSON response.
        tunisiepromo exposes less room detail than promohotel,
        so we extract what's available and mark missing fields clearly.
        """
        check_in   = response.meta["check_in"]
        check_out  = response.meta["check_out"]
        city_id    = response.meta["city_id"]
        adults     = response.meta["adults"]
        children   = response.meta["children"]
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

                # tunisiepromo only exposes one room per boarding in Pax[0]
                # We extract it and yield a single item per boarding type.
                price = self._extract_price(boarding)
                if price is None:
                    logger.debug(
                        "No price for %s / %s on %s",
                        hotel_name, boarding_name, check_in,
                    )
                    continue

                yield HotelPriceItem(
                    source        = self.SOURCE,
                    scraped_at    = scraped_at,
                    check_in      = check_in,
                    check_out     = check_out,
                    city_id       = city_id,
                    adults        = adults,
                    children      = children,
                    hotel_name    = hotel_name,
                    stars         = stars,
                    boarding_name = boarding_name,
                    room_name     = "N/A",        # not exposed by tunisiepromo
                    price         = price,
                    sur_demande   = False,
                    supplements   = [],
                )

    # ------------------------------------------------------------------ #
    #  Price extractor                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_price(boarding: dict) -> float | None:
        """
        tunisiepromo packs the price at Pax[0].Rooms[0].Price.
        Returns None if the path is missing or price is invalid.
        """
        try:
            raw = boarding["Pax"][0]["Rooms"][0]["Price"]
            return to_float(raw)
        except (KeyError, IndexError, TypeError):
            return None

    # ------------------------------------------------------------------ #
    #  Error handler                                                       #
    # ------------------------------------------------------------------ #

    def handle_error(self, failure):
        logger.error(
            "Request failed: %s | %s",
            failure.request.url,
            failure.getErrorMessage(),
        )
