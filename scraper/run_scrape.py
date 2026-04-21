"""
run_scrape.py

Launches all hotel spiders in a single process.
Designed to be called by the Windows Task Scheduler via run_scrape.bat.

Tiered scraping strategy (aligned to API updates at 10h and 15h):
  - Morning run (10h):  days 0-60 (all cities/nights) + days 60-180 (all cities/nights)
  - Afternoon run (15h): days 0-60 only (captures intra-day price changes for near-term)

Each run is tagged with a unique scrape_run_id (UUID) so every item
from the same session can be grouped together downstream.
"""

import sys
import uuid
import logging
from datetime import datetime, timezone

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

from hotel_scraper.constants import (
    PROMOHOTEL_CITIES, TUNISIEPROMO_CITIES, NIGHTS_OPTIONS,
)
from hotel_scraper.spiders.promohotel import PromoHotelSpider
from hotel_scraper.spiders.tunisiepromo import TunisiePromoSpider


logger = logging.getLogger(__name__)

# Tiered day ranges: (start_day, number_of_days)
TIER_NEAR = (0, 60)      # every run   — high-value window for price dynamics
TIER_FAR  = (60, 120)    # morning only — days 60-180, slower-moving prices


def _schedule_crawls(process, tiers):
    """Queue crawls for all cities × nights × day-tiers."""
    for start_day, num_days in tiers:
        for city_id in PROMOHOTEL_CITIES:
            for nights in NIGHTS_OPTIONS:
                process.crawl(
                    PromoHotelSpider,
                    city_id=city_id, days=num_days, nights=nights,
                    start_day=start_day,
                )
        for city_id in TUNISIEPROMO_CITIES:
            for nights in NIGHTS_OPTIONS:
                process.crawl(
                    TunisiePromoSpider,
                    city_id=city_id, days=num_days, nights=nights,
                    start_day=start_day,
                )


def main():
    run_id = uuid.uuid4().hex[:12]
    now = datetime.now()
    is_morning = now.hour < 12

    logger.info(
        "=== Scrape run %s started at %s (%s run) ===",
        run_id, datetime.now(timezone.utc).isoformat(),
        "morning" if is_morning else "afternoon",
    )

    settings = get_project_settings()
    settings.set("SCRAPE_RUN_ID", run_id)

    process = CrawlerProcess(settings)

    if is_morning:
        _schedule_crawls(process, [TIER_NEAR, TIER_FAR])
    else:
        _schedule_crawls(process, [TIER_NEAR])

    process.start()

    logger.info("=== Scrape run %s finished ===", run_id)


if __name__ == "__main__":
    sys.exit(main() or 0)
