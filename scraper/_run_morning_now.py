"""One-off: force a full morning run (both tiers) regardless of current hour."""
import uuid, logging
from datetime import datetime, timezone
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from hotel_scraper.constants import PROMOHOTEL_CITIES, TUNISIEPROMO_CITIES, NIGHTS_OPTIONS
from hotel_scraper.spiders.promohotel import PromoHotelSpider
from hotel_scraper.spiders.tunisiepromo import TunisiePromoSpider
from run_scrape import TIER_NEAR, _schedule_crawls

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logging.getLogger("pymongo").setLevel(logging.WARNING)
run_id = uuid.uuid4().hex[:12]
print(f"=== Forced morning run {run_id} started at {datetime.now(timezone.utc).isoformat()} ===", flush=True)
settings = get_project_settings()
settings.set("SCRAPE_RUN_ID", run_id)
process = CrawlerProcess(settings)
_schedule_crawls(process, [TIER_NEAR])
process.start()
print(f"=== Forced morning run {run_id} finished ===", flush=True)
