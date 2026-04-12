"""
Scrapy settings for hotel_scraper project.
Production-ready configuration with polite crawling defaults.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

BOT_NAME = "hotel_scraper"

SPIDER_MODULES = ["hotel_scraper.spiders"]
NEWSPIDER_MODULE = "hotel_scraper.spiders"

# ------------------------------------------------------------------ #
#  Identity                                                            #
# ------------------------------------------------------------------ #
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

# ------------------------------------------------------------------ #
#  Politeness / rate limiting  ← CRITICAL for real sites              #
# ------------------------------------------------------------------ #
ROBOTSTXT_OBEY          = False      # these sites have no robots.txt for APIs
DOWNLOAD_DELAY          = 1.5        # seconds between requests (per domain)
RANDOMIZE_DOWNLOAD_DELAY = True      # actual delay = 0.5–1.5× DOWNLOAD_DELAY
CONCURRENT_REQUESTS     = 4
CONCURRENT_REQUESTS_PER_DOMAIN = 2

# AutoThrottle: dynamically adjusts delay based on server latency
AUTOTHROTTLE_ENABLED         = True
AUTOTHROTTLE_START_DELAY     = 1.0
AUTOTHROTTLE_MAX_DELAY       = 10.0
AUTOTHROTTLE_TARGET_CONCURRENCY = 1.5
AUTOTHROTTLE_DEBUG           = False

# ------------------------------------------------------------------ #
#  Retry logic                                                         #
# ------------------------------------------------------------------ #
RETRY_ENABLED   = True
RETRY_TIMES     = 3                  # retry failed requests 3×
RETRY_HTTP_CODES = [500, 502, 503, 504, 429]

# ------------------------------------------------------------------ #
#  Timeouts                                                            #
# ------------------------------------------------------------------ #
DOWNLOAD_TIMEOUT = 30

# ------------------------------------------------------------------ #
#  Item pipelines  (order = priority, lower = earlier)                #
# ------------------------------------------------------------------ #
ITEM_PIPELINES = {
    "hotel_scraper.pipelines.DuplicateFilterPipeline": 200,
    "hotel_scraper.pipelines.MongoDBPipeline":         300,
}

# ------------------------------------------------------------------ #
#  MongoDB                                                             #
# ------------------------------------------------------------------ #
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB", "hotel_scraper")

# ------------------------------------------------------------------ #
#  Feeds (optional JSON export alongside MongoDB)                      #
# ------------------------------------------------------------------ #
# Write a CSV file per run (per spider) in output/
FEEDS = {
    "output/%(name)s_%(time)s.csv": {
        "format": "csv",
        "encoding": "utf-8",
    },
}

# ------------------------------------------------------------------ #
#  Logging                                                             #
# ------------------------------------------------------------------ #
LOG_LEVEL  = "INFO"
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"

# ------------------------------------------------------------------ #
#  HTTP cache (useful during development — disable in production)      #
# ------------------------------------------------------------------ #
HTTPCACHE_ENABLED = False
# HTTPCACHE_EXPIRATION_SECS = 3600
# HTTPCACHE_DIR = ".scrapy/httpcache"

# Disable cookies (these are stateless JSON APIs)
COOKIES_ENABLED = False

# ------------------------------------------------------------------ #
#  Default request headers                                             #
# ------------------------------------------------------------------ #
DEFAULT_REQUEST_HEADERS = {
    "Accept":           "application/json, text/javascript, */*; q=0.01",
    "Accept-Language":  "fr-FR,fr;q=0.9,en;q=0.8",
    "X-Requested-With": "XMLHttpRequest",
}
