import logging
from datetime import datetime, timezone
from itemadapter import ItemAdapter
import pymongo


logger = logging.getLogger(__name__)


class MongoDBPipeline:
    """
    Persists every HotelPriceItem into MongoDB.

    Collection: hotel_prices
    Index: (source, hotel_name, check_in, boarding_name, room_name) → unique
           so re-running a scrape updates prices rather than duplicating them.
    """

    COLLECTION = "hotel_prices"

    def __init__(
        self,
        mongo_uri: str,
        mongo_db: str,
        server_selection_timeout_ms: int = 3000,
        connect_timeout_ms: int = 3000,
        socket_timeout_ms: int = 5000,
    ):
        self.mongo_uri = mongo_uri
        self.mongo_db  = mongo_db
        self.server_selection_timeout_ms = server_selection_timeout_ms
        self.connect_timeout_ms = connect_timeout_ms
        self.socket_timeout_ms = socket_timeout_ms
        self.client    = None
        self.db        = None
        self.enabled   = True

    # ------------------------------------------------------------------ #
    #  Scrapy lifecycle hooks                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            mongo_uri=crawler.settings.get("MONGO_URI", "mongodb://localhost:27017"),
            mongo_db=crawler.settings.get("MONGO_DB",  "hotel_scraper"),
            server_selection_timeout_ms=crawler.settings.getint(
                "MONGO_SERVER_SELECTION_TIMEOUT_MS", 3000
            ),
            connect_timeout_ms=crawler.settings.getint("MONGO_CONNECT_TIMEOUT_MS", 3000),
            socket_timeout_ms=crawler.settings.getint("MONGO_SOCKET_TIMEOUT_MS", 5000),
        )

    def open_spider(self, spider=None):
        try:
            self.client = pymongo.MongoClient(
                self.mongo_uri,
                serverSelectionTimeoutMS=self.server_selection_timeout_ms,
                connectTimeoutMS=self.connect_timeout_ms,
                socketTimeoutMS=self.socket_timeout_ms,
            )
            # Force an early connection check so failures happen at startup.
            self.client.admin.command("ping")
            self.db = self.client[self.mongo_db]
            self._ensure_indexes()
            self.enabled = True
            logger.info("MongoDB pipeline connected → %s / %s", self.mongo_db, self.COLLECTION)
        except pymongo.errors.PyMongoError as exc:
            self.enabled = False
            self.db = None
            if self.client:
                self.client.close()
            self.client = None
            logger.error(
                "MongoDB unavailable (%s). Continuing crawl without DB writes.",
                exc,
            )

    def close_spider(self, spider=None):
        if self.client:
            self.client.close()
            logger.info("MongoDB pipeline disconnected.")

    # ------------------------------------------------------------------ #
    #  Item processing                                                     #
    # ------------------------------------------------------------------ #

    def process_item(self, item, spider=None):
        if not self.enabled or self.db is None:
            return item

        adapter = ItemAdapter(item)
        doc = dict(adapter)

        try:
            self.db[self.COLLECTION].insert_one(doc)
        except pymongo.errors.DuplicateKeyError:
            logger.debug("Duplicate item skipped: %s", doc)
        except pymongo.errors.PyMongoError as exc:
            logger.error("MongoDB write failed; item skipped in DB only: %s", exc)
        return item

    # ------------------------------------------------------------------ #
    #  Index management                                                    #
    # ------------------------------------------------------------------ #

    def _ensure_indexes(self):
        col = self.db[self.COLLECTION]
        col.create_index(
            [
                ("source",        pymongo.ASCENDING),
                ("hotel_name",    pymongo.ASCENDING),
                ("check_in",      pymongo.ASCENDING),
                ("boarding_name", pymongo.ASCENDING),
                ("room_name",     pymongo.ASCENDING),
                ("scraped_at",    pymongo.ASCENDING),
            ],
            name="idx_hotel_price_time",
        )
        # Fast range query on check_in for ML feature pipelines
        col.create_index([("check_in", pymongo.ASCENDING)], name="idx_check_in")
        col.create_index([("source",   pymongo.ASCENDING)], name="idx_source")


# ------------------------------------------------------------------ #
#  Optional: duplicate filter in-memory (within a single run)         #
# ------------------------------------------------------------------ #

class DuplicateFilterPipeline:
    """
    Drops exact price duplicates within the same crawl session
    before they reach the MongoDB pipeline.
    """

    def __init__(self):
        self.seen = set()

    def process_item(self, item, spider=None):
        key = (
            item.get("source"),
            item.get("hotel_name"),
            item.get("check_in"),
            item.get("boarding_name"),
            item.get("room_name"),
            item.get("scraped_at"),
        )
        if key in self.seen:
            logger.debug("Duplicate dropped: %s", key)
            from scrapy.exceptions import DropItem
            raise DropItem(f"Duplicate item: {key}")
        self.seen.add(key)
        return item
