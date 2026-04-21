"""
Persists scraped items into MongoDB.
"""

import logging

from itemadapter import ItemAdapter
import pymongo

logger = logging.getLogger(__name__)


class MongoDBPipeline:
    """
    Persists every HotelPriceItem into MongoDB.

    Collection: hotel_prices
    Index: (source, hotel_name, check_in, boarding_name, room_name, scraped_at)
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
        self.client  = None
        self.db      = None
        self.enabled = True

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
            self.client.admin.command("ping")
            self.db = self.client[self.mongo_db]
            self._ensure_indexes()
            self.enabled = True
            logger.info("MongoDB connected → %s / %s", self.mongo_db, self.COLLECTION)
        except pymongo.errors.PyMongoError as exc:
            self.enabled = False
            self.db = None
            if self.client:
                self.client.close()
            self.client = None
            logger.error("MongoDB unavailable (%s). Continuing without DB writes.", exc)

    def close_spider(self, spider=None):
        if self.client:
            self.client.close()
            logger.info("MongoDB disconnected.")

    def process_item(self, item, spider=None):
        if not self.enabled or self.db is None:
            return item

        doc = dict(ItemAdapter(item))
        try:
            self.db[self.COLLECTION].insert_one(doc)
        except pymongo.errors.DuplicateKeyError:
            logger.debug("Duplicate item skipped: %s", doc.get("hotel_name"))
        except pymongo.errors.PyMongoError as exc:
            logger.error("MongoDB write failed: %s", exc)
        return item

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
        col.create_index([("check_in", pymongo.ASCENDING)], name="idx_check_in")
        col.create_index([("source",   pymongo.ASCENDING)], name="idx_source")
