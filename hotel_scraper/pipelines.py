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

    def __init__(self, mongo_uri: str, mongo_db: str):
        self.mongo_uri = mongo_uri
        self.mongo_db  = mongo_db
        self.client    = None
        self.db        = None

    # ------------------------------------------------------------------ #
    #  Scrapy lifecycle hooks                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            mongo_uri=crawler.settings.get("MONGO_URI", "mongodb://localhost:27017"),
            mongo_db=crawler.settings.get("MONGO_DB",  "hotel_scraper"),
        )

    def open_spider(self, spider):
        self.client = pymongo.MongoClient(self.mongo_uri)
        self.db     = self.client[self.mongo_db]
        self._ensure_indexes()
        logger.info("MongoDB pipeline connected → %s / %s", self.mongo_db, self.COLLECTION)

    def close_spider(self, spider):
        if self.client:
            self.client.close()
            logger.info("MongoDB pipeline disconnected.")

    # ------------------------------------------------------------------ #
    #  Item processing                                                     #
    # ------------------------------------------------------------------ #

    def process_item(self, item, spider):
        adapter = ItemAdapter(item)
        doc = dict(adapter)

        # Upsert: update if same hotel+date+room already scraped
        filter_key = {
            "source":        doc.get("source"),
            "hotel_name":    doc.get("hotel_name"),
            "check_in":      doc.get("check_in"),
            "boarding_name": doc.get("boarding_name"),
            "room_name":     doc.get("room_name"),
            "city_id":       doc.get("city_id"),
        }
        doc["updated_at"] = datetime.now(timezone.utc).isoformat()

        self.db[self.COLLECTION].update_one(
            filter_key,
            {"$set": doc},
            upsert=True,
        )
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
            ],
            unique=True,
            name="unique_hotel_price",
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

    def process_item(self, item, spider):
        key = (
            item.get("source"),
            item.get("hotel_name"),
            item.get("check_in"),
            item.get("boarding_name"),
            item.get("room_name"),
        )
        if key in self.seen:
            logger.debug("Duplicate dropped: %s", key)
            from scrapy.exceptions import DropItem
            raise DropItem(f"Duplicate item: {key}")
        self.seen.add(key)
        return item
