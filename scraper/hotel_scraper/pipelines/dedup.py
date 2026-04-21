"""
In-memory duplicate filter within a single crawl session.
"""

import logging

from scrapy.exceptions import DropItem

logger = logging.getLogger(__name__)


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
            raise DropItem(f"Duplicate item: {key}")
        self.seen.add(key)
        return item
