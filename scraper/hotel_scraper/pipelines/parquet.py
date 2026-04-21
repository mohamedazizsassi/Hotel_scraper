"""
Exports scraped items to Parquet files for ML consumption.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from itemadapter import ItemAdapter
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

PARQUET_SCHEMA = pa.schema([
    ("source",                  pa.string()),
    ("scraped_at",              pa.string()),
    ("scrape_run_id",           pa.string()),
    ("check_in",                pa.string()),
    ("check_out",               pa.string()),
    ("nights",                  pa.int16()),
    ("days_until_checkin",      pa.int16()),
    ("city_id",                 pa.int32()),
    ("city_name",               pa.string()),
    ("adults",                  pa.int8()),
    ("children",                pa.int8()),
    ("hotel_name",              pa.string()),
    ("hotel_name_normalized",   pa.string()),
    ("stars",                   pa.string()),
    ("boarding_name",           pa.string()),
    ("room_name",               pa.string()),
    ("price",                   pa.float32()),
    ("price_per_night",         pa.float32()),
    ("sur_demande",             pa.bool_()),
    ("supplements",             pa.string()),
])


class ParquetExportPipeline:
    """
    Buffers items and writes .parquet files with Snappy compression.
    One file per spider per run.

    Output: <PARQUET_OUTPUT_DIR>/<spider_name>_<timestamp>.parquet
    """

    BATCH_SIZE = 10_000

    @classmethod
    def from_crawler(cls, crawler):
        output_dir = crawler.settings.get("PARQUET_OUTPUT_DIR", "output")
        return cls(output_dir=Path(output_dir))

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.writer = None
        self.file_path = None
        self.items_buffer = []

    def open_spider(self, spider):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        self.file_path = self.output_dir / f"{spider.name}_{timestamp}.parquet"
        self.writer = pq.ParquetWriter(
            str(self.file_path), PARQUET_SCHEMA, compression="snappy",
        )
        self.items_buffer = []
        logger.info("Parquet export → %s", self.file_path)

    def process_item(self, item, spider):
        row = dict(ItemAdapter(item))
        row["supplements"] = json.dumps(row.get("supplements") or [])
        self.items_buffer.append(row)

        if len(self.items_buffer) >= self.BATCH_SIZE:
            self._flush()
        return item

    def close_spider(self, spider):
        if self.items_buffer:
            self._flush()
        if self.writer:
            self.writer.close()
            logger.info("Parquet file written → %s", self.file_path)

    def _flush(self):
        if not self.items_buffer:
            return
        columns = {field.name: [] for field in PARQUET_SCHEMA}
        for row in self.items_buffer:
            for col in columns:
                columns[col].append(row.get(col))
        table = pa.table(columns, schema=PARQUET_SCHEMA)
        self.writer.write_table(table)
        self.items_buffer.clear()
