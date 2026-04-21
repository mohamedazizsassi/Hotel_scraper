"""
hotel_scraper.pipelines

All item pipelines, importable by their short path for settings.py.
"""

from hotel_scraper.pipelines.normalization import NormalizationPipeline
from hotel_scraper.pipelines.dedup import DuplicateFilterPipeline
from hotel_scraper.pipelines.mongodb import MongoDBPipeline
from hotel_scraper.pipelines.parquet import ParquetExportPipeline

__all__ = [
    "NormalizationPipeline",
    "DuplicateFilterPipeline",
    "MongoDBPipeline",
    "ParquetExportPipeline",
]
