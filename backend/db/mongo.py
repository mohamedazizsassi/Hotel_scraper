"""Thin async MongoDB client. Used ONLY for the live hotel_prices total via
estimated_document_count() (reads collection metadata — no scan). Degrades to
None if Mongo is unreachable so the rest of the (PG-sourced) API still works.
"""
from __future__ import annotations

from motor.motor_asyncio import AsyncIOMotorClient

from core.config import settings

_client: AsyncIOMotorClient | None = None


def get_mongo_client() -> AsyncIOMotorClient:
    global _client
    if _client is None:
        _client = AsyncIOMotorClient(settings.mongo_uri, serverSelectionTimeoutMS=3000)
    return _client


def get_mongo_db():
    return get_mongo_client()[settings.mongo_db]


async def count_hotel_prices(db) -> int | None:
    """Estimated row count of hotel_prices. Returns None if Mongo errors out."""
    try:
        return await db["hotel_prices"].estimated_document_count()
    except Exception:
        return None


def close_mongo_client() -> None:
    global _client
    if _client is not None:
        _client.close()
        _client = None
