from unittest.mock import AsyncMock, MagicMock
from db.mongo import count_hotel_prices


async def test_count_hotel_prices_returns_count():
    coll = MagicMock()
    coll.estimated_document_count = AsyncMock(return_value=24_400_000)
    fake_db = {"hotel_prices": coll}
    assert await count_hotel_prices(fake_db) == 24_400_000


async def test_count_hotel_prices_returns_none_on_error():
    coll = MagicMock()
    coll.estimated_document_count = AsyncMock(side_effect=RuntimeError("mongo down"))
    fake_db = {"hotel_prices": coll}
    assert await count_hotel_prices(fake_db) is None
