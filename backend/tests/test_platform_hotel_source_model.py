from sqlalchemy import select
from db.models import PlatformHotelSource


async def test_platform_hotel_source_round_trip(db_session):
    src = PlatformHotelSource(
        platform_hotel_id=1, source="promohotel",
        source_hotel_name="Hotel Comp 1 (promo)", source_city_id=10,
    )
    db_session.add(src)
    await db_session.commit()
    res = await db_session.execute(
        select(PlatformHotelSource).where(PlatformHotelSource.platform_hotel_id == 1)
    )
    row = res.scalars().first()
    assert row is not None
    assert row.source == "promohotel"
