from sqlalchemy import select
from db.models import PlatformHotel


async def test_hotel_contact_fields_persist(db_session):
    res = await db_session.execute(
        select(PlatformHotel).where(PlatformHotel.hotel_name_normalized == "hotel_comp_1")
    )
    hotel = res.scalar_one()
    hotel.contact_email = "contact@hotelcomp1.tn"
    hotel.contact_phone = "+216 71 000 000"
    await db_session.commit()

    # Force the re-read to hit the DB instead of the session identity map. Without
    # this, SQLAlchemy returns the same in-memory object and the assertions would
    # pass even if the columns were never mapped/persisted (false positive).
    db_session.expunge_all()

    res2 = await db_session.execute(
        select(PlatformHotel).where(PlatformHotel.hotel_name_normalized == "hotel_comp_1")
    )
    reread = res2.scalar_one()
    assert reread.contact_email == "contact@hotelcomp1.tn"
    assert reread.contact_phone == "+216 71 000 000"
