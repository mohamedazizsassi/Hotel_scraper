from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, update as sa_update
from db.models import User
from schemas.profile import ManagerProfile, HotelBrief, ProfileUpdate


async def get_profile(user: User, db: AsyncSession) -> ManagerProfile:
    row = (await db.execute(text("""
        SELECT ph.id AS hotel_id,
               ph.hotel_name_display,
               c.name_display AS city_name,
               ph.stars_int,
               a.max_competitors,
               (SELECT COUNT(*) FROM user_competitor_selections ucs
                 WHERE ucs.user_id = :uid) AS competitor_count
        FROM user_hotel_assignments a
        JOIN platform_hotels ph ON ph.id = a.hotel_id
        JOIN cities c           ON c.id  = ph.city_id
        WHERE a.user_id = :uid AND a.is_active = TRUE
    """), {"uid": str(user.id)})).mappings().first()

    hotel = None
    max_comp = 4
    comp_count = 0
    if row is not None:
        hotel = HotelBrief(
            id=row["hotel_id"],
            hotel_name_display=row["hotel_name_display"],
            city_name=row["city_name"],
            stars_int=row["stars_int"],
        )
        max_comp = row["max_competitors"]
        comp_count = row["competitor_count"]

    return ManagerProfile(
        full_name=user.full_name,
        email=str(user.email),
        role=user.role,
        preferences=user.preferences or {},
        hotel=hotel,
        competitor_count=comp_count,
        max_competitors=max_comp,
        last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
    )


async def update_profile(user: User, patch: ProfileUpdate, db: AsyncSession) -> ManagerProfile:
    values: dict = {}
    if patch.full_name is not None:
        values["full_name"] = patch.full_name
    if patch.preferences is not None:
        values["preferences"] = patch.preferences
    if values:
        await db.execute(sa_update(User).where(User.id == user.id).values(**values))
        await db.commit()
        await db.refresh(user)
    return await get_profile(user, db)
