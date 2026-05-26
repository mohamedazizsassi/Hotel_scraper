# backend/services/common.py
from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from db.models import User, UserHotelAssignment, PlatformHotel
from core.exceptions import NotFoundError

async def get_manager_hotel_name(user: User, db: AsyncSession) -> str:
    """Return hotel_name_normalized for the manager's assigned hotel."""
    result = await db.execute(
        select(PlatformHotel.hotel_name_normalized)
        .join(UserHotelAssignment, UserHotelAssignment.hotel_id == PlatformHotel.id)
        .where(
            UserHotelAssignment.user_id == user.id,
            UserHotelAssignment.is_active.is_(True),
        )
    )
    name = result.scalar_one_or_none()
    if name is None:
        raise NotFoundError("Hotel assignment not found")
    return name
