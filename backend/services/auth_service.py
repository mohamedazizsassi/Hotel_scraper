from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from sqlalchemy.sql import func
from db.models import User, UserHotelAssignment
from core.security import verify_password, create_access_token
from core.exceptions import AuthError

async def login(email: str, password: str, db: AsyncSession) -> str:
    result = await db.execute(
        select(User).where(User.email == email, User.is_active.is_(True))
    )
    user = result.scalar_one_or_none()
    if user is None or not verify_password(password, user.password_hash):
        raise AuthError("Invalid email or password")

    assign = await db.execute(
        select(UserHotelAssignment).where(
            UserHotelAssignment.user_id == user.id,
            UserHotelAssignment.is_active.is_(True),
        )
    )
    row = assign.scalar_one_or_none()
    hotel_id = row.hotel_id if row else None

    await db.execute(
        update(User).where(User.id == user.id).values(last_login_at=func.now())
    )
    await db.commit()

    return create_access_token(str(user.id), hotel_id=hotel_id, role=user.role)
