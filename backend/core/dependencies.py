# backend/core/dependencies.py
from __future__ import annotations
from typing import AsyncGenerator
import uuid
import jwt
from fastapi import Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from db.session import AsyncSessionLocal
from db.models import User, UserHotelAssignment
from ml_store.store import MLStore
from core.security import decode_access_token
from core.exceptions import AuthError, ForbiddenError, MLStoreNotReadyError

_ml_store: MLStore | None = None

def set_ml_store(store: MLStore) -> None:
    global _ml_store
    _ml_store = store

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

async def get_ml_store() -> MLStore:
    if _ml_store is None:
        raise MLStoreNotReadyError()
    return _ml_store

async def get_current_user(
    authorization: str = Header(...),
    db: AsyncSession = Depends(get_db),
) -> User:
    if not authorization.startswith("Bearer "):
        raise AuthError("Missing or malformed Authorization header")
    token = authorization.removeprefix("Bearer ")
    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        raise AuthError("Token expired")
    except jwt.InvalidTokenError:
        raise AuthError("Invalid token")

    user_id = payload.get("sub")
    try:
        user_id = uuid.UUID(str(user_id))
    except (ValueError, AttributeError):
        raise AuthError("Invalid token subject")
    result = await db.execute(
        select(User).where(User.id == user_id, User.is_active.is_(True))
    )
    user = result.scalar_one_or_none()
    if user is None:
        raise AuthError("User not found or inactive")
    return user

async def get_current_manager(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> User:
    if user.role != "manager":
        raise ForbiddenError("Manager role required")
    result = await db.execute(
        select(UserHotelAssignment).where(
            UserHotelAssignment.user_id == user.id,
            UserHotelAssignment.is_active.is_(True),
        )
    )
    assignment = result.scalar_one_or_none()
    if assignment is None:
        raise ForbiddenError("No active hotel assignment for this manager")
    return user
