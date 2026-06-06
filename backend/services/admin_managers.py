from __future__ import annotations
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from core.exceptions import NotFoundError
from schemas.admin_manager import AdminManagerRow

_LIST_SQL = text("""
    SELECT u.id::text                     AS id,
           u.email,
           u.full_name,
           u.is_active,
           u.last_login_at::text          AS last_login_at,
           uha.hotel_id                   AS assigned_hotel_id,
           ph.hotel_name_display          AS assigned_hotel_name
    FROM users u
    LEFT JOIN user_hotel_assignments uha ON uha.user_id = u.id AND uha.is_active = TRUE
    LEFT JOIN platform_hotels ph ON ph.id = uha.hotel_id
    WHERE u.role = 'manager'
    ORDER BY u.full_name NULLS LAST, u.email
""")


async def list_managers(db: AsyncSession) -> list[AdminManagerRow]:
    rows = (await db.execute(_LIST_SQL)).mappings().fetchall()
    return [AdminManagerRow(**dict(r)) for r in rows]


async def get_manager(db: AsyncSession, manager_id: str) -> AdminManagerRow:
    rows = (await db.execute(_LIST_SQL)).mappings().fetchall()
    m = next((r for r in rows if r["id"] == str(manager_id)), None)
    if m is None:
        raise NotFoundError(f"Manager {manager_id} not found")
    return AdminManagerRow(**dict(m))
