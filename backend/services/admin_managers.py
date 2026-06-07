from __future__ import annotations
import uuid
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from core.exceptions import NotFoundError, ConflictError
from core.security import hash_password
from schemas.admin_manager import AdminManagerRow, ManagerCreate, ManagerUpdate, PasswordReset

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


async def create_manager(db: AsyncSession, body: ManagerCreate) -> AdminManagerRow:
    dup = await db.scalar(text("SELECT 1 FROM users WHERE email = :e"), {"e": str(body.email)})
    if dup:
        raise ConflictError(f"Email '{body.email}' already in use")
    new_id = str(uuid.uuid4())
    await db.execute(
        text("""INSERT INTO users (id, email, password_hash, full_name, role, is_active)
                VALUES (:id, :e, :ph, :fn, 'manager', TRUE)"""),
        {"id": new_id, "e": str(body.email), "ph": hash_password(body.initial_password), "fn": body.full_name})
    await db.commit()
    return await get_manager(db, new_id)


async def update_manager(db: AsyncSession, manager_id: str, body: ManagerUpdate) -> AdminManagerRow:
    exists = await db.scalar(
        text("SELECT 1 FROM users WHERE id::text = :id AND role = 'manager'"),
        {"id": str(manager_id)})
    if not exists:
        raise NotFoundError(f"Manager {manager_id} not found")
    fields = body.model_dump(exclude_unset=True)
    if "email" in fields and fields["email"] is not None:
        dup = await db.scalar(
            text("SELECT 1 FROM users WHERE email = :e AND id::text <> :id"),
            {"e": str(fields["email"]), "id": str(manager_id)})
        if dup:
            raise ConflictError(f"Email '{fields['email']}' already in use")
    allowed = {"email", "full_name", "is_active"}
    sets = ", ".join(f"{k} = :{k}" for k in fields if k in allowed)
    if sets:
        params = {k: (str(v) if k == "email" else v) for k, v in fields.items() if k in allowed}
        params["id"] = str(manager_id)
        await db.execute(text(f"UPDATE users SET {sets} WHERE id::text = :id"), params)
        await db.commit()
    return await get_manager(db, manager_id)


async def reset_password(db: AsyncSession, manager_id: str, body: PasswordReset) -> None:
    exists = await db.scalar(
        text("SELECT 1 FROM users WHERE id::text = :id AND role = 'manager'"),
        {"id": str(manager_id)})
    if not exists:
        raise NotFoundError(f"Manager {manager_id} not found")
    await db.execute(
        text("UPDATE users SET password_hash = :ph WHERE id::text = :id"),
        {"ph": hash_password(body.new_password), "id": str(manager_id)})
    await db.commit()
