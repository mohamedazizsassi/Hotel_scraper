# backend/services/admin_assignments.py
from __future__ import annotations
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from core.exceptions import NotFoundError, ConflictError
from schemas.admin_assignment import AdminAssignmentRow, AssignmentCreate, AssignmentUpdate

_LIST_SQL = text("""
    SELECT uha.id,
           uha.user_id::text          AS user_id,
           u.email                    AS manager_email,
           u.full_name                AS manager_name,
           uha.hotel_id,
           ph.hotel_name_display      AS hotel_name,
           uha.max_competitors,
           uha.is_active
    FROM user_hotel_assignments uha
    JOIN users u ON u.id = uha.user_id
    JOIN platform_hotels ph ON ph.id = uha.hotel_id
    ORDER BY u.full_name NULLS LAST, u.email
""")


async def list_assignments(db: AsyncSession) -> list[AdminAssignmentRow]:
    rows = (await db.execute(_LIST_SQL)).mappings().fetchall()
    return [AdminAssignmentRow(**dict(r)) for r in rows]


async def _get_row(db: AsyncSession, assignment_id: int) -> AdminAssignmentRow:
    rows = (await db.execute(_LIST_SQL)).mappings().fetchall()
    m = next((r for r in rows if r["id"] == assignment_id), None)
    if m is None:
        raise NotFoundError(f"Assignment {assignment_id} not found")
    return AdminAssignmentRow(**dict(m))


async def update_assignment(db: AsyncSession, assignment_id: int,
                            body: AssignmentUpdate) -> AdminAssignmentRow:
    exists = await db.scalar(
        text("SELECT 1 FROM user_hotel_assignments WHERE id = :id"), {"id": assignment_id})
    if not exists:
        raise NotFoundError(f"Assignment {assignment_id} not found")
    fields = body.model_dump(exclude_unset=True)
    allowed = {"hotel_id", "max_competitors", "is_active"}
    sets = ", ".join(f"{k} = :{k}" for k in fields if k in allowed)
    if sets:
        params = {k: v for k, v in fields.items() if k in allowed}
        params["id"] = assignment_id
        await db.execute(
            text(f"UPDATE user_hotel_assignments SET {sets} WHERE id = :id"), params)
        await db.commit()
    return await _get_row(db, assignment_id)


async def delete_assignment(db: AsyncSession, assignment_id: int) -> None:
    exists = await db.scalar(
        text("SELECT 1 FROM user_hotel_assignments WHERE id = :id"), {"id": assignment_id})
    if not exists:
        raise NotFoundError(f"Assignment {assignment_id} not found")
    await db.execute(
        text("DELETE FROM user_hotel_assignments WHERE id = :id"), {"id": assignment_id})
    await db.commit()


async def create_assignment(db: AsyncSession, body: AssignmentCreate) -> AdminAssignmentRow:
    mgr = await db.scalar(
        text("SELECT 1 FROM users WHERE id::text = :id AND role = 'manager' AND is_active = TRUE"),
        {"id": body.user_id})
    if not mgr:
        raise NotFoundError(f"Active manager {body.user_id} not found")
    hotel = await db.scalar(text("SELECT 1 FROM platform_hotels WHERE id = :h"),
                            {"h": body.hotel_id})
    if not hotel:
        raise NotFoundError(f"Hotel {body.hotel_id} not found")
    dup = await db.scalar(
        text("SELECT 1 FROM user_hotel_assignments WHERE user_id::text = :id"),
        {"id": body.user_id})
    if dup:
        raise ConflictError(f"Manager {body.user_id} already has an assignment")
    new_id = await db.scalar(
        text("""INSERT INTO user_hotel_assignments (user_id, hotel_id, max_competitors, is_active)
                VALUES (CAST(:uid AS uuid), :h, :mc, TRUE) RETURNING id"""),
        {"uid": body.user_id, "h": body.hotel_id, "mc": body.max_competitors})
    await db.commit()
    return await _get_row(db, new_id)
