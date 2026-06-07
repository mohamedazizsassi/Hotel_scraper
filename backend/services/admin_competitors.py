from __future__ import annotations
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from core.exceptions import NotFoundError, BadRequestError
from schemas.admin_competitor import CompetitorRow, SelectableHotel, CompetitorSelectionUpdate

_SELECTION_SQL = text("""
    SELECT ucs.hotel_id,
           ph.hotel_name_display,
           c.name_normalized      AS city_name,
           ph.stars_int,
           ucs.display_order
    FROM user_competitor_selections ucs
    JOIN platform_hotels ph ON ph.id = ucs.hotel_id
    JOIN cities c ON c.id = ph.city_id
    WHERE ucs.user_id::text = :uid
    ORDER BY ucs.display_order
""")

_SELECTABLE_SQL = text("""
    SELECT ph.id AS hotel_id,
           ph.hotel_name_display,
           c.name_normalized AS city_name,
           ph.stars_int
    FROM platform_hotels ph
    JOIN cities c ON c.id = ph.city_id
    WHERE ph.is_active = TRUE AND ph.id <> COALESCE(:own, -1)
    ORDER BY ph.hotel_name_display
""")


async def _ensure_manager(db: AsyncSession, manager_id: str) -> None:
    ok = await db.scalar(
        text("SELECT 1 FROM users WHERE id::text = :id AND role = 'manager'"),
        {"id": str(manager_id)})
    if not ok:
        raise NotFoundError(f"Manager {manager_id} not found")


async def _own_hotel_id(db: AsyncSession, manager_id: str) -> int | None:
    return await db.scalar(
        text("SELECT hotel_id FROM user_hotel_assignments "
             "WHERE user_id::text = :id AND is_active = TRUE"),
        {"id": str(manager_id)})


async def get_selection(db: AsyncSession, manager_id: str) -> list[CompetitorRow]:
    await _ensure_manager(db, manager_id)
    rows = (await db.execute(_SELECTION_SQL, {"uid": str(manager_id)})).mappings().fetchall()
    return [CompetitorRow(**dict(r)) for r in rows]


async def get_selectable(db: AsyncSession, manager_id: str) -> list[SelectableHotel]:
    await _ensure_manager(db, manager_id)
    own = await _own_hotel_id(db, manager_id)
    rows = (await db.execute(_SELECTABLE_SQL, {"own": own})).mappings().fetchall()
    return [SelectableHotel(**dict(r)) for r in rows]


async def set_selection(db: AsyncSession, manager_id: str,
                        body: CompetitorSelectionUpdate) -> list[CompetitorRow]:
    await _ensure_manager(db, manager_id)
    row = (await db.execute(
        text("SELECT hotel_id, max_competitors FROM user_hotel_assignments "
             "WHERE user_id::text = :id AND is_active = TRUE"),
        {"id": str(manager_id)})).mappings().first()
    if row is None:
        raise BadRequestError("Manager has no active hotel assignment; assign a hotel first")
    own_hotel_id, cap = row["hotel_id"], row["max_competitors"]

    hotel_ids = body.hotel_ids
    if len(hotel_ids) != len(set(hotel_ids)):
        raise BadRequestError("Duplicate hotels in selection")
    if len(hotel_ids) > cap:
        raise BadRequestError(f"At most {cap} competitors allowed")
    if own_hotel_id in hotel_ids:
        raise BadRequestError("A manager cannot select their own hotel as a competitor")
    for hid in hotel_ids:
        active = await db.scalar(
            text("SELECT 1 FROM platform_hotels WHERE id = :h AND is_active = TRUE"),
            {"h": hid})
        if not active:
            raise BadRequestError(f"Hotel {hid} is not an active platform hotel")

    await db.execute(
        text("DELETE FROM user_competitor_selections WHERE user_id::text = :id"),
        {"id": str(manager_id)})
    for order, hid in enumerate(hotel_ids, start=1):
        await db.execute(
            text("""INSERT INTO user_competitor_selections (user_id, hotel_id, display_order)
                    VALUES (CAST(:uid AS uuid), :h, :o)"""),
            {"uid": str(manager_id), "h": hid, "o": order})
    await db.commit()
    return await get_selection(db, manager_id)
