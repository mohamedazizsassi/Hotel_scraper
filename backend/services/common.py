# backend/services/common.py
from __future__ import annotations
from datetime import date, datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from db.models import User, UserHotelAssignment, PlatformHotel
from core.exceptions import NotFoundError


def build_feature_query(
    hotel_name: str,
    *,
    check_in_from: Optional[date] = None,
    check_in_to: Optional[date] = None,
    nights: Optional[int] = None,
    adults: Optional[int] = None,
    room_base: Optional[str] = None,
    room_view: Optional[str] = None,
    room_tier: Optional[str] = None,
    room_occupancy: Optional[str] = None,
    boarding_canonical: Optional[str] = None,
    best_peer_granularity_used: Optional[str] = None,
    scrape_date: Optional[str] = None,
) -> tuple[str, dict]:
    """Build a WHERE clause + bind params for hotel_features[_full] queries.

    Shared by the calendar, recommendation, and anomaly services so all three
    scope to the same product configuration. Taxonomy columns use
    COALESCE(col,'') so the frontend's empty-string sentinel matches both NULL
    and '' rows (some hotels have no tier/view/occupancy). None / "any" mean
    "do not constrain".
    """
    conditions = ["hotel_name_normalized = :hotel_name"]
    params: dict = {"hotel_name": hotel_name}

    if check_in_from:
        conditions.append("check_in >= :check_in_from"); params["check_in_from"] = check_in_from
    if check_in_to:
        conditions.append("check_in <= :check_in_to"); params["check_in_to"] = check_in_to

    def add_num(col: str, value):
        if value is not None and value != "any":
            conditions.append(f"{col} = :{col}"); params[col] = value

    def add_str(col: str, value):
        if value is not None and value != "any":
            conditions.append(f"COALESCE({col}, '') = :{col}"); params[col] = value

    add_num("nights", nights)
    add_num("adults", adults)
    add_str("room_base", room_base)
    add_str("room_view", room_view)
    add_str("room_tier", room_tier)
    add_str("room_occupancy", room_occupancy)
    add_str("boarding_canonical", boarding_canonical)
    add_str("best_peer_granularity_used", best_peer_granularity_used)
    if scrape_date:
        # scrape_date is a timestamp column; asyncpg rejects raw strings, so
        # parse the frontend value ("2026-05-18" or "2026-05-18 00:00:00") to a
        # datetime before binding.
        sd = datetime.fromisoformat(scrape_date) if isinstance(scrape_date, str) else scrape_date
        conditions.append("scrape_date = :scrape_date"); params["scrape_date"] = sd

    return " AND ".join(conditions), params

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
