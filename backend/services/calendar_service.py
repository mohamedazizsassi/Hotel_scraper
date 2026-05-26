# backend/services/calendar_service.py
from __future__ import annotations
from datetime import date
from typing import Optional
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from db.models import User
from services.common import get_manager_hotel_name
from schemas.calendar import CalendarRow

async def get_calendar(
    user: User,
    db: AsyncSession,
    check_in_from: Optional[date] = None,
    check_in_to: Optional[date] = None,
    nights: Optional[int] = None,
) -> list[CalendarRow]:
    hotel_name = await get_manager_hotel_name(user, db)

    conditions = ["hotel_name_normalized = :hotel_name"]
    params: dict = {"hotel_name": hotel_name}
    if check_in_from:
        conditions.append("check_in >= :check_in_from")
        params["check_in_from"] = check_in_from
    if check_in_to:
        conditions.append("check_in <= :check_in_to")
        params["check_in_to"] = check_in_to
    if nights is not None:
        conditions.append("nights = :nights")
        params["nights"] = nights

    where = " AND ".join(conditions)
    sql = text(f"""
        SELECT hotel_name_normalized, city_name, stars_int,
               check_in, nights, adults, boarding_canonical,
               price, price_per_night, scraped_at::text,
               peer_medium_median, peer_medium_count
        FROM hotel_features
        WHERE {where}
        ORDER BY check_in, nights
    """)
    result = await db.execute(sql, params)
    rows = result.fetchall()
    if not rows:
        return []
    df = pd.DataFrame(rows, columns=list(result.keys()))
    return [CalendarRow(**r) for r in df.to_dict(orient="records")]
