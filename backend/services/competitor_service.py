# backend/services/competitor_service.py
from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from db.models import User
from schemas.competitor import CompetitorSummary

async def get_competitors(user: User, db: AsyncSession) -> list[CompetitorSummary]:
    sql = text("""
        SELECT
            ph.hotel_name_normalized,
            ph.hotel_name_display,
            c.name_normalized        AS city_name,
            ph.stars_int,
            ucs.display_order,
            AVG(hf.price_per_night)  AS avg_price_per_night,
            MAX(hf.scraped_at)::text AS latest_scraped_at
        FROM user_competitor_selections ucs
        JOIN platform_hotels ph ON ph.id = ucs.hotel_id
        JOIN cities c           ON c.id  = ph.city_id
        LEFT JOIN hotel_features hf
               ON hf.hotel_name_normalized = ph.hotel_name_normalized
        WHERE ucs.user_id = :user_id
        GROUP BY ph.hotel_name_normalized, ph.hotel_name_display,
                 c.name_normalized, ph.stars_int, ucs.display_order
        ORDER BY ucs.display_order
    """)
    result = await db.execute(sql, {"user_id": str(user.id)})
    rows = result.mappings().fetchall()
    return [CompetitorSummary(**dict(r)) for r in rows]
