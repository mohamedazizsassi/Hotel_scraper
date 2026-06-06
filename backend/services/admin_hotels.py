from __future__ import annotations
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from schemas.admin_hotel import AdminHotelRow, DiscoverableHotel

_LIST_SQL = text("""
    SELECT ph.id,
           ph.hotel_name_normalized,
           ph.hotel_name_display,
           c.name_normalized                              AS city_name,
           ph.stars_int,
           ph.is_active,
           sd.macro_region                                AS region,
           ph.contact_email,
           ph.contact_phone,
           COALESCE(string_agg(DISTINCT phs.source, ',' ORDER BY phs.source), '') AS sources,
           u.id::text                                     AS manager_id,
           u.full_name                                    AS manager_name,
           MAX(hf.scraped_at)                             AS latest_scraped_at
    FROM platform_hotels ph
    JOIN cities c ON c.id = ph.city_id
    LEFT JOIN segment_dim sd
           ON sd.city_name = c.name_normalized AND sd.stars_int = ph.stars_int
    LEFT JOIN user_hotel_assignments uha
           ON uha.hotel_id = ph.id AND uha.is_active = TRUE
    LEFT JOIN users u ON u.id = uha.user_id
    LEFT JOIN platform_hotel_sources phs ON phs.platform_hotel_id = ph.id
    LEFT JOIN hotel_features hf ON hf.hotel_name_normalized = ph.hotel_name_normalized
    GROUP BY ph.id, c.name_normalized, sd.macro_region, u.id, u.full_name
    ORDER BY ph.hotel_name_display
""")

_DISCOVERABLE_SQL = text("""
    SELECT DISTINCT hf.hotel_name_normalized, hf.city_name, hf.stars_int
    FROM hotel_features hf
    WHERE NOT EXISTS (
        SELECT 1 FROM platform_hotels ph
        JOIN cities c ON c.id = ph.city_id
        WHERE ph.hotel_name_normalized = hf.hotel_name_normalized
          AND c.name_normalized = hf.city_name
    )
    ORDER BY hf.city_name, hf.hotel_name_normalized
""")


async def list_hotels(db: AsyncSession) -> list[AdminHotelRow]:
    rows = (await db.execute(_LIST_SQL)).mappings().fetchall()
    return [AdminHotelRow(**dict(r)) for r in rows]


async def list_discoverable(db: AsyncSession) -> list[DiscoverableHotel]:
    rows = (await db.execute(_DISCOVERABLE_SQL)).mappings().fetchall()
    return [DiscoverableHotel(**dict(r)) for r in rows]
