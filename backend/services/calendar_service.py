# backend/services/calendar_service.py
from __future__ import annotations
from datetime import date
from typing import Optional
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from db.models import User
from services.common import get_manager_hotel_name, build_feature_query
from ml_store.store import MLStore, prepare_serve_frame
from schemas.calendar import CalendarRow, CalendarOptions, CalendarConfig


async def get_calendar_options(user: User, db: AsyncSession) -> CalendarOptions:
    """Distinct filter values + a default product config for the hotel, so the
    frontend never offers a combination that returns zero rows."""
    hotel_name = await get_manager_hotel_name(user, db)
    p = {"hotel_name": hotel_name}

    agg = (await db.execute(text("""
        SELECT
            array_agg(DISTINCT room_base)          AS room_base,
            array_agg(DISTINCT room_view)          AS room_view,
            array_agg(DISTINCT room_tier)          AS room_tier,
            array_agg(DISTINCT room_occupancy)     AS room_occupancy,
            array_agg(DISTINCT boarding_canonical) AS boarding_canonical,
            array_agg(DISTINCT nights)             AS nights,
            array_agg(DISTINCT adults)             AS adults,
            MIN(check_in) AS check_in_min,
            MAX(check_in) AS check_in_max
        FROM hotel_features_full
        WHERE hotel_name_normalized = :hotel_name
    """), p)).mappings().first()

    scrape_dates = [
        str(r[0]) for r in (await db.execute(text("""
            SELECT DISTINCT scrape_date FROM hotel_features_full
            WHERE hotel_name_normalized = :hotel_name
            ORDER BY scrape_date DESC
        """), p)).fetchall()
    ]

    top = (await db.execute(text("""
        SELECT room_base, room_view, room_tier, room_occupancy,
               boarding_canonical, nights, adults
        FROM hotel_features_full
        WHERE hotel_name_normalized = :hotel_name
        GROUP BY 1,2,3,4,5,6,7
        ORDER BY COUNT(*) DESC
        LIMIT 1
    """), p)).mappings().first()

    def clean_str(values) -> list[str]:
        return sorted({("" if v is None else str(v)) for v in (values or [])})

    def clean_int(values) -> list[int]:
        return sorted({int(v) for v in (values or []) if v is not None})

    if agg is None:
        agg = {}
    return CalendarOptions(
        room_base=clean_str(agg.get("room_base")),
        room_view=clean_str(agg.get("room_view")),
        room_tier=clean_str(agg.get("room_tier")),
        room_occupancy=clean_str(agg.get("room_occupancy")),
        boarding_canonical=clean_str(agg.get("boarding_canonical")),
        nights=clean_int(agg.get("nights")),
        adults=clean_int(agg.get("adults")),
        scrape_dates=scrape_dates,
        check_in_min=agg.get("check_in_min"),
        check_in_max=agg.get("check_in_max"),
        default=CalendarConfig(
            room_base=("" if top["room_base"] is None else str(top["room_base"])),
            room_view=("" if top["room_view"] is None else str(top["room_view"])),
            room_tier=("" if top["room_tier"] is None else str(top["room_tier"])),
            room_occupancy=("" if top["room_occupancy"] is None else str(top["room_occupancy"])),
            boarding_canonical=("" if top["boarding_canonical"] is None else str(top["boarding_canonical"])),
            nights=top["nights"],
            adults=top["adults"],
        ) if top else CalendarConfig(
            room_base=None, room_view=None, room_tier=None, room_occupancy=None,
            boarding_canonical=None, nights=None, adults=None,
        ),
    )


def _confidence_from_interval(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray) -> np.ndarray:
    """UI proxy: narrower calibrated 80% interval -> higher confidence.

    Not a calibrated probability — it maps the relative interval width
    (q90_cal - q10_cal) / q50 to a [0.05, 0.99] band for display only.
    """
    q50_safe = np.where(np.abs(q50) < 1e-9, 1e-9, q50)
    rel_width = (q90 - q10) / q50_safe
    conf = 1.0 - rel_width / 2.0
    return np.clip(conf, 0.05, 0.99)


def _b(v: object) -> bool:
    return bool(v) if v is not None else False


async def get_calendar(
    user: User,
    db: AsyncSession,
    ml_store: MLStore,
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
) -> list[CalendarRow]:
    hotel_name = await get_manager_hotel_name(user, db)
    where, params = build_feature_query(
        hotel_name,
        check_in_from=check_in_from, check_in_to=check_in_to,
        nights=nights, adults=adults,
        room_base=room_base, room_view=room_view, room_tier=room_tier,
        room_occupancy=room_occupancy, boarding_canonical=boarding_canonical,
        best_peer_granularity_used=best_peer_granularity_used, scrape_date=scrape_date,
    )
    # One row per check_in: the freshest snapshot matching the product config.
    sql = text(f"""
        SELECT DISTINCT ON (check_in) *
        FROM hotel_features_full
        WHERE {where}
        ORDER BY check_in, scraped_at DESC
    """)
    result = await db.execute(sql, params)
    rows = result.fetchall()
    if not rows:
        return []

    df = pd.DataFrame(rows, columns=list(result.keys())).reset_index(drop=True)
    df = prepare_serve_frame(df)

    # Recommender provides recommended_price_tnd + calibrated quantiles (TND).
    rec = ml_store.recommender.score(df, test_indices=np.arange(len(df)))
    nights_arr = df["nights"].to_numpy(dtype=float)
    rec_ppn = rec["recommended_price_tnd"].to_numpy(dtype=float) / np.where(nights_arr == 0, 1, nights_arr)
    confidence = _confidence_from_interval(
        rec["q10_cal_tnd"].to_numpy(dtype=float),
        rec["q50_tnd"].to_numpy(dtype=float),
        rec["q90_cal_tnd"].to_numpy(dtype=float),
    )

    out: list[CalendarRow] = []
    for i in range(len(df)):
        r = df.iloc[i]
        pm = r["peer_medium_median"]
        pc = r["peer_medium_count"]
        sur = r.get("sur_demande_rate_city_stars_checkin")
        out.append(CalendarRow(
            hotel_name_normalized=str(r["hotel_name_normalized"]),
            city_name=str(r["city_name"]),
            stars_int=int(r["stars_int"]) if pd.notna(r["stars_int"]) else None,
            check_in=r["check_in"],
            scrape_date=str(r["scrape_date"]),
            scraped_at=str(r["scraped_at"]),
            nights=int(r["nights"]),
            adults=int(r["adults"]),
            children=int(r["children"]) if pd.notna(r["children"]) else 0,
            boarding_canonical=str(r["boarding_canonical"]),
            room_base=str(r["room_base"]),
            room_view=str(r["room_view"]),
            room_tier=str(r["room_tier"]),
            room_occupancy=str(r["room_occupancy"]),
            is_supplement_variant=_b(r["is_supplement_variant"]),
            has_free_view_upgrade=_b(r["has_free_view_upgrade"]),
            price_per_night=float(r["price_per_night"]),
            peer_medium_median=float(pm) if pd.notna(pm) else None,
            peer_medium_count=int(pc) if pd.notna(pc) else None,
            best_peer_granularity_used=str(r["best_peer_granularity_used"]) if pd.notna(r["best_peer_granularity_used"]) else None,
            recommended_price_per_night=float(rec_ppn[i]),
            forecaster_confidence=float(confidence[i]),
            sur_demande_rate_city_stars_checkin=float(sur) if sur is not None and pd.notna(sur) else None,
            days_until_checkin=int(r["days_until_checkin"]) if pd.notna(r["days_until_checkin"]) else 0,
            is_weekend_checkin=_b(r["is_weekend_checkin"]),
            is_ramadan=_b(r["is_ramadan"]),
            is_tunisia_public_holiday=_b(r["is_tunisia_public_holiday"]),
            is_tunisia_school_holiday=_b(r["is_tunisia_school_holiday"]),
            is_school_holiday_france=_b(r["is_school_holiday_france"]),
            is_school_holiday_germany=_b(r["is_school_holiday_germany"]),
            is_school_holiday_uk=_b(r["is_school_holiday_uk"]),
        ))
    return out
