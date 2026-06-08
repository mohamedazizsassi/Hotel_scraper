# backend/services/recommendation_service.py
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
from schemas.recommendation import RecommendationRow

async def get_recommendations(
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
    scrape_date: Optional[str] = None,
) -> list[RecommendationRow]:
    hotel_name = await get_manager_hotel_name(user, db)
    where, params = build_feature_query(
        hotel_name,
        check_in_from=check_in_from, check_in_to=check_in_to,
        nights=nights, adults=adults,
        room_base=room_base, room_view=room_view, room_tier=room_tier,
        room_occupancy=room_occupancy, boarding_canonical=boarding_canonical,
        scrape_date=scrape_date,
    )

    # Latest snapshot per (date, product config) — avoids scoring ~28 daily
    # snapshots per check-in through the recommender's per-row loop.
    key = ("check_in, nights, adults, room_base, room_view, room_tier, "
           "room_occupancy, boarding_canonical")
    sql = text(f"""
        SELECT DISTINCT ON ({key}) *
        FROM hotel_features_full
        WHERE {where}
        ORDER BY {key}, scraped_at DESC
    """)
    result = await db.execute(sql, params)
    rows = result.fetchall()
    if not rows:
        return []

    df = pd.DataFrame(rows, columns=list(result.keys())).reset_index(drop=True)
    df = prepare_serve_frame(df)

    result_df = ml_store.recommender.score(df, test_indices=np.arange(len(df)))
    result_df["scraped_at"] = result_df["scraped_at"].astype(str)

    out: list[RecommendationRow] = []
    for rec in result_df.to_dict(orient="records"):
        rec["reasons"] = list(rec.get("reasons", []))
        out.append(RecommendationRow(**rec))

    from services.decision_service import get_decision_map
    decisions = await get_decision_map(user, db)
    for row in out:
        row.decision_status = decisions.get(
            (row.check_in, row.nights, row.adults, row.boarding_canonical)
        )
    return out
