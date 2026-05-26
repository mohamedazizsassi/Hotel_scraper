# backend/services/recommendation_service.py
from __future__ import annotations
from datetime import date
from typing import Optional
import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from db.models import User
from services.common import get_manager_hotel_name
from ml_store.store import MLStore, prepare_serve_frame
from schemas.recommendation import RecommendationRow

async def get_recommendations(
    user: User,
    db: AsyncSession,
    ml_store: MLStore,
    check_in_from: Optional[date] = None,
    check_in_to: Optional[date] = None,
    nights: Optional[int] = None,
) -> list[RecommendationRow]:
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

    sql = text(f"SELECT * FROM hotel_features_full WHERE {' AND '.join(conditions)}")
    result = await db.execute(sql, params)
    rows = result.fetchall()
    if not rows:
        return []

    df = pd.DataFrame(rows, columns=list(result.keys()))
    df = prepare_serve_frame(df)

    result_df = ml_store.recommender.score(df, test_indices=np.arange(len(df)))
    result_df["scraped_at"] = result_df["scraped_at"].astype(str)

    out: list[RecommendationRow] = []
    for rec in result_df.to_dict(orient="records"):
        rec["reasons"] = list(rec.get("reasons", []))
        out.append(RecommendationRow(**rec))
    return out
