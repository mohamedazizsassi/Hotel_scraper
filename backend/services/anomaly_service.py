# backend/services/anomaly_service.py
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
from schemas.anomaly import AnomalyRow

async def get_anomalies(
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
) -> list[AnomalyRow]:
    hotel_name = await get_manager_hotel_name(user, db)
    where, params = build_feature_query(
        hotel_name,
        check_in_from=check_in_from, check_in_to=check_in_to,
        nights=nights, adults=adults,
        room_base=room_base, room_view=room_view, room_tier=room_tier,
        room_occupancy=room_occupancy, boarding_canonical=boarding_canonical,
        scrape_date=scrape_date,
    )

    # Latest snapshot per (date, product config).
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

    feat_cols = list(ml_store.forecaster.feature_names_)
    X = df[feat_cols]
    y_log = np.log(df["price"].to_numpy(dtype=np.float64))
    scores_df = ml_store.detector.score(X, y_log)

    flagged_idx = scores_df.index[scores_df["is_anomaly"]].tolist()
    if not flagged_idx:
        return []

    out: list[AnomalyRow] = []
    for i in flagged_idx:
        row = df.iloc[i]
        sc = scores_df.iloc[i]
        out.append(AnomalyRow(
            hotel_name_normalized=str(row["hotel_name_normalized"]),
            city_name=str(row["city_name"]),
            stars_int=int(row["stars_int"]) if row["stars_int"] is not None else None,
            scraped_at=str(row["scraped_at"]),
            check_in=row["check_in"],
            nights=int(row["nights"]),
            adults=int(row["adults"]),
            boarding_canonical=str(row["boarding_canonical"]),
            price=float(row["price"]),
            price_per_night=float(row["price_per_night"]),
            q10_cal_tnd=float(np.exp(sc["q10_cal_log"])),
            q90_cal_tnd=float(np.exp(sc["q90_cal_log"])),
            anomaly_score=float(sc["anomaly_score"]),
            interval_status="below_band" if sc["anomaly_score"] < 0 else "above_band",
        ))
    return out
