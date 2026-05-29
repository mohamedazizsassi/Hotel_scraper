# backend/routers/anomalies.py
from __future__ import annotations
from datetime import date
from typing import Optional
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_manager, get_ml_store
from db.models import User
from ml_store.store import MLStore
from schemas.anomaly import AnomalyResponse
from services.anomaly_service import get_anomalies

router = APIRouter(prefix="/manager", tags=["manager"])

@router.get("/anomalies", response_model=AnomalyResponse)
async def anomalies_endpoint(
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
    user: User = Depends(get_current_manager),
    db: AsyncSession = Depends(get_db),
    ml_store: MLStore = Depends(get_ml_store),
):
    rows = await get_anomalies(
        user, db, ml_store,
        check_in_from=check_in_from, check_in_to=check_in_to,
        nights=nights, adults=adults,
        room_base=room_base, room_view=room_view, room_tier=room_tier,
        room_occupancy=room_occupancy, boarding_canonical=boarding_canonical,
        scrape_date=scrape_date,
    )
    return AnomalyResponse(data=rows, count=len(rows))
