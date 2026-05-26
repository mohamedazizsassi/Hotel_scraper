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
    user: User = Depends(get_current_manager),
    db: AsyncSession = Depends(get_db),
    ml_store: MLStore = Depends(get_ml_store),
):
    rows = await get_anomalies(user, db, ml_store, check_in_from, check_in_to, nights)
    return AnomalyResponse(data=rows, count=len(rows))
