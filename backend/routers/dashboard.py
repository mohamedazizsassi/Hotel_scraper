# backend/routers/dashboard.py
from __future__ import annotations
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_manager, get_ml_store
from db.models import User
from ml_store.store import MLStore
from schemas.dashboard import DashboardResponse
from services.dashboard_service import get_dashboard

router = APIRouter(prefix="/manager", tags=["manager"])

@router.get("/dashboard", response_model=DashboardResponse)
async def dashboard_endpoint(
    days: int = 30,
    user: User = Depends(get_current_manager),
    db: AsyncSession = Depends(get_db),
    ml_store: MLStore = Depends(get_ml_store),
):
    return await get_dashboard(user, db, ml_store, days=days)
