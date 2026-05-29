# backend/routers/calendar.py
from __future__ import annotations
from datetime import date
from typing import Optional
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_manager, get_ml_store
from db.models import User
from ml_store.store import MLStore
from schemas.calendar import CalendarResponse, CalendarOptions
from services.calendar_service import get_calendar, get_calendar_options

router = APIRouter(prefix="/manager", tags=["manager"])

@router.get("/calendar/options", response_model=CalendarOptions)
async def calendar_options_endpoint(
    user: User = Depends(get_current_manager),
    db: AsyncSession = Depends(get_db),
):
    return await get_calendar_options(user, db)

@router.get("/calendar", response_model=CalendarResponse)
async def calendar_endpoint(
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
    user: User = Depends(get_current_manager),
    db: AsyncSession = Depends(get_db),
    ml_store: MLStore = Depends(get_ml_store),
):
    rows = await get_calendar(
        user, db, ml_store,
        check_in_from=check_in_from, check_in_to=check_in_to,
        nights=nights, adults=adults,
        room_base=room_base, room_view=room_view, room_tier=room_tier,
        room_occupancy=room_occupancy, boarding_canonical=boarding_canonical,
        best_peer_granularity_used=best_peer_granularity_used,
        scrape_date=scrape_date,
    )
    return CalendarResponse(data=rows, count=len(rows))
