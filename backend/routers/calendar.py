# backend/routers/calendar.py
from __future__ import annotations
from datetime import date
from typing import Optional
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_manager
from db.models import User
from schemas.calendar import CalendarResponse
from services.calendar_service import get_calendar

router = APIRouter(prefix="/manager", tags=["manager"])

@router.get("/calendar", response_model=CalendarResponse)
async def calendar_endpoint(
    check_in_from: Optional[date] = None,
    check_in_to: Optional[date] = None,
    nights: Optional[int] = None,
    user: User = Depends(get_current_manager),
    db: AsyncSession = Depends(get_db),
):
    rows = await get_calendar(user, db, check_in_from, check_in_to, nights)
    return CalendarResponse(data=rows, count=len(rows))
