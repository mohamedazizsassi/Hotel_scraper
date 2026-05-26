# backend/routers/competitors.py
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_manager
from db.models import User
from schemas.competitor import CompetitorResponse
from services.competitor_service import get_competitors

router = APIRouter(prefix="/manager", tags=["manager"])

@router.get("/competitors", response_model=CompetitorResponse)
async def competitors_endpoint(
    user: User = Depends(get_current_manager),
    db: AsyncSession = Depends(get_db),
):
    rows = await get_competitors(user, db)
    return CompetitorResponse(data=rows, count=len(rows))
