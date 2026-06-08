# backend/routers/recommendation_decisions.py
from __future__ import annotations
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_manager
from db.models import User
from schemas.decision import DecisionIn, DecisionBulkIn, DecisionRow
from services.decision_service import set_decision, set_decisions_bulk

router = APIRouter(prefix="/manager", tags=["manager"])

@router.post("/recommendations/decision", response_model=DecisionRow)
async def decision_endpoint(
    body: DecisionIn,
    user: User = Depends(get_current_manager),
    db: AsyncSession = Depends(get_db),
):
    return await set_decision(user, body, db)

@router.post("/recommendations/decision/bulk")
async def decision_bulk_endpoint(
    body: DecisionBulkIn,
    user: User = Depends(get_current_manager),
    db: AsyncSession = Depends(get_db),
):
    count = await set_decisions_bulk(user, body, db)
    return {"count": count}
