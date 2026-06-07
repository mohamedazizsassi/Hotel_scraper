from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_admin
from db.models import User
from schemas.admin_competitor import CompetitorSelectionResponse, SelectableResponse
from services.admin_competitors import get_selection, get_selectable

router = APIRouter(prefix="/admin/managers", tags=["admin"])


@router.get("/{manager_id}/competitors", response_model=CompetitorSelectionResponse)
async def competitors_get(
    manager_id: str,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await get_selection(db, manager_id)
    return CompetitorSelectionResponse(data=rows, count=len(rows))


@router.get("/{manager_id}/selectable-competitors", response_model=SelectableResponse)
async def competitors_selectable(
    manager_id: str,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await get_selectable(db, manager_id)
    return SelectableResponse(data=rows, count=len(rows))
