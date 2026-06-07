from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_admin
from db.models import User
from schemas.admin_alert import AlertListResponse
from services.admin_alerts import list_alerts

router = APIRouter(prefix="/admin/alerts", tags=["admin"])


@router.get("", response_model=AlertListResponse)
async def alerts_list(
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await list_alerts(db)
    return AlertListResponse(data=rows, count=len(rows))
