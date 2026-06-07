from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_admin, get_hotel_prices_total
from db.models import User
from schemas.admin_monitoring import MonitoringSummary, ScrapeRunListResponse, DailyResponse
from services.admin_monitoring import build_summary, list_runs, daily_rollup

router = APIRouter(prefix="/admin/monitoring", tags=["admin"])


@router.get("/summary", response_model=MonitoringSummary)
async def monitoring_summary(
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
    total_rows: int | None = Depends(get_hotel_prices_total),
):
    return await build_summary(db, total_rows)


@router.get("/runs", response_model=ScrapeRunListResponse)
async def monitoring_runs(
    limit: int = 50,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await list_runs(db, limit)
    return ScrapeRunListResponse(data=rows, count=len(rows))


@router.get("/daily", response_model=DailyResponse)
async def monitoring_daily(
    days: int = 30,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await daily_rollup(db, days)
    return DailyResponse(data=rows, count=len(rows))
