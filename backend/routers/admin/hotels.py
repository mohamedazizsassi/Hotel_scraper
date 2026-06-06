from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_admin
from db.models import User
from schemas.admin_hotel import AdminHotelListResponse, DiscoverableResponse
from services.admin_hotels import list_hotels, list_discoverable

router = APIRouter(prefix="/admin/hotels", tags=["admin"])


@router.get("", response_model=AdminHotelListResponse)
async def hotels_list(
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await list_hotels(db)
    return AdminHotelListResponse(data=rows, count=len(rows))


@router.get("/discoverable", response_model=DiscoverableResponse)
async def hotels_discoverable(
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await list_discoverable(db)
    return DiscoverableResponse(data=rows, count=len(rows))
