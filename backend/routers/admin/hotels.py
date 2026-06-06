from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_admin
from db.models import User
from schemas.admin_hotel import AdminHotelListResponse, AdminHotelRow, DiscoverableResponse, HotelCreate, HotelUpdate
from services.admin_hotels import list_hotels, list_discoverable, create_hotel, get_hotel, update_hotel

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


@router.post("", response_model=AdminHotelRow, status_code=status.HTTP_201_CREATED)
async def hotels_create(
    body: HotelCreate,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    return await create_hotel(db, body)


@router.get("/{hotel_id}", response_model=AdminHotelRow)
async def hotels_detail(
    hotel_id: int,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    return await get_hotel(db, hotel_id)


@router.patch("/{hotel_id}", response_model=AdminHotelRow)
async def hotels_update(
    hotel_id: int,
    body: HotelUpdate,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    return await update_hotel(db, hotel_id, body)
