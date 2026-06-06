from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_admin
from db.models import User
from schemas.admin_manager import AdminManagerRow, AdminManagerListResponse
from services.admin_managers import list_managers, get_manager

router = APIRouter(prefix="/admin/managers", tags=["admin"])


@router.get("", response_model=AdminManagerListResponse)
async def managers_list(
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await list_managers(db)
    return AdminManagerListResponse(data=rows, count=len(rows))


@router.get("/{manager_id}", response_model=AdminManagerRow)
async def managers_detail(
    manager_id: str,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    return await get_manager(db, manager_id)
