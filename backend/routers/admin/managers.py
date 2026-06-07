from fastapi import APIRouter, Depends, status, Response
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_admin
from db.models import User
from schemas.admin_manager import AdminManagerRow, AdminManagerListResponse, ManagerCreate, ManagerUpdate, PasswordReset
from services.admin_managers import list_managers, get_manager, create_manager, update_manager, reset_password

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


@router.post("", response_model=AdminManagerRow, status_code=status.HTTP_201_CREATED)
async def managers_create(
    body: ManagerCreate,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    return await create_manager(db, body)


@router.patch("/{manager_id}", response_model=AdminManagerRow)
async def managers_update(
    manager_id: str,
    body: ManagerUpdate,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    return await update_manager(db, manager_id, body)


@router.post("/{manager_id}/reset-password", status_code=status.HTTP_204_NO_CONTENT)
async def managers_reset_password(
    manager_id: str,
    body: PasswordReset,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    await reset_password(db, manager_id, body)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
