# backend/routers/admin/assignments.py
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_admin
from db.models import User
from schemas.admin_assignment import (
    AdminAssignmentRow, AdminAssignmentListResponse, AssignmentCreate,
)
from services.admin_assignments import list_assignments, create_assignment

router = APIRouter(prefix="/admin/assignments", tags=["admin"])


@router.get("", response_model=AdminAssignmentListResponse)
async def assignments_list(
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await list_assignments(db)
    return AdminAssignmentListResponse(data=rows, count=len(rows))


@router.post("", response_model=AdminAssignmentRow, status_code=status.HTTP_201_CREATED)
async def assignments_create(
    body: AssignmentCreate,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    return await create_assignment(db, body)
