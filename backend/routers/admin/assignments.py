# backend/routers/admin/assignments.py
from fastapi import APIRouter, Depends, status, Response
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_admin
from db.models import User
from schemas.admin_assignment import (
    AdminAssignmentRow, AdminAssignmentListResponse, AssignmentCreate, AssignmentUpdate,
)
from services.admin_assignments import list_assignments, create_assignment, update_assignment, delete_assignment

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


@router.patch("/{assignment_id}", response_model=AdminAssignmentRow)
async def assignments_update(
    assignment_id: int,
    body: AssignmentUpdate,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    return await update_assignment(db, assignment_id, body)


@router.delete("/{assignment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def assignments_delete(
    assignment_id: int,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    await delete_assignment(db, assignment_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
