# backend/schemas/admin_assignment.py
from __future__ import annotations
from pydantic import BaseModel
from schemas.common import DataResponse


class AdminAssignmentRow(BaseModel):
    id: int
    user_id: str
    manager_email: str
    manager_name: str | None
    hotel_id: int
    hotel_name: str
    max_competitors: int
    is_active: bool


class AssignmentCreate(BaseModel):
    user_id: str
    hotel_id: int
    max_competitors: int = 4


class AssignmentUpdate(BaseModel):
    hotel_id: int | None = None
    max_competitors: int | None = None
    is_active: bool | None = None


AdminAssignmentListResponse = DataResponse[AdminAssignmentRow]
