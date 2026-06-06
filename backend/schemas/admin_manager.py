from __future__ import annotations
from pydantic import BaseModel, EmailStr
from schemas.common import DataResponse


class AdminManagerRow(BaseModel):
    id: str
    email: str
    full_name: str | None
    is_active: bool
    last_login_at: str | None
    assigned_hotel_id: int | None
    assigned_hotel_name: str | None


AdminManagerListResponse = DataResponse[AdminManagerRow]


class ManagerCreate(BaseModel):
    email: EmailStr
    full_name: str | None = None
    initial_password: str


class ManagerUpdate(BaseModel):
    email: EmailStr | None = None
    full_name: str | None = None
    is_active: bool | None = None


class PasswordReset(BaseModel):
    new_password: str
