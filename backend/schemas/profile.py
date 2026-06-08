from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class HotelBrief(BaseModel):
    id: int
    hotel_name_display: str
    city_name: str
    stars_int: int | None


class ManagerProfile(BaseModel):
    full_name: str | None
    email: str
    role: str
    preferences: dict
    hotel: HotelBrief | None
    competitor_count: int
    max_competitors: int
    last_login_at: str | None


class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    preferences: Optional[dict] = None
