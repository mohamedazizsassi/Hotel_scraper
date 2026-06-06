from __future__ import annotations
from pydantic import BaseModel
from schemas.common import DataResponse


class AdminHotelRow(BaseModel):
    id: int
    hotel_name_normalized: str
    hotel_name_display: str
    city_name: str
    stars_int: int | None
    is_active: bool
    region: str | None
    contact_email: str | None
    contact_phone: str | None
    sources: str
    manager_id: str | None
    manager_name: str | None
    latest_scraped_at: str | None


class DiscoverableHotel(BaseModel):
    hotel_name_normalized: str
    city_name: str
    stars_int: int | None


AdminHotelListResponse = DataResponse[AdminHotelRow]
DiscoverableResponse = DataResponse[DiscoverableHotel]
