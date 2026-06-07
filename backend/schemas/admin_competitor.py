from __future__ import annotations
from pydantic import BaseModel
from schemas.common import DataResponse


class CompetitorRow(BaseModel):
    hotel_id: int
    hotel_name_display: str
    city_name: str
    stars_int: int | None
    display_order: int


class SelectableHotel(BaseModel):
    hotel_id: int
    hotel_name_display: str
    city_name: str
    stars_int: int | None


class CompetitorSelectionUpdate(BaseModel):
    hotel_ids: list[int]


CompetitorSelectionResponse = DataResponse[CompetitorRow]
SelectableResponse = DataResponse[SelectableHotel]
