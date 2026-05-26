# backend/schemas/common.py
from __future__ import annotations
from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class DataResponse(BaseModel, Generic[T]):
    data: list[T]
    count: int

class HotelMeta(BaseModel):
    hotel_name_normalized: str
    city_name: str
    stars_int: int | None
