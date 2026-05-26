# backend/schemas/calendar.py
from __future__ import annotations
from datetime import date
from pydantic import BaseModel
from schemas.common import HotelMeta, DataResponse

class CalendarRow(HotelMeta):
    check_in: date
    nights: int
    adults: int
    boarding_canonical: str
    price: float
    price_per_night: float
    scraped_at: str
    peer_medium_median: float | None
    peer_medium_count: int | None

CalendarResponse = DataResponse[CalendarRow]
