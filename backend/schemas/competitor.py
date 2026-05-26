# backend/schemas/competitor.py
from __future__ import annotations
from pydantic import BaseModel
from schemas.common import HotelMeta, DataResponse

class CompetitorSummary(HotelMeta):
    hotel_name_display: str
    display_order: int
    avg_price_per_night: float | None
    latest_scraped_at: str | None

CompetitorResponse = DataResponse[CompetitorSummary]
