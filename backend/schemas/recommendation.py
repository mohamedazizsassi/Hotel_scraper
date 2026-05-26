# backend/schemas/recommendation.py
from __future__ import annotations
from datetime import date
from pydantic import BaseModel
from schemas.common import HotelMeta, DataResponse

class RecommendationRow(HotelMeta):
    scraped_at: str
    check_in: date
    nights: int
    adults: int
    boarding_canonical: str
    current_price_tnd: float
    q10_cal_tnd: float
    q50_tnd: float
    q90_cal_tnd: float
    interval_status: str
    direction: str
    recommended_price_tnd: float
    delta_pct_vs_current: float
    peer_medium_median: float | None
    peer_medium_count: int | None
    reasons: list[str]

RecommendationResponse = DataResponse[RecommendationRow]
