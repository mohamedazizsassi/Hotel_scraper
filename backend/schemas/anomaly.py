# backend/schemas/anomaly.py
from __future__ import annotations
from datetime import date
from pydantic import BaseModel
from schemas.common import HotelMeta, DataResponse

class AnomalyRow(HotelMeta):
    scraped_at: str
    check_in: date
    nights: int
    adults: int
    boarding_canonical: str
    price: float
    price_per_night: float
    q10_cal_tnd: float
    q90_cal_tnd: float
    anomaly_score: float
    interval_status: str  # "below_band" | "above_band"

AnomalyResponse = DataResponse[AnomalyRow]
