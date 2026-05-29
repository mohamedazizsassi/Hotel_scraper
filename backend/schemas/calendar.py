# backend/schemas/calendar.py
from __future__ import annotations
from datetime import date
from pydantic import BaseModel
from schemas.common import HotelMeta, DataResponse

class CalendarRow(HotelMeta):
    # Identity / freshness (hotel_features_full)
    check_in: date
    scrape_date: str
    scraped_at: str

    # Stay + room configuration
    nights: int
    adults: int
    children: int
    boarding_canonical: str
    room_base: str
    room_view: str
    room_tier: str
    room_occupancy: str
    is_supplement_variant: bool
    has_free_view_upgrade: bool

    # Pricing (TND, per-night) + peer aggregate
    price_per_night: float
    peer_medium_median: float | None
    peer_medium_count: int | None
    best_peer_granularity_used: str | None

    # Recommendation overlay (derived from forecaster + CQR + recommender)
    recommended_price_per_night: float
    forecaster_confidence: float  # 0..1, UI proxy from calibrated interval width

    # Demand proxy
    sur_demande_rate_city_stars_checkin: float | None
    days_until_checkin: int

    # Calendar overlays (calendar_dim, joined on check_in)
    is_weekend_checkin: bool
    is_ramadan: bool
    is_tunisia_public_holiday: bool
    is_tunisia_school_holiday: bool
    is_school_holiday_france: bool
    is_school_holiday_germany: bool
    is_school_holiday_uk: bool

CalendarResponse = DataResponse[CalendarRow]


class CalendarConfig(BaseModel):
    """The most-populated product configuration for the manager's hotel —
    used by the frontend as sensible default filter values."""
    room_base: str | None
    room_view: str | None
    room_tier: str | None
    room_occupancy: str | None
    boarding_canonical: str | None
    nights: int | None
    adults: int | None


class CalendarOptions(BaseModel):
    """Filter values that actually exist for the manager's hotel, so the
    calendar filter bar offers only configurations that return data."""
    room_base: list[str]
    room_view: list[str]
    room_tier: list[str]
    room_occupancy: list[str]
    boarding_canonical: list[str]
    nights: list[int]
    adults: list[int]
    scrape_dates: list[str]
    check_in_min: date | None
    check_in_max: date | None
    default: CalendarConfig
