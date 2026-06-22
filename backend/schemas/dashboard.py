# backend/schemas/dashboard.py
from __future__ import annotations
from datetime import date
from pydantic import BaseModel
from schemas.recommendation import RecommendationRow
from schemas.competitor import CompetitorSummary
from schemas.anomaly import AnomalyRow

class DashboardKpis(BaseModel):
    avg_listed_rate_tnd: float | None
    market_position_pct: float | None
    vs_competitor_pct: float | None
    opportunity_tnd: float | None
    opportunity_pct: float | None
    open_recommendations: int
    active_alerts: int

class PriceSeriesPoint(BaseModel):
    check_in: date
    price_per_night: float
    peer_medium_median: float | None
    recommended_price_per_night: float

class DashboardResponse(BaseModel):
    kpis: DashboardKpis
    price_series: list[PriceSeriesPoint]
    top_recommendations: list[RecommendationRow]
    competitors: list[CompetitorSummary]
    recent_alerts: list[AnomalyRow]
