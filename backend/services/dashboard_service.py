# backend/services/dashboard_service.py
from __future__ import annotations
from datetime import date, timedelta
from statistics import mean
from sqlalchemy.ext.asyncio import AsyncSession
from db.models import User
from ml_store.store import MLStore
from services.calendar_service import get_calendar, get_calendar_options
from services.competitor_service import get_competitors
from services.recommendation_service import get_recommendations
from services.anomaly_service import get_anomalies
from schemas.dashboard import DashboardResponse, DashboardKpis, PriceSeriesPoint

def _safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None

async def get_dashboard(user: User, db: AsyncSession, ml_store: MLStore,
                        days: int = 30) -> DashboardResponse:
    today = date.today()
    to = today + timedelta(days=days)
    opts = await get_calendar_options(user, db)
    d = opts.default
    cfg = dict(nights=d.nights, adults=d.adults, room_base=d.room_base,
               room_view=d.room_view, room_tier=d.room_tier,
               room_occupancy=d.room_occupancy, boarding_canonical=d.boarding_canonical)

    cal = await get_calendar(user, db, ml_store, check_in_from=today, check_in_to=to, **cfg)
    competitors = await get_competitors(user, db)
    recs = await get_recommendations(user, db, ml_store, check_in_from=today, check_in_to=to, **cfg)
    alerts = await get_anomalies(user, db, ml_store, check_in_from=today, check_in_to=to, **cfg)

    avg_price = _safe_mean([r.price_per_night for r in cal])
    avg_peer = _safe_mean([r.peer_medium_median for r in cal])
    comp_avg = _safe_mean([c.avg_price_per_night for c in competitors])
    avg_rec = _safe_mean([r.recommended_price_per_night for r in cal])

    market_pos = ((avg_price - avg_peer) / avg_peer * 100) if avg_price and avg_peer else None
    vs_comp = ((avg_price - comp_avg) / comp_avg * 100) if avg_price and comp_avg else None
    opp_tnd = (avg_rec - avg_price) if avg_rec is not None and avg_price is not None else None
    opp_pct = (opp_tnd / avg_price * 100) if opp_tnd is not None and avg_price else None

    open_recs = sum(1 for r in recs if r.direction != "hold" and r.decision_status is None)

    kpis = DashboardKpis(
        avg_listed_rate_tnd=avg_price, market_position_pct=market_pos,
        vs_competitor_pct=vs_comp, opportunity_tnd=opp_tnd, opportunity_pct=opp_pct,
        open_recommendations=open_recs, active_alerts=len(alerts),
    )
    series = [PriceSeriesPoint(
        check_in=r.check_in, price_per_night=r.price_per_night,
        peer_medium_median=r.peer_medium_median,
        recommended_price_per_night=r.recommended_price_per_night,
    ) for r in sorted(cal, key=lambda x: x.check_in)[:14]]
    top_recs = sorted(recs, key=lambda r: abs(r.delta_pct_vs_current), reverse=True)[:5]
    recent_alerts = sorted(alerts, key=lambda a: abs(a.anomaly_score), reverse=True)[:5]

    return DashboardResponse(
        kpis=kpis, price_series=series, top_recommendations=top_recs,
        competitors=competitors, recent_alerts=recent_alerts,
    )
