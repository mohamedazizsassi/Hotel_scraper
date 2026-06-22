import pytest
from sqlalchemy import select
from db.models import User
from core.security import create_access_token

async def _manager_token(db_session) -> str:
    result = await db_session.execute(select(User).where(User.email == "manager@test.com"))
    user = result.scalar_one()
    return create_access_token(str(user.id), hotel_id=1, role="manager")

@pytest.mark.asyncio
async def test_dashboard_shape(client, db_session):
    token = await _manager_token(db_session)
    resp = await client.get("/manager/dashboard", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    for key in ("kpis", "price_series", "top_recommendations", "competitors", "recent_alerts"):
        assert key in body
    k = body["kpis"]
    for key in ("avg_listed_rate_tnd", "market_position_pct", "vs_competitor_pct",
                "opportunity_tnd", "opportunity_pct", "open_recommendations", "active_alerts"):
        assert key in k

@pytest.mark.asyncio
async def test_dashboard_requires_auth(client):
    resp = await client.get("/manager/dashboard")
    assert resp.status_code == 401
