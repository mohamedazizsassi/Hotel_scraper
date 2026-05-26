import pytest
from sqlalchemy import select
from db.models import User
from core.security import create_access_token

async def _manager_token(db_session) -> str:
    result = await db_session.execute(select(User).where(User.email == "manager@test.com"))
    user = result.scalar_one()
    return create_access_token(str(user.id), hotel_id=1, role="manager")

@pytest.mark.asyncio
async def test_recommendations_returns_200(client, db_session):
    token = await _manager_token(db_session)
    resp = await client.get(
        "/manager/recommendations",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    assert "count" in body
    assert body["count"] == len(body["data"])

@pytest.mark.asyncio
async def test_recommendations_schema_shape(client, db_session):
    token = await _manager_token(db_session)
    resp = await client.get(
        "/manager/recommendations",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    rows = resp.json()["data"]
    if rows:
        row = rows[0]
        for field in ("direction", "recommended_price_tnd", "delta_pct_vs_current",
                      "check_in", "nights", "boarding_canonical", "reasons"):
            assert field in row, f"Missing field: {field}"
        assert row["direction"] in ("raise", "hold", "lower")

@pytest.mark.asyncio
async def test_recommendations_requires_auth(client):
    resp = await client.get("/manager/recommendations")
    assert resp.status_code == 422

@pytest.mark.asyncio
async def test_recommendations_rejects_bad_token(client):
    resp = await client.get(
        "/manager/recommendations",
        headers={"Authorization": "Bearer not.a.valid.token"},
    )
    assert resp.status_code == 401
