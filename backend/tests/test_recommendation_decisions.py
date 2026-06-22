import pytest
from sqlalchemy import select
from db.models import User
from core.security import create_access_token

async def _manager_token(db_session) -> str:
    result = await db_session.execute(select(User).where(User.email == "manager@test.com"))
    user = result.scalar_one()
    return create_access_token(str(user.id), hotel_id=1, role="manager")

def _payload(status="accepted"):
    return {"check_in": "2026-07-01", "nights": 3, "adults": 2,
            "boarding_canonical": "BB", "recommended_price_tnd": 450.0, "status": status}

@pytest.mark.asyncio
async def test_decision_upsert_insert_then_update(client, db_session):
    token = await _manager_token(db_session)
    h = {"Authorization": f"Bearer {token}"}
    r1 = await client.post("/manager/recommendations/decision", headers=h, json=_payload("accepted"))
    assert r1.status_code == 200
    assert r1.json()["status"] == "accepted"
    r2 = await client.post("/manager/recommendations/decision", headers=h, json=_payload("dismissed"))
    assert r2.status_code == 200
    assert r2.json()["status"] == "dismissed"

@pytest.mark.asyncio
async def test_decision_bad_status_rejected(client, db_session):
    token = await _manager_token(db_session)
    h = {"Authorization": f"Bearer {token}"}
    r = await client.post("/manager/recommendations/decision", headers=h, json=_payload("maybe"))
    assert r.status_code == 422

@pytest.mark.asyncio
async def test_decision_requires_auth(client):
    r = await client.post("/manager/recommendations/decision", json=_payload())
    assert r.status_code == 401

@pytest.mark.asyncio
async def test_decision_bulk(client, db_session):
    token = await _manager_token(db_session)
    h = {"Authorization": f"Bearer {token}"}
    body = {"status": "accepted", "items": [
        {"check_in": "2026-07-01", "nights": 3, "adults": 2, "boarding_canonical": "BB", "recommended_price_tnd": 450.0},
        {"check_in": "2026-07-02", "nights": 3, "adults": 2, "boarding_canonical": "BB", "recommended_price_tnd": 460.0},
    ]}
    r = await client.post("/manager/recommendations/decision/bulk", headers=h, json=body)
    assert r.status_code == 200
    assert r.json()["count"] == 2

@pytest.mark.asyncio
async def test_recommendations_include_decision_status(client, db_session):
    token = await _manager_token(db_session)
    h = {"Authorization": f"Bearer {token}"}
    await client.post("/manager/recommendations/decision", headers=h, json=_payload("accepted"))
    resp = await client.get("/manager/recommendations", headers=h)
    rows = resp.json()["data"]
    assert rows, "expected at least one recommendation row"
    match = [r for r in rows if r["check_in"] == "2026-07-01"]
    assert match and match[0]["decision_status"] == "accepted"
