import pytest
from sqlalchemy import select
from db.models import User
from core.security import create_access_token

async def _manager_token(db_session) -> str:
    result = await db_session.execute(select(User).where(User.email == "manager@test.com"))
    user = result.scalar_one()
    return create_access_token(str(user.id), hotel_id=1, role="manager")

@pytest.mark.asyncio
async def test_me_returns_profile(client, db_session):
    token = await _manager_token(db_session)
    resp = await client.get("/manager/me", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "manager@test.com"
    assert body["full_name"] == "Test Manager"
    assert body["role"] == "manager"
    assert body["hotel"]["hotel_name_display"] == "Hotel Manager Test"
    assert body["hotel"]["city_name"] == "Hammamet"
    assert body["competitor_count"] == 2
    assert body["max_competitors"] == 4
    assert isinstance(body["preferences"], dict)

@pytest.mark.asyncio
async def test_me_requires_auth(client):
    resp = await client.get("/manager/me")
    assert resp.status_code == 401

@pytest.mark.asyncio
async def test_patch_me_updates_name_and_prefs(client, db_session):
    token = await _manager_token(db_session)
    resp = await client.patch(
        "/manager/me",
        headers={"Authorization": f"Bearer {token}"},
        json={"full_name": "Renamed Manager",
              "preferences": {"language": "fr", "alerts": {"price_spike": True}}},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["full_name"] == "Renamed Manager"
    assert body["preferences"]["language"] == "fr"
    assert body["preferences"]["alerts"]["price_spike"] is True

@pytest.mark.asyncio
async def test_patch_me_partial_keeps_other_fields(client, db_session):
    token = await _manager_token(db_session)
    await client.patch("/manager/me", headers={"Authorization": f"Bearer {token}"},
                       json={"preferences": {"language": "en"}})
    resp = await client.patch("/manager/me", headers={"Authorization": f"Bearer {token}"},
                              json={"full_name": "Only Name"})
    body = resp.json()
    assert body["full_name"] == "Only Name"
    assert body["preferences"]["language"] == "en"
