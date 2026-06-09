import pytest
from sqlalchemy import select
from db.models import User
from core.security import create_access_token


async def _manager_token(db_session) -> str:
    result = await db_session.execute(select(User).where(User.email == "manager@test.com"))
    user = result.scalar_one()
    return create_access_token(str(user.id), hotel_id=1, role="manager")


@pytest.mark.asyncio
async def test_calendar_returns_200(client, db_session):
    token = await _manager_token(db_session)
    resp = await client.get(
        "/manager/calendar",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    assert "count" in body
    assert body["count"] == len(body["data"])


@pytest.mark.asyncio
async def test_calendar_row_has_competitor_avg(client, db_session):
    token = await _manager_token(db_session)
    resp = await client.get(
        "/manager/calendar",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    rows = resp.json()["data"]
    assert len(rows) >= 1, "Expected at least one calendar row"
    row = rows[0]
    assert "competitor_avg_per_night" in row, "Field competitor_avg_per_night missing from response"
    # Both competitors have price_per_night 500 and 480 → avg = 490
    assert row["competitor_avg_per_night"] == pytest.approx(490.0, abs=1.0)


@pytest.mark.asyncio
async def test_calendar_row_schema(client, db_session):
    token = await _manager_token(db_session)
    resp = await client.get(
        "/manager/calendar",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    rows = resp.json()["data"]
    if rows:
        row = rows[0]
        for field in (
            "check_in", "price_per_night", "peer_medium_median",
            "recommended_price_per_night", "competitor_avg_per_night",
            "boarding_canonical", "nights", "adults",
        ):
            assert field in row, f"Missing field: {field}"
