from sqlalchemy import select
from db.models import User
from core.security import create_access_token


async def _admin_token(db_session) -> str:
    res = await db_session.execute(select(User).where(User.email == "admin@test.com"))
    return create_access_token(str(res.scalar_one().id), hotel_id=None, role="admin")


async def _manager_token(db_session) -> str:
    res = await db_session.execute(select(User).where(User.email == "manager@test.com"))
    return create_access_token(str(res.scalar_one().id), hotel_id=1, role="manager")


async def test_list_hotels_requires_admin(client, db_session):
    assert (await client.get("/admin/hotels")).status_code == 401
    tok = await _manager_token(db_session)
    r = await client.get("/admin/hotels", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 403


async def test_list_hotels_returns_pool(client, db_session):
    tok = await _admin_token(db_session)
    r = await client.get("/admin/hotels", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    body = r.json()
    assert "data" in body and "count" in body
    names = {h["hotel_name_normalized"] for h in body["data"]}
    assert "hotel_comp_1" in names
    assert "hotel_unregistered" not in names
    row = next(h for h in body["data"] if h["hotel_name_normalized"] == "hotel_comp_1")
    for f in ("id", "hotel_name_display", "city_name", "stars_int", "is_active",
              "region", "sources", "manager_name"):
        assert f in row


async def test_discoverable_excludes_registered(client, db_session):
    tok = await _admin_token(db_session)
    r = await client.get("/admin/hotels/discoverable", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    names = {h["hotel_name_normalized"] for h in r.json()["data"]}
    assert "hotel_unregistered" in names
    assert "hotel_comp_1" not in names
