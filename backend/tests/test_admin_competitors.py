from sqlalchemy import select
from db.models import User, PlatformHotel
from core.security import create_access_token


async def _admin_token(db_session) -> str:
    res = await db_session.execute(select(User).where(User.email == "admin@test.com"))
    return create_access_token(str(res.scalar_one().id), hotel_id=None, role="admin")


async def _manager_token(db_session) -> str:
    res = await db_session.execute(select(User).where(User.email == "manager@test.com"))
    return create_access_token(str(res.scalar_one().id), hotel_id=1, role="manager")


async def _uid(db_session, email) -> str:
    res = await db_session.execute(select(User).where(User.email == email))
    return str(res.scalar_one().id)


async def _hid(db_session, name) -> int:
    res = await db_session.execute(
        select(PlatformHotel).where(PlatformHotel.hotel_name_normalized == name))
    return res.scalar_one().id


async def test_competitors_requires_admin(client, db_session):
    uid = await _uid(db_session, "manager@test.com")
    assert (await client.get(f"/admin/managers/{uid}/competitors")).status_code == 401
    tok = await _manager_token(db_session)
    r = await client.get(f"/admin/managers/{uid}/competitors",
                         headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 403


async def test_get_selection(client, db_session):
    tok = await _admin_token(db_session)
    uid = await _uid(db_session, "manager@test.com")
    r = await client.get(f"/admin/managers/{uid}/competitors",
                         headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    rows = r.json()["data"]
    names = [c["hotel_name_display"] for c in rows]
    assert names == ["Hotel Comp 1", "Hotel Comp 2"]          # ordered by display_order
    for f in ("hotel_id", "hotel_name_display", "city_name", "stars_int", "display_order"):
        assert f in rows[0]


async def test_get_selectable_excludes_own_hotel(client, db_session):
    tok = await _admin_token(db_session)
    uid = await _uid(db_session, "manager@test.com")
    own = await _hid(db_session, "hotel_manager_test")
    r = await client.get(f"/admin/managers/{uid}/selectable-competitors",
                         headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    ids = {h["hotel_id"] for h in r.json()["data"]}
    assert own not in ids
    assert await _hid(db_session, "hotel_comp_1") in ids
