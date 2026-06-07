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


async def test_put_selection_replaces(client, db_session):
    tok = await _admin_token(db_session)
    uid = await _uid(db_session, "manager@test.com")
    comp2 = await _hid(db_session, "hotel_comp_2")
    r = await client.put(f"/admin/managers/{uid}/competitors",
                         json={"hotel_ids": [comp2]},
                         headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200, r.text
    rows = r.json()["data"]
    assert [c["hotel_id"] for c in rows] == [comp2]
    assert rows[0]["display_order"] == 1


async def test_put_selection_rejects_own_hotel(client, db_session):
    tok = await _admin_token(db_session)
    uid = await _uid(db_session, "manager@test.com")
    own = await _hid(db_session, "hotel_manager_test")
    r = await client.put(f"/admin/managers/{uid}/competitors",
                         json={"hotel_ids": [own]},
                         headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 400


async def test_put_selection_requires_assignment(client, db_session):
    tok = await _admin_token(db_session)
    uid = await _uid(db_session, "manager2@test.com")   # unassigned
    comp1 = await _hid(db_session, "hotel_comp_1")
    r = await client.put(f"/admin/managers/{uid}/competitors",
                         json={"hotel_ids": [comp1]},
                         headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 400


async def test_put_selection_enforces_cap(client, db_session):
    tok = await _admin_token(db_session)
    # assign manager2 to hotel_comp_2 with max_competitors = 1
    uid = await _uid(db_session, "manager2@test.com")
    hcomp2 = await _hid(db_session, "hotel_comp_2")
    asg = await client.post("/admin/assignments",
                            json={"user_id": uid, "hotel_id": hcomp2, "max_competitors": 1},
                            headers={"Authorization": f"Bearer {tok}"})
    assert asg.status_code == 201
    # now try to set 2 competitors → exceeds cap of 1
    own_excluded = [await _hid(db_session, "hotel_manager_test"),
                    await _hid(db_session, "hotel_comp_1")]
    r = await client.put(f"/admin/managers/{uid}/competitors",
                         json={"hotel_ids": own_excluded},
                         headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 400
