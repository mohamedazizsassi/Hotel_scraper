# backend/tests/test_admin_assignments.py
from sqlalchemy import select
from db.models import User, PlatformHotel
from core.security import create_access_token


async def _admin_token(db_session) -> str:
    res = await db_session.execute(select(User).where(User.email == "admin@test.com"))
    return create_access_token(str(res.scalar_one().id), hotel_id=None, role="admin")


async def _uid(db_session, email) -> str:
    res = await db_session.execute(select(User).where(User.email == email))
    return str(res.scalar_one().id)


async def _hid(db_session, name) -> int:
    res = await db_session.execute(
        select(PlatformHotel).where(PlatformHotel.hotel_name_normalized == name))
    return res.scalar_one().id


async def test_list_assignments(client, db_session):
    tok = await _admin_token(db_session)
    r = await client.get("/admin/assignments", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    rows = r.json()["data"]
    assert any(a["manager_email"] == "manager@test.com"
               and a["hotel_name"] == "Hotel Manager Test" for a in rows)
    for f in ("id", "user_id", "manager_email", "manager_name", "hotel_id",
              "hotel_name", "max_competitors", "is_active"):
        assert f in rows[0]


async def test_create_assignment(client, db_session):
    tok = await _admin_token(db_session)
    uid = await _uid(db_session, "manager2@test.com")
    hid = await _hid(db_session, "hotel_comp_2")
    r = await client.post("/admin/assignments",
                          json={"user_id": uid, "hotel_id": hid, "max_competitors": 3},
                          headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["manager_email"] == "manager2@test.com"
    assert body["hotel_name"] == "Hotel Comp 2"
    assert body["max_competitors"] == 3


async def test_create_assignment_duplicate(client, db_session):
    tok = await _admin_token(db_session)
    uid = await _uid(db_session, "manager@test.com")   # already assigned
    hid = await _hid(db_session, "hotel_comp_1")
    r = await client.post("/admin/assignments",
                          json={"user_id": uid, "hotel_id": hid},
                          headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 409


async def _assignment_id(client, db_session, email="manager@test.com"):
    tok = await _admin_token(db_session)
    r = await client.get("/admin/assignments", headers={"Authorization": f"Bearer {tok}"})
    return next(a["id"] for a in r.json()["data"] if a["manager_email"] == email)


async def test_update_assignment(client, db_session):
    tok = await _admin_token(db_session)
    aid = await _assignment_id(client, db_session)
    new_hotel = await _hid(db_session, "hotel_comp_1")
    r = await client.patch(f"/admin/assignments/{aid}",
                           json={"hotel_id": new_hotel, "max_competitors": 2},
                           headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    assert r.json()["hotel_name"] == "Hotel Comp 1"
    assert r.json()["max_competitors"] == 2


async def test_delete_assignment(client, db_session):
    tok = await _admin_token(db_session)
    aid = await _assignment_id(client, db_session)
    r = await client.delete(f"/admin/assignments/{aid}",
                            headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 204
    lst = await client.get("/admin/assignments", headers={"Authorization": f"Bearer {tok}"})
    assert all(a["id"] != aid for a in lst.json()["data"])


async def test_update_assignment_404(client, db_session):
    tok = await _admin_token(db_session)
    r = await client.patch("/admin/assignments/999999", json={"max_competitors": 1},
                           headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 404
