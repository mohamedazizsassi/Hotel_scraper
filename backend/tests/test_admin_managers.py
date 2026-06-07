from sqlalchemy import select
from db.models import User
from core.security import create_access_token


async def _admin_token(db_session) -> str:
    res = await db_session.execute(select(User).where(User.email == "admin@test.com"))
    return create_access_token(str(res.scalar_one().id), hotel_id=None, role="admin")


async def _manager_token(db_session) -> str:
    res = await db_session.execute(select(User).where(User.email == "manager@test.com"))
    return create_access_token(str(res.scalar_one().id), hotel_id=1, role="manager")


async def _manager_id(client, db_session, email="manager@test.com"):
    tok = await _admin_token(db_session)
    r = await client.get("/admin/managers", headers={"Authorization": f"Bearer {tok}"})
    return next(m["id"] for m in r.json()["data"] if m["email"] == email)


async def test_list_managers_requires_admin(client, db_session):
    assert (await client.get("/admin/managers")).status_code == 401
    tok = await _manager_token(db_session)
    r = await client.get("/admin/managers", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 403


async def test_list_managers(client, db_session):
    tok = await _admin_token(db_session)
    r = await client.get("/admin/managers", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    data = r.json()["data"]
    by_email = {m["email"]: m for m in data}
    assert "manager@test.com" in by_email and "manager2@test.com" in by_email
    assert by_email["manager@test.com"]["assigned_hotel_name"] == "Hotel Manager Test"
    assert by_email["manager2@test.com"]["assigned_hotel_id"] is None
    for f in ("id", "email", "full_name", "is_active", "last_login_at",
              "assigned_hotel_id", "assigned_hotel_name"):
        assert f in by_email["manager@test.com"]


async def test_manager_detail_and_404(client, db_session):
    tok = await _admin_token(db_session)
    mid = await _manager_id(client, db_session)
    r = await client.get(f"/admin/managers/{mid}", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    assert r.json()["email"] == "manager@test.com"
    r404 = await client.get("/admin/managers/00000000-0000-0000-0000-000000000000",
                            headers={"Authorization": f"Bearer {tok}"})
    assert r404.status_code == 404


async def test_create_manager(client, db_session):
    tok = await _admin_token(db_session)
    r = await client.post("/admin/managers",
                          json={"email": "newmgr@test.com", "full_name": "New Mgr",
                                "initial_password": "secret123"},
                          headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["email"] == "newmgr@test.com"
    assert body["assigned_hotel_id"] is None
    assert "password_hash" not in body
    login = await client.post("/auth/login",
                              json={"email": "newmgr@test.com", "password": "secret123"})
    assert login.status_code == 200


async def test_create_manager_duplicate_email(client, db_session):
    tok = await _admin_token(db_session)
    r = await client.post("/admin/managers",
                          json={"email": "manager@test.com", "full_name": "Dup",
                                "initial_password": "x1234567"},
                          headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 409


async def test_update_manager(client, db_session):
    tok = await _admin_token(db_session)
    mid = await _manager_id(client, db_session, "manager2@test.com")
    r = await client.patch(f"/admin/managers/{mid}",
                           json={"full_name": "Renamed Two", "is_active": False},
                           headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    assert r.json()["full_name"] == "Renamed Two"
    assert r.json()["is_active"] is False


async def test_reset_password(client, db_session):
    tok = await _admin_token(db_session)
    mid = await _manager_id(client, db_session, "manager2@test.com")
    r = await client.post(f"/admin/managers/{mid}/reset-password",
                          json={"new_password": "brandnew9"},
                          headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 204
    # With per-test isolation (Task 1), manager2 is fresh here regardless of other tests.
