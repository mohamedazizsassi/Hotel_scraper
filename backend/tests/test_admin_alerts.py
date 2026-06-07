from sqlalchemy import select
from db.models import User
from core.security import create_access_token


async def _admin_token(db_session) -> str:
    res = await db_session.execute(select(User).where(User.email == "admin@test.com"))
    return create_access_token(str(res.scalar_one().id), hotel_id=None, role="admin")


async def _manager_token(db_session) -> str:
    res = await db_session.execute(select(User).where(User.email == "manager@test.com"))
    return create_access_token(str(res.scalar_one().id), hotel_id=1, role="manager")


async def test_alerts_requires_admin(client, db_session):
    assert (await client.get("/admin/alerts")).status_code == 401
    tok = await _manager_token(db_session)
    r = await client.get("/admin/alerts", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 403


async def test_alerts_derived(client, db_session):
    # seeded scrape_runs: items 25000/24000/3000 finished (median 24000) + 1 failed.
    tok = await _admin_token(db_session)
    r = await client.get("/admin/alerts", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    alerts = r.json()["data"]
    types = {a["type"] for a in alerts}
    assert "failed_run" in types        # the 2026-06-02 15:00 failed run
    assert "low_volume" in types        # the 3000-item run < 50% of median 24000
    for f in ("type", "severity", "message", "run_ts", "log_filename"):
        assert f in alerts[0]
