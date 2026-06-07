from sqlalchemy import select
from db.models import User
from core.security import create_access_token
from core.dependencies import get_hotel_prices_total
from main import app


async def _admin_token(db_session) -> str:
    res = await db_session.execute(select(User).where(User.email == "admin@test.com"))
    return create_access_token(str(res.scalar_one().id), hotel_id=None, role="admin")


async def _manager_token(db_session) -> str:
    res = await db_session.execute(select(User).where(User.email == "manager@test.com"))
    return create_access_token(str(res.scalar_one().id), hotel_id=1, role="manager")


async def test_summary_requires_admin(client, db_session):
    assert (await client.get("/admin/monitoring/summary")).status_code == 401
    tok = await _manager_token(db_session)
    r = await client.get("/admin/monitoring/summary", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 403


async def test_summary(client, db_session):
    app.dependency_overrides[get_hotel_prices_total] = lambda: 24_400_000
    try:
        tok = await _admin_token(db_session)
        r = await client.get("/admin/monitoring/summary",
                             headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200, r.text
        b = r.json()
        assert b["total_rows"] == 24_400_000
        assert b["logged_window_items"] == 52000      # 25000+24000+3000+0
        assert b["runs_count"] == 4
        assert b["finished_runs"] == 3
        assert b["failed_runs"] == 1
        assert b["last_run_status"] == "failed"        # latest run_ts is 2026-06-02 15:00
        assert b["hotels_scraped_distinct"] == 2       # from hotel_features
        assert b["latest_scrape_at"].startswith("2026-06-02")
    finally:
        app.dependency_overrides.pop(get_hotel_prices_total, None)


async def test_summary_total_null_when_mongo_down(client, db_session):
    app.dependency_overrides[get_hotel_prices_total] = lambda: None
    try:
        tok = await _admin_token(db_session)
        r = await client.get("/admin/monitoring/summary",
                             headers={"Authorization": f"Bearer {tok}"})
        assert r.status_code == 200
        assert r.json()["total_rows"] is None
    finally:
        app.dependency_overrides.pop(get_hotel_prices_total, None)


async def test_runs(client, db_session):
    tok = await _admin_token(db_session)
    r = await client.get("/admin/monitoring/runs?limit=10",
                         headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    rows = r.json()["data"]
    assert len(rows) == 4
    # newest first
    assert rows[0]["log_filename"] == "run_2026-06-02_15-00.log"
    for f in ("run_ts", "log_filename", "source", "items_total", "errors_total",
              "duration_s", "status"):
        assert f in rows[0]


async def test_daily(client, db_session):
    tok = await _admin_token(db_session)
    r = await client.get("/admin/monitoring/daily?days=3650",
                         headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    by_day = {d["day"]: d for d in r.json()["data"]}
    assert by_day["2026-06-01"]["items_total"] == 49000   # 25000 + 24000
    assert by_day["2026-06-01"]["runs"] == 2
    assert by_day["2026-06-02"]["items_total"] == 3000
