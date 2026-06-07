# Admin API — Competitors + Monitoring + Alerts — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the admin backend: admin selects each manager's 3–4 competitors (D11, admin-only), plus scraping-monitoring endpoints (Mongo total + `scrape_runs` aggregates) and derived collection alerts.

**Architecture:** Same patterns as Plans 2–3: `/admin/*` routers gated by `get_current_admin`, raw-SQL services, `DataResponse[T]`, custom exceptions. Mongo total comes through an **overridable dependency** (`get_hotel_prices_total`) so tests don't touch a real Mongo. The **test DB has no DB triggers** (it's ORM-built), so the competitor service must do all validation itself — the migration's triggers (`check_competitor_not_self`, `check_competitor_count`) are only a production backstop.

**Tech Stack:** FastAPI, SQLAlchemy 2.0 async, asyncpg, PostgreSQL, motor (Mongo), pytest.

**Plan map:** Plans 1–3 done. **Plan 4 = competitors + monitoring + alerts (this doc).** Plan 5 = frontend.

---

## Conventions for every backend command

- Work in the `feat/admin-platform` **worktree**:
  `C:\Users\ASUS\Desktop\PFE\revway\.claude\worktrees\feat+admin-platform`. Never
  touch the main checkout. Verify before committing: `git rev-parse --show-toplevel`
  ends in `.claude/worktrees/feat+admin-platform`; `git branch --show-current` =
  `feat/admin-platform`.
- No local `.venv` — use `C:\Users\ASUS\Desktop\PFE\revway\backend\.venv\Scripts\python.exe`
  from the worktree `backend/` dir.
- Commits: Conventional Commits, **no `Co-Authored-By: Claude` trailer**.

## Recurring gotcha (from Plans 2–3)

Raw `INSERT`s vs the ORM-built test DB must set Python-default `NOT NULL` columns
explicitly. `user_competitor_selections.id` is an autoincrement integer (omit it);
`display_order` is required (set it). UUID columns: bind via `CAST(:uid AS uuid)`
(inserts) / `col::text = :uid` (reads). conftest already seeds: manager
`manager@test.com` (assigned to `hotel_manager_test`, `max_competitors=4`) WITH two
competitor selections (`hotel_comp_1` order 1, `hotel_comp_2` order 2);
`manager2@test.com` (unassigned); hotels `hotel_manager_test`/`hotel_comp_1`/
`hotel_comp_2`; `hotel_features` (2 distinct hotels). `scrape_runs` is **empty** in
the test DB until Task 3 seeds it.

---

### Task 1: Competitors — GET selection + GET selectable

**Files:**
- Create: `backend/schemas/admin_competitor.py`
- Create: `backend/services/admin_competitors.py`
- Create: `backend/routers/admin/competitors.py`
- Modify: `backend/main.py`
- Test: `backend/tests/test_admin_competitors.py`

- [ ] **Step 1: Write the failing test** — Create `backend/tests/test_admin_competitors.py`:

```python
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
```

- [ ] **Step 2: Run test → FAIL** (`pytest tests/test_admin_competitors.py -v`; 404/import).

- [ ] **Step 3: Schemas** — Create `backend/schemas/admin_competitor.py`:

```python
from __future__ import annotations
from pydantic import BaseModel
from schemas.common import DataResponse


class CompetitorRow(BaseModel):
    hotel_id: int
    hotel_name_display: str
    city_name: str
    stars_int: int | None
    display_order: int


class SelectableHotel(BaseModel):
    hotel_id: int
    hotel_name_display: str
    city_name: str
    stars_int: int | None


class CompetitorSelectionUpdate(BaseModel):
    hotel_ids: list[int]


CompetitorSelectionResponse = DataResponse[CompetitorRow]
SelectableResponse = DataResponse[SelectableHotel]
```

- [ ] **Step 4: Service** — Create `backend/services/admin_competitors.py`:

```python
from __future__ import annotations
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from core.exceptions import NotFoundError
from schemas.admin_competitor import CompetitorRow, SelectableHotel

_SELECTION_SQL = text("""
    SELECT ucs.hotel_id,
           ph.hotel_name_display,
           c.name_normalized      AS city_name,
           ph.stars_int,
           ucs.display_order
    FROM user_competitor_selections ucs
    JOIN platform_hotels ph ON ph.id = ucs.hotel_id
    JOIN cities c ON c.id = ph.city_id
    WHERE ucs.user_id::text = :uid
    ORDER BY ucs.display_order
""")

_SELECTABLE_SQL = text("""
    SELECT ph.id AS hotel_id,
           ph.hotel_name_display,
           c.name_normalized AS city_name,
           ph.stars_int
    FROM platform_hotels ph
    JOIN cities c ON c.id = ph.city_id
    WHERE ph.is_active = TRUE AND ph.id <> COALESCE(:own, -1)
    ORDER BY ph.hotel_name_display
""")


async def _ensure_manager(db: AsyncSession, manager_id: str) -> None:
    ok = await db.scalar(
        text("SELECT 1 FROM users WHERE id::text = :id AND role = 'manager'"),
        {"id": str(manager_id)})
    if not ok:
        raise NotFoundError(f"Manager {manager_id} not found")


async def _own_hotel_id(db: AsyncSession, manager_id: str) -> int | None:
    return await db.scalar(
        text("SELECT hotel_id FROM user_hotel_assignments "
             "WHERE user_id::text = :id AND is_active = TRUE"),
        {"id": str(manager_id)})


async def get_selection(db: AsyncSession, manager_id: str) -> list[CompetitorRow]:
    await _ensure_manager(db, manager_id)
    rows = (await db.execute(_SELECTION_SQL, {"uid": str(manager_id)})).mappings().fetchall()
    return [CompetitorRow(**dict(r)) for r in rows]


async def get_selectable(db: AsyncSession, manager_id: str) -> list[SelectableHotel]:
    await _ensure_manager(db, manager_id)
    own = await _own_hotel_id(db, manager_id)
    rows = (await db.execute(_SELECTABLE_SQL, {"own": own})).mappings().fetchall()
    return [SelectableHotel(**dict(r)) for r in rows]
```

- [ ] **Step 5: Router + mount** — Create `backend/routers/admin/competitors.py`:

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_admin
from db.models import User
from schemas.admin_competitor import CompetitorSelectionResponse, SelectableResponse
from services.admin_competitors import get_selection, get_selectable

router = APIRouter(prefix="/admin/managers", tags=["admin"])


@router.get("/{manager_id}/competitors", response_model=CompetitorSelectionResponse)
async def competitors_get(
    manager_id: str,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await get_selection(db, manager_id)
    return CompetitorSelectionResponse(data=rows, count=len(rows))


@router.get("/{manager_id}/selectable-competitors", response_model=SelectableResponse)
async def competitors_selectable(
    manager_id: str,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await get_selectable(db, manager_id)
    return SelectableResponse(data=rows, count=len(rows))
```

In `backend/main.py`, add:

```python
from routers.admin import competitors as admin_competitors
app.include_router(admin_competitors.router)
```

- [ ] **Step 6: Run tests → PASS** — `pytest tests/test_admin_competitors.py -v` (3 tests),
then `pytest -q` (expect 50 + 3 = 53).

- [ ] **Step 7: Commit**

```bash
git add backend/schemas/admin_competitor.py backend/services/admin_competitors.py backend/routers/admin/competitors.py backend/main.py backend/tests/test_admin_competitors.py
git commit -m "feat(api): admin competitor selection read endpoints"
```

---

### Task 2: Competitors — PUT replace (admin sets the 3–4)

**Files:**
- Modify: `backend/services/admin_competitors.py`
- Modify: `backend/routers/admin/competitors.py`
- Test: `backend/tests/test_admin_competitors.py` (extend)

- [ ] **Step 1: Write the failing tests** — APPEND to `backend/tests/test_admin_competitors.py`:

```python
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
```

- [ ] **Step 2: Run test → FAIL** (`pytest tests/test_admin_competitors.py -k put -v`).

- [ ] **Step 3: Service** — In `backend/services/admin_competitors.py`: extend the
exceptions import to `from core.exceptions import NotFoundError, BadRequestError`,
extend the schema import to include `CompetitorSelectionUpdate`, and ADD:

```python
async def set_selection(db: AsyncSession, manager_id: str,
                        body: CompetitorSelectionUpdate) -> list[CompetitorRow]:
    await _ensure_manager(db, manager_id)
    row = (await db.execute(
        text("SELECT hotel_id, max_competitors FROM user_hotel_assignments "
             "WHERE user_id::text = :id AND is_active = TRUE"),
        {"id": str(manager_id)})).mappings().first()
    if row is None:
        raise BadRequestError("Manager has no active hotel assignment; assign a hotel first")
    own_hotel_id, cap = row["hotel_id"], row["max_competitors"]

    hotel_ids = body.hotel_ids
    if len(hotel_ids) != len(set(hotel_ids)):
        raise BadRequestError("Duplicate hotels in selection")
    if len(hotel_ids) > cap:
        raise BadRequestError(f"At most {cap} competitors allowed")
    if own_hotel_id in hotel_ids:
        raise BadRequestError("A manager cannot select their own hotel as a competitor")
    for hid in hotel_ids:
        active = await db.scalar(
            text("SELECT 1 FROM platform_hotels WHERE id = :h AND is_active = TRUE"),
            {"h": hid})
        if not active:
            raise BadRequestError(f"Hotel {hid} is not an active platform hotel")

    await db.execute(
        text("DELETE FROM user_competitor_selections WHERE user_id::text = :id"),
        {"id": str(manager_id)})
    for order, hid in enumerate(hotel_ids, start=1):
        await db.execute(
            text("""INSERT INTO user_competitor_selections (user_id, hotel_id, display_order)
                    VALUES (CAST(:uid AS uuid), :h, :o)"""),
            {"uid": str(manager_id), "h": hid, "o": order})
    await db.commit()
    return await get_selection(db, manager_id)
```

- [ ] **Step 4: Router** — In `backend/routers/admin/competitors.py`: extend the
schema import to include `CompetitorSelectionUpdate`, extend the service import to
include `set_selection`, and ADD:

```python
@router.put("/{manager_id}/competitors", response_model=CompetitorSelectionResponse)
async def competitors_set(
    manager_id: str,
    body: CompetitorSelectionUpdate,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await set_selection(db, manager_id, body)
    return CompetitorSelectionResponse(data=rows, count=len(rows))
```

- [ ] **Step 5: Run tests** — `pytest tests/test_admin_competitors.py -v` (7 tests),
then `pytest -q` (expect 53 + 4 = 57).

- [ ] **Step 6: Commit**

```bash
git add backend/services/admin_competitors.py backend/routers/admin/competitors.py backend/tests/test_admin_competitors.py
git commit -m "feat(api): admin competitor selection PUT (validated replace)"
```

---

### Task 3: Monitoring — summary (Mongo total + scrape_runs aggregates)

**Files:**
- Modify: `backend/core/dependencies.py` (add `get_hotel_prices_total`)
- Modify: `backend/main.py` (close Mongo client on shutdown)
- Modify: `backend/tests/conftest.py` (seed `scrape_runs` sample)
- Create: `backend/schemas/admin_monitoring.py`
- Create: `backend/services/admin_monitoring.py`
- Create: `backend/routers/admin/monitoring.py`
- Test: `backend/tests/test_admin_monitoring.py`

- [ ] **Step 1: Seed `scrape_runs` sample in conftest** — In
`backend/tests/conftest.py` `setup_test_db`, after the existing seeds, add:

```python
        # scrape_runs sample (monitoring + alerts). 3 finished + 1 failed.
        await conn.execute(text("""
            INSERT INTO scrape_runs
              (run_ts, log_filename, source, spiders_count, items_total, errors_total, duration_s, status)
            VALUES
              ('2026-06-01 10:00+00','run_2026-06-01_10-00.log','promohotel',200,25000,40000,700,'finished'),
              ('2026-06-01 15:00+00','run_2026-06-01_15-00.log','promohotel',200,24000,41000,710,'finished'),
              ('2026-06-02 10:00+00','run_2026-06-02_10-00.log','promohotel',200, 3000,42000,300,'finished'),
              ('2026-06-02 15:00+00','run_2026-06-02_15-00.log', NULL,         0,    0,    0, NULL,'failed')
        """))
```

- [ ] **Step 2: Write the failing test** — Create `backend/tests/test_admin_monitoring.py`:

```python
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
```

- [ ] **Step 3: Run test → FAIL** (`pytest tests/test_admin_monitoring.py -v`).

- [ ] **Step 4: Add the Mongo-total dependency** — In `backend/core/dependencies.py`,
add (near the other dependencies):

```python
from db.mongo import get_mongo_db, count_hotel_prices

async def get_hotel_prices_total() -> int | None:
    return await count_hotel_prices(get_mongo_db())
```

- [ ] **Step 5: Schema** — Create `backend/schemas/admin_monitoring.py`:

```python
from __future__ import annotations
from pydantic import BaseModel


class MonitoringSummary(BaseModel):
    total_rows: int | None
    logged_window_items: int
    runs_count: int
    finished_runs: int
    failed_runs: int
    latest_scrape_at: str | None
    last_run_status: str | None
    last_run_items: int | None
    hotels_scraped_distinct: int
```

- [ ] **Step 6: Service** — Create `backend/services/admin_monitoring.py`:

```python
from __future__ import annotations
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from schemas.admin_monitoring import MonitoringSummary

_SUMMARY_SQL = text("""
    SELECT
        COALESCE(SUM(items_total), 0)                              AS logged_window_items,
        COUNT(*)                                                   AS runs_count,
        COUNT(*) FILTER (WHERE status = 'finished')                AS finished_runs,
        COUNT(*) FILTER (WHERE status <> 'finished')               AS failed_runs,
        MAX(run_ts)::text                                          AS latest_scrape_at
    FROM scrape_runs
""")

_LAST_RUN_SQL = text("""
    SELECT status, items_total
    FROM scrape_runs ORDER BY run_ts DESC LIMIT 1
""")


async def build_summary(db: AsyncSession, total_rows: int | None) -> MonitoringSummary:
    agg = (await db.execute(_SUMMARY_SQL)).mappings().one()
    last = (await db.execute(_LAST_RUN_SQL)).mappings().first()
    hotels = await db.scalar(
        text("SELECT COUNT(DISTINCT hotel_name_normalized) FROM hotel_features"))
    return MonitoringSummary(
        total_rows=total_rows,
        logged_window_items=int(agg["logged_window_items"]),
        runs_count=int(agg["runs_count"]),
        finished_runs=int(agg["finished_runs"]),
        failed_runs=int(agg["failed_runs"]),
        latest_scrape_at=agg["latest_scrape_at"],
        last_run_status=last["status"] if last else None,
        last_run_items=last["items_total"] if last else None,
        hotels_scraped_distinct=int(hotels or 0),
    )
```

- [ ] **Step 7: Router + mount** — Create `backend/routers/admin/monitoring.py`:

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_admin, get_hotel_prices_total
from db.models import User
from schemas.admin_monitoring import MonitoringSummary
from services.admin_monitoring import build_summary

router = APIRouter(prefix="/admin/monitoring", tags=["admin"])


@router.get("/summary", response_model=MonitoringSummary)
async def monitoring_summary(
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
    total_rows: int | None = Depends(get_hotel_prices_total),
):
    return await build_summary(db, total_rows)
```

In `backend/main.py`, add the mount and close Mongo on shutdown. Add the router
include:

```python
from routers.admin import monitoring as admin_monitoring
app.include_router(admin_monitoring.router)
```

And in the `lifespan` function, after the `yield` (shutdown section), add:

```python
    from db.mongo import close_mongo_client
    close_mongo_client()
```

- [ ] **Step 8: Run tests** — `pytest tests/test_admin_monitoring.py -v` (3 tests),
then `pytest -q` (expect 57 + 3 = 60).

- [ ] **Step 9: Commit**

```bash
git add backend/core/dependencies.py backend/main.py backend/tests/conftest.py backend/schemas/admin_monitoring.py backend/services/admin_monitoring.py backend/routers/admin/monitoring.py backend/tests/test_admin_monitoring.py
git commit -m "feat(api): admin monitoring summary (Mongo total + scrape_runs aggregates)"
```

---

### Task 4: Monitoring — runs + daily

**Files:**
- Modify: `backend/schemas/admin_monitoring.py`
- Modify: `backend/services/admin_monitoring.py`
- Modify: `backend/routers/admin/monitoring.py`
- Test: `backend/tests/test_admin_monitoring.py` (extend)

- [ ] **Step 1: Write the failing tests** — APPEND to `backend/tests/test_admin_monitoring.py`:

```python
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
```

- [ ] **Step 2: Run test → FAIL** (`pytest tests/test_admin_monitoring.py -k "runs or daily" -v`).

- [ ] **Step 3: Schema** — In `backend/schemas/admin_monitoring.py`, ADD:

```python
from schemas.common import DataResponse


class ScrapeRunRow(BaseModel):
    run_ts: str
    log_filename: str
    source: str | None
    items_total: int
    errors_total: int
    duration_s: float | None
    status: str


class DailyRow(BaseModel):
    day: str
    items_total: int
    runs: int


ScrapeRunListResponse = DataResponse[ScrapeRunRow]
DailyResponse = DataResponse[DailyRow]
```

- [ ] **Step 4: Service** — In `backend/services/admin_monitoring.py`: extend the
schema import to include `ScrapeRunRow, DailyRow`, and ADD:

```python
_RUNS_SQL = text("""
    SELECT run_ts::text AS run_ts, log_filename, source,
           items_total, errors_total, duration_s, status
    FROM scrape_runs
    ORDER BY run_ts DESC
    LIMIT :limit
""")

_DAILY_SQL = text("""
    SELECT to_char(run_ts, 'YYYY-MM-DD') AS day,
           COALESCE(SUM(items_total), 0) AS items_total,
           COUNT(*)                      AS runs
    FROM scrape_runs
    WHERE run_ts >= now() - make_interval(days => :days)
    GROUP BY 1
    ORDER BY 1
""")


async def list_runs(db: AsyncSession, limit: int) -> list[ScrapeRunRow]:
    rows = (await db.execute(_RUNS_SQL, {"limit": limit})).mappings().fetchall()
    return [ScrapeRunRow(**dict(r)) for r in rows]


async def daily_rollup(db: AsyncSession, days: int) -> list[DailyRow]:
    rows = (await db.execute(_DAILY_SQL, {"days": days})).mappings().fetchall()
    return [DailyRow(day=r["day"], items_total=int(r["items_total"]), runs=int(r["runs"]))
            for r in rows]
```

- [ ] **Step 5: Router** — In `backend/routers/admin/monitoring.py`: extend the schema
import to include `ScrapeRunListResponse, DailyResponse`, extend the service import
to include `list_runs, daily_rollup`, and ADD:

```python
@router.get("/runs", response_model=ScrapeRunListResponse)
async def monitoring_runs(
    limit: int = 50,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await list_runs(db, limit)
    return ScrapeRunListResponse(data=rows, count=len(rows))


@router.get("/daily", response_model=DailyResponse)
async def monitoring_daily(
    days: int = 30,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await daily_rollup(db, days)
    return DailyResponse(data=rows, count=len(rows))
```

- [ ] **Step 6: Run tests** — `pytest tests/test_admin_monitoring.py -v` (5 tests),
then `pytest -q` (expect 60 + 2 = 62).

- [ ] **Step 7: Commit**

```bash
git add backend/schemas/admin_monitoring.py backend/services/admin_monitoring.py backend/routers/admin/monitoring.py backend/tests/test_admin_monitoring.py
git commit -m "feat(api): admin monitoring runs + daily endpoints"
```

---

### Task 5: Collection Alerts (derived from scrape_runs)

**Files:**
- Create: `backend/schemas/admin_alert.py`
- Create: `backend/services/admin_alerts.py`
- Create: `backend/routers/admin/alerts.py`
- Modify: `backend/main.py`
- Test: `backend/tests/test_admin_alerts.py`

Alert types (read-only, derived from `scrape_runs`): **failed_run** (status ≠
'finished'); **low_volume** (a finished run whose `items_total` is below 50% of the
median `items_total` of finished runs — the report's "significantly fewer records"
NFR). Chronological (newest first). (Missing-scheduled and error-rate-baseline are
deferred — noted in spec §7.2.)

- [ ] **Step 1: Write the failing test** — Create `backend/tests/test_admin_alerts.py`:

```python
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
```

- [ ] **Step 2: Run test → FAIL** (`pytest tests/test_admin_alerts.py -v`).

- [ ] **Step 3: Schema** — Create `backend/schemas/admin_alert.py`:

```python
from __future__ import annotations
from pydantic import BaseModel
from schemas.common import DataResponse


class Alert(BaseModel):
    type: str
    severity: str
    message: str
    run_ts: str
    log_filename: str


AlertListResponse = DataResponse[Alert]
```

- [ ] **Step 4: Service** — Create `backend/services/admin_alerts.py`:

```python
from __future__ import annotations
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from schemas.admin_alert import Alert

# One row per run with: the run fields + the median items_total over FINISHED runs.
_RUNS_SQL = text("""
    SELECT run_ts::text AS run_ts, log_filename, items_total, errors_total, status,
           (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY items_total)
            FROM scrape_runs WHERE status = 'finished') AS median_items
    FROM scrape_runs
    ORDER BY run_ts DESC
""")


async def list_alerts(db: AsyncSession) -> list[Alert]:
    rows = (await db.execute(_RUNS_SQL)).mappings().fetchall()
    alerts: list[Alert] = []
    for r in rows:
        if r["status"] != "finished":
            alerts.append(Alert(
                type="failed_run", severity="error",
                message=f"Run {r['log_filename']} did not finish (status={r['status']}).",
                run_ts=r["run_ts"], log_filename=r["log_filename"]))
            continue
        median = r["median_items"]
        if median and r["items_total"] < 0.5 * float(median):
            alerts.append(Alert(
                type="low_volume", severity="warning",
                message=(f"Run {r['log_filename']} collected {r['items_total']} rows, "
                         f"well below the median ({int(median)})."),
                run_ts=r["run_ts"], log_filename=r["log_filename"]))
    return alerts
```

- [ ] **Step 5: Router + mount** — Create `backend/routers/admin/alerts.py`:

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_admin
from db.models import User
from schemas.admin_alert import AlertListResponse
from services.admin_alerts import list_alerts

router = APIRouter(prefix="/admin/alerts", tags=["admin"])


@router.get("", response_model=AlertListResponse)
async def alerts_list(
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await list_alerts(db)
    return AlertListResponse(data=rows, count=len(rows))
```

In `backend/main.py`, add:

```python
from routers.admin import alerts as admin_alerts
app.include_router(admin_alerts.router)
```

- [ ] **Step 6: Run tests** — `pytest tests/test_admin_alerts.py -v` (2 tests),
then `pytest -q` (expect 62 + 2 = 64).

- [ ] **Step 7: Commit**

```bash
git add backend/schemas/admin_alert.py backend/services/admin_alerts.py backend/routers/admin/alerts.py backend/main.py backend/tests/test_admin_alerts.py
git commit -m "feat(api): admin collection alerts (failed + low-volume, derived)"
```

---

## Self-review

**Spec coverage (vs §5.4 competitors/monitoring/alerts, §7, D11):**
- Competitor read (selection + selectable, own-hotel excluded) → Task 1. ✓
- Competitor PUT replace, admin-only, validates not-self / cap / active / dups / requires-assignment (D11) → Task 2. ✓
- Monitoring summary: Mongo total (overridable dep) + scrape_runs aggregates + distinct hotels → Task 3. ✓
- Monitoring runs + daily rollup → Task 4. ✓
- Alerts: failed_run + low_volume derived from scrape_runs → Task 5. ✓
- Mongo client closed on shutdown (lifespan) → Task 3. ✓
- Deferred: missing-scheduled + error-rate-baseline alerts (noted), frontend (Plan 5).

**Deviation from spec §5.4 monitoring fields:** the summary returns a concrete,
testable field set (`total_rows, logged_window_items, runs_count, finished_runs,
failed_runs, latest_scrape_at, last_run_status, last_run_items,
hotels_scraped_distinct`) instead of the looser draft list (dropped the
dynamic/fuzzy `rows_added_today`/`error_rate`). Rationale: deterministic to test;
the daily endpoint covers per-day volume.

**Triggers note:** the test DB has no DB triggers (ORM-built), so the competitor
service validates not-self + cap in Python (Task 2). In production those are also
enforced by the migration's `check_competitor_not_self` / `check_competitor_count`
triggers as a backstop — the service runs first and returns clean 400s.

**Placeholder scan:** none. **Type consistency:** every `*Row` schema matches its
SQL columns; UUID params use `CAST(:uid AS uuid)` (inserts) / `::text` (reads);
`user_competitor_selections` INSERT sets `display_order` and omits the autoincrement
`id`; the Mongo total flows through `get_hotel_prices_total` (overridden in tests,
real Mongo in prod, `None` when down).
