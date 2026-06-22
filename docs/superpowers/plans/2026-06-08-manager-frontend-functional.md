# Manager Frontend Functional — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every RevWay manager page fully functional against the live FastAPI backend — wire the mock Dashboard and static Settings to real endpoints, and make the recommendation Accept/Dismiss actions persist.

**Architecture:** Three backend additions — `GET/PATCH /manager/me` (profile + a `preferences` JSONB column on `users`), `GET /manager/dashboard` (composes the existing calendar/competitor/recommendation/anomaly services into KPIs + panels, one request), and recommendation-decision persistence (new `manager_recommendation_decisions` table, POST single + bulk, status merged into the existing recommendations read). Frontend then wires Dashboard, Settings, Recommendations actions, and an Alerts deep-link.

**Tech Stack:** FastAPI · SQLAlchemy async + asyncpg · PostgreSQL (raw migrations) · Pydantic v2 · pytest/pytest-asyncio/httpx · Angular 19 standalone + signals.

**Spec:** `docs/superpowers/specs/2026-06-08-manager-frontend-functional-design.md`

---

## Conventions (read before any task)

- **Base branch `main`; isolated worktree on `feat/manager-functional`.** Do NOT
  work on `feat/forecaster-bakeoff`.
- Backend layering (`backend/CLAUDE.md`): routers thin → services own SQL/mapping
  → schemas are the wire contract. Raw `text()` SQL + `.mappings()` live in
  services. Auth/ownership centralised in `core/dependencies.py`.
- Use `ForbiddenError`, `NotFoundError`, `BadRequestError`, `ConflictError` from
  `core/exceptions.py` so the frontend surfaces exact messages.
- Manager endpoints depend on `get_current_manager` (role + active assignment).
- Frontend (`frontend/CLAUDE.md`): standalone components, signals, `@if/@for`,
  selector prefix `rw-`, Postgres column names verbatim, taxonomy values from
  `core/data/mock.ts`, no new libraries.
- Run backend tests from `backend/`: `./.venv/Scripts/python.exe -m pytest`.
- Apply SQL migrations with the venv Python (no `psql` on this machine) — see Task 2.
- Commit after every green step. No `Co-Authored-By: Claude` trailer (project rule).

---

## File Structure

**Backend (create):**
- `database/postgres/migrations/006_add_users_preferences.sql`
- `database/postgres/migrations/007_create_recommendation_decisions.sql`
- `backend/schemas/profile.py` — `ManagerProfile`, `HotelBrief`, `ProfileUpdate`.
- `backend/schemas/dashboard.py` — `DashboardKpis`, `PriceSeriesPoint`, `DashboardResponse`.
- `backend/schemas/decision.py` — `DecisionIn`, `DecisionBulkIn`, `DecisionRow`.
- `backend/services/profile_service.py` — `get_profile`, `update_profile`.
- `backend/services/dashboard_service.py` — `get_dashboard`.
- `backend/services/decision_service.py` — `set_decision`, `set_decisions_bulk`, `get_decision_map`.
- `backend/routers/profile.py`, `backend/routers/dashboard.py`, `backend/routers/recommendation_decisions.py`
- `backend/tests/test_profile.py`, `test_dashboard.py`, `test_recommendation_decisions.py`

**Backend (modify):**
- `backend/db/models.py` — add `preferences` to `User`; add `ManagerRecommendationDecision`.
- `backend/main.py` — include the 3 new routers.
- `backend/schemas/recommendation.py` — add `decision_status` to `RecommendationRow`.
- `backend/services/recommendation_service.py` — merge decision status into rows.
- `backend/tests/conftest.py` — add `preferences` + decisions table to the test schema.

**Frontend (modify):**
- `frontend/src/app/core/api/dto.ts` — add profile/dashboard/decision DTOs; extend `RecommendationDto`.
- `frontend/src/app/core/api/api.service.ts` — add the new endpoint methods.
- `frontend/src/app/core/api/adapters.ts` — map `decision_status` into `Recommendation.status`.
- `frontend/src/app/features/manager/dashboard/manager-dashboard.component.ts` — mock → live.
- `frontend/src/app/features/manager/settings/manager-settings.component.ts` — static → live.
- `frontend/src/app/features/manager/recommendations/manager-recommendations.component.ts` — wire actions.
- `frontend/src/app/features/manager/alerts/manager-alerts.component.ts` — Investigate deep-link.

---

## Task 1: Worktree + runtime deps + commit the spec

**Files:** none edited; workspace setup only.

- [ ] **Step 1: Create the worktree off `main`**

From the main checkout `C:\Users\ASUS\Desktop\PFE\revway`:

```bash
git worktree add -b feat/manager-functional ../revway-manager main
```

(If the `superpowers:using-git-worktrees` skill is driving execution, let it create
the worktree on branch `feat/manager-functional` from `main` instead.)

- [ ] **Step 2: Provide gitignored runtime deps in the worktree**

```bash
# .env (small — copy)
cp backend/.env ../revway-manager/backend/.env
# node_modules (large — junction instead of reinstall; no admin needed on Win)
cmd /c mklink /J ..\\revway-manager\\frontend\\node_modules ..\\revway\\frontend\\node_modules
# ML model dir (MODEL_DIR=../ml/models/...): junction the whole trained-model folder
cmd /c mklink /J ..\\revway-manager\\ml\\models\\forecasting ..\\revway\\ml\\models\\forecasting
```

Expected: `../revway-manager/backend/.env` exists; the two junctions resolve.
(If `mklink /J` is unavailable, `npm install` in the worktree frontend and copy the
model dir instead.)

- [ ] **Step 3: Copy the spec into the worktree and commit**

The spec currently lives only in the `feat/forecaster-bakeoff` working tree. Copy
both design + plan docs into the worktree:

```bash
mkdir -p ../revway-manager/docs/superpowers/specs ../revway-manager/docs/superpowers/plans
cp docs/superpowers/specs/2026-06-08-manager-frontend-functional-design.md ../revway-manager/docs/superpowers/specs/
cp docs/superpowers/plans/2026-06-08-manager-frontend-functional.md ../revway-manager/docs/superpowers/plans/
cd ../revway-manager
git add docs/superpowers
git commit -m "docs(manager): functional-frontend design spec + implementation plan"
```

Expected: clean commit on `feat/manager-functional`. **All subsequent tasks run in
`../revway-manager`.**

- [ ] **Step 4: Smoke the stack boots**

```bash
cd backend && ./.venv/Scripts/python.exe -m pytest -q 2>&1 | tail -5
```

Expected: existing suite passes (baseline green before changes).

---

## Task 2: Migration 006 — `users.preferences` JSONB

**Files:**
- Create: `database/postgres/migrations/006_add_users_preferences.sql`
- Modify: `backend/db/models.py`

- [ ] **Step 1: Write the migration SQL**

`database/postgres/migrations/006_add_users_preferences.sql`:

```sql
-- =========================================================
-- 006_add_users_preferences.sql
-- =========================================================
-- Per-user UI preferences (language + alert toggles) for the manager Settings
-- page. Stored as JSONB so Settings is functional without a per-setting table.
-- Honest scope: preferences only — nothing consumes them to send notifications.

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS preferences JSONB NOT NULL DEFAULT '{}'::jsonb;

INSERT INTO schema_migrations (version, description)
VALUES ('006', 'users.preferences JSONB: manager language + alert toggles')
ON CONFLICT (version) DO NOTHING;
```

- [ ] **Step 2: Apply the migration to the `revway` DB**

Run from `backend/`:

```bash
./.venv/Scripts/python.exe -c "
import asyncio, pathlib
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
sql = pathlib.Path('../database/postgres/migrations/006_add_users_preferences.sql').read_text()
async def main():
    e = create_async_engine('postgresql+asyncpg://revway:REDACTED@localhost:5432/revway')
    async with e.begin() as c:
        for stmt in [s for s in sql.split(';') if s.strip()]:
            await c.execute(text(stmt))
    await e.dispose()
asyncio.run(main())
print('006 applied')
"
```

Expected: prints `006 applied`.

- [ ] **Step 3: Add `preferences` to the `User` ORM model**

In `backend/db/models.py`, add the import and column. After the existing
`last_login_at` line in `class User`:

```python
    last_login_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime(timezone=True))
    preferences: Mapped[dict] = mapped_column(JSONB, nullable=False, default=dict)
```

And add to the imports at the top:

```python
from sqlalchemy.dialects.postgresql import UUID, JSONB
```

- [ ] **Step 4: Verify the column exists**

```bash
./.venv/Scripts/python.exe -c "
import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
async def main():
    e=create_async_engine('postgresql+asyncpg://revway:REDACTED@localhost:5432/revway')
    async with e.connect() as c:
        r=(await c.execute(text(\"SELECT column_name FROM information_schema.columns WHERE table_name='users' AND column_name='preferences'\"))).fetchall()
        print(r)
    await e.dispose()
asyncio.run(main())
"
```

Expected: `[('preferences',)]`.

- [ ] **Step 5: Commit**

```bash
git add database/postgres/migrations/006_add_users_preferences.sql backend/db/models.py
git commit -m "feat(db): users.preferences JSONB (migration 006)"
```

---

## Task 3: Migration 007 — recommendation decisions table

**Files:**
- Create: `database/postgres/migrations/007_create_recommendation_decisions.sql`
- Modify: `backend/db/models.py`

- [ ] **Step 1: Write the migration SQL**

`database/postgres/migrations/007_create_recommendation_decisions.sql`:

```sql
-- =========================================================
-- 007_create_recommendation_decisions.sql
-- =========================================================
-- Persists a manager's Accept/Dismiss decision on a recommendation. Decisions
-- are decision-support tracking only (no PMS / price execution). Key matches the
-- fields the recommendation row exposes and the frontend's recommendation id
-- (check_in + nights + adults + boarding_canonical).

CREATE TABLE IF NOT EXISTS manager_recommendation_decisions (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    hotel_id INTEGER NOT NULL REFERENCES platform_hotels(id) ON DELETE CASCADE,
    check_in DATE NOT NULL,
    nights SMALLINT NOT NULL,
    adults SMALLINT NOT NULL,
    boarding_canonical TEXT NOT NULL,
    recommended_price_tnd NUMERIC,
    status TEXT NOT NULL CHECK (status IN ('accepted', 'dismissed')),
    decided_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, check_in, nights, adults, boarding_canonical)
);

CREATE INDEX IF NOT EXISTS idx_mrd_user_checkin
    ON manager_recommendation_decisions (user_id, check_in);

CREATE TRIGGER trg_mrd_updated_at
    BEFORE UPDATE ON manager_recommendation_decisions
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

INSERT INTO schema_migrations (version, description)
VALUES ('007', 'manager_recommendation_decisions: accept/dismiss tracking')
ON CONFLICT (version) DO NOTHING;
```

- [ ] **Step 2: Apply it** (same runner as Task 2, Step 2, with the 007 path)

```bash
./.venv/Scripts/python.exe -c "
import asyncio, pathlib
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
sql = pathlib.Path('../database/postgres/migrations/007_create_recommendation_decisions.sql').read_text()
async def main():
    e = create_async_engine('postgresql+asyncpg://revway:REDACTED@localhost:5432/revway')
    async with e.begin() as c:
        for stmt in [s for s in sql.split(';') if s.strip()]:
            await c.execute(text(stmt))
    await e.dispose()
asyncio.run(main())
print('007 applied')
"
```

Expected: prints `007 applied`. (NOTE: the trigger body contains no `;`-split
hazard because `set_updated_at()` already exists from migration 001; the simple
split-on-`;` runner is safe here.)

- [ ] **Step 3: Add the ORM model**

Append to `backend/db/models.py`:

```python
class ManagerRecommendationDecision(Base):
    __tablename__ = "manager_recommendation_decisions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))
    hotel_id: Mapped[int] = mapped_column(Integer, ForeignKey("platform_hotels.id"))
    check_in: Mapped[datetime.date] = mapped_column(nullable=False)
    nights: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    adults: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    boarding_canonical: Mapped[str] = mapped_column(String, nullable=False)
    recommended_price_tnd: Mapped[Optional[float]] = mapped_column()
    status: Mapped[str] = mapped_column(String, nullable=False)
    decided_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
```

Ensure `datetime` is imported (it is) and `Date` mapping works via
`Mapped[datetime.date]` (SQLAlchemy infers `DATE`).

- [ ] **Step 4: Verify the table**

```bash
./.venv/Scripts/python.exe -c "
import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
async def main():
    e=create_async_engine('postgresql+asyncpg://revway:REDACTED@localhost:5432/revway')
    async with e.connect() as c:
        print((await c.execute(text(\"SELECT to_regclass('public.manager_recommendation_decisions')\"))).fetchall())
    await e.dispose()
asyncio.run(main())
"
```

Expected: `[('manager_recommendation_decisions',)]`.

- [ ] **Step 5: Commit**

```bash
git add database/postgres/migrations/007_create_recommendation_decisions.sql backend/db/models.py
git commit -m "feat(db): manager_recommendation_decisions table (migration 007)"
```

---

## Task 4: Update test conftest schema (preferences + decisions)

The test DB schema is built by `Base.metadata.create_all` (ORM), so `preferences`
and the decisions table come for free once Tasks 2–3 land — EXCEPT the
`set_updated_at` trigger and `preferences` default, which the ORM does create the
column for. Verify the suite still builds the schema.

**Files:** `backend/tests/conftest.py` (only if a seed needs preferences).

- [ ] **Step 1: Run the existing suite to confirm schema still builds**

```bash
cd backend && ./.venv/Scripts/python.exe -m pytest -q 2>&1 | tail -8
```

Expected: PASS (ORM `create_all` now includes the new column + table; no seed
references them yet).

- [ ] **Step 2: Commit (only if conftest changed)**

```bash
git add backend/tests/conftest.py
git commit -m "test: include preferences + decisions in test schema"
```

(If conftest needed no change, skip this commit.)

---

## Task 5: `GET /manager/me`

**Files:**
- Create: `backend/schemas/profile.py`, `backend/services/profile_service.py`, `backend/routers/profile.py`, `backend/tests/test_profile.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_profile.py`:

```python
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
```

- [ ] **Step 2: Run it — expect failure (404, route missing)**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_profile.py -q 2>&1 | tail -10
```

Expected: FAIL (404 / route not found).

- [ ] **Step 3: Write the schema**

`backend/schemas/profile.py`:

```python
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel

class HotelBrief(BaseModel):
    id: int
    hotel_name_display: str
    city_name: str
    stars_int: int | None

class ManagerProfile(BaseModel):
    full_name: str | None
    email: str
    role: str
    preferences: dict
    hotel: HotelBrief | None
    competitor_count: int
    max_competitors: int
    last_login_at: str | None

class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    preferences: Optional[dict] = None
```

- [ ] **Step 4: Write the service**

`backend/services/profile_service.py`:

```python
from __future__ import annotations
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from db.models import User
from schemas.profile import ManagerProfile, HotelBrief, ProfileUpdate

async def get_profile(user: User, db: AsyncSession) -> ManagerProfile:
    row = (await db.execute(text("""
        SELECT ph.id AS hotel_id,
               ph.hotel_name_display,
               c.name_display AS city_name,
               ph.stars_int,
               a.max_competitors,
               (SELECT COUNT(*) FROM user_competitor_selections ucs
                 WHERE ucs.user_id = :uid) AS competitor_count
        FROM user_hotel_assignments a
        JOIN platform_hotels ph ON ph.id = a.hotel_id
        JOIN cities c           ON c.id  = ph.city_id
        WHERE a.user_id = :uid AND a.is_active = TRUE
    """), {"uid": str(user.id)})).mappings().first()

    hotel = None
    max_comp = 4
    comp_count = 0
    if row is not None:
        hotel = HotelBrief(
            id=row["hotel_id"],
            hotel_name_display=row["hotel_name_display"],
            city_name=row["city_name"],
            stars_int=row["stars_int"],
        )
        max_comp = row["max_competitors"]
        comp_count = row["competitor_count"]

    return ManagerProfile(
        full_name=user.full_name,
        email=str(user.email),
        role=user.role,
        preferences=user.preferences or {},
        hotel=hotel,
        competitor_count=comp_count,
        max_competitors=max_comp,
        last_login_at=user.last_login_at.isoformat() if user.last_login_at else None,
    )
```

- [ ] **Step 5: Write the router**

`backend/routers/profile.py`:

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_manager
from db.models import User
from schemas.profile import ManagerProfile
from services.profile_service import get_profile

router = APIRouter(prefix="/manager", tags=["manager"])

@router.get("/me", response_model=ManagerProfile)
async def me_endpoint(
    user: User = Depends(get_current_manager),
    db: AsyncSession = Depends(get_db),
):
    return await get_profile(user, db)
```

- [ ] **Step 6: Register the router in `backend/main.py`**

Change the import + include block at the bottom:

```python
from routers import auth, calendar, competitors, recommendations, anomalies, profile
app.include_router(auth.router)
app.include_router(calendar.router)
app.include_router(competitors.router)
app.include_router(recommendations.router)
app.include_router(anomalies.router)
app.include_router(profile.router)
```

- [ ] **Step 7: Run the tests — expect PASS**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_profile.py -q 2>&1 | tail -10
```

Expected: 2 passed.

- [ ] **Step 8: Commit**

```bash
git add backend/schemas/profile.py backend/services/profile_service.py backend/routers/profile.py backend/main.py backend/tests/test_profile.py
git commit -m "feat(api): GET /manager/me profile endpoint"
```

---

## Task 6: `PATCH /manager/me`

**Files:** Modify `backend/services/profile_service.py`, `backend/routers/profile.py`, `backend/tests/test_profile.py`.

- [ ] **Step 1: Add the failing test** (append to `test_profile.py`)

```python
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
```

- [ ] **Step 2: Run — expect failure (405/404 on PATCH)**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_profile.py -k patch -q 2>&1 | tail -10
```

Expected: FAIL.

- [ ] **Step 3: Add the service function** (append to `profile_service.py`)

```python
from sqlalchemy import update as sa_update

async def update_profile(user: User, patch: ProfileUpdate, db: AsyncSession) -> ManagerProfile:
    values: dict = {}
    if patch.full_name is not None:
        values["full_name"] = patch.full_name
    if patch.preferences is not None:
        values["preferences"] = patch.preferences
    if values:
        await db.execute(sa_update(User).where(User.id == user.id).values(**values))
        await db.commit()
        await db.refresh(user)
    return await get_profile(user, db)
```

- [ ] **Step 4: Add the route** (append to `routers/profile.py`)

```python
from schemas.profile import ProfileUpdate
from services.profile_service import update_profile

@router.patch("/me", response_model=ManagerProfile)
async def update_me_endpoint(
    patch: ProfileUpdate,
    user: User = Depends(get_current_manager),
    db: AsyncSession = Depends(get_db),
):
    return await update_profile(user, patch, db)
```

- [ ] **Step 5: Run — expect PASS**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_profile.py -q 2>&1 | tail -10
```

Expected: all profile tests pass.

- [ ] **Step 6: Commit**

```bash
git add backend/services/profile_service.py backend/routers/profile.py backend/tests/test_profile.py
git commit -m "feat(api): PATCH /manager/me (full_name + preferences)"
```

---

## Task 7: Recommendation decisions — service + endpoints

**Files:**
- Create: `backend/schemas/decision.py`, `backend/services/decision_service.py`, `backend/routers/recommendation_decisions.py`, `backend/tests/test_recommendation_decisions.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_recommendation_decisions.py`:

```python
import pytest
from sqlalchemy import select
from db.models import User
from core.security import create_access_token

async def _manager_token(db_session) -> str:
    result = await db_session.execute(select(User).where(User.email == "manager@test.com"))
    user = result.scalar_one()
    return create_access_token(str(user.id), hotel_id=1, role="manager")

def _payload(status="accepted"):
    return {"check_in": "2026-07-01", "nights": 3, "adults": 2,
            "boarding_canonical": "BB", "recommended_price_tnd": 450.0, "status": status}

@pytest.mark.asyncio
async def test_decision_upsert_insert_then_update(client, db_session):
    token = await _manager_token(db_session)
    h = {"Authorization": f"Bearer {token}"}
    r1 = await client.post("/manager/recommendations/decision", headers=h, json=_payload("accepted"))
    assert r1.status_code == 200
    assert r1.json()["status"] == "accepted"
    # same key, new status -> update, not duplicate
    r2 = await client.post("/manager/recommendations/decision", headers=h, json=_payload("dismissed"))
    assert r2.status_code == 200
    assert r2.json()["status"] == "dismissed"

@pytest.mark.asyncio
async def test_decision_bad_status_rejected(client, db_session):
    token = await _manager_token(db_session)
    h = {"Authorization": f"Bearer {token}"}
    r = await client.post("/manager/recommendations/decision", headers=h, json=_payload("maybe"))
    assert r.status_code == 422

@pytest.mark.asyncio
async def test_decision_requires_auth(client):
    r = await client.post("/manager/recommendations/decision", json=_payload())
    assert r.status_code == 401

@pytest.mark.asyncio
async def test_decision_bulk(client, db_session):
    token = await _manager_token(db_session)
    h = {"Authorization": f"Bearer {token}"}
    body = {"status": "accepted", "items": [
        {"check_in": "2026-07-01", "nights": 3, "adults": 2, "boarding_canonical": "BB", "recommended_price_tnd": 450.0},
        {"check_in": "2026-07-02", "nights": 3, "adults": 2, "boarding_canonical": "BB", "recommended_price_tnd": 460.0},
    ]}
    r = await client.post("/manager/recommendations/decision/bulk", headers=h, json=body)
    assert r.status_code == 200
    assert r.json()["count"] == 2

@pytest.mark.asyncio
async def test_recommendations_include_decision_status(client, db_session):
    token = await _manager_token(db_session)
    h = {"Authorization": f"Bearer {token}"}
    # the mock recommender returns a row for check_in 2026-07-01 / 3n / BB / 2a
    await client.post("/manager/recommendations/decision", headers=h, json=_payload("accepted"))
    resp = await client.get("/manager/recommendations", headers=h)
    rows = resp.json()["data"]
    assert rows, "expected at least one recommendation row"
    match = [r for r in rows if r["check_in"] == "2026-07-01"]
    assert match and match[0]["decision_status"] == "accepted"
```

- [ ] **Step 2: Run — expect failure**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_recommendation_decisions.py -q 2>&1 | tail -12
```

Expected: FAIL (routes missing; `decision_status` missing).

- [ ] **Step 3: Write the schema**

`backend/schemas/decision.py`:

```python
from __future__ import annotations
from datetime import date
from typing import Literal, Optional
from pydantic import BaseModel

DecisionStatus = Literal["accepted", "dismissed"]

class DecisionKey(BaseModel):
    check_in: date
    nights: int
    adults: int
    boarding_canonical: str
    recommended_price_tnd: Optional[float] = None

class DecisionIn(DecisionKey):
    status: DecisionStatus

class DecisionBulkIn(BaseModel):
    status: DecisionStatus
    items: list[DecisionKey]

class DecisionRow(BaseModel):
    check_in: date
    nights: int
    adults: int
    boarding_canonical: str
    recommended_price_tnd: float | None
    status: DecisionStatus
```

- [ ] **Step 4: Write the service**

`backend/services/decision_service.py`:

```python
from __future__ import annotations
from datetime import date
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from db.models import User
from services.common import get_manager_hotel_name
from schemas.decision import DecisionIn, DecisionBulkIn, DecisionRow, DecisionStatus

# asyncpg + the UNIQUE key drive an UPSERT. hotel_id is resolved from the
# manager's active assignment (never trusted from the client).
async def _hotel_id(user: User, db: AsyncSession) -> int:
    row = (await db.execute(text("""
        SELECT hotel_id FROM user_hotel_assignments
        WHERE user_id = :uid AND is_active = TRUE
    """), {"uid": str(user.id)})).first()
    return int(row[0])

_UPSERT = text("""
    INSERT INTO manager_recommendation_decisions
        (user_id, hotel_id, check_in, nights, adults, boarding_canonical,
         recommended_price_tnd, status)
    VALUES (:uid, :hid, :check_in, :nights, :adults, :boarding, :price, :status)
    ON CONFLICT (user_id, check_in, nights, adults, boarding_canonical)
    DO UPDATE SET status = EXCLUDED.status,
                  recommended_price_tnd = EXCLUDED.recommended_price_tnd,
                  updated_at = now()
""")

async def set_decision(user: User, body: DecisionIn, db: AsyncSession) -> DecisionRow:
    hid = await _hotel_id(user, db)
    await db.execute(_UPSERT, {
        "uid": str(user.id), "hid": hid, "check_in": body.check_in,
        "nights": body.nights, "adults": body.adults,
        "boarding": body.boarding_canonical, "price": body.recommended_price_tnd,
        "status": body.status,
    })
    await db.commit()
    return DecisionRow(**body.model_dump())

async def set_decisions_bulk(user: User, body: DecisionBulkIn, db: AsyncSession) -> int:
    hid = await _hotel_id(user, db)
    for item in body.items:
        await db.execute(_UPSERT, {
            "uid": str(user.id), "hid": hid, "check_in": item.check_in,
            "nights": item.nights, "adults": item.adults,
            "boarding": item.boarding_canonical, "price": item.recommended_price_tnd,
            "status": body.status,
        })
    await db.commit()
    return len(body.items)

async def get_decision_map(user: User, db: AsyncSession) -> dict[tuple, str]:
    """(check_in, nights, adults, boarding_canonical) -> status, for merging
    into recommendation reads."""
    rows = (await db.execute(text("""
        SELECT check_in, nights, adults, boarding_canonical, status
        FROM manager_recommendation_decisions
        WHERE user_id = :uid
    """), {"uid": str(user.id)})).fetchall()
    return {(r[0], r[1], r[2], r[3]): r[4] for r in rows}
```

- [ ] **Step 5: Write the router**

`backend/routers/recommendation_decisions.py`:

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_manager
from db.models import User
from schemas.decision import DecisionIn, DecisionBulkIn, DecisionRow
from services.decision_service import set_decision, set_decisions_bulk

router = APIRouter(prefix="/manager", tags=["manager"])

@router.post("/recommendations/decision", response_model=DecisionRow)
async def decision_endpoint(
    body: DecisionIn,
    user: User = Depends(get_current_manager),
    db: AsyncSession = Depends(get_db),
):
    return await set_decision(user, body, db)

@router.post("/recommendations/decision/bulk")
async def decision_bulk_endpoint(
    body: DecisionBulkIn,
    user: User = Depends(get_current_manager),
    db: AsyncSession = Depends(get_db),
):
    count = await set_decisions_bulk(user, body, db)
    return {"count": count}
```

- [ ] **Step 6: Register router in `main.py`**

```python
from routers import (auth, calendar, competitors, recommendations, anomalies,
                     profile, recommendation_decisions)
...
app.include_router(profile.router)
app.include_router(recommendation_decisions.router)
```

- [ ] **Step 7: Merge `decision_status` into recommendations**

In `backend/schemas/recommendation.py`, add to `RecommendationRow`:

```python
    reasons: list[str]
    decision_status: str | None = None
```

In `backend/services/recommendation_service.py`, after building `out` and before
`return out`, merge the decision map:

```python
    from services.decision_service import get_decision_map
    decisions = await get_decision_map(user, db)
    for row in out:
        row.decision_status = decisions.get(
            (row.check_in, row.nights, row.adults, row.boarding_canonical)
        )
    return out
```

(Place the `import` at the top of the file instead if preferred; inline keeps the
diff local. Ensure `out` items are mutable `RecommendationRow` instances — they
are.)

- [ ] **Step 8: Run — expect PASS**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_recommendation_decisions.py tests/test_recommendations.py -q 2>&1 | tail -12
```

Expected: all pass (incl. the existing recommendation tests, now with the extra
nullable field).

- [ ] **Step 9: Commit**

```bash
git add backend/schemas/decision.py backend/services/decision_service.py backend/routers/recommendation_decisions.py backend/schemas/recommendation.py backend/services/recommendation_service.py backend/main.py backend/tests/test_recommendation_decisions.py
git commit -m "feat(api): persist recommendation accept/dismiss decisions + merge status"
```

---

## Task 8: `GET /manager/dashboard`

**Files:**
- Create: `backend/schemas/dashboard.py`, `backend/services/dashboard_service.py`, `backend/routers/dashboard.py`, `backend/tests/test_dashboard.py`
- Modify: `backend/main.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_dashboard.py`:

```python
import pytest
from sqlalchemy import select
from db.models import User
from core.security import create_access_token

async def _manager_token(db_session) -> str:
    result = await db_session.execute(select(User).where(User.email == "manager@test.com"))
    user = result.scalar_one()
    return create_access_token(str(user.id), hotel_id=1, role="manager")

@pytest.mark.asyncio
async def test_dashboard_shape(client, db_session):
    token = await _manager_token(db_session)
    resp = await client.get("/manager/dashboard", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    body = resp.json()
    for key in ("kpis", "price_series", "top_recommendations", "competitors", "recent_alerts"):
        assert key in body
    k = body["kpis"]
    for key in ("avg_listed_rate_tnd", "market_position_pct", "vs_competitor_pct",
                "opportunity_tnd", "opportunity_pct", "open_recommendations", "active_alerts"):
        assert key in k

@pytest.mark.asyncio
async def test_dashboard_requires_auth(client):
    resp = await client.get("/manager/dashboard")
    assert resp.status_code == 401
```

- [ ] **Step 2: Run — expect failure**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_dashboard.py -q 2>&1 | tail -10
```

Expected: FAIL.

- [ ] **Step 3: Write the schema**

`backend/schemas/dashboard.py`:

```python
from __future__ import annotations
from datetime import date
from pydantic import BaseModel
from schemas.recommendation import RecommendationRow
from schemas.competitor import CompetitorSummary
from schemas.anomaly import AnomalyRow

class DashboardKpis(BaseModel):
    avg_listed_rate_tnd: float | None
    market_position_pct: float | None
    vs_competitor_pct: float | None
    opportunity_tnd: float | None
    opportunity_pct: float | None
    open_recommendations: int
    active_alerts: int

class PriceSeriesPoint(BaseModel):
    check_in: date
    price_per_night: float
    peer_medium_median: float | None
    recommended_price_per_night: float

class DashboardResponse(BaseModel):
    kpis: DashboardKpis
    price_series: list[PriceSeriesPoint]
    top_recommendations: list[RecommendationRow]
    competitors: list[CompetitorSummary]
    recent_alerts: list[AnomalyRow]
```

(Confirm the competitor schema class name is `CompetitorSummary` — see
`backend/schemas/competitor.py`. If different, import the actual name.)

- [ ] **Step 4: Write the service (composes existing services)**

`backend/services/dashboard_service.py`:

```python
from __future__ import annotations
from datetime import date, timedelta
from statistics import mean
from sqlalchemy.ext.asyncio import AsyncSession
from db.models import User
from ml_store.store import MLStore
from services.calendar_service import get_calendar, get_calendar_options
from services.competitor_service import get_competitors
from services.recommendation_service import get_recommendations
from services.anomaly_service import get_anomalies
from schemas.dashboard import DashboardResponse, DashboardKpis, PriceSeriesPoint

def _safe_mean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None

async def get_dashboard(user: User, db: AsyncSession, ml_store: MLStore,
                        days: int = 30) -> DashboardResponse:
    today = date.today()
    to = today + timedelta(days=days)
    opts = await get_calendar_options(user, db)
    d = opts.default
    cfg = dict(nights=d.nights, adults=d.adults, room_base=d.room_base,
               room_view=d.room_view, room_tier=d.room_tier,
               room_occupancy=d.room_occupancy, boarding_canonical=d.boarding_canonical)

    cal = await get_calendar(user, db, ml_store, check_in_from=today, check_in_to=to, **cfg)
    competitors = await get_competitors(user, db)
    recs = await get_recommendations(user, db, ml_store, check_in_from=today, check_in_to=to, **cfg)
    alerts = await get_anomalies(user, db, ml_store, check_in_from=today, check_in_to=to, **cfg)

    # KPIs (per-night, consistent units)
    avg_price = _safe_mean([r.price_per_night for r in cal])
    avg_peer = _safe_mean([r.peer_medium_median for r in cal])
    comp_avg = _safe_mean([c.avg_price_per_night for c in competitors])
    avg_rec = _safe_mean([r.recommended_price_per_night for r in cal])

    market_pos = ((avg_price - avg_peer) / avg_peer * 100) if avg_price and avg_peer else None
    vs_comp = ((avg_price - comp_avg) / comp_avg * 100) if avg_price and comp_avg else None
    opp_tnd = (avg_rec - avg_price) if avg_rec is not None and avg_price is not None else None
    opp_pct = (opp_tnd / avg_price * 100) if opp_tnd is not None and avg_price else None

    open_recs = sum(1 for r in recs if r.direction != "hold" and r.decision_status is None)

    kpis = DashboardKpis(
        avg_listed_rate_tnd=avg_price,
        market_position_pct=market_pos,
        vs_competitor_pct=vs_comp,
        opportunity_tnd=opp_tnd,
        opportunity_pct=opp_pct,
        open_recommendations=open_recs,
        active_alerts=len(alerts),
    )

    series = [PriceSeriesPoint(
        check_in=r.check_in, price_per_night=r.price_per_night,
        peer_medium_median=r.peer_medium_median,
        recommended_price_per_night=r.recommended_price_per_night,
    ) for r in sorted(cal, key=lambda x: x.check_in)[:14]]

    top_recs = sorted(recs, key=lambda r: abs(r.delta_pct_vs_current), reverse=True)[:5]
    recent_alerts = sorted(alerts, key=lambda a: abs(a.anomaly_score), reverse=True)[:5]

    return DashboardResponse(
        kpis=kpis, price_series=series, top_recommendations=top_recs,
        competitors=competitors, recent_alerts=recent_alerts,
    )
```

(Verify the `get_anomalies` signature/keyword args in
`backend/services/anomaly_service.py` match those passed here; adjust kwargs if it
omits e.g. `room_view`.)

- [ ] **Step 5: Write the router**

`backend/routers/dashboard.py`:

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_manager, get_ml_store
from db.models import User
from ml_store.store import MLStore
from schemas.dashboard import DashboardResponse
from services.dashboard_service import get_dashboard

router = APIRouter(prefix="/manager", tags=["manager"])

@router.get("/dashboard", response_model=DashboardResponse)
async def dashboard_endpoint(
    days: int = 30,
    user: User = Depends(get_current_manager),
    db: AsyncSession = Depends(get_db),
    ml_store: MLStore = Depends(get_ml_store),
):
    return await get_dashboard(user, db, ml_store, days=days)
```

- [ ] **Step 6: Register in `main.py`** (add `dashboard` to the import + an `include_router`).

- [ ] **Step 7: Run — expect PASS**

```bash
./.venv/Scripts/python.exe -m pytest tests/test_dashboard.py -q 2>&1 | tail -10
```

Expected: 2 passed. Then run the full suite:

```bash
./.venv/Scripts/python.exe -m pytest -q 2>&1 | tail -8
```

Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add backend/schemas/dashboard.py backend/services/dashboard_service.py backend/routers/dashboard.py backend/main.py backend/tests/test_dashboard.py
git commit -m "feat(api): GET /manager/dashboard (KPIs + panels)"
```

---

## Task 9: Frontend — DTOs, ApiService, adapters

**Files:** Modify `frontend/src/app/core/api/dto.ts`, `api.service.ts`, `adapters.ts`.

- [ ] **Step 1: Add DTOs** (append to `dto.ts`)

```typescript
/** GET /manager/me */
export interface HotelBriefDto {
  id: number;
  hotel_name_display: string;
  city_name: string;
  stars_int: number | null;
}
export interface ManagerProfileDto {
  full_name: string | null;
  email: string;
  role: string;
  preferences: Record<string, any>;
  hotel: HotelBriefDto | null;
  competitor_count: number;
  max_competitors: number;
  last_login_at: string | null;
}
export interface ProfileUpdateDto {
  full_name?: string;
  preferences?: Record<string, any>;
}

/** GET /manager/dashboard */
export interface DashboardKpisDto {
  avg_listed_rate_tnd: number | null;
  market_position_pct: number | null;
  vs_competitor_pct: number | null;
  opportunity_tnd: number | null;
  opportunity_pct: number | null;
  open_recommendations: number;
  active_alerts: number;
}
export interface PriceSeriesPointDto {
  check_in: string;
  price_per_night: number;
  peer_medium_median: number | null;
  recommended_price_per_night: number;
}
export interface DashboardDto {
  kpis: DashboardKpisDto;
  price_series: PriceSeriesPointDto[];
  top_recommendations: RecommendationDto[];
  competitors: CompetitorDto[];
  recent_alerts: AnomalyDto[];
}

/** POST /manager/recommendations/decision */
export interface DecisionDto {
  check_in: string;
  nights: number;
  adults: number;
  boarding_canonical: string;
  recommended_price_tnd?: number;
  status: 'accepted' | 'dismissed';
}
export interface DecisionBulkDto {
  status: 'accepted' | 'dismissed';
  items: Omit<DecisionDto, 'status'>[];
}
```

Then extend `RecommendationDto` (add one line):

```typescript
  reasons: string[];
  decision_status: 'accepted' | 'dismissed' | null;
```

- [ ] **Step 2: Add ApiService methods** (in `api.service.ts`, add imports + methods)

```typescript
import {
  AnomalyDto, CalendarOptionsDto, CalendarQuery, CalendarRowDto,
  CompetitorDto, DataResponse, DateRangeQuery, RecommendationDto,
  ManagerProfileDto, ProfileUpdateDto, DashboardDto, DecisionDto, DecisionBulkDto,
} from './dto';

  getMe(): Observable<ManagerProfileDto> {
    return this.http.get<ManagerProfileDto>(`${API_BASE}/manager/me`);
  }
  updateMe(patch: ProfileUpdateDto): Observable<ManagerProfileDto> {
    return this.http.patch<ManagerProfileDto>(`${API_BASE}/manager/me`, patch);
  }
  getDashboard(days = 30): Observable<DashboardDto> {
    return this.http.get<DashboardDto>(`${API_BASE}/manager/dashboard`, { params: toParams({ days }) });
  }
  postDecision(body: DecisionDto): Observable<DecisionDto> {
    return this.http.post<DecisionDto>(`${API_BASE}/manager/recommendations/decision`, body);
  }
  postDecisionBulk(body: DecisionBulkDto): Observable<{ count: number }> {
    return this.http.post<{ count: number }>(`${API_BASE}/manager/recommendations/decision/bulk`, body);
  }
```

- [ ] **Step 3: Update the adapter** (in `adapters.ts`, change the `status` line)

```typescript
    rationale: d.reasons,
    status: d.decision_status ?? 'new',
```

- [ ] **Step 4: Build to typecheck**

```bash
cd frontend && npm run build 2>&1 | tail -15
```

Expected: build succeeds (no TS errors). Decision DTO/method types resolve.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/core/api/dto.ts frontend/src/app/core/api/api.service.ts frontend/src/app/core/api/adapters.ts
git commit -m "feat(frontend): manager profile/dashboard/decision API client"
```

---

## Task 10: Frontend — Settings page wired to /manager/me

**Files:** Modify `frontend/src/app/features/manager/settings/manager-settings.component.ts`.

- [ ] **Step 1: Rewrite the component to load + save profile**

Replace the class body and bind the template to signals. Key changes:
- inject `ApiService`, `AuthService`, `Router`.
- `profile = signal<ManagerProfileDto | null>(null)`, `saving`, `saved`, `error`.
- On init `api.getMe()` → populate `fullName`, `language`, and an `alerts` object
  from `preferences.alerts` (default each toggle to a sensible value if missing).
- `save()` → `api.updateMe({ full_name, preferences: { language, alerts } })`,
  set `saved` on success, surface `err.error.detail` on failure.
- `signOut()` → `auth.logout(); router.navigateByUrl('/login')`.
- Template: bind name/email/hotel from `profile()`, toggles to `alerts`, the Save
  button `(click)="save()"` with disabled+spinner on `saving()`, and a success line
  when `saved()`. Email + hotel fields stay `disabled`.

Use the existing template markup as the visual base; only swap hardcoded `value=...`
for `[value]`/`[(ngModel)]`-free signal bindings (the project uses `(input)` +
`.set()`, not `ngModel` — follow the calendar component's pattern for inputs).

Concrete init + save:

```typescript
import { Component, OnInit, inject, signal } from '@angular/core';
import { Router } from '@angular/router';
import { ApiService } from '../../../core/api/api.service';
import { AuthService } from '../../../core/auth/auth.service';
import { ManagerProfileDto } from '../../../core/api/dto';

interface AlertPrefs {
  competitor_undercut: boolean; price_spike: boolean;
  anomaly_digest: boolean; data_quality: boolean;
}

export class ManagerSettingsComponent implements OnInit {
  private api = inject(ApiService);
  private auth = inject(AuthService);
  private router = inject(Router);

  profile = signal<ManagerProfileDto | null>(null);
  fullName = signal('');
  language = signal('en');
  alerts = signal<AlertPrefs>({ competitor_undercut: true, price_spike: true, anomaly_digest: false, data_quality: true });
  saving = signal(false);
  saved = signal(false);
  error = signal<string | null>(null);

  ngOnInit(): void {
    this.api.getMe().subscribe({
      next: p => {
        this.profile.set(p);
        this.fullName.set(p.full_name ?? '');
        this.language.set(p.preferences?.['language'] ?? 'en');
        const a = p.preferences?.['alerts'] ?? {};
        this.alerts.set({
          competitor_undercut: a.competitor_undercut ?? true,
          price_spike: a.price_spike ?? true,
          anomaly_digest: a.anomaly_digest ?? false,
          data_quality: a.data_quality ?? true,
        });
      },
      error: () => this.error.set('Could not load your profile.'),
    });
  }

  toggleAlert(key: keyof AlertPrefs) {
    this.alerts.update(a => ({ ...a, [key]: !a[key] }));
    this.saved.set(false);
  }

  save(): void {
    this.saving.set(true); this.saved.set(false); this.error.set(null);
    this.api.updateMe({
      full_name: this.fullName(),
      preferences: { language: this.language(), alerts: this.alerts() },
    }).subscribe({
      next: p => { this.profile.set(p); this.saving.set(false); this.saved.set(true); },
      error: err => { this.saving.set(false); this.error.set(err?.error?.detail ?? 'Save failed.'); },
    });
  }

  signOut(): void { this.auth.logout(); this.router.navigateByUrl('/login'); }
}
```

Update the template: replace `value="Sami Bouazizi"` → `[value]="fullName()" (input)="fullName.set($any($event.target).value)"`; email → `[value]="profile()?.email"`; hotel block → `[value]="profile()?.hotel?.hotel_name_display"` etc.; the `prefs` `@for` → explicit toggles bound to `alerts()` via `toggleAlert(...)`; Save button → `(click)="save()" [disabled]="saving()"`; add `@if (saved()) { <span class="muted small">Saved.</span> }` and `@if (error()) {...}`; Danger-zone "Sign out" button → `(click)="signOut()"`.

- [ ] **Step 2: Build**

```bash
cd frontend && npm run build 2>&1 | tail -15
```

Expected: builds clean.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/features/manager/settings/manager-settings.component.ts
git commit -m "feat(frontend): wire manager Settings to /manager/me (load + save + sign out)"
```

---

## Task 11: Frontend — Dashboard wired to live data

**Files:** Modify `frontend/src/app/features/manager/dashboard/manager-dashboard.component.ts`.

- [ ] **Step 1: Replace mock imports with live calls**

- Remove `import { ALERTS, CALENDAR, COMPETITORS, RECOMMENDATIONS } from '.../mock'`.
- inject `ApiService`; on init call `getMe()` (header) and `getDashboard(30)` (data).
- Replace hardcoded KPI card values with computed signals from `dashboard().kpis`,
  rendering `'—'` when a value is `null`. Rename the first card label from
  **"Your ADR (next 7d)"** to **"Your avg listed rate (30d)"**.
- Build the chart `ownPath/mktPath/recPath` from `dashboard().price_series`.
- Top recommendations panel ← `dashboard().top_recommendations` (map via
  `recommendationFromDto`); competitor panel ← `dashboard().competitors`; alerts
  panel ← `dashboard().recent_alerts` (map via `alertFromDto`).
- Greeting: `Hello, {{ profile()?.full_name }}` and hotel line from
  `profile()?.hotel`.
- Add loading + error signals; render a loading state while either request is in
  flight.

Concrete class skeleton:

```typescript
import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { DatePipe, DecimalPipe } from '@angular/common';
import { RouterLink } from '@angular/router';
import { ApiService } from '../../../core/api/api.service';
import { recommendationFromDto } from '../../../core/api/adapters';
import { DashboardDto, ManagerProfileDto } from '../../../core/api/dto';
// ...shared component imports as before...

export class ManagerDashboardComponent implements OnInit {
  private api = inject(ApiService);
  profile = signal<ManagerProfileDto | null>(null);
  dash = signal<DashboardDto | null>(null);
  loading = signal(true);
  error = signal<string | null>(null);

  recs = computed(() => (this.dash()?.top_recommendations ?? []).slice(0, 4).map(recommendationFromDto));
  competitors = computed(() => this.dash()?.competitors ?? []);
  kpis = computed(() => this.dash()?.kpis ?? null);
  series = computed(() => this.dash()?.price_series ?? []);

  ngOnInit(): void {
    this.api.getMe().subscribe({ next: p => this.profile.set(p), error: () => {} });
    this.api.getDashboard(30).subscribe({
      next: d => { this.dash.set(d); this.loading.set(false); },
      error: () => { this.error.set('Could not load dashboard.'); this.loading.set(false); },
    });
  }

  fmt(v: number | null | undefined, digits = 0): string {
    return v === null || v === undefined ? '—' : v.toFixed(digits);
  }

  line(values: number[]): string { /* keep existing polyline math, fed from series() */ }
}
```

Update the template to read from `kpis()`, `series()`, `recs()`, `competitors()`,
and the alerts list; keep the existing CSS. Competitor panel fields use the DTO
shape (`hotel_name_display`, `city_name`, `stars_int`, `avg_price_per_night`) — note
this differs from the old mock fields (`name`, `city`, `stars`, `avgPrice7d`,
`trend`); the sparkline `trend` is no longer available, so drop the sparkline or
show the single `avg_price_per_night` value.

- [ ] **Step 2: Build**

```bash
cd frontend && npm run build 2>&1 | tail -15
```

Expected: clean build (watch for `strictTemplates` errors on the changed bindings).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/features/manager/dashboard/manager-dashboard.component.ts
git commit -m "feat(frontend): wire manager Dashboard to /manager/dashboard + /me"
```

---

## Task 12: Frontend — Recommendations Accept/Dismiss + bulk

**Files:** Modify `frontend/src/app/features/manager/recommendations/manager-recommendations.component.ts`.

- [ ] **Step 1: Add decision handlers**

- Keep the existing list (now `status` reflects persisted `decision_status` via the
  adapter change in Task 9).
- Store the raw DTOs alongside domain rows so handlers can build the decision key.
  Simplest: keep a `Map<string, RecommendationDto>` keyed by the domain `id`
  (`${check_in}-${nights}n-${boarding}-${adults}a`) populated in the subscribe.
- `accept(r)` / `dismiss(r)` → look up the DTO by `r.id`, call
  `api.postDecision({ check_in, nights, adults, boarding_canonical, recommended_price_tnd, status })`,
  and on success update that row's `status` in the `all` signal optimistically.
- `acceptAllNew()` / `dismissAll()` → build a `DecisionBulkDto` from the currently
  visible rows (for accept-all, only those with `status==='new'`), call
  `api.postDecisionBulk(...)`, then update local statuses.

Concrete handlers:

```typescript
import { DecisionDto } from '../../../core/api/dto';

  private dtoById = new Map<string, RecommendationDto>();

  // in the subscribe next-handler, after building latestPerDate:
  // latestPerDate.forEach((row, i) => this.dtoById.set(row.id, rawSorted[i]))
  // (capture the raw DTO list in the same order you map domain rows)

  private keyOf(d: RecommendationDto): DecisionDto {
    return { check_in: d.check_in, nights: d.nights, adults: d.adults,
             boarding_canonical: d.boarding_canonical,
             recommended_price_tnd: d.recommended_price_tnd, status: 'accepted' };
  }

  decide(r: Recommendation, status: 'accepted' | 'dismissed') {
    const dto = this.dtoById.get(r.id);
    if (!dto) return;
    this.api.postDecision({ ...this.keyOf(dto), status }).subscribe({
      next: () => this.all.update(rows => rows.map(x => x.id === r.id ? { ...x, status } : x)),
      error: () => {},
    });
  }

  bulk(status: 'accepted' | 'dismissed') {
    const targets = this.all().filter(r => status === 'accepted' ? r.status === 'new' : true);
    const items = targets.map(r => this.dtoById.get(r.id)).filter(Boolean)
      .map(d => ({ check_in: d!.check_in, nights: d!.nights, adults: d!.adults,
                   boarding_canonical: d!.boarding_canonical,
                   recommended_price_tnd: d!.recommended_price_tnd }));
    if (!items.length) return;
    this.api.postDecisionBulk({ status, items }).subscribe({
      next: () => this.all.update(rows => rows.map(x =>
        targets.some(t => t.id === x.id) ? { ...x, status } : x)),
      error: () => {},
    });
  }
```

Wire template buttons: row `Accept` → `(click)="decide(r,'accepted')"`, `Dismiss`
→ `(click)="decide(r,'dismissed')"`; header `Accept all new` →
`(click)="bulk('accepted')"`, `Dismiss all` → `(click)="bulk('dismissed')"`.

**Important:** to build the `dtoById` map you need the raw DTO in the same order as
the mapped domain rows. Refactor the subscribe to keep `latestPerDateDtos` (the
deduped DTO array) and map both: `this.all.set(latestPerDateDtos.map(recommendationFromDto)); latestPerDateDtos.forEach(d => this.dtoById.set(`${d.check_in}-${d.nights}n-${d.boarding_canonical}-${d.adults}a`, d));`

- [ ] **Step 2: Build**

```bash
cd frontend && npm run build 2>&1 | tail -15
```

Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/features/manager/recommendations/manager-recommendations.component.ts
git commit -m "feat(frontend): persist recommendation accept/dismiss (single + bulk)"
```

---

## Task 13: Frontend — Alerts Investigate deep-link

**Files:** Modify `frontend/src/app/features/manager/alerts/manager-alerts.component.ts`.

- [ ] **Step 1: Make Investigate route to the calendar at the alert's check-in**

- The `Alert` domain model's `id` encodes `anom-${check_in}-...`; keep the raw
  `check_in` available (extend `alertFromDto` to also return `checkIn`, or store a
  parallel map like Task 12).
- Inject `Router`; `investigate(a)` → `router.navigate(['/manager/calendar'],
  { queryParams: { check_in: <checkIn> } })`.
- Wire the row button `Investigate` → `(click)="investigate(a)"`.
- Remove (or `disabled` with a `title="Coming soon"`) the "Mark all read" button —
  out of scope (spec §3).

Minimal approach: add `checkIn: string` to the `Alert` model and set it in
`alertFromDto` (`checkIn: d.check_in`). Then `investigate` uses `a.checkIn`.

(Optional, only if quick) In `manager-calendar.component.ts` `ngOnInit`, read
`check_in` from `ActivatedRoute.snapshot.queryParamMap` and, if present, bias the
window start to that date. If non-trivial, skip — landing on the calendar is the
core of the feature.

- [ ] **Step 2: Build**

```bash
cd frontend && npm run build 2>&1 | tail -15
```

Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/features/manager/alerts/manager-alerts.component.ts frontend/src/app/core/models/domain.ts frontend/src/app/core/api/adapters.ts
git commit -m "feat(frontend): Alerts Investigate deep-links into the calendar"
```

---

## Task 14: Live end-to-end verification

**Files:** none (verification + a possible password-reset one-off).

- [ ] **Step 1: Resolve the manager password**

The real manager is `manager@revway.tn`. If the password is unknown, reset it with
the venv (uses the project's `hash_password`):

```bash
cd backend && ./.venv/Scripts/python.exe -c "
import asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from core.security import hash_password
async def main():
    e=create_async_engine('postgresql+asyncpg://revway:REDACTED@localhost:5432/revway')
    async with e.begin() as c:
        await c.execute(text('UPDATE users SET password_hash=:h WHERE email=:em'),
                        {'h': hash_password('REDACTED_DEV_PASSWORD'), 'em':'manager@revway.tn'})
    await e.dispose(); print('password set to REDACTED_DEV_PASSWORD')
asyncio.run(main())
"
```

- [ ] **Step 2: Start backend + frontend (two terminals, from the worktree)**

```bash
cd backend && ./.venv/Scripts/python.exe -m uvicorn main:app --reload --port 8000
# other terminal:
cd frontend && npm start
```

Confirm `GET http://localhost:8000/health` → `{"status":"ok","ml_store":"loaded"}`.

- [ ] **Step 3: Manual smoke at http://localhost:4200**

Verify against the spec §8 checklist:
1. Log in as `manager@revway.tn` / `REDACTED_DEV_PASSWORD` → lands on manager dashboard.
2. Dashboard KPIs populated (not "—" everywhere), chart renders, panels show data.
3. Calendar / Competitors / Recommendations / Alerts load live data.
4. Accept a recommendation → reload → status persists; filter chips reflect it.
5. Settings: change name + a toggle → Save → reload → persists.
6. Alerts: Investigate → lands on calendar.
7. Sign out → `/login`; revisiting `/manager` redirects to login.

- [ ] **Step 4: Final full backend test run**

```bash
cd backend && ./.venv/Scripts/python.exe -m pytest -q 2>&1 | tail -8
```

Expected: all green.

- [ ] **Step 5: Update frontend CLAUDE.md note**

Update `frontend/CLAUDE.md` "Current state" to record that the manager side is now
fully wired (Dashboard + Settings + recommendation decisions), not mock-backed.

```bash
git add frontend/CLAUDE.md
git commit -m "docs(frontend): manager side fully wired to live API"
```

---

## Self-review (completed against the spec)

- **Spec §5.1 profile** → Tasks 5–6. **§5.2 dashboard** → Task 8. **§5.3
  decisions** → Tasks 3, 7. **§5.4 preferences** → Tasks 2, 6. **§5.5 migrations**
  → Tasks 2–3 (006, 007 confirmed as next numbers). **§6.1 API layer** → Task 9.
  **§6.2 dashboard UI** → Task 11. **§6.3 recs actions** → Task 12. **§6.4 alerts**
  → Task 13. **§6.5 settings** → Task 10. **§7 tests** → Tasks 5–8.
  **§8 verification** → Task 14. **§4 workspace** → Task 1.
- **Type consistency:** `decision_status` added to `RecommendationRow`
  (backend, Task 7) and `RecommendationDto` (frontend, Task 9); adapter maps
  `?? 'new'`. Decision key fields identical across migration 007, ORM,
  `DecisionKey`/`DecisionIn`, `get_decision_map`, and the frontend `keyOf`.
  `CompetitorSummary`/`CompetitorDto` and `AnomalyRow`/`AnomalyDto` reused, not
  redefined.
- **Open verification points flagged inline** (not placeholders): confirm
  `get_anomalies` kwargs (Task 8 Step 4), confirm competitor schema class name
  (Task 8 Step 3), confirm calendar query-param handling for the alerts deep-link
  (Task 13, optional).
- **Risks** (spec §9): manager password (Task 14 Step 1), worktree runtime deps
  (Task 1 Step 2).
```
