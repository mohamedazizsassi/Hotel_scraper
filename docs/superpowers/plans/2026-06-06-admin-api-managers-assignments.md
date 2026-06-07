# Admin API — Managers + Assignments — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Admin manages **manager accounts** (list, detail, create with initial password, update, reset password) and **manager↔hotel assignments** (list, assign, change, remove) — the onboarding steps before competitor selection.

**Architecture:** Same patterns as Plan 2: `/admin/*` routers gated by `get_current_admin`, raw-SQL services via `text()` + `.mappings()`, `DataResponse[T]` envelopes, custom exceptions (`ConflictError`=409, `NotFoundError`=404). New routers `routers/admin/managers.py` and `routers/admin/assignments.py`, mounted in `main.py`.

**Tech Stack:** FastAPI, SQLAlchemy 2.0 async, asyncpg, PostgreSQL, bcrypt (via `core.security.hash_password`), pytest + pytest-asyncio.

**Plan map:** Plan 1 = data layer (done). Plan 2 = foundation + hotels (done). **Plan 3 = managers + assignments (this doc).** Plan 4 = competitors (admin-only, D11) + monitoring + alerts. Plan 5 = frontend.

---

## Conventions for every backend command

- Work in the `feat/admin-platform` **worktree**:
  `C:\Users\ASUS\Desktop\PFE\revway\.claude\worktrees\feat+admin-platform`. Never
  touch the main checkout. Verify before committing: `git rev-parse --show-toplevel`
  ends in `.claude/worktrees/feat+admin-platform`; `git branch --show-current` =
  `feat/admin-platform`.
- No local `.venv` — use `C:\Users\ASUS\Desktop\PFE\revway\backend\.venv\Scripts\python.exe`
  for every python/pytest command, run from the worktree `backend/` dir.
- Commits: Conventional Commits, **no `Co-Authored-By: Claude` trailer**.

## Critical gotcha (from Plan 2)

Raw-SQL `INSERT`s run against the **ORM-built test DB** must explicitly set any
`NOT NULL` column whose ORM default is Python-side (`default=`), because
`Base.metadata.create_all` does not emit a server default. For this plan that
means **`user_hotel_assignments.max_competitors` and `is_active` must be in every
INSERT** (this plan's INSERTs already include them). Also: `users.id` /
`user_hotel_assignments.user_id` are UUID columns — bind string params with
`CAST(:p AS uuid)` or compare via `col::text = :p` (avoid `::uuid` next to a
bindparam in `text()`).

---

### Task 1: Per-test transaction isolation

The seed in `conftest.setup_test_db` is **session-scoped**, so without isolation a
mutating test (create/update/delete) leaks state into later tests and breaks their
assertions. This task wraps every test in an outer transaction rolled back at
teardown: each test starts from the same committed seed, sees its own writes via
savepoints, and leaves nothing behind. SQLAlchemy 2.0's
`join_transaction_mode="create_savepoint"` turns the app's `session.commit()` calls
into savepoint releases inside that outer transaction (so endpoints that commit
then re-read still work).

**Files:** Modify `backend/tests/conftest.py`. No new tests — verification is the
existing suite passing, twice.

- [ ] **Step 1: Replace the `db_session` fixture** — In `backend/tests/conftest.py`,
replace the existing `db_session` fixture body with:

```python
@pytest_asyncio.fixture(loop_scope="function")
async def db_session():
    engine = create_async_engine(settings.test_db_url, echo=False, poolclass=NullPool)
    conn = await engine.connect()
    trans = await conn.begin()
    session = AsyncSession(
        bind=conn, expire_on_commit=False, join_transaction_mode="create_savepoint")
    try:
        yield session
    finally:
        await session.close()
        if trans.is_active:
            await trans.rollback()
        await conn.close()
        await engine.dispose()
```

(`create_async_engine`, `AsyncSession`, `NullPool`, `settings`, `pytest_asyncio`
are already imported in conftest.)

- [ ] **Step 2: Verify existing suite stays green (run twice for determinism)**

```bash
"C:\Users\ASUS\Desktop\PFE\revway\backend\.venv\Scripts\python.exe" -m pytest -q
"C:\Users\ASUS\Desktop\PFE\revway\backend\.venv\Scripts\python.exe" -m pytest -q
```
Expected: **37 passed** both runs. If any test fails under savepoint isolation,
STOP and report rather than forcing it.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/conftest.py
git commit -m "test: isolate each test in a rolled-back transaction (savepoints)"
```

---

### Task 2: Managers — list + detail (GET) + unassigned-manager fixture

**Files:**
- Create: `backend/schemas/admin_manager.py`
- Create: `backend/services/admin_managers.py`
- Create: `backend/routers/admin/managers.py`
- Modify: `backend/main.py`
- Modify: `backend/tests/conftest.py`
- Test: `backend/tests/test_admin_managers.py`

- [ ] **Step 1: Add an unassigned manager to conftest** — In
`backend/tests/conftest.py` `setup_test_db`, after the admin user seed, add a
second manager with NO assignment (used by managers-list + assignment-create
tests):

```python
        # Second manager, intentionally UNASSIGNED (for list + assignment tests)
        mgr2_id = str(uuid.uuid4())
        mgr2_pw = hash_password("testpass2")
        await conn.execute(text(
            f"INSERT INTO users (id, email, password_hash, full_name, role, is_active) "
            f"VALUES ('{mgr2_id}', 'manager2@test.com', '{mgr2_pw}', 'Manager Two', 'manager', true)"
        ))
```

- [ ] **Step 2: Write the failing test** — Create `backend/tests/test_admin_managers.py`:

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
    # the seeded manager is assigned to hotel_manager_test; manager2 is not
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
```

- [ ] **Step 3: Run test → FAIL** (`pytest tests/test_admin_managers.py -v`; 404/import).

- [ ] **Step 4: Schemas** — Create `backend/schemas/admin_manager.py`:

```python
from __future__ import annotations
from pydantic import BaseModel, EmailStr
from schemas.common import DataResponse


class AdminManagerRow(BaseModel):
    id: str
    email: str
    full_name: str | None
    is_active: bool
    last_login_at: str | None
    assigned_hotel_id: int | None
    assigned_hotel_name: str | None


AdminManagerListResponse = DataResponse[AdminManagerRow]
```

- [ ] **Step 5: Service** — Create `backend/services/admin_managers.py`:

```python
from __future__ import annotations
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from core.exceptions import NotFoundError
from schemas.admin_manager import AdminManagerRow

_LIST_SQL = text("""
    SELECT u.id::text                     AS id,
           u.email,
           u.full_name,
           u.is_active,
           u.last_login_at::text          AS last_login_at,
           uha.hotel_id                   AS assigned_hotel_id,
           ph.hotel_name_display          AS assigned_hotel_name
    FROM users u
    LEFT JOIN user_hotel_assignments uha ON uha.user_id = u.id AND uha.is_active = TRUE
    LEFT JOIN platform_hotels ph ON ph.id = uha.hotel_id
    WHERE u.role = 'manager'
    ORDER BY u.full_name NULLS LAST, u.email
""")


async def list_managers(db: AsyncSession) -> list[AdminManagerRow]:
    rows = (await db.execute(_LIST_SQL)).mappings().fetchall()
    return [AdminManagerRow(**dict(r)) for r in rows]


async def get_manager(db: AsyncSession, manager_id: str) -> AdminManagerRow:
    rows = (await db.execute(_LIST_SQL)).mappings().fetchall()
    m = next((r for r in rows if r["id"] == str(manager_id)), None)
    if m is None:
        raise NotFoundError(f"Manager {manager_id} not found")
    return AdminManagerRow(**dict(m))
```

- [ ] **Step 6: Router + mount** — Create `backend/routers/admin/managers.py`:

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_admin
from db.models import User
from schemas.admin_manager import AdminManagerRow, AdminManagerListResponse
from services.admin_managers import list_managers, get_manager

router = APIRouter(prefix="/admin/managers", tags=["admin"])


@router.get("", response_model=AdminManagerListResponse)
async def managers_list(
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await list_managers(db)
    return AdminManagerListResponse(data=rows, count=len(rows))


@router.get("/{manager_id}", response_model=AdminManagerRow)
async def managers_detail(
    manager_id: str,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    return await get_manager(db, manager_id)
```

In `backend/main.py`, add to the admin imports/includes:

```python
from routers.admin import managers as admin_managers
app.include_router(admin_managers.router)
```

- [ ] **Step 7: Run tests → PASS** — `pytest tests/test_admin_managers.py -v` then
`pytest -q` (expect all green: 37 + 3 = 40).

- [ ] **Step 8: Commit**

```bash
git add backend/schemas/admin_manager.py backend/services/admin_managers.py backend/routers/admin/managers.py backend/main.py backend/tests/conftest.py backend/tests/test_admin_managers.py
git commit -m "feat(api): admin managers list + detail endpoints"
```

---

### Task 3: Managers — create + update + reset-password

**Files:**
- Modify: `backend/schemas/admin_manager.py`
- Modify: `backend/services/admin_managers.py`
- Modify: `backend/routers/admin/managers.py`
- Test: `backend/tests/test_admin_managers.py` (extend)

- [ ] **Step 1: Write the failing test** — APPEND to `backend/tests/test_admin_managers.py`:

```python
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
    # the new manager can log in with the initial password
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
```

- [ ] **Step 2: Run test → FAIL** (`pytest tests/test_admin_managers.py -k "create or update or reset" -v`).

- [ ] **Step 3: Schemas** — In `backend/schemas/admin_manager.py`, ADD:

```python
class ManagerCreate(BaseModel):
    email: EmailStr
    full_name: str | None = None
    initial_password: str


class ManagerUpdate(BaseModel):
    email: EmailStr | None = None
    full_name: str | None = None
    is_active: bool | None = None


class PasswordReset(BaseModel):
    new_password: str
```

- [ ] **Step 4: Service** — In `backend/services/admin_managers.py`: extend the
exceptions import to `from core.exceptions import NotFoundError, ConflictError`,
add `from core.security import hash_password`, extend the schema import to include
`ManagerCreate, ManagerUpdate, PasswordReset`, and ADD:

```python
async def create_manager(db: AsyncSession, body: ManagerCreate) -> AdminManagerRow:
    dup = await db.scalar(text("SELECT 1 FROM users WHERE email = :e"), {"e": str(body.email)})
    if dup:
        raise ConflictError(f"Email '{body.email}' already in use")
    new_id = await db.scalar(
        text("""INSERT INTO users (email, password_hash, full_name, role, is_active)
                VALUES (:e, :ph, :fn, 'manager', TRUE) RETURNING id::text"""),
        {"e": str(body.email), "ph": hash_password(body.initial_password), "fn": body.full_name})
    await db.commit()
    return await get_manager(db, new_id)


async def update_manager(db: AsyncSession, manager_id: str, body: ManagerUpdate) -> AdminManagerRow:
    exists = await db.scalar(
        text("SELECT 1 FROM users WHERE id::text = :id AND role = 'manager'"),
        {"id": str(manager_id)})
    if not exists:
        raise NotFoundError(f"Manager {manager_id} not found")
    fields = body.model_dump(exclude_unset=True)
    if "email" in fields and fields["email"] is not None:
        dup = await db.scalar(
            text("SELECT 1 FROM users WHERE email = :e AND id::text <> :id"),
            {"e": str(fields["email"]), "id": str(manager_id)})
        if dup:
            raise ConflictError(f"Email '{fields['email']}' already in use")
    allowed = {"email", "full_name", "is_active"}
    sets = ", ".join(f"{k} = :{k}" for k in fields if k in allowed)
    if sets:
        params = {k: (str(v) if k == "email" else v) for k, v in fields.items() if k in allowed}
        params["id"] = str(manager_id)
        await db.execute(text(f"UPDATE users SET {sets} WHERE id::text = :id"), params)
        await db.commit()
    return await get_manager(db, manager_id)


async def reset_password(db: AsyncSession, manager_id: str, body: PasswordReset) -> None:
    exists = await db.scalar(
        text("SELECT 1 FROM users WHERE id::text = :id AND role = 'manager'"),
        {"id": str(manager_id)})
    if not exists:
        raise NotFoundError(f"Manager {manager_id} not found")
    await db.execute(
        text("UPDATE users SET password_hash = :ph WHERE id::text = :id"),
        {"ph": hash_password(body.new_password), "id": str(manager_id)})
    await db.commit()
```

- [ ] **Step 5: Router** — In `backend/routers/admin/managers.py`: add `status, Response`
to the fastapi import (`from fastapi import APIRouter, Depends, status, Response`),
extend the schema import to include `ManagerCreate, ManagerUpdate, PasswordReset`,
extend the service import to include `create_manager, update_manager, reset_password`,
and ADD (the create/update before the `/{manager_id}` GET is not required since
methods differ, but keep create POST near the top for clarity):

```python
@router.post("", response_model=AdminManagerRow, status_code=status.HTTP_201_CREATED)
async def managers_create(
    body: ManagerCreate,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    return await create_manager(db, body)


@router.patch("/{manager_id}", response_model=AdminManagerRow)
async def managers_update(
    manager_id: str,
    body: ManagerUpdate,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    return await update_manager(db, manager_id, body)


@router.post("/{manager_id}/reset-password", status_code=status.HTTP_204_NO_CONTENT)
async def managers_reset_password(
    manager_id: str,
    body: PasswordReset,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    await reset_password(db, manager_id, body)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
```

- [ ] **Step 6: Run tests** — `pytest tests/test_admin_managers.py -v` then `pytest -q`
(expect all green: 40 + 4 = 44).

- [ ] **Step 7: Commit**

```bash
git add backend/schemas/admin_manager.py backend/services/admin_managers.py backend/routers/admin/managers.py backend/tests/test_admin_managers.py
git commit -m "feat(api): admin manager create + update + reset-password"
```

---

### Task 4: Assignments — list + create (GET, POST)

**Files:**
- Create: `backend/schemas/admin_assignment.py`
- Create: `backend/services/admin_assignments.py`
- Create: `backend/routers/admin/assignments.py`
- Modify: `backend/main.py`
- Test: `backend/tests/test_admin_assignments.py`

- [ ] **Step 1: Write the failing test** — Create `backend/tests/test_admin_assignments.py`:

```python
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
```

- [ ] **Step 2: Run test → FAIL** (`pytest tests/test_admin_assignments.py -v`).

- [ ] **Step 3: Schemas** — Create `backend/schemas/admin_assignment.py`:

```python
from __future__ import annotations
from pydantic import BaseModel
from schemas.common import DataResponse


class AdminAssignmentRow(BaseModel):
    id: int
    user_id: str
    manager_email: str
    manager_name: str | None
    hotel_id: int
    hotel_name: str
    max_competitors: int
    is_active: bool


class AssignmentCreate(BaseModel):
    user_id: str
    hotel_id: int
    max_competitors: int = 4


AdminAssignmentListResponse = DataResponse[AdminAssignmentRow]
```

- [ ] **Step 4: Service** — Create `backend/services/admin_assignments.py`:

```python
from __future__ import annotations
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from core.exceptions import NotFoundError, ConflictError
from schemas.admin_assignment import AdminAssignmentRow, AssignmentCreate

_LIST_SQL = text("""
    SELECT uha.id,
           uha.user_id::text          AS user_id,
           u.email                    AS manager_email,
           u.full_name                AS manager_name,
           uha.hotel_id,
           ph.hotel_name_display      AS hotel_name,
           uha.max_competitors,
           uha.is_active
    FROM user_hotel_assignments uha
    JOIN users u ON u.id = uha.user_id
    JOIN platform_hotels ph ON ph.id = uha.hotel_id
    ORDER BY u.full_name NULLS LAST, u.email
""")


async def list_assignments(db: AsyncSession) -> list[AdminAssignmentRow]:
    rows = (await db.execute(_LIST_SQL)).mappings().fetchall()
    return [AdminAssignmentRow(**dict(r)) for r in rows]


async def _get_row(db: AsyncSession, assignment_id: int) -> AdminAssignmentRow:
    rows = (await db.execute(_LIST_SQL)).mappings().fetchall()
    m = next((r for r in rows if r["id"] == assignment_id), None)
    if m is None:
        raise NotFoundError(f"Assignment {assignment_id} not found")
    return AdminAssignmentRow(**dict(m))


async def create_assignment(db: AsyncSession, body: AssignmentCreate) -> AdminAssignmentRow:
    mgr = await db.scalar(
        text("SELECT 1 FROM users WHERE id::text = :id AND role = 'manager' AND is_active = TRUE"),
        {"id": body.user_id})
    if not mgr:
        raise NotFoundError(f"Active manager {body.user_id} not found")
    hotel = await db.scalar(text("SELECT 1 FROM platform_hotels WHERE id = :h"),
                            {"h": body.hotel_id})
    if not hotel:
        raise NotFoundError(f"Hotel {body.hotel_id} not found")
    dup = await db.scalar(
        text("SELECT 1 FROM user_hotel_assignments WHERE user_id::text = :id"),
        {"id": body.user_id})
    if dup:
        raise ConflictError(f"Manager {body.user_id} already has an assignment")
    new_id = await db.scalar(
        text("""INSERT INTO user_hotel_assignments (user_id, hotel_id, max_competitors, is_active)
                VALUES (CAST(:uid AS uuid), :h, :mc, TRUE) RETURNING id"""),
        {"uid": body.user_id, "h": body.hotel_id, "mc": body.max_competitors})
    await db.commit()
    return await _get_row(db, new_id)
```

- [ ] **Step 5: Router + mount** — Create `backend/routers/admin/assignments.py`:

```python
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_admin
from db.models import User
from schemas.admin_assignment import (
    AdminAssignmentRow, AdminAssignmentListResponse, AssignmentCreate,
)
from services.admin_assignments import list_assignments, create_assignment

router = APIRouter(prefix="/admin/assignments", tags=["admin"])


@router.get("", response_model=AdminAssignmentListResponse)
async def assignments_list(
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await list_assignments(db)
    return AdminAssignmentListResponse(data=rows, count=len(rows))


@router.post("", response_model=AdminAssignmentRow, status_code=status.HTTP_201_CREATED)
async def assignments_create(
    body: AssignmentCreate,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    return await create_assignment(db, body)
```

In `backend/main.py`, add:

```python
from routers.admin import assignments as admin_assignments
app.include_router(admin_assignments.router)
```

- [ ] **Step 6: Run tests** — `pytest tests/test_admin_assignments.py -v` then
`pytest -q` (expect all green: 44 + 3 = 47).

- [ ] **Step 7: Commit**

```bash
git add backend/schemas/admin_assignment.py backend/services/admin_assignments.py backend/routers/admin/assignments.py backend/main.py backend/tests/test_admin_assignments.py
git commit -m "feat(api): admin assignments list + create endpoints"
```

---

### Task 5: Assignments — update + delete (PATCH, DELETE)

**Files:**
- Modify: `backend/schemas/admin_assignment.py`
- Modify: `backend/services/admin_assignments.py`
- Modify: `backend/routers/admin/assignments.py`
- Test: `backend/tests/test_admin_assignments.py` (extend)

- [ ] **Step 1: Write the failing test** — APPEND to `backend/tests/test_admin_assignments.py`:

```python
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
    # gone from the list
    lst = await client.get("/admin/assignments", headers={"Authorization": f"Bearer {tok}"})
    assert all(a["id"] != aid for a in lst.json()["data"])


async def test_update_assignment_404(client, db_session):
    tok = await _admin_token(db_session)
    r = await client.patch("/admin/assignments/999999", json={"max_competitors": 1},
                           headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 404
```

- [ ] **Step 2: Run test → FAIL**
(`pytest tests/test_admin_assignments.py -k "update or delete" -v`).

- [ ] **Step 3: Schema** — In `backend/schemas/admin_assignment.py`, ADD:

```python
class AssignmentUpdate(BaseModel):
    hotel_id: int | None = None
    max_competitors: int | None = None
    is_active: bool | None = None
```

- [ ] **Step 4: Service** — In `backend/services/admin_assignments.py`: extend the
schema import to include `AssignmentUpdate`, and ADD:

```python
async def update_assignment(db: AsyncSession, assignment_id: int,
                            body: AssignmentUpdate) -> AdminAssignmentRow:
    exists = await db.scalar(
        text("SELECT 1 FROM user_hotel_assignments WHERE id = :id"), {"id": assignment_id})
    if not exists:
        raise NotFoundError(f"Assignment {assignment_id} not found")
    fields = body.model_dump(exclude_unset=True)
    allowed = {"hotel_id", "max_competitors", "is_active"}
    sets = ", ".join(f"{k} = :{k}" for k in fields if k in allowed)
    if sets:
        params = {k: v for k, v in fields.items() if k in allowed}
        params["id"] = assignment_id
        await db.execute(
            text(f"UPDATE user_hotel_assignments SET {sets} WHERE id = :id"), params)
        await db.commit()
    return await _get_row(db, assignment_id)


async def delete_assignment(db: AsyncSession, assignment_id: int) -> None:
    exists = await db.scalar(
        text("SELECT 1 FROM user_hotel_assignments WHERE id = :id"), {"id": assignment_id})
    if not exists:
        raise NotFoundError(f"Assignment {assignment_id} not found")
    await db.execute(
        text("DELETE FROM user_hotel_assignments WHERE id = :id"), {"id": assignment_id})
    await db.commit()
```

- [ ] **Step 5: Router** — In `backend/routers/admin/assignments.py`: add `Response`
to the fastapi import (`from fastapi import APIRouter, Depends, status, Response`),
extend the schema import to include `AssignmentUpdate`, extend the service import
to include `update_assignment, delete_assignment`, and ADD:

```python
@router.patch("/{assignment_id}", response_model=AdminAssignmentRow)
async def assignments_update(
    assignment_id: int,
    body: AssignmentUpdate,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    return await update_assignment(db, assignment_id, body)


@router.delete("/{assignment_id}", status_code=status.HTTP_204_NO_CONTENT)
async def assignments_delete(
    assignment_id: int,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    await delete_assignment(db, assignment_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
```

- [ ] **Step 6: Run tests** — `pytest tests/test_admin_assignments.py -v` then
`pytest -q` (expect all green: 47 + 3 = 50).

- [ ] **Step 7: Commit**

```bash
git add backend/schemas/admin_assignment.py backend/services/admin_assignments.py backend/routers/admin/assignments.py backend/tests/test_admin_assignments.py
git commit -m "feat(api): admin assignments update + delete endpoints"
```

---

## Self-review

**Spec coverage (vs §5.4 managers + assignments):**
- Per-test isolation so the mutating tests don't contaminate each other → Task 1. ✓
- `GET /admin/managers` + `/{id}` → Task 2. ✓
- `POST /admin/managers` (hash pw, 409 dup email) → Task 3. ✓
- `PATCH /admin/managers/{id}` + `POST /{id}/reset-password` → Task 3. ✓
- `GET /admin/assignments` + `POST` (409 already-assigned, validates manager+hotel) → Task 4. ✓
- `PATCH /admin/assignments/{id}` + `DELETE` → Task 5. ✓
- Deferred: competitors, monitoring, alerts (Plan 4), frontend (Plan 5).

**Placeholder scan:** none — every step has complete code/commands.

**Type consistency:** `AdminManagerRow` fields match `_LIST_SQL` (managers) columns;
`AdminAssignmentRow` matches `_LIST_SQL` (assignments) columns. UUID params bound
via `id::text = :id` (reads) and `CAST(:uid AS uuid)` (assignment insert). All
`user_hotel_assignments` INSERTs set `max_competitors` + `is_active` explicitly
(test-DB gotcha). `password_hash` is never selected, so it never leaks in responses.
Reset-password and delete return 204 via `Response`. `EmailStr` requires
`email-validator` (already in requirements.txt).

**Behaviour note:** with Task 1's per-test rollback isolation, every test starts
from the same committed seed and its writes are undone at teardown — so the
mutating tests (deactivate manager2, assign/delete) no longer leak across tests or
files. Tests still read IDs by email (not by position) for clarity.
