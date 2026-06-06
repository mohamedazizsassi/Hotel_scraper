# Admin API — Foundation + Hotels — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the server-side Admin gate and the complete Hotel-management API: list the pool, discover unregistered hotels from `hotel_features`, promote (create) a hotel, view its detail, and edit it.

**Architecture:** New `get_current_admin` dependency (mirrors `get_current_manager`) protects an `/admin/*` router group. Services use raw SQL via `text()` + `.mappings()` (the existing manager-service pattern). The test DB is built from ORM `create_all`, so a new `PlatformHotelSource` ORM model + conftest additions (admin user, `segment_dim`, an unregistered `hotel_features` row) are what make admin tests exercisable.

**Tech Stack:** FastAPI, SQLAlchemy 2.0 async, asyncpg, PostgreSQL, pytest + pytest-asyncio (`asyncio_mode=auto`).

**Plan map:** Plan 2 = foundation + hotels (this doc). Plan 3 = managers + assignments. Plan 4 = competitors (admin-only selection, D11) + monitoring + alerts. Plan 5 = frontend.

---

## Conventions for every backend command

- This work is in the `feat/admin-platform` **git worktree**:
  `C:\Users\ASUS\Desktop\PFE\revway\.claude\worktrees\feat+admin-platform`.
  NEVER touch the main checkout. Verify before committing:
  `git rev-parse --show-toplevel` ends in `.claude/worktrees/feat+admin-platform`;
  `git branch --show-current` prints `feat/admin-platform`.
- The worktree has **no local `.venv`** — use the existing interpreter for every
  python/pytest command:
  `C:\Users\ASUS\Desktop\PFE\revway\backend\.venv\Scripts\python.exe`
- Run tests from the worktree `backend/` dir, e.g.:
  `"C:\Users\ASUS\Desktop\PFE\revway\backend\.venv\Scripts\python.exe" -m pytest tests/test_admin_hotels.py -v`
- `backend/.env` (copied) makes `test_db_url` → `revway_test` (running). Migrations
  001–004 are already applied to dev `revway`; the test DB is rebuilt from ORM
  models each session by the autouse `setup_test_db` fixture.
- Commits: Conventional Commits, **no `Co-Authored-By: Claude` trailer**.

---

### Task 1: Admin auth gate + 409/400 exceptions

**Files:**
- Modify: `backend/core/exceptions.py`
- Modify: `backend/core/dependencies.py`
- Modify: `backend/tests/conftest.py`
- Test: `backend/tests/test_admin_auth.py`

- [ ] **Step 1: Write the failing test** — Create `backend/tests/test_admin_auth.py`:

```python
import uuid
import pytest
from db.models import User
from core.dependencies import get_current_admin
from core.exceptions import ForbiddenError


def _user(role: str) -> User:
    return User(id=uuid.uuid4(), email="x@y.tn", password_hash="h", full_name="X",
                role=role, is_active=True)


async def test_get_current_admin_allows_admin():
    u = _user("admin")
    assert await get_current_admin(u) is u


async def test_get_current_admin_rejects_manager():
    with pytest.raises(ForbiddenError):
        await get_current_admin(_user("manager"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_admin_auth.py -v`
Expected: FAIL — `ImportError: cannot import name 'get_current_admin'`.

- [ ] **Step 3: Add the dependency** — In `backend/core/dependencies.py`, append:

```python
async def get_current_admin(
    user: User = Depends(get_current_user),
) -> User:
    if user.role != "admin":
        raise ForbiddenError("Admin role required")
    return user
```

(`User`, `Depends`, and `ForbiddenError` are already imported in that file.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_admin_auth.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Add 409/400 exceptions (used by later admin tasks)** — In
`backend/core/exceptions.py`, add the classes (after `NotFoundError`):

```python
class ConflictError(Exception):
    def __init__(self, detail: str = "Conflict"):
        self.detail = detail

class BadRequestError(Exception):
    def __init__(self, detail: str = "Bad request"):
        self.detail = detail
```

And register their handlers inside `register_exception_handlers` (next to the others):

```python
    @app.exception_handler(ConflictError)
    async def conflict_error_handler(request: Request, exc: ConflictError):
        return JSONResponse(status_code=409, content={"detail": exc.detail})

    @app.exception_handler(BadRequestError)
    async def bad_request_error_handler(request: Request, exc: BadRequestError):
        return JSONResponse(status_code=400, content={"detail": exc.detail})
```

- [ ] **Step 6: Seed an admin user + admin-token helper in conftest** — In
`backend/tests/conftest.py`, inside `setup_test_db` (after the manager user is
seeded), add an admin user:

```python
        # Seed admin user (for /admin/* tests)
        admin_id = str(uuid.uuid4())
        admin_pw = hash_password("adminpass")
        await conn.execute(text(
            f"INSERT INTO users (id, email, password_hash, full_name, role, is_active) "
            f"VALUES ('{admin_id}', 'admin@test.com', '{admin_pw}', 'Test Admin', 'admin', true)"
        ))
```

(`uuid`, `hash_password`, and `text` are already imported in conftest.) Admin
tests mint a token directly, mirroring `test_recommendations._manager_token`:
`create_access_token(str(admin.id), hotel_id=None, role="admin")`.

- [ ] **Step 7: Run the full suite (no regressions)**

Run: `pytest -q`
Expected: all green (26 prior + 2 new = 28).

- [ ] **Step 8: Commit**

```bash
git add backend/core/exceptions.py backend/core/dependencies.py backend/tests/conftest.py backend/tests/test_admin_auth.py
git commit -m "feat(api): admin auth dependency + 409/400 exceptions + admin test seed"
```

---

### Task 2: `PlatformHotelSource` ORM + admin test data (sources, segment_dim, discoverable hotel)

**Files:**
- Modify: `backend/db/models.py`
- Modify: `backend/tests/conftest.py`
- Test: `backend/tests/test_platform_hotel_source_model.py`

- [ ] **Step 1: Write the failing test** — Create
`backend/tests/test_platform_hotel_source_model.py`:

```python
from sqlalchemy import select
from db.models import PlatformHotelSource


async def test_platform_hotel_source_round_trip(db_session):
    src = PlatformHotelSource(
        platform_hotel_id=1, source="promohotel",
        source_hotel_name="Hotel Comp 1 (promo)", source_city_id=10,
    )
    db_session.add(src)
    await db_session.commit()
    res = await db_session.execute(
        select(PlatformHotelSource).where(PlatformHotelSource.platform_hotel_id == 1)
    )
    row = res.scalars().first()
    assert row is not None
    assert row.source == "promohotel"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_platform_hotel_source_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'PlatformHotelSource'`.

- [ ] **Step 3: Add the ORM model** — In `backend/db/models.py`, append:

```python
class PlatformHotelSource(Base):
    __tablename__ = "platform_hotel_sources"
    platform_hotel_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("platform_hotels.id"), primary_key=True)
    source: Mapped[str] = mapped_column(String, primary_key=True)
    source_hotel_name: Mapped[str] = mapped_column(String, nullable=False)
    source_city_id: Mapped[Optional[int]] = mapped_column(Integer)
    last_seen_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now())
```

(`Integer`, `String`, `ForeignKey`, `DateTime`, `func`, `Optional`, `datetime`,
`Mapped`, `mapped_column` are already imported.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_platform_hotel_source_model.py -v`
Expected: PASS.

- [ ] **Step 5: Add test data for hotels list/discoverable/region** — In
`backend/tests/conftest.py` `setup_test_db`, after the existing `hotel_features`
INSERT, add (a) a `segment_dim` table + row (region derivation), (b) a source row
for a registered hotel, and (c) an extra `hotel_features` row for a hotel NOT in
`platform_hotels` (so "discoverable" returns something):

```python
        # segment_dim (region source for admin hotel list); ML-owned in prod.
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS segment_dim (
                city_name text, stars_int int, macro_region text,
                stars_band text, market_segment_id int
            )
        """))
        await conn.execute(text(
            "INSERT INTO segment_dim (city_name, stars_int, macro_region, stars_band, market_segment_id) "
            "VALUES ('hammamet', 4, 'cap_bon', '4-5', 1)"
        ))
        # A source link for a registered hotel
        await conn.execute(text("""
            INSERT INTO platform_hotel_sources (platform_hotel_id, source, source_hotel_name)
            VALUES ((SELECT id FROM platform_hotels WHERE hotel_name_normalized='hotel_comp_1'),
                    'promohotel', 'Hotel Comp 1')
        """))
        # A hotel present in features but NOT registered → must appear in /discoverable
        await conn.execute(text("""
            INSERT INTO hotel_features
              (hotel_name_normalized, city_name, stars_int, check_in, nights, adults,
               boarding_canonical, room_base, room_view, room_tier, room_occupancy,
               price, price_per_night, scraped_at, peer_medium_median, peer_medium_count)
            VALUES
              ('hotel_unregistered', 'sousse', 3, DATE '2026-07-01', 2, 2,
               'BB', 'chambre', 'mer', '', 'double', 600.0, 300.0,
               '2026-05-18T10:00:00', 320.0, 5)
        """))
```

Also drop `segment_dim` in the fixture teardown (next to the existing
`DROP TABLE IF EXISTS hotel_features`):

```python
        await conn.execute(text("DROP TABLE IF EXISTS segment_dim"))
```

- [ ] **Step 6: Run the full suite (no regressions)**

Run: `pytest -q`
Expected: all green (29 now).

- [ ] **Step 7: Commit**

```bash
git add backend/db/models.py backend/tests/conftest.py backend/tests/test_platform_hotel_source_model.py
git commit -m "feat(db): PlatformHotelSource ORM + admin hotel test fixtures"
```

---

### Task 3: Hotels — list + discoverable (GET)

**Files:**
- Create: `backend/schemas/admin_hotel.py`
- Create: `backend/services/admin_hotels.py`
- Create: `backend/routers/admin/__init__.py`
- Create: `backend/routers/admin/hotels.py`
- Modify: `backend/main.py`
- Test: `backend/tests/test_admin_hotels.py`

- [ ] **Step 1: Write the failing test** — Create `backend/tests/test_admin_hotels.py`:

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


async def test_list_hotels_requires_admin(client, db_session):
    # no token
    assert (await client.get("/admin/hotels")).status_code == 401
    # manager token → forbidden
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
    assert "hotel_comp_1" in names          # registered
    assert "hotel_unregistered" not in names  # not in the pool
    row = next(h for h in body["data"] if h["hotel_name_normalized"] == "hotel_comp_1")
    for f in ("id", "hotel_name_display", "city_name", "stars_int", "is_active",
              "region", "sources", "manager_name"):
        assert f in row


async def test_discoverable_excludes_registered(client, db_session):
    tok = await _admin_token(db_session)
    r = await client.get("/admin/hotels/discoverable", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    names = {h["hotel_name_normalized"] for h in r.json()["data"]}
    assert "hotel_unregistered" in names    # in features, not registered
    assert "hotel_comp_1" not in names      # already registered
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_admin_hotels.py -v`
Expected: FAIL — 404s (router not mounted) / import errors.

- [ ] **Step 3: Schemas** — Create `backend/schemas/admin_hotel.py`:

```python
from __future__ import annotations
from pydantic import BaseModel
from schemas.common import DataResponse


class AdminHotelRow(BaseModel):
    id: int
    hotel_name_normalized: str
    hotel_name_display: str
    city_name: str
    stars_int: int | None
    is_active: bool
    region: str | None
    contact_email: str | None
    contact_phone: str | None
    sources: str
    manager_id: str | None
    manager_name: str | None
    latest_scraped_at: str | None


class DiscoverableHotel(BaseModel):
    hotel_name_normalized: str
    city_name: str
    stars_int: int | None


AdminHotelListResponse = DataResponse[AdminHotelRow]
DiscoverableResponse = DataResponse[DiscoverableHotel]
```

- [ ] **Step 4: Service** — Create `backend/services/admin_hotels.py`:

```python
from __future__ import annotations
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from schemas.admin_hotel import AdminHotelRow, DiscoverableHotel

_LIST_SQL = text("""
    SELECT ph.id,
           ph.hotel_name_normalized,
           ph.hotel_name_display,
           c.name_normalized                              AS city_name,
           ph.stars_int,
           ph.is_active,
           sd.macro_region                                AS region,
           ph.contact_email,
           ph.contact_phone,
           COALESCE(string_agg(DISTINCT phs.source, ',' ORDER BY phs.source), '') AS sources,
           u.id::text                                     AS manager_id,
           u.full_name                                    AS manager_name,
           MAX(hf.scraped_at)                             AS latest_scraped_at
    FROM platform_hotels ph
    JOIN cities c ON c.id = ph.city_id
    LEFT JOIN segment_dim sd
           ON sd.city_name = c.name_normalized AND sd.stars_int = ph.stars_int
    LEFT JOIN user_hotel_assignments uha
           ON uha.hotel_id = ph.id AND uha.is_active = TRUE
    LEFT JOIN users u ON u.id = uha.user_id
    LEFT JOIN platform_hotel_sources phs ON phs.platform_hotel_id = ph.id
    LEFT JOIN hotel_features hf ON hf.hotel_name_normalized = ph.hotel_name_normalized
    GROUP BY ph.id, c.name_normalized, sd.macro_region, u.id, u.full_name
    ORDER BY ph.hotel_name_display
""")

_DISCOVERABLE_SQL = text("""
    SELECT DISTINCT hf.hotel_name_normalized, hf.city_name, hf.stars_int
    FROM hotel_features hf
    WHERE NOT EXISTS (
        SELECT 1 FROM platform_hotels ph
        JOIN cities c ON c.id = ph.city_id
        WHERE ph.hotel_name_normalized = hf.hotel_name_normalized
          AND c.name_normalized = hf.city_name
    )
    ORDER BY hf.city_name, hf.hotel_name_normalized
""")


async def list_hotels(db: AsyncSession) -> list[AdminHotelRow]:
    rows = (await db.execute(_LIST_SQL)).mappings().fetchall()
    return [AdminHotelRow(**dict(r)) for r in rows]


async def list_discoverable(db: AsyncSession) -> list[DiscoverableHotel]:
    rows = (await db.execute(_DISCOVERABLE_SQL)).mappings().fetchall()
    return [DiscoverableHotel(**dict(r)) for r in rows]
```

- [ ] **Step 5: Router + package** — Create `backend/routers/admin/__init__.py` (empty),
then `backend/routers/admin/hotels.py`:

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from core.dependencies import get_db, get_current_admin
from db.models import User
from schemas.admin_hotel import AdminHotelListResponse, DiscoverableResponse
from services.admin_hotels import list_hotels, list_discoverable

router = APIRouter(prefix="/admin/hotels", tags=["admin"])


@router.get("", response_model=AdminHotelListResponse)
async def hotels_list(
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await list_hotels(db)
    return AdminHotelListResponse(data=rows, count=len(rows))


@router.get("/discoverable", response_model=DiscoverableResponse)
async def hotels_discoverable(
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    rows = await list_discoverable(db)
    return DiscoverableResponse(data=rows, count=len(rows))
```

- [ ] **Step 6: Mount the router** — In `backend/main.py`, extend the import +
includes block at the bottom:

```python
from routers import auth, calendar, competitors, recommendations, anomalies
from routers.admin import hotels as admin_hotels
app.include_router(auth.router)
app.include_router(calendar.router)
app.include_router(competitors.router)
app.include_router(recommendations.router)
app.include_router(anomalies.router)
app.include_router(admin_hotels.router)
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `pytest tests/test_admin_hotels.py -v`
Expected: PASS (3 tests). Then `pytest -q` → all green.

- [ ] **Step 8: Commit**

```bash
git add backend/schemas/admin_hotel.py backend/services/admin_hotels.py backend/routers/admin/__init__.py backend/routers/admin/hotels.py backend/main.py backend/tests/test_admin_hotels.py
git commit -m "feat(api): admin hotels list + discoverable endpoints"
```

---

### Task 4: Hotels — create / promote (POST)

**Files:**
- Modify: `backend/schemas/admin_hotel.py`
- Modify: `backend/services/admin_hotels.py`
- Modify: `backend/routers/admin/hotels.py`
- Test: `backend/tests/test_admin_hotels.py` (extend)

- [ ] **Step 1: Write the failing test** — Append to `backend/tests/test_admin_hotels.py`:

```python
async def test_create_hotel_promotes_discoverable(client, db_session):
    tok = await _admin_token(db_session)
    payload = {
        "hotel_name_normalized": "hotel_unregistered",
        "hotel_name_display": "Hotel Unregistered",
        "city_name": "sousse",
        "stars_int": 3,
        "contact_email": "info@unreg.tn",
        "sources": ["promohotel"],
    }
    r = await client.post("/admin/hotels", json=payload,
                          headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 201, r.text
    created = r.json()
    assert created["hotel_name_normalized"] == "hotel_unregistered"
    assert created["id"] > 0
    # now it disappears from discoverable and appears in the pool
    disc = await client.get("/admin/hotels/discoverable",
                            headers={"Authorization": f"Bearer {tok}"})
    assert "hotel_unregistered" not in {h["hotel_name_normalized"] for h in disc.json()["data"]}


async def test_create_hotel_duplicate_conflicts(client, db_session):
    tok = await _admin_token(db_session)
    payload = {"hotel_name_normalized": "hotel_comp_1",
               "hotel_name_display": "Dup", "city_name": "hammamet", "stars_int": 4}
    r = await client.post("/admin/hotels", json=payload,
                          headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 409
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_admin_hotels.py -k create -v`
Expected: FAIL (404/422 — endpoint/schema absent).

- [ ] **Step 3: Schema** — In `backend/schemas/admin_hotel.py`, add:

```python
class HotelCreate(BaseModel):
    hotel_name_normalized: str
    hotel_name_display: str
    city_name: str
    stars_int: int | None = None
    contact_email: str | None = None
    contact_phone: str | None = None
    sources: list[str] = []
```

- [ ] **Step 4: Service** — In `backend/services/admin_hotels.py`, add (and add
`from schemas.admin_hotel import HotelCreate, AdminHotelRow` is already importing
the row type; extend the import line to include `HotelCreate`). Implement:

```python
from core.exceptions import ConflictError

_VALID_SOURCES = {"promohotel", "tunisiepromo"}


async def create_hotel(db: AsyncSession, body: HotelCreate) -> AdminHotelRow:
    # resolve or insert city
    city_id = await db.scalar(
        text("SELECT id FROM cities WHERE name_normalized = :c"), {"c": body.city_name})
    if city_id is None:
        city_id = await db.scalar(
            text("INSERT INTO cities (name_normalized, name_display) "
                 "VALUES (:c, :d) RETURNING id"),
            {"c": body.city_name, "d": body.city_name.title()})

    dup = await db.scalar(
        text("SELECT 1 FROM platform_hotels "
             "WHERE hotel_name_normalized = :n AND city_id = :cid"),
        {"n": body.hotel_name_normalized, "cid": city_id})
    if dup:
        raise ConflictError(f"Hotel '{body.hotel_name_normalized}' already registered in this city")

    hotel_id = await db.scalar(
        text("""INSERT INTO platform_hotels
                  (hotel_name_normalized, hotel_name_display, city_id, stars_int,
                   contact_email, contact_phone)
                VALUES (:n, :d, :cid, :s, :ce, :cp) RETURNING id"""),
        {"n": body.hotel_name_normalized, "d": body.hotel_name_display, "cid": city_id,
         "s": body.stars_int, "ce": body.contact_email, "cp": body.contact_phone})

    for src in body.sources:
        if src not in _VALID_SOURCES:
            continue
        await db.execute(
            text("""INSERT INTO platform_hotel_sources
                      (platform_hotel_id, source, source_hotel_name)
                    VALUES (:hid, :src, :name)"""),
            {"hid": hotel_id, "src": src, "name": body.hotel_name_display})

    await db.commit()
    row = (await db.execute(_LIST_SQL)).mappings().fetchall()
    created = next(r for r in row if r["id"] == hotel_id)
    return AdminHotelRow(**dict(created))
```

- [ ] **Step 5: Router** — In `backend/routers/admin/hotels.py`, add the POST
(extend imports with `AdminHotelRow`, `HotelCreate`, `create_hotel`):

```python
from fastapi import status

@router.post("", response_model=AdminHotelRow, status_code=status.HTTP_201_CREATED)
async def hotels_create(
    body: HotelCreate,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    return await create_hotel(db, body)
```

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_admin_hotels.py -v` then `pytest -q`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/schemas/admin_hotel.py backend/services/admin_hotels.py backend/routers/admin/hotels.py backend/tests/test_admin_hotels.py
git commit -m "feat(api): admin hotel create/promote endpoint"
```

---

### Task 5: Hotels — detail + update (GET {id}, PATCH {id})

**Files:**
- Modify: `backend/schemas/admin_hotel.py`
- Modify: `backend/services/admin_hotels.py`
- Modify: `backend/routers/admin/hotels.py`
- Test: `backend/tests/test_admin_hotels.py` (extend)

- [ ] **Step 1: Write the failing test** — Append to `backend/tests/test_admin_hotels.py`:

```python
async def _hotel_id(client, db_session, name="hotel_comp_1"):
    tok = await _admin_token(db_session)
    r = await client.get("/admin/hotels", headers={"Authorization": f"Bearer {tok}"})
    return next(h["id"] for h in r.json()["data"] if h["hotel_name_normalized"] == name)


async def test_hotel_detail(client, db_session):
    tok = await _admin_token(db_session)
    hid = await _hotel_id(client, db_session)
    r = await client.get(f"/admin/hotels/{hid}", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    assert r.json()["hotel_name_normalized"] == "hotel_comp_1"


async def test_hotel_detail_404(client, db_session):
    tok = await _admin_token(db_session)
    r = await client.get("/admin/hotels/999999", headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 404


async def test_hotel_update(client, db_session):
    tok = await _admin_token(db_session)
    hid = await _hotel_id(client, db_session)
    r = await client.patch(f"/admin/hotels/{hid}",
                           json={"hotel_name_display": "Renamed", "is_active": False,
                                 "contact_phone": "+216 99 999 999"},
                           headers={"Authorization": f"Bearer {tok}"})
    assert r.status_code == 200
    body = r.json()
    assert body["hotel_name_display"] == "Renamed"
    assert body["is_active"] is False
    assert body["contact_phone"] == "+216 99 999 999"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_admin_hotels.py -k "detail or update" -v`
Expected: FAIL (404 route / missing).

- [ ] **Step 3: Schema** — In `backend/schemas/admin_hotel.py`, add (note: at least
one field required so an empty PATCH is rejected by validation is optional; all
optional is fine — we only update provided fields):

```python
class HotelUpdate(BaseModel):
    hotel_name_display: str | None = None
    stars_int: int | None = None
    is_active: bool | None = None
    contact_email: str | None = None
    contact_phone: str | None = None
```

- [ ] **Step 4: Service** — In `backend/services/admin_hotels.py`, add (extend
imports with `HotelUpdate` and `from core.exceptions import NotFoundError`):

```python
async def _get_hotel_row(db: AsyncSession, hotel_id: int) -> AdminHotelRow:
    rows = (await db.execute(_LIST_SQL)).mappings().fetchall()
    match = next((r for r in rows if r["id"] == hotel_id), None)
    if match is None:
        raise NotFoundError(f"Hotel {hotel_id} not found")
    return AdminHotelRow(**dict(match))


async def get_hotel(db: AsyncSession, hotel_id: int) -> AdminHotelRow:
    return await _get_hotel_row(db, hotel_id)


async def update_hotel(db: AsyncSession, hotel_id: int, body: HotelUpdate) -> AdminHotelRow:
    fields = body.model_dump(exclude_unset=True)
    if not fields:
        return await _get_hotel_row(db, hotel_id)
    exists = await db.scalar(text("SELECT 1 FROM platform_hotels WHERE id = :id"),
                             {"id": hotel_id})
    if not exists:
        raise NotFoundError(f"Hotel {hotel_id} not found")
    allowed = {"hotel_name_display", "stars_int", "is_active", "contact_email", "contact_phone"}
    sets = ", ".join(f"{k} = :{k}" for k in fields if k in allowed)
    params = {k: v for k, v in fields.items() if k in allowed}
    params["id"] = hotel_id
    await db.execute(text(f"UPDATE platform_hotels SET {sets} WHERE id = :id"), params)
    await db.commit()
    return await _get_hotel_row(db, hotel_id)
```

- [ ] **Step 5: Router** — In `backend/routers/admin/hotels.py`, add (extend
imports with `HotelUpdate`, `get_hotel`, `update_hotel`):

```python
@router.get("/{hotel_id}", response_model=AdminHotelRow)
async def hotels_detail(
    hotel_id: int,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    return await get_hotel(db, hotel_id)


@router.patch("/{hotel_id}", response_model=AdminHotelRow)
async def hotels_update(
    hotel_id: int,
    body: HotelUpdate,
    _: User = Depends(get_current_admin),
    db: AsyncSession = Depends(get_db),
):
    return await update_hotel(db, hotel_id, body)
```

Note: define `/discoverable` BEFORE `/{hotel_id}` in the file (it already is) so
the literal route wins over the path param.

- [ ] **Step 6: Run tests**

Run: `pytest tests/test_admin_hotels.py -v` then `pytest -q`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/schemas/admin_hotel.py backend/services/admin_hotels.py backend/routers/admin/hotels.py backend/tests/test_admin_hotels.py
git commit -m "feat(api): admin hotel detail + update endpoints"
```

---

## Self-review

**Spec coverage (vs §5.1, §5.4 hotels, §6.4 region):**
- `get_current_admin` (§5.1) → Task 1. ✓
- 409/400 exceptions for later concerns → Task 1. ✓
- `PlatformHotelSource` ORM (§5.2) → Task 2. ✓
- `GET /admin/hotels` + `/discoverable` (§5.4) → Task 3. ✓
- `POST /admin/hotels` promote + source links + city resolve (§5.4) → Task 4. ✓
- `GET /admin/hotels/{id}` + `PATCH /admin/hotels/{id}` (§5.4) → Task 5. ✓
- Region derived from `segment_dim` (§6.4, D8) → Task 3 query + Task 2 fixture. ✓
- Admin-gating (manager → 403, anon → 401) → Task 3 test. ✓
- Deferred (Plan 3+): managers, assignments, competitors, monitoring, alerts, frontend. Listed in the plan map.

**Placeholder scan:** none — every step has complete code/SQL/commands.

**Type consistency:** `AdminHotelRow` fields match `_LIST_SQL` output columns
(`id, hotel_name_normalized, hotel_name_display, city_name, stars_int, is_active,
region, contact_email, contact_phone, sources, manager_id, manager_name,
latest_scraped_at`); `create_hotel`/`update_hotel`/`get_hotel` all re-read via
`_LIST_SQL` so every returned object is the same shape. `manager_id` is cast to
text in SQL to match the `str | None` schema. The router orders `/discoverable`
before `/{hotel_id}`. `HotelCreate.sources` validated against `_VALID_SOURCES`.

**Risk note:** `_LIST_SQL` is re-run after writes to return the row — fine at this
scale (small pool); if the pool grows large, switch to a single-id filtered query.
