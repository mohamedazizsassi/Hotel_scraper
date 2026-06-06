# RevWay — Admin / Platform Management — Design Spec

- **Date:** 2026-06-06
- **Status:** Approved (brainstorm complete) — ready for implementation plan
- **Author:** brainstorm session (mohamedazizsassi)
- **Scope owner:** RevWay PFE, defense ~2026-06-15

---

## 1. Problem & goal

RevWay has a Manager experience but **no working Admin**. The platform DB and the
Angular admin *pages* exist, but there is no admin backend and the pages run on
mock data. This work makes the **Admin role fully functional end-to-end**:
manage the hotel pool, manage manager accounts, assign managers to hotels,
monitor the scraping pipeline, and view collection alerts.

This satisfies the report's Chapter 3 admin use cases plus the more detailed
functional breakdown the user supplied (modules 1, 2.1–2.5, and the admin
dashboard in §5).

### Goals
- Admin can create/edit/(de)activate hotels, sourced from real scraped data.
- Admin can create/update/enable/disable manager accounts and reset passwords.
- Admin can assign / change / remove a manager↔hotel link.
- Admin can monitor scraping: real run history, total rows, rows added per
  scrape and per day, success/error stats.
- Admin can see derived collection alerts (failed / low-volume / missing runs).
- All of the above is real (no mock) and respects the frozen training snapshot
  (the scraper stays paused and untouched).

### Non-goals (explicitly out of scope)
- **No scraper changes** — no live instrumentation; the scraper, `items.py`, and
  the dedup/observation key are untouched.
- **No hard delete** of hotels — soft `is_active=false` only.
- **No separate `username`** — email is the login identifier.
- **No email/invite flow or password self-service** — admin sets/resets passwords.
- **No alert acknowledge/dismiss persistence** — alerts are computed read-only.
- **No manager-module enhancements** (calendar day/week/month, competitor
  search/filter, "distance" filter — the schema has no geo data, so distance is
  not buildable anyway). The manager module already exists and is out of scope.

---

## 2. Current state (verified 2026-06-06)

- **Database** — `001_create_platform_tables.sql` already defines: `cities`,
  `users` (`role ∈ {admin,manager}`), `platform_hotels`, `platform_hotel_sources`,
  `user_hotel_assignments` (1:1 via `UNIQUE(user_id)`), `user_competitor_selections`,
  with `updated_at` triggers and domain triggers (competitor-not-self, max cap).
  `002_create_hotel_features_view.sql` defines `hotel_features_full`.
  Seeds exist for cities, admin (`02_seed_admin.sql`), manager.
- **Backend** — FastAPI mounts `auth` + four `/manager/*` routers (calendar,
  competitors, recommendations, anomalies). `core/dependencies.py` has
  `get_current_user`, `get_current_manager` — **no `get_current_admin`**.
  `core/security.py` has `hash_password`/`verify_password` (bcrypt) and
  `create_access_token(user_id, hotel_id, role)`. Services use raw SQL via
  `text()` + `.mappings()`. **PG-only — no Mongo client.**
- **Frontend** — Angular 19 standalone/signals. Admin pages exist
  (`features/admin/{dashboard,hotels,managers,assignments,scrapers}`) on
  `core/data/mock.ts`. `roleGuard('admin')` already protects `/admin/*`.
  `AuthService` (real login → JWT `{sub,role,hotel_id,exp}`), `ApiService`
  (manager endpoints, signals loading/error pattern) already exist.
- **Scraper** — paused (frozen snapshot for model training). `scraper/logs/`
  holds ~88 `run_<date>_<time>.log` files. Each ends with Scrapy stats dumps
  (`item_scraped_count`, `log_count/ERROR`, `elapsed_time_seconds`,
  `finish_reason`, `finish_time`). Pipeline: Normalization → DuplicateFilter
  (in-memory, per-run) → **MongoDB `insert_one` (plain insert, no unique index)**
  → Parquet. **Therefore `item_scraped_count` == rows inserted into
  `hotel_prices` for that run.**

---

## 3. Key decisions (locked)

| # | Decision | Rationale |
|---|----------|-----------|
| D1 | Full-stack, all 3 admin areas + monitoring + alerts | User chose full scope |
| D2 | Scraper monitoring sourced from **logs + one Mongo count** | Real data, no scraper change, respects frozen snapshot |
| D3 | Monitoring history materialized in a **`scrape_runs` PG table** via an idempotent loader | Fast reads, durable, queryable, parsing out of request path; strong data-model story for the defense |
| D4 | Hotel onboarding is **discovery-based** | Admin promotes hotels that exist in `hotel_features`, guaranteeing real backing data; no orphan hotels |
| D5 | Rows-added series from `scrape_runs.items_total`; total from Mongo `estimated_document_count()` | Insert-only pipeline makes log counts exact; estimated count is instant (no scan) |
| D6 | Add Mongo client to backend (**`motor`**), used **only** for the total count | User wants the raw total; single metadata call ⇒ no full-scan risk |
| D7 | **Collection Alerts** included, derived read-only from `scrape_runs` | In the breakdown + report reliability NFR; demonstrable on historical data |
| D8 | Hotel **detail views** + **Region** (derived from `segment_dim.macro_region`) | Cheap; no schema change for region |
| D9 | Hotel **contact fields** (`contact_email`, `contact_phone`) added via migration | User selected; cosmetic but requested |
| D10 | Soft-deactivate only; **no hard delete**; **email = username** | Safer (FKs/history); avoids redundant unique field |

---

## 4. Architecture

Admin completes a vertical that already exists in 3 of 4 tiers:

```
DB (ready + 2 migrations)  →  Backend (NEW /admin/*)  →  Frontend (wire pages)
Scraper logs (read-only)   →  loader → scrape_runs (PG)
MongoDB hotel_prices       →  motor: estimated_document_count() only
```

Auth needs no structural change: JWT already carries `role`; admin token is
`role='admin'`, `hotel_id=null`; `02_seed_admin.sql` provides the login;
`roleGuard('admin')` already exists. We only add the **server-side**
`get_current_admin` gate (UI guards are UX-only).

**Data source per feature:** CRUD → PG platform tables · hotel discovery → PG
`hotel_features` (distinct identities) · region → PG `segment_dim` ·
monitoring history → PG `scrape_runs` (from logs) · total rows → Mongo ·
alerts → derived from PG `scrape_runs`.

---

## 5. Backend design (`backend/`)

### 5.1 Auth dependency
`core/dependencies.py` → add:
```python
async def get_current_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise ForbiddenError("Admin role required")
    return user
```

### 5.2 ORM (`db/models.py`)
- Extend `PlatformHotel` with `contact_email`, `contact_phone` (nullable).
- Add `PlatformHotelSource` model (maps to existing table).
- Add `ScrapeRun` model (new table, §6).

### 5.3 Mongo client
- `db/mongo.py` — lazy `motor.motor_asyncio.AsyncIOMotorClient`; expose
  `get_hotel_prices_count()` → `estimated_document_count()`.
- `core/config.py` — add `mongo_uri`, `mongo_db` (env `MONGO_URI`, `MONGO_DB`).
- `requirements.txt` — add `motor`.
- `main.py` lifespan — open/close the Mongo client; `/health` reports mongo
  reachability (degrade gracefully if Mongo is down → total shows `null`).

### 5.4 Routers (`routers/admin/`, all `Depends(get_current_admin)`, tag `admin`)
One file per concern (mirrors existing layout):

**hotels.py**
- `GET  /admin/hotels` — registered pool: id, display name, city, region
  (derived), stars, is_active, assigned manager, source(s), latest scrape.
- `GET  /admin/hotels/discoverable` — distinct
  `(hotel_name_normalized, city_name, stars_int)` in `hotel_features` not yet
  in `platform_hotels`.
- `GET  /admin/hotels/{id}` — detail (fields + sources + assignment + recent
  scrape volume).
- `POST /admin/hotels` — promote a discovered hotel: resolve/insert city,
  insert `platform_hotels`, insert `platform_hotel_sources` link(s); accept
  optional contact fields.
- `PATCH /admin/hotels/{id}` — display name, stars, contact fields, is_active.

**managers.py**
- `GET  /admin/managers` — users where `role='manager'` + assignment + status.
- `GET  /admin/managers/{id}` — detail.
- `POST /admin/managers` — create user (`hash_password(initial_password)`).
- `PATCH /admin/managers/{id}` — full_name, email, is_active.
- `POST /admin/managers/{id}/reset-password` — set new password hash.

**assignments.py**
- `GET    /admin/assignments` — manager↔hotel list.
- `POST   /admin/assignments` — assign (respect `UNIQUE(user_id)`,
  set `max_competitors`; 409 if manager already assigned).
- `PATCH  /admin/assignments/{id}` — change hotel / max_competitors / is_active.
- `DELETE /admin/assignments/{id}` — remove assignment.

**monitoring.py**
- `GET /admin/monitoring/summary` — `{ total_rows (Mongo), rows_added_today,
  rows_added_last_run, latest_scrape_at, last_run_status, error_rate,
  runs_count, hotels_scraped_distinct, integrity_logged_vs_mongo }`.
- `GET /admin/monitoring/runs?limit=N` — per-scrape: `{ run_ts, source,
  items_total (=rows added), errors_total, duration_s, status }`.
- `GET /admin/monitoring/daily?days=N` — per-day rows-added rollup for the chart.

**alerts.py**
- `GET /admin/alerts` — derived alert list + history (§7).

### 5.5 Services (`services/admin/`)
`hotels_service`, `managers_service`, `assignments_service`,
`monitoring_service`, `alerts_service` — raw SQL + `.mappings()` per existing
pattern; `monitoring_service` also calls `db/mongo.py` for the total.

### 5.6 Schemas (`schemas/admin/` or `schemas/admin.py`)
Pydantic request/response models per concern (HotelCreate/Update/Detail/List,
ManagerCreate/Update/Detail/List, AssignmentCreate/Update, RunRow, DailyRow,
MonitoringSummary, Alert). Reuse the existing `DataResponse[T]` envelope where
the manager endpoints already use it.

### 5.7 Tests (`tests/test_admin_*.py`)
- Auth gating: manager/anon → 403/401 on every `/admin/*` route.
- Hotels: discoverable excludes already-registered; promote inserts hotel +
  source; PATCH updates contact/active.
- Managers: create hashes password; reset-password changes hash; disable sets
  is_active.
- Assignments: assign honours 1:1 (409 on second); remove works.
- Monitoring: summary shape; runs/daily ordering; total falls back to `null`
  when Mongo unavailable.
- Alerts: synthetic `scrape_runs` rows produce the expected alert types.

---

## 6. Database design (`database/postgres/`)

### 6.1 `migrations/003_create_scrape_runs.sql`
```sql
CREATE TABLE scrape_runs (
    id            INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_ts        TIMESTAMPTZ NOT NULL,           -- parsed from filename
    log_filename  TEXT NOT NULL UNIQUE,           -- idempotency key
    source        TEXT,                            -- promohotel/tunisiepromo/mixed
    spiders_count INTEGER NOT NULL DEFAULT 0,
    items_total   INTEGER NOT NULL DEFAULT 0,      -- = rows added to hotel_prices
    errors_total  INTEGER NOT NULL DEFAULT 0,
    duration_s    NUMERIC,
    status        TEXT NOT NULL,                   -- finished / partial / failed
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_scrape_runs_ts ON scrape_runs (run_ts);
INSERT INTO schema_migrations (version, description)
VALUES ('003', 'scrape_runs: per-run scrape stats parsed from logs');
```
Granularity: **one row per log file** (= one scheduled run). A run log may
contain both sources' spiders → `source` may be `mixed`; per-source breakdown is
derivable later if needed.

### 6.2 `migrations/004_add_hotel_contact_columns.sql`
```sql
ALTER TABLE platform_hotels
    ADD COLUMN contact_email TEXT,
    ADD COLUMN contact_phone TEXT;
INSERT INTO schema_migrations (version, description)
VALUES ('004', 'platform_hotels: contact_email, contact_phone');
```

### 6.3 Loader `backend/scripts/load_scrape_runs.py`
Idempotent: for each `scraper/logs/run_*.log`, parse every Scrapy stats block,
sum `item_scraped_count`/`log_count/ERROR`, take max `elapsed_time_seconds`,
derive `status` (`finished` if all `finish_reason='finished'`, else
`partial`/`failed`), parse `run_ts` from the filename, `UPSERT ON CONFLICT
(log_filename)`. Re-runnable; configurable logs path. One pass backfills all 88
runs. Read-only on the scraper.

### 6.4 Region derivation (no migration)
Region = `segment_dim.macro_region` joined on `(city_name, stars_int)`:
`platform_hotels ph JOIN cities c ON c.id=ph.city_id LEFT JOIN segment_dim sd ON
sd.city_name=c.name_normalized AND sd.stars_int=ph.stars_int`.

---

## 7. Monitoring & alerts logic

### 7.1 Totals & series
- **Total rows** = Mongo `estimated_document_count('hotel_prices')` (instant).
- **Rows added per run** = `scrape_runs.items_total` (exact; insert-only).
- **Rows added per day** = `SUM(items_total) GROUP BY date(run_ts)`.
- **Integrity cross-check** = `SUM(items_total)` vs Mongo total (≈ equal).
- **Hotels scraped (distinct)** = `COUNT(DISTINCT hotel_name_normalized)` from
  `hotel_features` (what is actually serveable).
- **Scraper status** = healthy / failed / **stale (paused)** based on the latest
  `run_ts` vs expected cadence; honestly shows "paused since <last run>".

### 7.2 Alert types (derived read-only from `scrape_runs`)
1. **Failed run** (error) — `status != 'finished'`.
2. **Low-volume run** (warning, = report's reliability NFR) — `items_total` below
   ~50% of the trailing median for that scrape slot.
3. **Missing scheduled run** (warning) — an expected ~10:00/~15:00 slot has no
   run on a date **within the active window** (evaluation stops at the last known
   run date so the pause doesn't fire infinite "missing" alerts).
4. **High error rate** (info/warning) — `errors_total` deviates above the rolling
   baseline (logs sit ~180/spider, so this is *relative*, not absolute).

`GET /admin/alerts` returns the chronological list (the "notification history").
No persistence/ack in v1.

---

## 8. Frontend design (`frontend/`)

### 8.1 API layer
- `core/api/admin-api.service.ts` — hotels (list/discoverable/detail/create/update),
  managers (list/detail/create/update/reset-password), assignments
  (list/create/update/delete), monitoring (summary/runs/daily), alerts (list).
- `core/api/admin-dto.ts`, `core/api/admin-adapters.ts` — wire DTOs + mapping.
- Reuse `AuthService` token via the existing `auth.interceptor`.

### 8.2 Wire existing pages (replace `mock.ts`, add forms; signals loading/error)
- **dashboard** — KPI cards: total rows, rows added today, last scrape freshness,
  error rate, total hotels, total managers, scraper status; **recent alerts** list.
- **hotels** — live table (name, city, **region**, stars, sources, latest scrape,
  manager, status). "+ Add hotel" modal (load discoverable → POST, optional
  contact). Row → **detail view**. Edit (name/stars/contact/active). Drop the
  mock "rooms" column (no such field).
- **managers** — live table; "Create manager" form (email, full_name, initial
  password); edit; enable/disable; reset password; row → **detail view**.
- **assignments** — live table; assign form (manager + hotel + max_competitors);
  change / remove.
- **scrapers** — runs table (date · scrape time · rows added · errors · duration
  · status) + per-day rows-added sparkline/bars (inline SVG, no chart lib) +
  total/freshness/error cards.

### 8.3 New page
- `features/admin/alerts/admin-alerts.component.ts` + route under `/admin/alerts`
  + nav entry in `admin-shell`. (Adding a route is allowed; do not rearrange the
  shell.) Design tokens stay locked.

---

## 9. Build order

1. **DB**: `003_create_scrape_runs.sql`, `004_add_hotel_contact_columns.sql`;
   run `load_scrape_runs.py` (backfill 88 runs).
2. **Backend**: config + `db/mongo.py` + `requirements` → ORM additions →
   `get_current_admin` → schemas → services → routers → mount in `main.py` →
   tests (pytest green).
3. **Frontend**: `admin-api.service` + dto/adapters → wire dashboard → hotels →
   managers → assignments → scrapers → new alerts page → nav.
4. **Verify**: pytest; `ng build`; manual admin login → exercise each page.

---

## 10. Report alignment (separate task, not code)

The detailed breakdown lists **5** admin use cases (Manage Hotels, Manage Manager
Accounts, Assign Manager, Monitor Scraping, View Collection Alerts) vs the
Chapter 3 diagram's **3**. After the build, update the report's global use-case
diagram + add UC-A1…A5 tables to match. English only, no custom boxes
(per project rules). Note for the manager UCs: the competitor **distance** filter
is not buildable (no geo data) — keep it out of the report.

---

## 11. Risks / open points

- **Mongo availability in backend** — if Mongo is down, total degrades to `null`;
  everything else (PG-sourced) still works.
- **`source` granularity in `scrape_runs`** — per-file `mixed` is acceptable for
  v1; per-source split is a later enhancement.
- **Alert thresholds** (low-volume %, error-rate baseline) — start with the
  values in §7, tune against the real 88-run history during implementation.
- **`hotel_features` freshness** — discovery + region + distinct-hotel counts
  reflect the last pipeline load, not live Mongo (acceptable; snapshot is frozen).
