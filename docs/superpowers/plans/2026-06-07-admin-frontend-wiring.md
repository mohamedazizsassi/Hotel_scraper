# Admin Frontend — Wiring to Real Data — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mocked admin pages (`core/data/mock.ts` → `HOTELS`,
`MANAGERS`, `SCRAPER_RUNS`) with live data from the `/admin/*` API built in
Plans 1–4 (64 backend tests passing, tip `48144ca`). Add the one screen that
doesn't exist yet: the admin's competitor-selection UI (D11 — admin picks each
manager's 3–4 competitors; the manager's own view stays read-only).

**Architecture:** Same pattern already proven in `ManagerCompetitorsComponent`
/ `ManagerAlertsComponent`: `inject(ApiService)`, `signal()` for
`data/loading/error`, `ngOnInit` subscribes, template uses `@if (loading())` /
`@else if (error())` / `@for`. Wire-format DTOs go in `core/api/dto.ts`
(snake_case, mirror the backend `*Row`/`*Summary` Pydantic schemas exactly —
see frontend `CLAUDE.md` rule "column names mirror Postgres", which extends
naturally to "DTO fields mirror the API schema"). `ApiService` gets one typed
method per endpoint, returning `Observable<T[]>` (list endpoints, unwrapping
`DataResponse<T>`) or `Observable<T>` (single-object endpoints like
`/admin/monitoring/summary`).

**Tech stack:** Angular 19 (standalone components, signals, new control flow),
RxJS for HTTP, the existing `authInterceptor` (attaches the admin JWT and
redirects to `/login` on 401 — already wired, nothing to change there).

**Plan map:** Plans 1–4 = admin backend (DONE, 64 tests). **Plan 5 = this
plan — admin frontend.** No Plan 6 currently scoped.

---

## Conventions for every frontend task

- Work in the `feat/admin-platform` **worktree**:
  `C:\Users\ASUS\Desktop\PFE\revway\.claude\worktrees\feat+admin-platform`.
  Never touch the main checkout (`feat/forecaster-bakeoff` — has in-flight ML
  work). Verify before committing: `git rev-parse --show-toplevel` ends in
  `.claude/worktrees/feat+admin-platform`; `git branch --show-current` =
  `feat/admin-platform`.
- The frontend tree was just copied into this worktree fresh (commit
  `08378b4`, "chore(frontend): add baseline frontend tree") — `node_modules`
  is gitignored and **does not exist yet**. **Task 1, Step 0** runs
  `npm install` once; it persists for all later tasks (don't repeat it).
- **No automated frontend test suite exists** (`frontend/CLAUDE.md` confirms:
  static demo surface, no `.spec.ts` files, no `test` script in
  `package.json`). The closest thing to a "red/green" gate is `ng build`
  (catches TypeScript + Angular template errors under `strictTemplates:
  true`). Each task: (a) make the change, (b) run `npm run build` from
  `frontend/` and confirm it succeeds with **zero new errors/warnings**, (c)
  start the dev server (`npm start`, runs at `http://localhost:4200`), log in
  as `admin@test.com` (or whichever seeded admin credential exists in the dev
  Postgres — check `backend/.env`/seed scripts if unsure), and **manually
  verify the page renders real data** — note exactly what you checked when you
  report back (this replaces the pytest "N passed" checkpoint from the backend
  plans).
- The **backend must be running** (`uvicorn main:app --reload` from
  `backend/`, against the dev Postgres `revway`) for manual verification —
  the dev DB has real `platform_hotels`/`users`/`scrape_runs` rows from the
  Plan 1–3 backfills. If Mongo is down, `total_rows` in the monitoring summary
  will correctly render as `null` / "—" — that's expected, not a bug (see
  [[project-scrape-runs-monitoring]]).
- Commits: Conventional Commits, **no `Co-Authored-By: Claude` trailer**.
- Follow `frontend/CLAUDE.md` conventions throughout: standalone components
  only, signals (not `BehaviorSubject`), new control flow (`@if`/`@for`/
  `@switch`, never `*ngIf`/`*ngFor`), `DatePipe`/`DecimalPipe` imported
  explicitly, inline SVG icons (no emojis), `.mono` + tabular-nums for all
  numbers, snake_case DTO fields matching the backend schema exactly (do not
  rename for cosmetics — see "column names mirror Postgres", which is the
  same discipline applied to API field names).

---

### Task 1: API plumbing — admin DTOs + `ApiService` methods

**Files:**
- Modify: `frontend/src/app/core/api/dto.ts`
- Modify: `frontend/src/app/core/api/api.service.ts`

This task adds NO UI — it's the typed contract layer every later task builds
on. Mirror the existing manager-side DTOs/methods exactly in style (see
`CompetitorDto` / `getCompetitors()` as the template).

- [ ] **Step 0 (one-time):** `cd frontend && npm install` (this populates
  gitignored `node_modules/`; takes a couple of minutes; do this ONCE, not
  per-task).

- [ ] **Step 1:** In `dto.ts`, ADD these interfaces — field names and types
  copied verbatim from the Pydantic schemas in
  `backend/schemas/admin_*.py` (Plans 2–4):

```typescript
// --- Admin: hotels (backend/schemas/admin_hotel.py — AdminHotelRow / DiscoverableHotel / HotelCreate / HotelUpdate) ---
export interface AdminHotelDto {
  id: number;
  hotel_name_normalized: string;
  hotel_name_display: string;
  city_name: string;
  stars_int: number | null;
  is_active: boolean;
  region: string | null;
  contact_email: string | null;
  contact_phone: string | null;
  sources: string;                    // e.g. "promohotel,tunisiepromo" — comma-joined, NOT a list
  manager_id: string | null;           // null = unassigned
  manager_name: string | null;
  latest_scraped_at: string | null;
}

// NOTE: deliberately has NO hotel_name_display and NO source/sources field —
// these candidates come straight off distinct hotel_features identities,
// which only carry the normalized name. The display name gets chosen at
// promote time (see HotelCreate below).
export interface DiscoverableHotelDto {
  hotel_name_normalized: string;
  city_name: string;
  stars_int: number | null;
}

export interface HotelCreateBody {
  hotel_name_normalized: string;
  hotel_name_display: string;
  city_name: string;
  stars_int?: number | null;
  contact_email?: string | null;
  contact_phone?: string | null;
  sources?: string[];
}

export interface HotelUpdateBody {
  hotel_name_display?: string | null;
  stars_int?: number | null;
  is_active?: boolean | null;
  contact_email?: string | null;
  contact_phone?: string | null;
}

// --- Admin: managers (backend/schemas/admin_manager.py — AdminManagerRow / ManagerCreate / ManagerUpdate / PasswordReset) ---
export interface AdminManagerDto {
  id: string;
  email: string;
  full_name: string | null;
  is_active: boolean;
  last_login_at: string | null;
  assigned_hotel_id: number | null;    // null = unassigned
  assigned_hotel_name: string | null;
}

export interface ManagerCreateBody {
  email: string;
  full_name?: string | null;
  initial_password: string;            // admin sets the manager's first password — there is no auto-generated invite flow
}

export interface ManagerUpdateBody {
  email?: string | null;
  full_name?: string | null;
  is_active?: boolean | null;
}

// --- Admin: assignments (backend/schemas/admin_assignment.py — AdminAssignmentRow / AssignmentCreate / AssignmentUpdate) ---
export interface AssignmentDto {
  id: number;
  user_id: string;
  manager_email: string;
  manager_name: string | null;
  hotel_id: number;
  hotel_name: string;
  max_competitors: number;
  is_active: boolean;
}

// --- Admin: competitors (backend/schemas/admin_competitor.py) ---
export interface AdminCompetitorRowDto {
  hotel_id: number;
  hotel_name_display: string;
  city_name: string;
  stars_int: number | null;
  display_order: number;
}

export interface SelectableHotelDto {
  hotel_id: number;
  hotel_name_display: string;
  city_name: string;
  stars_int: number | null;
}

// --- Admin: monitoring (backend/schemas/admin_monitoring.py) ---
export interface MonitoringSummaryDto {
  total_rows: number | null;
  logged_window_items: number;
  runs_count: number;
  finished_runs: number;
  failed_runs: number;
  latest_scrape_at: string | null;
  last_run_status: string | null;
  last_run_items: number | null;
  hotels_scraped_distinct: number;
}

export interface ScrapeRunDto {
  run_ts: string;
  log_filename: string;
  source: string | null;
  items_total: number;
  errors_total: number;
  duration_s: number | null;
  status: string;
}

export interface DailyRollupDto {
  day: string;
  items_total: number;
  runs: number;
}

// --- Admin: alerts (backend/schemas/admin_alert.py) ---
export interface AdminAlertDto {
  type: string;
  severity: string;
  message: string;
  run_ts: string;
  log_filename: string;
}
```

> **Provenance note:** every field above was copied directly from the
> committed `backend/schemas/admin_*.py` files (not from memory of the design
> discussion) — `AdminHotelRow`, `DiscoverableHotel`, `HotelCreate`,
> `HotelUpdate`, `AdminManagerRow`, `ManagerCreate`, `ManagerUpdate`,
> `PasswordReset`, `AdminAssignmentRow`, `AssignmentCreate`,
> `AssignmentUpdate`. You can still spot-check by opening those files
> side-by-side, but you should find them matching exactly — if you don't,
> something changed since this plan was written and you should treat the
> live file as ground truth.
>
> **Three things that surprised the plan author and are easy to get wrong by
> guessing instead of reading:**
> 1. `AdminHotelDto.sources` is a **comma-joined string** (`"promohotel,tunisiepromo"`),
>    not a list — but `HotelCreateBody.sources` (the create/promote request)
>    IS a `string[]`. Different shapes on read vs write — don't conflate them.
> 2. `DiscoverableHotelDto` carries only `hotel_name_normalized`, `city_name`,
>    `stars_int` — **no display name, no source**. The admin supplies the
>    display name (and everything else) at promote time via `HotelCreateBody`.
>    Your "+ Add hotel" panel (Task 3) needs a small form at promote time, not
>    a one-click "Promote" button.
> 3. `PasswordReset` takes `new_password: string` in the request body — **the
>    admin chooses the new password**, the backend does not generate and
>    return a temporary one. Task 4's reset-password UI must be a small
>    "set new password" form, not a "click to generate" button.

- [ ] **Step 2:** In `api.service.ts`, extend the DTO import to bring in all
  the new types from Step 1 (including the `*Body` request-shape interfaces),
  and ADD one method per endpoint, following the exact `getCompetitors()`
  shape (`.pipe(map(r => r.data))` for list endpoints that return
  `DataResponse<T>`; plain `.get<T>(...)`/`.post<T>(...)`/`.patch<T>(...)`
  for single-object endpoints — confirmed against `response_model=` in each
  router file, see inline comments below):

```typescript
  // --- Admin: hotels ---
  getAdminHotels(): Observable<AdminHotelDto[]> {
    return this.http
      .get<DataResponse<AdminHotelDto>>(`${API_BASE}/admin/hotels`)
      .pipe(map(r => r.data));
  }

  getDiscoverableHotels(): Observable<DiscoverableHotelDto[]> {
    return this.http
      .get<DataResponse<DiscoverableHotelDto>>(`${API_BASE}/admin/hotels/discoverable`)
      .pipe(map(r => r.data));
  }

  // POST /admin/hotels returns the bare AdminHotelRow (201), NOT a DataResponse —
  // confirmed via `response_model=AdminHotelRow` in routers/admin/hotels.py.
  promoteHotel(body: HotelCreateBody): Observable<AdminHotelDto> {
    return this.http.post<AdminHotelDto>(`${API_BASE}/admin/hotels`, body);
  }

  updateHotel(id: number, body: HotelUpdateBody): Observable<AdminHotelDto> {
    return this.http.patch<AdminHotelDto>(`${API_BASE}/admin/hotels/${id}`, body);
  }

  // --- Admin: managers ---
  getAdminManagers(): Observable<AdminManagerDto[]> {
    return this.http
      .get<DataResponse<AdminManagerDto>>(`${API_BASE}/admin/managers`)
      .pipe(map(r => r.data));
  }

  // POST/PATCH return the bare AdminManagerRow (201/200) — response_model=AdminManagerRow.
  createManager(body: ManagerCreateBody): Observable<AdminManagerDto> {
    return this.http.post<AdminManagerDto>(`${API_BASE}/admin/managers`, body);
  }

  updateManager(id: string, body: ManagerUpdateBody): Observable<AdminManagerDto> {
    return this.http.patch<AdminManagerDto>(`${API_BASE}/admin/managers/${id}`, body);
  }

  // 204 No Content — admin SUPPLIES new_password (not server-generated).
  // Observable<void> is correct; Angular maps a 204 body to null.
  resetManagerPassword(id: string, new_password: string): Observable<void> {
    return this.http.post<void>(`${API_BASE}/admin/managers/${id}/reset-password`, { new_password });
  }

  // --- Admin: assignments ---
  getAssignments(): Observable<AssignmentDto[]> {
    return this.http
      .get<DataResponse<AssignmentDto>>(`${API_BASE}/admin/assignments`)
      .pipe(map(r => r.data));
  }

  // POST/PATCH return the bare AdminAssignmentRow — response_model=AdminAssignmentRow.
  createAssignment(body: { user_id: string; hotel_id: number; max_competitors?: number }): Observable<AssignmentDto> {
    return this.http.post<AssignmentDto>(`${API_BASE}/admin/assignments`, body);
  }

  // 204 No Content.
  deleteAssignment(id: number): Observable<void> {
    return this.http.delete<void>(`${API_BASE}/admin/assignments/${id}`);
  }

  // --- Admin: competitors (D11 — admin-only selection) ---
  getManagerCompetitors(managerId: string): Observable<AdminCompetitorRowDto[]> {
    return this.http
      .get<DataResponse<AdminCompetitorRowDto>>(`${API_BASE}/admin/managers/${managerId}/competitors`)
      .pipe(map(r => r.data));
  }

  getSelectableCompetitors(managerId: string): Observable<SelectableHotelDto[]> {
    return this.http
      .get<DataResponse<SelectableHotelDto>>(`${API_BASE}/admin/managers/${managerId}/selectable-competitors`)
      .pipe(map(r => r.data));
  }

  setManagerCompetitors(managerId: string, hotel_ids: number[]): Observable<AdminCompetitorRowDto[]> {
    return this.http
      .put<DataResponse<AdminCompetitorRowDto>>(`${API_BASE}/admin/managers/${managerId}/competitors`, { hotel_ids })
      .pipe(map(r => r.data));
  }

  // --- Admin: monitoring + alerts ---
  getMonitoringSummary(): Observable<MonitoringSummaryDto> {
    return this.http.get<MonitoringSummaryDto>(`${API_BASE}/admin/monitoring/summary`);
  }

  getMonitoringRuns(limit = 50): Observable<ScrapeRunDto[]> {
    return this.http
      .get<DataResponse<ScrapeRunDto>>(`${API_BASE}/admin/monitoring/runs`, { params: toParams({ limit }) })
      .pipe(map(r => r.data));
  }

  getMonitoringDaily(days = 30): Observable<DailyRollupDto[]> {
    return this.http
      .get<DataResponse<DailyRollupDto>>(`${API_BASE}/admin/monitoring/daily`, { params: toParams({ days }) })
      .pipe(map(r => r.data));
  }

  getAdminAlerts(): Observable<AdminAlertDto[]> {
    return this.http
      .get<DataResponse<AdminAlertDto>>(`${API_BASE}/admin/alerts`)
      .pipe(map(r => r.data));
  }
```

- [ ] **Step 3:** Run `npm run build` from `frontend/` — confirm it compiles
  clean (this task adds no template references yet, so a clean build here
  just means the new types/methods type-check).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/core/api/dto.ts frontend/src/app/core/api/api.service.ts
git commit -m "feat(frontend): admin API DTOs + ApiService methods (Plan 5 plumbing)"
```

---

### Task 2: Wire Monitoring + Alerts (`admin-scrapers` page) + Dashboard summary

**Files:**
- Modify: `frontend/src/app/features/admin/scrapers/admin-scrapers.component.ts`
- Modify: `frontend/src/app/features/admin/dashboard/admin-dashboard.component.ts`

This is the highest defense-value page — it's where the live Mongo count and
the real scrape-run history show up. Rename the page concept from "Scrapers"
to "Monitoring" in the heading/copy if you judge it reads better given the
data is now about collection health, not live spider control — but **do not
rename the route** (`/admin/scrapers` is wired in `app.routes.ts` and the
sidebar; renaming the route is an off-limits "rearrange the shell" change per
`frontend/CLAUDE.md`).

- [ ] **Step 1:** Rewrite `AdminScrapersComponent` to follow the
  `ManagerAlertsComponent`/`ManagerCompetitorsComponent` pattern:
  - Inject `ApiService`. Add `summary = signal<MonitoringSummaryDto | null>(null)`,
    `runs = signal<ScrapeRunDto[]>([])`, `alerts = signal<AdminAlertDto[]>([])`,
    `loading = signal(true)`, `error = signal<string | null>(null)`.
  - In `ngOnInit`, fetch all three in parallel (use `forkJoin` from `rxjs` —
    look at how other multi-source components in this codebase combine
    streams, or simply subscribe to each independently updating its own
    signal + a shared `loading`/`error` pair).
  - Replace the four `KpiCardComponent` cards (currently hardcoded "Total
    runs (7d)" / "Success rate" / "Avg duration" / "Items / day") with cards
    driven by `summary()`: e.g. `total_rows` ("Rows in collection", with a
    "—" fallback when `null`), `runs_count`/`finished_runs`/`failed_runs`,
    `last_run_status` + `latest_scrape_at`, `hotels_scraped_distinct`. Pick a
    sensible 4-card layout — there is no exact prescribed mapping, use
    judgment, but every number must come from `summary()`, never be
    hardcoded.
  - Replace the run-history table body (`@for (r of runs; ...)`) to read
    from `ScrapeRunDto` fields: `run_ts`, `log_filename`, `source`,
    `items_total`, `errors_total`, `duration_s`, `status`. Map `status` to
    the existing `<rw-status-pill>` tones — note the backend's `status`
    values (`finished`/`failed`, possibly others — check actual seeded/
    backfilled values in the dev DB) differ from the mock's
    (`success`/`partial`/`failed`/`running`); adjust the `@switch` cases to
    match real values, don't force-fit the old ones.
  - Add an **alerts section** (new — there's no equivalent in the current
    mocked page) rendering `alerts()`: one row per alert with `severity` →
    `<rw-status-pill>` tone (`error`→`err`, `warning`→`warn`), `message`,
    `run_ts`. Reuse `.card`/`.list`/`.item` classes already established in
    `admin-assignments.component.ts` for a consistent look, or the alert-card
    styling in `manager-alerts.component.ts`.
  - Keep `@if (loading())` / `@else if (error())` guards consistent with the
    rest of the codebase.

- [ ] **Step 2:** In `AdminDashboardComponent`, replace the hardcoded "Rows
  ingested today" / "Scrape success" KPI cards and the "Latest scrape runs"
  table with data from `getMonitoringSummary()` + `getMonitoringRuns(6)`
  (mirrors the existing `runs = SCRAPER_RUNS.slice(0, 6)` pattern — just
  swap the source). The "Hotels tracked" / "Active managers" cards and
  "Coverage by city" section can stay on `getAdminHotels()`/`getAdminManagers()`
  (wired in Tasks 3–4) — if you reach this task before those land, it's fine
  to leave those two sections on mock data for now and note it in your
  report; do NOT block this task on Tasks 3–4.

- [ ] **Step 3:** `npm run build` → clean. Start the dev server, log in as
  admin, open `/admin/scrapers` and `/admin/dashboard`. Confirm: real
  `total_rows` (or "—" if Mongo is down), real run history dates/items/status
  matching what's in the `scrape_runs` Postgres table, and the alerts list
  shows `failed_run`/`low_volume` entries consistent with
  `GET /admin/alerts` (you can hit the endpoint directly with `curl` +
  the admin JWT to cross-check).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/features/admin/scrapers/admin-scrapers.component.ts frontend/src/app/features/admin/dashboard/admin-dashboard.component.ts
git commit -m "feat(frontend): wire admin monitoring + alerts + dashboard summary to live API"
```

---

### Task 3: Wire Hotels page (`admin-hotels`)

**Files:**
- Modify: `frontend/src/app/features/admin/hotels/admin-hotels.component.ts`

- [ ] **Step 1:** Replace `all = HOTELS` with `ApiService`-backed signals
  (`hotels = signal<AdminHotelDto[]>([])`, `loading`, `error`), fetched via
  `getAdminHotels()` in `ngOnInit`.
- [ ] **Step 2:** Update the template/`filtered()` computed to use
  `AdminHotelDto` fields: `hotel_name_display`, `city_name`, `stars_int`,
  `is_active`, `region`, `sources` (a comma-joined string — e.g. render it
  directly as the "Source" badge text, or `.split(',')` if you want one badge
  per source), `manager_id`/`manager_name` (use `manager_name` for display,
  `manager_id` for the unassigned check) — instead of the mock `Hotel` shape
  (`name`, `city`, `stars`, `source`, `rooms`, `active`, `managerId`). The
  real schema has **no `rooms` column** — drop that table column entirely
  (don't invent a number). Consider surfacing `region` and/or
  `latest_scraped_at` as new columns since they're real, useful admin signals
  the mock never had.
- [ ] **Step 3:** Wire "+ Add hotel" to the **discovery → promote flow**:
  clicking it opens a simple inline panel (no modal component exists — a
  conditionally-rendered `<section class="card">` toggled by a signal is the
  established lightweight pattern) listing `getDiscoverableHotels()` results
  (`hotel_name_normalized`, `city_name`, `stars_int` — note: **no display
  name or source on these candidates**, the admin supplies them at promote
  time). Each row needs a small "Promote" form, not a one-click button —
  at minimum a display-name input (pre-fill a reasonable default, e.g.
  title-cased `hotel_name_normalized`, but let the admin edit it) — calling
  `promoteHotel({ hotel_name_normalized, hotel_name_display, city_name,
  stars_int, sources: [...] })`. Where does `sources` come from for a
  discovered candidate? Check whether `DiscoverableHotelDto` rows can be
  cross-referenced against `hotel_features`/`platform_hotel_sources` for
  their originating source(s) — if the discoverable endpoint genuinely
  doesn't expose it, a single-checkbox-per-known-source (`promohotel`,
  `tunisiepromo`) in the promote form is an acceptable minimal UI; don't
  silently default to an empty list if the backend actually has the data
  available elsewhere. On success, refresh the hotel list. This is the one
  genuinely new interaction on this page — keep it minimal, it just needs to
  prove the discovery → promote flow end to end.
- [ ] **Step 4:** `npm run build` → clean. Verify in the browser: hotel list
  loads from Postgres, search/city filters still work against real fields,
  the discovery panel lists real `hotel_features`-derived candidates, and
  promoting one adds it to the pool (refresh confirms it persists).
- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/features/admin/hotels/admin-hotels.component.ts
git commit -m "feat(frontend): wire admin hotels page (list, filters, discovery promote) to live API"
```

---

### Task 4: Wire Managers page (`admin-managers`)

**Files:**
- Modify: `frontend/src/app/features/admin/managers/admin-managers.component.ts`

- [ ] **Step 1:** Replace `managers = MANAGERS` with `ApiService`-backed
  signals fetched via `getAdminManagers()`.
- [ ] **Step 2:** Update the template to `AdminManagerDto` fields:
  `email`, `full_name`, `is_active`, `assigned_hotel_id`/`assigned_hotel_name`
  (use `_name` for display, `_id === null` for the "unassigned" branch),
  `last_login_at`. Note the mock had `competitorIds.length` and `lastSeen` —
  the real schema has **no per-manager competitor count on the list row**
  (that's a separate `/admin/managers/{id}/competitors` call, scoped
  per-manager — don't N+1-fetch it here; drop that column, or replace it
  with an "Competitors →" link that opens Task 6's screen, which doubles as
  this page's entry point into that screen) and `last_login_at` may be `null`
  for managers who've never logged in (render "never" or "—", not a blank
  date — `DatePipe` on `null` renders empty, which looks like a bug).
- [ ] **Step 3:** Wire "+ Invite manager" to a minimal inline create form
  (`email`, `full_name`, `initial_password` — the admin sets the manager's
  first password directly; **there is no email-invite flow**, the naming
  "Invite manager" in the existing mock copy is slightly misleading for what
  the backend actually does — consider relabeling the button "+ New manager"
  to match reality, or keep "Invite" if you'd rather not touch copy; your
  call, note which) calling `createManager({ email, full_name,
  initial_password })`; on success, refresh the list. Wire the row "Edit"
  button to a minimal inline edit (`email`, `full_name`, toggle `is_active`)
  via `updateManager(id, {...})`. Add a separate "Reset password" action that
  opens a small form for the admin to **type a new password** (NOT a
  generate-and-display flow — `PasswordReset` takes `new_password` in the
  request; the endpoint returns 204 with no body) calling
  `resetManagerPassword(id, new_password)`; on success show a simple
  confirmation (no toast component exists in this codebase — an inline
  `<div class="muted small">Password updated.</div>` that clears after a
  `setTimeout`, or similar minimal pattern, is fine).
- [ ] **Step 4:** `npm run build` → clean. Verify in browser: manager list
  loads from Postgres, create/edit/reset-password round-trip against the real
  API (check the dev DB or re-fetch the list to confirm persistence).
- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/features/admin/managers/admin-managers.component.ts
git commit -m "feat(frontend): wire admin managers page (list, create, edit, reset-password) to live API"
```

---

### Task 5: Wire Assignments page (`admin-assignments`)

**Files:**
- Modify: `frontend/src/app/features/admin/assignments/admin-assignments.component.ts`

- [ ] **Step 1:** Replace the `HOTELS`/`MANAGERS`-derived `unassigned`/
  `assigned` computed values with signals fetched from `getAdminHotels()` +
  `getAssignments()` (no need for `getAdminManagers()` here — `AssignmentDto`
  already carries display fields).
  - `unassigned` = active hotels (`is_active === true`) with `manager_id ===
    null` from `getAdminHotels()` — this field is the direct, intended
    signal for "does this hotel have a manager", no cross-referencing needed.
  - `assigned` = `getAssignments()` rows directly — `AssignmentDto` carries
    `manager_name`/`manager_email` and `hotel_name` already joined server-side
    (note: it's `hotel_name`, not `hotel_name_display` — a different column
    alias than the hotels/competitors endpoints use; bind exactly what the
    DTO has, don't assume consistency across endpoints).
- [ ] **Step 2:** Wire "Assign…" on an unassigned hotel to a minimal inline
  picker. This DOES need `getAdminManagers()` (fetch alongside the other two
  on init) — to offer a choice of manager, filter to those with
  `assigned_hotel_id === null` (a manager owns at most one hotel per the
  data model). Pick one + optionally override `max_competitors` (defaults to
  4 server-side per `AssignmentCreate`), call `createAssignment({ user_id:
  manager.id, hotel_id, max_competitors? })`. Wire "Unassign" to
  `deleteAssignment(assignment.id)` (the assignment row's `id`, not the
  hotel's or manager's). Refresh all three lists on success. Keep "Save changes" in the header or remove it if
  the per-row actions make it redundant — your call, note which you chose and
  why.
- [ ] **Step 3:** `npm run build` → clean. Verify in browser: both panels
  reflect real Postgres state, assign/unassign round-trips persist (re-fetch
  to confirm), and a manager who already has `max_competitors` competitors
  selected still shows correctly after reassignment (no stale client state).
- [ ] **Step 4: Commit**

```bash
git add frontend/src/app/features/admin/assignments/admin-assignments.component.ts
git commit -m "feat(frontend): wire admin assignments page (assign/unassign) to live API"
```

---

### Task 6: NEW — Admin competitor-selection screen (D11)

**Files:**
- Create: `frontend/src/app/features/admin/managers/admin-manager-competitors.component.ts`
  (or fold into the manager edit view from Task 4 — your call; if you fold it
  in, name the section clearly and keep it visually distinct since it's a
  materially different concern from editing the account itself)
- Modify: `frontend/src/app/app.routes.ts` (only if you create a separate
  routed page — add a child route under `admin`, e.g.
  `managers/:id/competitors`; do NOT restructure the existing `admin/*`
  shape, only add to it)
- Modify: `frontend/src/app/features/admin/managers/admin-managers.component.ts`
  (add an entry point — a button/link per manager row to open this screen)

This is the one genuinely new admin capability (confirmed D11: admin-only,
manager view stays read-only — see [[project-scrape-runs-monitoring]] and the
spec's D11 decision). There is no mocked precedent to mirror; design it with
the same restraint as the rest of the admin UI (`.card`, `.tbl`, `.btn`, no
new shared components, no chart libraries).

- [ ] **Step 1:** Build the screen around three calls:
  - `getManagerCompetitors(managerId)` → the manager's current N picks,
    ordered by `display_order`.
  - `getSelectableCompetitors(managerId)` → active platform hotels excluding
    the manager's own assigned hotel.
  - `setManagerCompetitors(managerId, hotel_ids)` → full-replace PUT.
- [ ] **Step 2:** UI shape: show the current selection (ordered list, each
  row removable), a way to add from the selectable pool (e.g. a searchable
  list or `<select>` with an "Add" button appending to a working list), and a
  "Save" button that calls `setManagerCompetitors` with the final
  `hotel_ids` array in order. **Surface backend validation errors verbatim**
  — the service raises `BadRequestError` (HTTP 400) with messages like "At
  most {cap} competitors allowed" / "A manager cannot select their own hotel
  as a competitor" / "Manager has no active hotel assignment; assign a hotel
  first" — these are exactly the messages the admin needs to see to correct
  their input; don't replace them with generic "Something went wrong" copy.
  Show the manager's `max_competitors` cap so the admin knows the limit
  before hitting Save: fetch `getAssignments()` and find the row where
  `manager.assigned_hotel_id === assignment.hotel_id` (or simpler, where
  `assignment.user_id === managerId`) — `AssignmentDto.max_competitors` is
  the live cap (defaults to 4, but admins can override it per-assignment per
  `AssignmentCreate`/`AssignmentUpdate`).
- [ ] **Step 3:** Wire the entry point from the managers list (Task 4): a
  button/link per row, e.g. "Competitors →", navigating to this screen (or
  expanding inline — your call, document which).
- [ ] **Step 4:** `npm run build` → clean. Verify in browser end-to-end: open
  a manager's competitor screen, see their current picks, add/remove to reach
  a new set within the cap, save, reload the page and confirm the new
  selection persisted from Postgres (`user_competitor_selections`). Also
  verify the validation-error path: try to save a set that exceeds the cap or
  includes the manager's own hotel, and confirm the backend's exact error
  message surfaces in the UI.
- [ ] **Step 5: Commit**

```bash
git add frontend/src/app/features/admin/managers/ frontend/src/app/app.routes.ts
git commit -m "feat(frontend): admin competitor-selection screen (D11 — admin-only, manager view stays read-only)"
```

---

## Self-review

**Spec coverage (vs the brainstormed admin scope + D11):**
- Hotels (discovery-based onboarding, list, filters) → Task 3. ✓
- Managers (CRUD, reset-password) → Task 4. ✓
- Assignments (assign/unassign) → Task 5. ✓
- Monitoring (live Mongo total + scrape_runs aggregates/history) + Collection
  Alerts → Task 2. ✓
- Admin-only competitor selection (D11) → Task 6. ✓
- Dashboard tying it together → Task 2 (summary cards + recent runs); city
  coverage / active-manager cards remain on Tasks 3–4's data once those land.

**Ordering rationale (defense value first):** Task 1 (plumbing) must be
first — everything depends on it. Task 2 (monitoring/alerts) goes next
because it's the most visually compelling proof that "the platform watches
itself" (live row count ticking up, real run history, derived alerts) — the
single best demo moment if time runs out after this task. Hotels → Managers →
Assignments follow in the order an admin would naturally use them (you can't
assign a manager to a hotel that doesn't exist in the pool yet). Task 6 (new
competitor UI) is last because it's the only screen requiring net-new design
rather than wiring an existing one — highest risk, so it's sequenced after
the "safe" wiring wins are banked.

**Deviation from the backend plans' rigor:** Tasks 2–6 do not give
line-exact code blocks the way Plans 1–4 did, because UI wiring isn't TDD —
there's no failing test to write first, and the "right" template diff depends
on what the existing mocked markup looks like once you're staring at it.
Instead each task gives: the exact DTO/method contract (Task 1, which IS
exact), the exact fields to bind, explicit call-outs where the mock shape
diverges from the real schema (so you don't accidentally keep rendering
`h.rooms` when `AdminHotelDto` has no `rooms`), and a concrete manual
verification checklist. This is a deliberate, named trade — flagging it
per the recommendation-hygiene rule rather than silently presenting prose
tasks as if they had the same rigor as the SQL-exact backend tasks.

**DTO contract verification (corrected before implementation, not left as
guesses):** the first draft of Task 1's DTOs was written from memory of the
Plan 2–3 design discussions and contained five real mismatches —
`AdminHotelDto` actually has `region`/`sources`(comma-string)/`manager_id`/
`manager_name`/`latest_scraped_at` (not the guessed `manager_email`/`rooms`/
`source`); `DiscoverableHotel` has only 3 fields (no display name, no
source — promote-time requires a small form, not one click);
`AdminManagerDto` uses `assigned_hotel_id`/`assigned_hotel_name` (not
`hotel_name_display`) and has no competitor count; `AssignmentDto` uses
`manager_name`/`hotel_name` (not `hotel_name_display`); and critically,
password reset takes admin-supplied `new_password` (HTTP 204, no generated
password returned) — a UX-shape difference, not just a naming one. **All of
these were caught by reading the committed `backend/schemas/admin_*.py`
files directly before finalizing this plan** (not deferred to the
implementer to discover mid-task) — the corrected, verified shapes are what
appear in Task 1 now. The remaining genuine unknowns (e.g. exactly how
`sources` should be populated for a freshly-discovered hotel at promote
time, since `DiscoverableHotelDto` doesn't carry it) are flagged inline at
the point they matter, with a pointer to the specific file to check.

**Placeholder scan:** none — every data-bearing template expression in this
plan is traced to a named DTO field from a named backend schema file.
