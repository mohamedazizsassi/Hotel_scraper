# Manager frontend — make it fully functional

**Date:** 2026-06-08
**Author:** brainstormed with the user (mohamedaziz.sassi)
**Status:** design, pending user review
**Defense deadline:** ~2026-06-15 (~7 days) — scope is bounded by this.

---

## 1. Product framing

RevWay is repositioned as **"Competitive Pricing Intelligence for Hotel Revenue
Managers."** It is a **decision-support tool**, not a price-execution tool: there
is no PMS / channel-manager integration, so no manager action publishes a real
price anywhere. Every feature below is honest to that framing — the tool surfaces
market signals, the manager triages and records decisions, and RevWay tracks that
decision state.

This framing also matches the repo's scientific framing (model of *public OTA
listed prices*, not realized occupancy). It is the reason the dashboard must NOT
use the term "ADR" (which implies revenue ÷ rooms sold).

---

## 2. Current state (verified 2026-06-08)

Runtime prerequisites all present: Postgres up on 5432; ML models present at
`ml/models/forecasting/lgbm_quantile_2026-05-23/hotel_wise/`; a real manager user
exists: `manager@revway.tn` ("Iberostar Averroes Manager") with 1 active hotel
assignment, 4 competitor selections, 29.4M `hotel_features` rows.

| Manager page    | Backend endpoint                         | State |
| --------------- | ---------------------------------------- | ----- |
| Calendar        | `GET /manager/calendar` + `/options`     | ✅ Wired (filters call API) |
| Competitors     | `GET /manager/competitors`               | ✅ Wired (read-only) |
| Recommendations | `GET /manager/recommendations`           | ✅ Read wired — **Accept/Dismiss dead** |
| Alerts          | `GET /manager/anomalies`                 | ✅ Read wired — **actions dead** |
| Dashboard       | *(none)*                                 | ❌ **Fully mock** (`core/data/mock.ts`) |
| Settings        | *(none)*                                 | ❌ **Fully static** (hardcoded) |

**Genuine gaps:** Dashboard, Settings, and all write/action buttons (Accept,
Dismiss, Save, Investigate, Export). The reason they are static is that no backend
exists for them: no profile endpoint, no recommendation-decision persistence, no
preferences storage.

---

## 3. Scope

Make the manager experience fully functional. Three backend additions + frontend
wiring:

1. **Manager profile** — `GET /manager/me`, `PATCH /manager/me`.
2. **Dashboard** — `GET /manager/dashboard` (composes existing services).
3. **Recommendation decisions** — new table + `POST` (single + bulk) + status
   merged into the existing `GET /manager/recommendations`.

Explicitly **out of scope** (noted, not silently dropped):
- Alerts "Mark all read" persistence (anomalies are computed on the fly; this is
  the lowest-value, highest-ambiguity feature — would need a second decision
  table). "Investigate" becomes a deep-link into the calendar instead.
- Any real notification delivery (alert preferences are stored as *preferences*
  only; nothing sends emails).
- "Applied price" tracking on accepted recommendations (richer, deferred).
- PMS integration (does not exist; not planned for the defense).

---

## 4. Base branch & workspace

- Base branch: **`main`**. Verified 2026-06-08: `main` already contains the
  `feat/admin-platform` merge (full backend + frontend, manager pages already
  API-wired), so it is the most complete branch. `feat/admin-platform` has 0
  commits not in `main` (redundant). NOT `feat/forecaster-bakeoff` — that is a
  9-commit ML-only bake-off branch, 9 commits *behind* `main` (branched before the
  admin merge), with no backend/frontend; basing on it would be merging backwards.
  The bake-off branch stays separate and merges to `main` on its own later.
- New branch: `feat/manager-functional`, in an **isolated git worktree off
  `main`** (user decision 2026-06-08). The current checkout stays on
  `feat/forecaster-bakeoff` with its uncommitted ML work untouched.
- The worktree must be given the gitignored runtime deps before it can run:
  `backend/.env`, `frontend/node_modules` (or `npm install`), and the ML model
  dir referenced by `MODEL_DIR`. Concretely: copy `backend/.env`, copy/junction
  the model dir from the main checkout, and run `npm install` (or junction
  `node_modules`) in the worktree. These steps belong in the implementation
  plan's setup task. (Windows junctions: `cmd /c mklink /J` works without admin.)

---

## 5. Backend design

### 5.1 `GET /manager/me` and `PATCH /manager/me`

Returns the manager's identity + assigned hotel + preferences. Read joins
`users` ⋈ `user_hotel_assignments` (active) ⋈ `platform_hotels` ⋈ `cities`, plus a
count of `user_competitor_selections`.

`schemas/profile.py` → `ManagerProfile`:

```
full_name: str | None
email: str
role: str
preferences: dict            # JSONB blob, see 5.4
hotel: {                     # null only if no active assignment (shouldn't happen for a manager)
  id: int
  hotel_name_display: str
  city_name: str             # cities.name_display
  stars_int: int | None
}
competitor_count: int
max_competitors: int
last_login_at: str | None
```

`PATCH /manager/me` body (all optional): `full_name`, `preferences`. Updates only
provided fields. `email`, `role`, hotel are **not** editable here (admin-managed).
Returns the updated `ManagerProfile`. Service: `services/profile_service.py`.

Router: `routers/profile.py`, prefix `/manager`, `Depends(get_current_manager)`.

### 5.2 `GET /manager/dashboard`

One endpoint, computed server-side, composing the existing service functions
(no logic duplication, one request for the frontend).

Query: `days: int = 30` (forward window from today). Internally it calls, scoped
to the hotel's **default product config** (from calendar options, same pattern the
recommendations/alerts pages already use):
- `get_calendar(...)` over `[today, today+days]` → per-night rows.
- `get_competitors(...)` → competitor set with `avg_price_per_night`.
- `get_recommendations(...)` over the window → for the open-recs count + top list.
- `get_anomalies(...)` over the window → for the active-alerts count + recent list.

`schemas/dashboard.py` → `DashboardResponse`:

```
kpis: {
  avg_listed_rate_tnd: float | None        # avg(price_per_night) over window
  market_position_pct: float | None        # (avg price - avg peer_medium_median)/avg peer_medium_median * 100
  vs_competitor_pct: float | None          # (avg price - mean(competitor avg_price_per_night))/... * 100
  opportunity_tnd: float | None            # avg(recommended_price_per_night - price_per_night)
  opportunity_pct: float | None            # same as % of price
  open_recommendations: int                # recs with direction != 'hold' AND no decision yet
  active_alerts: int                        # anomalies in window
}
price_series: [                            # next 14 days, latest snapshot per check_in
  { check_in, price_per_night, peer_medium_median, recommended_price_per_night }
]
top_recommendations: [RecommendationRow]   # top N=5 by |delta_pct_vs_current|, decision status merged
competitors: [CompetitorSummary]           # the manager's set (reuse existing schema)
recent_alerts: [AnomalyRow]                # top N=5 by |anomaly_score|
```

KPI honesty notes:
- All price KPIs use **per-night** figures (`price_per_night`,
  `peer_medium_median`, `recommended_price_per_night`) for consistent units.
- `market_position` uses the **statistical peer neighborhood**
  (`peer_medium_median`); `vs_competitor` uses the manager's **hand-picked set**
  (`user_competitor_selections` averages). These are deliberately two different
  numbers shown side by side — the project's differentiator. Keep them distinct in
  code and copy (repo CLAUDE.md rule 5).
- Any KPI whose inputs are all null (e.g., no peer data in window) returns `null`,
  and the UI renders "—" rather than a fake 0.
- No deltas/trend arrows are computed server-side in v1 (would need a previous
  window comparison). KPI cards render without `delta` or with a sparkline from
  `price_series` only.

### 5.3 Recommendation decisions

New table `manager_recommendation_decisions`:

| column                | type        | notes |
| --------------------- | ----------- | ----- |
| id                    | serial PK   | |
| user_id               | uuid FK users | |
| hotel_id              | int FK platform_hotels | |
| check_in              | date        | |
| nights                | smallint    | part of decision key |
| adults                | smallint    | part of decision key |
| boarding_canonical    | text        | part of decision key |
| recommended_price_tnd | numeric     | snapshot at decision time (audit) |
| status                | text        | `accepted` \| `dismissed` |
| decided_at            | timestamptz | default now() |
| updated_at            | timestamptz | default now(), bumped on upsert |

**Unique key:** `(user_id, check_in, nights, adults, boarding_canonical)`.

Rationale for this key (trade-off, per repo CLAUDE.md recommendation hygiene):
- These are exactly the fields `RecommendationRow` exposes and exactly the fields
  the frontend already uses to build a recommendation `id`
  (`${check_in}-${nights}n-${boarding}-${adults}a` in `adapters.ts`). So the key
  matches the UI's existing notion of recommendation identity.
- **Preserves:** correct per-(date, board, nights, adults) decisions; matches the
  recs page which scopes to the default config and shows one row per check_in.
- **Destroys:** nothing the UI currently exposes. It does NOT distinguish
  room_base/view/tier/occupancy because `RecommendationRow` does not carry them;
  if per-room-class decisions are ever exposed, extend the key + migration then.
- **Irreversible state:** a DB migration adding one table; additive, reversible by
  dropping the table.

Endpoints (router `routers/recommendation_decisions.py`, prefix `/manager`):
- `POST /manager/recommendations/decision` — body
  `{ check_in, nights, adults, boarding_canonical, recommended_price_tnd, status }`.
  Upsert on the unique key. `status='accepted'|'dismissed'`. Returns the row.
- `POST /manager/recommendations/decision/bulk` — body
  `{ status, items: [{check_in, nights, adults, boarding_canonical, recommended_price_tnd}] }`.
  Powers "Accept all new" / "Dismiss all". Upserts each.
- `DELETE /manager/recommendations/decision` — body with the key fields; clears a
  decision (status back to implicit `new`). Optional; include if cheap.

Merge into reads: `get_recommendations` LEFT JOINs (or post-joins in Python)
`manager_recommendation_decisions` on the key and adds `decision_status` to each
`RecommendationRow` (`accepted` / `dismissed` / `null`=new). The frontend adapter
maps `null → 'new'`. This is what finally makes the existing
`new / accepted / dismissed` filter chips real.

### 5.4 Preferences (Settings)

Add a `preferences JSONB NOT NULL DEFAULT '{}'` column to `users`. Shape:

```
{
  "language": "en" | "fr",
  "alerts": {
    "competitor_undercut": bool,
    "price_spike": bool,
    "anomaly_digest": bool,
    "data_quality": bool
  }
}
```

Read via `GET /manager/me.preferences`; written via `PATCH /manager/me`. Stored
honestly as preferences — nothing consumes them to send notifications (v1). One
column, no extra table, makes the whole Settings page functional.

### 5.5 Migrations

New SQL files in `database/postgres/migrations/`, next sequential numbers (the
backend CLAUDE.md references migration 005, so confirm the latest on
`feat/admin-platform` and continue from there):
- `NNN_add_users_preferences.sql` — `ALTER TABLE users ADD COLUMN preferences ...`.
- `NNN_create_recommendation_decisions.sql` — table + unique index + FKs.

---

## 6. Frontend design

All components stay standalone + signals + new control flow, per `frontend/CLAUDE.md`.

### 6.1 API + DTO + adapters

`api.service.ts` — add: `getMe()`, `updateMe(patch)`, `getDashboard(days)`,
`postRecommendationDecision(body)`, `postRecommendationDecisionBulk(body)`,
(optional) `deleteRecommendationDecision(body)`.

`dto.ts` — add `ManagerProfileDto`, `DashboardDto`, `RecommendationDecisionDto`;
extend `RecommendationDto` with `decision_status: 'accepted'|'dismissed'|null`.

`adapters.ts` — `recommendationFromDto` maps `decision_status ?? 'new'` into
`status` (replaces the hardcoded `status: 'new'`).

### 6.2 Dashboard (`features/manager/dashboard/`)

Replace all `core/data/mock` imports. On init: call `getMe()` (header greeting +
hotel line) and `getDashboard(30)` (KPIs + panels). Loading + error states like
the other wired pages.

- Header: `Hello, {full_name}` + `{hotel_name_display} · {city_name} · {stars}★`.
- 4 KPI cards: **Your avg listed rate** (TND), **Market position** (% vs peer
  median), **Vs your competitor set** (% vs personal selection), **Pricing
  opportunity** (TND + %). Render "—" for null KPIs. Rename the old "ADR" card.
- Secondary: open-recommendations count (links to /manager/recommendations),
  active-alerts count (links to /manager/alerts).
- Price-vs-market chart: from `price_series` (reuse existing inline-SVG polyline).
- Top recommendations panel: from `top_recommendations`, showing decision status;
  Accept/Dismiss buttons wired to the decision endpoint (see 6.3).
- Competitor set panel: from `competitors`.
- Recent alerts panel: from `recent_alerts`.
- Export button: client-side CSV of the current dashboard view.

### 6.3 Recommendations (`features/manager/recommendations/`)

- Wire **Accept** → `postRecommendationDecision({...key, recommended_price_tnd,
  status:'accepted'})`; **Dismiss** → same with `'dismissed'`. On success, update
  the row's `status` signal optimistically; on error surface the backend message.
- Wire **Accept all new** / **Dismiss all** → bulk endpoint over the currently
  visible "new" rows.
- The `new/accepted/dismissed` filter chips already exist and now reflect real,
  persisted state (survives reload).
- Build the decision key from the DTO fields (`check_in, nights, adults,
  boarding_canonical`) — same fields used for the row `id`.

### 6.4 Alerts (`features/manager/alerts/`)

- **Investigate** → `routerLink` to `/manager/calendar` with the anomaly's
  `check_in` (calendar focuses/scrolls to that date; minimally, pre-set the
  window to include it). No persistence.
- Remove or disable "Mark all read" (out of scope; do not leave a dead button —
  either hide it or render it disabled with a tooltip).

### 6.5 Settings (`features/manager/settings/`)

- Load from `getMe()`: full name (editable), email (read-only), language
  (editable), hotel block (read-only), alert toggles (editable, from
  `preferences.alerts`).
- **Save** → `updateMe({ full_name, preferences:{ language, alerts } })`; success
  + error toast/inline. Survives reload.
- **Sign out** → `AuthService.logout()` + navigate to `/login`.

---

## 7. Testing

Backend (pytest, mirror existing `backend/tests/` patterns, async + seeded test
DB `revway_test`):
- `test_profile.py` — `GET /manager/me` shape + auth tiers; `PATCH` updates
  full_name + preferences; email/role/hotel not editable; manager-only.
- `test_dashboard.py` — KPI math on a seeded slice (incl. null handling when no
  peer/competitor data); manager-only; respects `days`.
- `test_recommendation_decisions.py` — upsert (insert + update same key); bulk;
  status merged into `GET /manager/recommendations`; unique-key dedup;
  manager-only; cannot decide for another manager's hotel.

Frontend: at minimum `npm run build` (strict templates) passes; manual smoke of
each manager page against the live backend (see verification).

---

## 8. Verification (live)

1. Start backend (`uvicorn main:app`) and frontend (`npm start`) in the worktree.
2. Log in as `manager@revway.tn` (obtain/confirm the seeded password).
3. Confirm each page loads live data: Dashboard KPIs non-empty, Calendar grid,
   Competitors, Recommendations, Alerts.
4. Accept a recommendation → reload → status persists; filter chips reflect it.
5. Edit settings (name, a toggle) → Save → reload → persists.
6. Investigate an alert → lands on calendar at that check-in.
7. Sign out → returns to /login; protected routes redirect.

---

## 9. Risks

- **Manager password unknown** — needed for live login. Resolve in setup (reset
  via a one-off script / known seed value) before verification.
- **Worktree runtime deps** — `.env`, models, `node_modules` must be present
  (section 4) or the backend won't start and pages will 500.
- **Dashboard endpoint latency** — composes 4 services over a 30-day window;
  reuse the same default-config + latest-snapshot scoping the recs/alerts pages
  use so it stays bounded. Watch the `hotel_features` full-scan rule (backend
  CLAUDE.md) — rely on the existing indexed access patterns.
- **7-day budget** — if time runs short, ship order: profile+settings (smallest,
  visible) → dashboard → recommendation decisions (largest). Each is independently
  demoable.
