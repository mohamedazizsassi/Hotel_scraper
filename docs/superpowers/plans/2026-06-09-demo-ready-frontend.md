# Demo-Ready Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a demo-ready RevWay platform by branching off `feat/admin-platform`, adding `competitor_avg_per_night` to the calendar API, redesigning the calendar cell with human-readable labels, simplifying the filter bar, and removing dark mode.

**Architecture:** All changes are additive or cosmetic — no existing endpoints, routes, or auth flows change. The backend gains one new nullable field (`competitor_avg_per_night`) on the existing `GET /manager/calendar` response. The frontend rewrites the calendar component template and styles in-place; all other manager and admin pages are inherited from `feat/admin-platform` untouched.

**Tech Stack:** Python 3.11 · FastAPI · SQLAlchemy async · pytest-asyncio · Angular 19 (standalone components, signals, `@if`/`@for` control flow) · CSS custom properties

---

## File map

| File | Change |
|---|---|
| *(git)* | Create `feat/demo-ready` from `feat/admin-platform` |
| `backend/schemas/calendar.py` | Add `competitor_avg_per_night: float \| None` to `CalendarRow` |
| `backend/services/calendar_service.py` | Query competitor avg per check_in, merge into output list |
| `backend/tests/conftest.py` | Seed hotel_features rows for competitor hotels |
| `backend/tests/test_calendar.py` | New file — tests for calendar endpoint including new field |
| `frontend/src/app/core/api/dto.ts` | Add `competitor_avg_per_night: number \| null` to `CalendarRowDto` |
| `frontend/src/styles.css` | Remove `@media (prefers-color-scheme: dark)` block |
| `frontend/src/app/features/manager/calendar/manager-calendar.component.ts` | Redesign cell template + styles + action badge + simplified filter bar |

---

## Task 1: Create the working branch

**Files:** *(git only)*

- [ ] **Step 1: Verify feat/admin-platform tip**

```bash
git log --oneline feat/admin-platform -5
```

Expected: see the 5 most recent admin-platform commits, latest being the D11 competitor-selection screen.

- [ ] **Step 2: Create feat/demo-ready off feat/admin-platform**

```bash
git checkout feat/admin-platform
git checkout -b feat/demo-ready
```

Expected: `Switched to a new branch 'feat/demo-ready'`

- [ ] **Step 3: Verify the branch has admin pages wired**

```bash
git log --oneline -3
```

Expected: top commit is the D11 screen commit (`feat(frontend): admin competitor-selection screen`).

---

## Task 2: Backend schema — add `competitor_avg_per_night`

**Files:**
- Modify: `backend/schemas/calendar.py`

- [ ] **Step 1: Add the field to CalendarRow**

Open `backend/schemas/calendar.py`. After the `best_peer_granularity_used` line, add:

```python
# Pricing (TND, per-night) + peer aggregate
price_per_night: float
peer_medium_median: float | None
peer_medium_count: int | None
best_peer_granularity_used: str | None
competitor_avg_per_night: float | None   # avg of manager's selected competitors, same product config
```

Full updated pricing block (lines 26–30):

```python
    # Pricing (TND, per-night) + peer aggregate
    price_per_night: float
    peer_medium_median: float | None
    peer_medium_count: int | None
    best_peer_granularity_used: str | None
    competitor_avg_per_night: float | None   # avg of manager's selected competitors, same product config
```

- [ ] **Step 2: Verify Pydantic accepts the schema**

```bash
cd backend
.venv\Scripts\python -c "from schemas.calendar import CalendarRow; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add backend/schemas/calendar.py
git commit -m "feat(backend): add competitor_avg_per_night field to CalendarRow schema"
```

---

## Task 3: Backend service — compute competitor averages

**Files:**
- Modify: `backend/services/calendar_service.py`

- [ ] **Step 1: Add the competitor-avg query after the main hotel query**

In `get_calendar`, after `if not rows: return []` and before `df = pd.DataFrame(...)`, insert:

```python
    # Competitor averages: avg price_per_night of the manager's selected competitors
    # for each check_in date, filtered to the same product config.
    comp_conditions = ["ucs.user_id = :comp_user_id"]
    comp_params: dict = {"comp_user_id": str(user.id)}
    if check_in_from:
        comp_conditions.append("hf.check_in >= :comp_from")
        comp_params["comp_from"] = check_in_from
    if check_in_to:
        comp_conditions.append("hf.check_in <= :comp_to")
        comp_params["comp_to"] = check_in_to
    if boarding_canonical:
        comp_conditions.append("COALESCE(hf.boarding_canonical,'') = :comp_boarding")
        comp_params["comp_boarding"] = boarding_canonical
    if nights is not None:
        comp_conditions.append("hf.nights = :comp_nights")
        comp_params["comp_nights"] = nights
    if adults is not None:
        comp_conditions.append("hf.adults = :comp_adults")
        comp_params["comp_adults"] = adults

    comp_sql = text(f"""
        SELECT hf.check_in, AVG(hf.price_per_night) AS avg_ppn
        FROM user_competitor_selections ucs
        JOIN platform_hotels ph ON ph.id = ucs.hotel_id
        JOIN hotel_features hf  ON hf.hotel_name_normalized = ph.hotel_name_normalized
        WHERE {" AND ".join(comp_conditions)}
        GROUP BY hf.check_in
    """)
    comp_result = await db.execute(comp_sql, comp_params)
    comp_avg: dict[str, float] = {
        str(row["check_in"]): float(row["avg_ppn"])
        for row in comp_result.mappings().fetchall()
    }
```

- [ ] **Step 2: Thread competitor_avg_per_night into the output loop**

In the `for i in range(len(df)):` loop, find the `out.append(CalendarRow(...))` call. Add `competitor_avg_per_night` as the last field before the closing `)`:

```python
            competitor_avg_per_night=comp_avg.get(str(r["check_in"])),
```

Full updated `out.append` block (replacing the existing one in full):

```python
        out.append(CalendarRow(
            hotel_name_normalized=str(r["hotel_name_normalized"]),
            city_name=str(r["city_name"]),
            stars_int=int(r["stars_int"]) if pd.notna(r["stars_int"]) else None,
            check_in=r["check_in"],
            scrape_date=str(r["scrape_date"]),
            scraped_at=str(r["scraped_at"]),
            nights=int(r["nights"]),
            adults=int(r["adults"]),
            children=int(r["children"]) if pd.notna(r["children"]) else 0,
            boarding_canonical=str(r["boarding_canonical"]),
            room_base=str(r["room_base"]),
            room_view=str(r["room_view"]),
            room_tier=str(r["room_tier"]),
            room_occupancy=str(r["room_occupancy"]),
            is_supplement_variant=_b(r["is_supplement_variant"]),
            has_free_view_upgrade=_b(r["has_free_view_upgrade"]),
            price_per_night=float(r["price_per_night"]),
            peer_medium_median=float(pm) if pd.notna(pm) else None,
            peer_medium_count=int(pc) if pd.notna(pc) else None,
            best_peer_granularity_used=str(r["best_peer_granularity_used"]) if pd.notna(r["best_peer_granularity_used"]) else None,
            competitor_avg_per_night=comp_avg.get(str(r["check_in"])),
            recommended_price_per_night=float(rec_ppn[i]),
            forecaster_confidence=float(confidence[i]),
            sur_demande_rate_city_stars_checkin=float(sur) if sur is not None and pd.notna(sur) else None,
            days_until_checkin=int(r["days_until_checkin"]) if pd.notna(r["days_until_checkin"]) else 0,
            is_weekend_checkin=_b(r["is_weekend_checkin"]),
            is_ramadan=_b(r["is_ramadan"]),
            is_tunisia_public_holiday=_b(r["is_tunisia_public_holiday"]),
            is_tunisia_school_holiday=_b(r["is_tunisia_school_holiday"]),
            is_school_holiday_france=_b(r["is_school_holiday_france"]),
            is_school_holiday_germany=_b(r["is_school_holiday_germany"]),
            is_school_holiday_uk=_b(r["is_school_holiday_uk"]),
        ))
```

- [ ] **Step 3: Commit**

```bash
git add backend/services/calendar_service.py
git commit -m "feat(backend): compute competitor_avg_per_night in calendar service"
```

---

## Task 4: Backend test — calendar includes competitor_avg_per_night

**Files:**
- Modify: `backend/tests/conftest.py` (add competitor hotel_features rows)
- Create: `backend/tests/test_calendar.py`

- [ ] **Step 1: Seed competitor hotel_features rows in conftest**

In `conftest.py`, in the `setup_test_db` fixture, after the existing `INSERT INTO hotel_features` statement (which inserts `hotel_manager_test`), add competitor rows:

```python
        # Seed competitor hotel_features so competitor_avg_per_night is computable
        await conn.execute(text("""
            INSERT INTO hotel_features
              (hotel_name_normalized, city_name, stars_int, check_in, nights, adults,
               boarding_canonical, room_base, room_view, room_tier, room_occupancy,
               price, price_per_night, scraped_at,
               peer_medium_median, peer_medium_count)
            VALUES
              ('hotel_comp_1', 'hammamet', 4, DATE '2026-07-01', 3, 2,
               'BB', 'chambre', 'mer', '', 'double', 1500.0, 500.0, '2026-05-18T10:00:00', NULL, NULL),
              ('hotel_comp_2', 'hammamet', 4, DATE '2026-07-01', 3, 2,
               'BB', 'chambre', 'mer', '', 'double', 1440.0, 480.0, '2026-05-18T10:00:00', NULL, NULL)
        """))
```

Place this immediately after the existing hotel_features INSERT (before `CREATE OR REPLACE VIEW`).

- [ ] **Step 2: Create test_calendar.py**

Create `backend/tests/test_calendar.py`:

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
async def test_calendar_returns_200(client, db_session):
    token = await _manager_token(db_session)
    resp = await client.get(
        "/manager/calendar",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "data" in body
    assert "count" in body
    assert body["count"] == len(body["data"])


@pytest.mark.asyncio
async def test_calendar_row_has_competitor_avg(client, db_session):
    token = await _manager_token(db_session)
    resp = await client.get(
        "/manager/calendar",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    rows = resp.json()["data"]
    assert len(rows) >= 1, "Expected at least one calendar row"
    row = rows[0]
    assert "competitor_avg_per_night" in row, "Field competitor_avg_per_night missing from response"
    # Both competitors have price_per_night 500 and 480 → avg = 490
    assert row["competitor_avg_per_night"] == pytest.approx(490.0, abs=1.0)


@pytest.mark.asyncio
async def test_calendar_row_schema(client, db_session):
    token = await _manager_token(db_session)
    resp = await client.get(
        "/manager/calendar",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
    rows = resp.json()["data"]
    if rows:
        row = rows[0]
        for field in (
            "check_in", "price_per_night", "peer_medium_median",
            "recommended_price_per_night", "competitor_avg_per_night",
            "boarding_canonical", "nights", "adults",
        ):
            assert field in row, f"Missing field: {field}"
```

- [ ] **Step 3: Run the failing tests**

```bash
cd backend
.venv\Scripts\pytest tests/test_calendar.py -v
```

Expected: `test_calendar_row_has_competitor_avg` FAILS because `competitor_avg_per_night` is not yet in the conftest hotel_features_full view. (If you completed Tasks 2 and 3, all three should PASS — proceed to step 4.)

- [ ] **Step 4: Run full test suite**

```bash
.venv\Scripts\pytest -v
```

Expected: all tests pass (the new calendar tests + existing 64 tests).

- [ ] **Step 5: Commit**

```bash
git add backend/tests/conftest.py backend/tests/test_calendar.py
git commit -m "test(backend): calendar endpoint — competitor_avg_per_night field"
```

---

## Task 5: Frontend DTO — add competitor_avg_per_night

**Files:**
- Modify: `frontend/src/app/core/api/dto.ts`

- [ ] **Step 1: Add the field to CalendarRowDto**

In `dto.ts`, find the `CalendarRowDto` interface. After `best_peer_granularity_used`, add:

```typescript
  peer_medium_median: number | null;
  peer_medium_count: number | null;
  best_peer_granularity_used: string | null;
  competitor_avg_per_night: number | null;
```

Full updated `CalendarRowDto` (replace the entire interface):

```typescript
/** GET /manager/calendar — enriched hotel_features_full row + recommendation. */
export interface CalendarRowDto {
  hotel_name_normalized: string;
  city_name: string;
  stars_int: number | null;
  check_in: string;
  scrape_date: string;
  scraped_at: string;
  nights: number;
  adults: number;
  children: number;
  boarding_canonical: string;
  room_base: string;
  room_view: string;
  room_tier: string;
  room_occupancy: string;
  is_supplement_variant: boolean;
  has_free_view_upgrade: boolean;
  price_per_night: number;
  peer_medium_median: number | null;
  peer_medium_count: number | null;
  best_peer_granularity_used: string | null;
  competitor_avg_per_night: number | null;
  recommended_price_per_night: number;
  forecaster_confidence: number;
  sur_demande_rate_city_stars_checkin: number | null;
  days_until_checkin: number;
  is_weekend_checkin: boolean;
  is_ramadan: boolean;
  is_tunisia_public_holiday: boolean;
  is_tunisia_school_holiday: boolean;
  is_school_holiday_france: boolean;
  is_school_holiday_germany: boolean;
  is_school_holiday_uk: boolean;
}
```

- [ ] **Step 2: Verify TypeScript compiles**

```bash
cd frontend
npm run build -- --configuration development 2>&1 | Select-String -Pattern "error"
```

Expected: no `error` lines printed.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/app/core/api/dto.ts
git commit -m "feat(frontend): add competitor_avg_per_night to CalendarRowDto"
```

---

## Task 6: Remove dark mode

**Files:**
- Modify: `frontend/src/styles.css`

- [ ] **Step 1: Delete the dark mode media query block**

In `frontend/src/styles.css`, find and delete the entire block from line 46 to 64 (inclusive):

```css
@media (prefers-color-scheme: dark) {
  :root {
    --color-background: #0B1220;
    --color-surface: #111A2E;
    --color-surface-2: #16223B;
    --color-foreground: #E6ECF7;
    --color-muted: #94A3B8;
    --color-muted-2: #64748B;
    --color-border: #1F2A44;
    --color-border-strong: #2A3958;
    --color-accent-soft: #053C2E;
    --color-destructive-soft: #3B0F12;
    --color-warning-soft: #3B2A0B;
    --color-success-soft: #053C2E;
    --shadow-sm: 0 1px 2px rgba(0,0,0,.5);
    --shadow-md: 0 4px 12px rgba(0,0,0,.45);
    --shadow-lg: 0 12px 32px rgba(0,0,0,.5);
  }
}
```

Remove exactly those 19 lines. Nothing else in the file changes.

- [ ] **Step 2: Verify the file still compiles**

```bash
cd frontend
npm run build -- --configuration development 2>&1 | Select-String -Pattern "error"
```

Expected: no error lines.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/styles.css
git commit -m "feat(frontend): remove dark mode — always use light theme"
```

---

## Task 7: Redesign calendar cell

**Files:**
- Modify: `frontend/src/app/features/manager/calendar/manager-calendar.component.ts`

This task rewrites the component template and styles for the calendar cell. The `CalendarFilters`, `Overlays` interfaces, signals, and `load()` / `set()` / `toggle()` / `reset()` logic stay **identical**. Only the template, a few helper methods, and the `styles` array change.

- [ ] **Step 1: Add helper methods to the component class**

Inside the `ManagerCalendarComponent` class body, after the `reset()` method, add:

```typescript
  /** Action badge derived from recommended vs current price. */
  badge(p: CalendarRowDto): { label: string; cls: string } {
    const delta = (p.recommended_price_per_night - p.price_per_night) / p.price_per_night;
    if (delta > 0.02)  return { label: '↑ Raise', cls: 'raise' };
    if (delta < -0.02) return { label: '↓ Lower', cls: 'lower' };
    return { label: '✓ Hold', cls: 'hold' };
  }

  /** Context tags shown inline in the date line. */
  tags(p: CalendarRowDto): string[] {
    const o = this.overlays();
    const t: string[] = [];
    if (p.is_weekend_checkin)                                     t.push('Weekend');
    if (o.is_tunisia_public_holiday && p.is_tunisia_public_holiday) t.push('TN Holiday');
    if (o.is_ramadan && p.is_ramadan)                             t.push('Ramadan');
    if (o.is_school_holiday_france && p.is_school_holiday_france) t.push('FR school');
    if (o.is_tunisia_school_holiday && p.is_tunisia_school_holiday) t.push('TN school');
    if (o.is_school_holiday_germany && p.is_school_holiday_germany) t.push('DE school');
    if (o.is_school_holiday_uk && p.is_school_holiday_uk)         t.push('UK school');
    return t;
  }

  /** Human-readable delta percentage string (e.g. "+5.2%" or "−3.1%"). */
  deltaPct(p: CalendarRowDto): string {
    const d = (p.recommended_price_per_night - p.price_per_night) / p.price_per_night * 100;
    return (d >= 0 ? '+' : '') + d.toFixed(1) + '%';
  }

  /** Tooltip for the granularity info (moved off the visible cell). */
  granularityTip(p: CalendarRowDto): string {
    if (!p.best_peer_granularity_used) return '';
    return `Market avg based on ${p.peer_medium_count ?? '?'} hotels (${p.best_peer_granularity_used} neighbourhood)`;
  }

  /** Combined school-holiday toggle state (FR + TN). */
  schoolActive = computed(() =>
    this.overlays().is_school_holiday_france || this.overlays().is_tunisia_school_holiday
  );

  toggleSchool() {
    const next = !this.schoolActive();
    this.overlays.update(o => ({
      ...o,
      is_school_holiday_france: next,
      is_tunisia_school_holiday: next,
    }));
  }
```

Also add `computed` to the imports at the top of the file if not already there (it already is — check the import line `import { Component, OnInit, computed, inject, signal } from '@angular/core';`).

- [ ] **Step 2: Replace the calendar cell section of the template**

Find the `<!-- Calendar grid -->` section in the template (everything from `<section class="card">` to the closing `</section>`). Replace it entirely with:

```html
    <!-- Calendar grid -->
    <section class="card">
      @if (error()) { <div class="cell empty wide muted">{{ error() }}</div> }
      <div class="cal-grid">
        <div class="cal-head">Mon</div>
        <div class="cal-head">Tue</div>
        <div class="cal-head">Wed</div>
        <div class="cal-head">Thu</div>
        <div class="cal-head">Fri</div>
        <div class="cal-head">Sat</div>
        <div class="cal-head">Sun</div>

        @for (s of spacers(); track $index) { <div class="cell empty"></div> }

        @for (p of visible(); track p.check_in) {
          <div class="cell" [class.weekend]="p.is_weekend_checkin"
               [title]="granularityTip(p)">
            <!-- Date line + context tags + action badge -->
            <div class="cell-top">
              <span class="cal-day mono">{{ p.check_in | date:'d MMM' }}</span>
              <span class="badge-action" [class]="badge(p).cls">{{ badge(p).label }}</span>
            </div>
            @if (tags(p).length > 0) {
              <div class="tag-row">
                @for (t of tags(p); track t) {
                  <span class="ctx-tag">{{ t }}</span>
                }
              </div>
            }

            <!-- Price rows -->
            <div class="prices">
              <div class="price-row">
                <span class="price-lbl">Your rate</span>
                <span class="mono fw">{{ p.price_per_night | number:'1.0-0' }} TND</span>
              </div>
              <div class="price-row muted">
                <span class="price-lbl">Market avg</span>
                <span class="mono">{{ p.peer_medium_median !== null ? (p.peer_medium_median | number:'1.0-0') + ' TND' : '—' }}</span>
              </div>
              <div class="price-row muted">
                <span class="price-lbl">Competitors</span>
                <span class="mono">{{ p.competitor_avg_per_night !== null ? (p.competitor_avg_per_night | number:'1.0-0') + ' TND' : '—' }}</span>
              </div>
              <div class="price-row suggested">
                <span class="price-lbl">Suggested</span>
                <span class="mono fw">{{ p.recommended_price_per_night | number:'1.0-0' }} TND</span>
              </div>
            </div>

            <div class="delta-bar" [class]="badge(p).cls">{{ deltaPct(p) }}</div>
          </div>
        }

        @if (!loading() && visible().length === 0) {
          <div class="cell empty wide muted">
            No rows for this configuration in the current window.
          </div>
        }
      </div>
    </section>
```

- [ ] **Step 3: Replace the cell-related CSS in the styles array**

Find the `styles: [\`...\`]` block. Replace everything **from** `.cal-grid {` **to the end** of the styles block with:

```css
    .cal-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 1px; background: var(--color-border); padding: 1px; }
    .cal-head { background: var(--color-surface-2); padding: 8px 10px; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; color: var(--color-muted); font-weight: 600; }

    .cell { background: var(--color-surface); padding: 10px; min-height: 150px; display: flex; flex-direction: column; gap: 6px; transition: background-color .12s ease; }
    .cell:hover { background: var(--color-surface-2); }
    .cell.empty { background: transparent; min-height: 80px; }
    .cell.empty.wide { grid-column: 1 / -1; min-height: 60px; display: flex; align-items: center; justify-content: center; font-size: 13px; padding: 18px; }
    .cell.weekend { background: linear-gradient(180deg, rgba(37,99,235,.04), var(--color-surface)); }

    .cell-top { display: flex; justify-content: space-between; align-items: center; gap: 4px; }
    .cal-day { font-size: 13px; font-weight: 700; color: var(--color-foreground); }

    .badge-action { font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 6px; white-space: nowrap; }
    .badge-action.hold  { background: var(--color-success-soft);      color: var(--color-success); }
    .badge-action.raise { background: var(--color-warning-soft);       color: var(--color-warning); }
    .badge-action.lower { background: var(--color-destructive-soft);   color: var(--color-destructive); }

    .tag-row { display: flex; flex-wrap: wrap; gap: 3px; }
    .ctx-tag { font-size: 9px; padding: 1px 5px; border-radius: 3px; background: var(--color-surface-2); color: var(--color-muted); border: 1px solid var(--color-border); }

    .prices { display: flex; flex-direction: column; gap: 3px; flex: 1; }
    .price-row { display: flex; justify-content: space-between; align-items: baseline; font-size: 12px; }
    .price-row.muted .price-lbl, .price-row.muted .mono { color: var(--color-muted-2); }
    .price-row.suggested { border-top: 1px solid var(--color-border); padding-top: 4px; margin-top: 2px; }
    .price-row.suggested .price-lbl { color: var(--color-primary); font-weight: 600; }
    .price-row.suggested .mono { color: var(--color-primary); }
    .price-lbl { color: var(--color-muted); }
    .fw { font-weight: 700; color: var(--color-foreground); }

    .delta-bar { font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; text-align: center; margin-top: auto; }
    .delta-bar.hold  { background: var(--color-surface-2); color: var(--color-muted); }
    .delta-bar.raise { background: var(--color-warning-soft); color: var(--color-warning); }
    .delta-bar.lower { background: var(--color-destructive-soft); color: var(--color-destructive); }

    @media (max-width: 900px) { .cell { min-height: 120px; } }
```

- [ ] **Step 4: Verify TypeScript compiles with no errors**

```bash
cd frontend
npm run build -- --configuration development 2>&1 | Select-String -Pattern "error TS|ERROR"
```

Expected: no output (no TypeScript errors).

- [ ] **Step 5: Start the dev server and visually check the calendar**

```bash
npm start
```

Open `http://localhost:4200`, log in as `manager@revway.tn / REDACTED_DEV_PASSWORD`, navigate to **Price calendar**. Verify:
- Each cell shows "Your rate / Market avg / Competitors / Suggested" in plain English
- Action badge (Hold/Raise/Lower) is visible with colour coding
- Weekends show "Weekend" tag inline
- Hovering a cell shows the granularity tooltip (inspect via DevTools if needed)
- `tight · n50` text is gone from the cell body

- [ ] **Step 6: Commit**

```bash
git add frontend/src/app/features/manager/calendar/manager-calendar.component.ts
git commit -m "feat(frontend): redesign calendar cell — human labels, action badge, competitor avg"
```

---

## Task 8: Simplify calendar filter bar

**Files:**
- Modify: `frontend/src/app/features/manager/calendar/manager-calendar.component.ts`

- [ ] **Step 1: Add the showMore signal, ROOM_BASE_LABELS constant, and boardingLabel helper**

At the top of the component class body (after the `readonly label = label;` line), add:

```typescript
  showMore = signal(false);

  private static readonly ROOM_BASE_LABELS: Record<string, string> = {
    chambre:      'Room',
    suite:        'Suite',
    studio:       'Studio',
    appartement:  'Apartment',
    bungalow:     'Bungalow',
    villa:        'Villa',
  };

  roomBaseLabel(v: string): string {
    return ManagerCalendarComponent.ROOM_BASE_LABELS[v] ?? v;
  }

  boardingLabel(v: string): string {
    const map: Record<string, string> = {
      BB:       'Bed & Breakfast',
      LOG:      'Room only',
      HDP:      'Half board',
      HDP_PLUS: 'Half board +',
      PC:       'Full board',
      PC_PLUS:  'Full board +',
      AI:       'All inclusive',
      AI_SOFT:  'AI soft',
      AI_ULTRA: 'AI ultra',
    };
    return map[v] ?? v;
  }
```

- [ ] **Step 2: Replace the filter bar section of the template**

Find the `<!-- Filter bar -->` section (from `<section class="card filter-bar">` to its closing `</section>`). Replace it entirely with:

```html
    <!-- Filter bar -->
    <section class="card filter-bar">
      <!-- Primary row: 4 key filters -->
      <div class="primary-filters">
        <div class="field">
          <label for="f-board">Meal plan</label>
          <select id="f-board" class="select" [value]="filters().boarding_canonical"
                  (change)="set('boarding_canonical', $any($event.target).value)">
            @for (b of opt().boarding_canonical; track b) {
              <option [value]="b">{{ boardingLabel(b) }}</option>
            }
          </select>
        </div>

        <div class="field">
          <label for="f-base">Room type</label>
          <select id="f-base" class="select" [value]="filters().room_base"
                  (change)="set('room_base', $any($event.target).value)">
            @for (v of opt().room_base; track v) {
              <option [value]="v">{{ roomBaseLabel(v) }}</option>
            }
          </select>
        </div>

        <div class="field f-xs">
          <label for="f-n">Nights</label>
          <select id="f-n" class="select mono" [value]="filters().nights"
                  (change)="set('nights', +$any($event.target).value)">
            @for (n of opt().nights; track n) { <option [value]="n">{{ n }}</option> }
          </select>
        </div>

        <div class="field f-xs">
          <label for="f-a">Adults</label>
          <select id="f-a" class="select mono" [value]="filters().adults"
                  (change)="set('adults', +$any($event.target).value)">
            @for (a of opt().adults; track a) { <option [value]="a">{{ a }}</option> }
          </select>
        </div>

        <div class="holiday-toggles">
          <label class="chip-toggle">
            <input type="checkbox" [checked]="overlays().is_tunisia_public_holiday"
                   (change)="toggle('is_tunisia_public_holiday')">
            <span>Public holidays</span>
          </label>
          <label class="chip-toggle">
            <input type="checkbox" [checked]="schoolActive()"
                   (change)="toggleSchool()">
            <span>School holidays</span>
          </label>
        </div>

        <button class="btn more-btn" (click)="showMore.update(v => !v)">
          {{ showMore() ? 'Less ▲' : 'More options ▼' }}
        </button>
      </div>

      <!-- Collapsible advanced filters -->
      @if (showMore()) {
        <div class="advanced-filters">
          <div class="field f-sm">
            <label for="f-view">room_view <span class="hint">view type</span></label>
            <select id="f-view" class="select" [value]="filters().room_view"
                    (change)="set('room_view', $any($event.target).value)">
              @for (v of opt().room_view; track v) { <option [value]="v">{{ label(v) }}</option> }
            </select>
          </div>

          <div class="field f-sm">
            <label for="f-tier">room_tier</label>
            <select id="f-tier" class="select" [value]="filters().room_tier"
                    (change)="set('room_tier', $any($event.target).value)">
              @for (v of opt().room_tier; track v) { <option [value]="v">{{ label(v) }}</option> }
            </select>
          </div>

          <div class="field f-sm">
            <label for="f-occ">room_occupancy</label>
            <select id="f-occ" class="select" [value]="filters().room_occupancy"
                    (change)="set('room_occupancy', $any($event.target).value)">
              @for (v of opt().room_occupancy; track v) { <option [value]="v">{{ label(v) }}</option> }
            </select>
          </div>

          <div class="field f-sm">
            <label for="f-scrape">scrape_date <span class="hint">freshness</span></label>
            <select id="f-scrape" class="select" [value]="filters().scrape_date"
                    (change)="set('scrape_date', $any($event.target).value)">
              <option value="">latest</option>
              @for (d of opt().scrape_dates; track d) { <option [value]="d">{{ d }}</option> }
            </select>
          </div>

          <div class="field f-sm">
            <label for="f-peer">best_peer_granularity_used</label>
            <select id="f-peer" class="select" [value]="filters().best_peer_granularity_used"
                    (change)="set('best_peer_granularity_used', $any($event.target).value)">
              <option value="any">any</option>
              <option value="tight">tight</option>
              <option value="medium">medium</option>
              <option value="loose">loose</option>
            </select>
          </div>

          <div class="adv-toggles">
            <label class="chip-toggle">
              <input type="checkbox" [checked]="overlays().is_ramadan" (change)="toggle('is_ramadan')">
              <span>Ramadan</span>
            </label>
            <label class="chip-toggle">
              <input type="checkbox" [checked]="overlays().is_school_holiday_germany"
                     (change)="toggle('is_school_holiday_germany')">
              <span>DE school holidays</span>
            </label>
            <label class="chip-toggle">
              <input type="checkbox" [checked]="overlays().is_school_holiday_uk"
                     (change)="toggle('is_school_holiday_uk')">
              <span>UK school holidays</span>
            </label>
          </div>
        </div>
      }
    </section>
```

- [ ] **Step 3: Replace the filter bar CSS in the styles array**

Find and replace the existing `.filter-bar`, `.filters`, `.overlays`, and `.chip-toggle` CSS rules with:

```css
    .filter-bar { padding: 14px 16px 12px; margin-bottom: 12px; }

    .primary-filters {
      display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: flex-end;
    }
    .primary-filters .field { display: flex; flex-direction: column; gap: 4px; }
    .primary-filters .field label { font-size: 12px; color: var(--color-muted); font-weight: 500; }
    .primary-filters .field label .hint { color: var(--color-muted-2); font-weight: 400; margin-left: 4px; font-size: 11px; }
    .primary-filters .select { height: 34px; font-size: 13px; padding: 0 10px; }
    .primary-filters .select.mono { font-family: var(--font-mono); }
    .f-xs { width: 80px; }
    .f-sm { width: 160px; }

    .holiday-toggles { display: flex; gap: 8px; align-items: center; padding-bottom: 2px; }
    .more-btn { align-self: flex-end; font-size: 12px; height: 34px; }

    .chip-toggle {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 5px 12px; border-radius: 999px;
      border: 1px solid var(--color-border);
      background: var(--color-surface-2); cursor: pointer;
      font-size: 12px; color: var(--color-muted);
      transition: all .15s ease;
    }
    .chip-toggle:hover { color: var(--color-foreground); border-color: var(--color-border-strong); }
    .chip-toggle input { accent-color: var(--color-primary); }
    .chip-toggle:has(input:checked) {
      background: rgba(37,99,235,.10); border-color: var(--color-primary); color: var(--color-primary);
    }

    .advanced-filters {
      display: flex; flex-wrap: wrap; gap: 10px 14px; align-items: flex-end;
      margin-top: 12px; padding-top: 12px;
      border-top: 1px dashed var(--color-border);
    }
    .advanced-filters .field { display: flex; flex-direction: column; gap: 4px; }
    .advanced-filters .field label { font-family: var(--font-mono); font-size: 11px; color: var(--color-muted); font-weight: 500; }
    .advanced-filters .field label .hint { color: var(--color-muted-2); font-weight: 400; margin-left: 4px; }
    .advanced-filters .select { height: 32px; font-size: 12px; padding: 0 8px; }
    .adv-toggles { display: flex; gap: 8px; align-items: center; padding-bottom: 2px; }
```

- [ ] **Step 4: Verify TypeScript compiles**

```bash
cd frontend
npm run build -- --configuration development 2>&1 | Select-String -Pattern "error TS|ERROR"
```

Expected: no output.

- [ ] **Step 5: Visual check in browser**

Reload `http://localhost:4200/manager/calendar`. Verify:
- Only 4 dropdowns visible (Meal plan, Room type, Nights, Adults) + 2 holiday chips + "More options ▼" button
- Clicking "More options ▼" expands the advanced row with technical filters
- Clicking "Less ▲" collapses it
- Changing any filter reloads the calendar grid
- "Reset filters" still works

- [ ] **Step 6: Commit**

```bash
git add frontend/src/app/features/manager/calendar/manager-calendar.component.ts
git commit -m "feat(frontend): simplify calendar filter bar — 4 primary filters + collapsible more options"
```

---

## Task 9: Final smoke test

- [ ] **Step 1: Run the full backend test suite**

```bash
cd backend
.venv\Scripts\pytest -v
```

Expected: all tests pass (no failures, no errors). Count should be the existing tests + 3 new calendar tests.

- [ ] **Step 2: Build the frontend for production**

```bash
cd frontend
npm run build
```

Expected: build succeeds. Check the summary — initial bundle should stay under the 1 MB warn budget.

- [ ] **Step 3: Demo walkthrough**

Start backend and frontend. Log in as `manager@revway.tn / REDACTED_DEV_PASSWORD`. Confirm:

| Screen | Check |
|---|---|
| Dashboard | KPI cards show real numbers, no "—" everywhere |
| Calendar | Cells show Hold/Raise/Lower badges, "Your rate / Market avg / Competitors / Suggested" labels, no raw Postgres column names |
| Calendar filters | 4 primary filters visible, "More options" collapses correctly |
| Calendar — dark mode check | Open DevTools → Rendering → Emulate "prefers-color-scheme: dark" → page stays light |
| Competitors | 3 hotel cards visible with real prices |
| Recommendations | List loads, Accept/Dismiss buttons work |
| Alerts | List loads |
| Settings | Profile editable |
| Admin — Hotels | Real hotel list (not mock names) |
| Admin — Managers | Real manager list |
| Admin — Assignments | Real assignment data |
| Admin — Scrapers | Real scrape run history |

- [ ] **Step 4: Commit if any last-minute fixes were made**

```bash
git add -p
git commit -m "fix(frontend): demo smoke test fixes"
```

---

## Summary of commits expected

```
feat(backend): add competitor_avg_per_night field to CalendarRow schema
feat(backend): compute competitor_avg_per_night in calendar service
test(backend): calendar endpoint — competitor_avg_per_night field
feat(frontend): add competitor_avg_per_night to CalendarRowDto
feat(frontend): remove dark mode — always use light theme
feat(frontend): redesign calendar cell — human labels, action badge, competitor avg
feat(frontend): simplify calendar filter bar — 4 primary filters + collapsible more options
```
