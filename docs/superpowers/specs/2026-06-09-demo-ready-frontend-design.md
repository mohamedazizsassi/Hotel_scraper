# Demo-Ready Frontend — Design Spec
**Date:** 2026-06-09  
**Branch:** `feat/demo-ready` (off `feat/admin-platform`)  
**Author:** mohamedazizsassi

---

## Goal

Prepare the RevWay platform for a live jury demo (~2026-06-15) by:
1. Connecting all frontend pages to real database data (admin pages are currently mock-backed)
2. Redesigning the calendar so a revenue manager — not a developer — can read it
3. Simplifying the calendar filter bar for a clean demo
4. Forcing the light theme (white background) regardless of system dark-mode setting

The manager side (calendar, competitors, recommendations, alerts, dashboard, settings) is already live-wired and stays untouched. The admin wiring is complete on `feat/admin-platform` — this branch starts there.

---

## Scope (5 changes)

| # | Layer | Change |
|---|---|---|
| 1 | Backend | Add `competitor_avg_per_night` to `CalendarRow` + calendar service |
| 2 | Frontend | Calendar cell redesign — human labels, action badge, two reference prices |
| 3 | Frontend | Calendar filter bar — 4 visible filters + collapsible "More options" |
| 4 | Frontend | Remove dark mode (`@media prefers-color-scheme: dark` block) |
| 5 | — | Demo hotel confirmed as Concorde Marco Polo (no DB changes needed) |

---

## Section 1 — Branch strategy

```
feat/admin-platform  (base — all admin API + frontend wiring, 64 tests)
         └── feat/demo-ready  (this work)
```

`feat/admin-platform` contains: admin hotels, managers, assignments, scrapers, competitor-selection (D11) pages all wired to the live FastAPI admin API. No re-implementation needed.

---

## Section 2 — Backend: `competitor_avg_per_night`

**Why:** The calendar cell design shows two reference prices — statistical market peers (`peer_medium_median`) and the manager's personally selected competitors (`competitor_avg_per_night`). Only the first is currently in the API response.

**What changes:**

`backend/schemas/calendar.py` — add one field to `CalendarRow`:
```python
competitor_avg_per_night: float | None
# Average price_per_night of the manager's user_competitor_selections
# for this check_in date + same boarding_canonical / nights / adults.
# null when no competitor has data for this date.
```

`backend/services/calendar_service.py` — after the main hotel_features query, run a second aggregation:
```sql
SELECT
    hf.check_in,
    AVG(hf.price_per_night) AS competitor_avg_per_night
FROM user_competitor_selections ucs
JOIN platform_hotels ph ON ph.id = ucs.hotel_id
JOIN hotel_features hf   ON hf.hotel_name_normalized = ph.hotel_name_normalized
WHERE ucs.user_id = :user_id
  AND hf.check_in BETWEEN :check_in_from AND :check_in_to
  AND hf.boarding_canonical = :boarding_canonical
  AND hf.nights = :nights
  AND hf.adults = :adults
GROUP BY hf.check_in
```
Merge into the output list by `check_in`. Where no row exists, `competitor_avg_per_night = None`.

`frontend/src/app/core/api/dto.ts` — add to `CalendarRowDto`:
```typescript
competitor_avg_per_night: number | null;
```

**Preserves:** all existing calendar fields and query logic. This is additive only.

---

## Section 3 — Frontend: calendar cell redesign

**File:** `frontend/src/app/features/manager/calendar/manager-calendar.component.ts`

### Cell layout (per check_in date)

```
┌─────────────────────────────────────┐
│  Jun 27  Weekend          ↑ Raise   │  ← date + context tag + action badge
│  Your rate          560 TND         │
│  Market avg         720 TND         │  ← peer_medium_median (null → —)
│  Competitors        695 TND         │  ← competitor_avg_per_night (null → —)
│  ─────────────────────────────────  │
│  Suggested          690 TND  +23%   │  ← recommended_price_per_night + delta%
└─────────────────────────────────────┘
```

### Action badge logic
Derived from `(recommended_price_per_night - price_per_night) / price_per_night`:
- `> +2%` → amber **↑ Raise**
- `< −2%` → red **↓ Lower**
- otherwise → green **✓ Hold**

### Context tags (inline in date line)
Shown when the overlay flag is true AND the corresponding overlay toggle is on:
- `is_weekend_checkin` → "Weekend" (amber)
- `is_tunisia_public_holiday` → "TN Holiday" (red)
- `is_ramadan` → "Ramadan" (purple)
- `is_school_holiday_france` → "FR school" (blue)
- `is_tunisia_school_holiday` → "TN school" (orange)
- `is_school_holiday_germany` → "DE school" (blue)
- `is_school_holiday_uk` → "UK school" (blue)

### Peer granularity pill
Moved to `title` attribute (hover tooltip) on the cell. No longer rendered visually.  
Format: `"Market avg based on N comparable hotels (tight / medium / loose neighbourhood)"`

---

## Section 4 — Frontend: filter bar simplification

**File:** `frontend/src/app/features/manager/calendar/manager-calendar.component.ts`

### Primary row (always visible — 4 filters)

| Label | Underlying param(s) | Values |
|---|---|---|
| Meal plan | `boarding_canonical` | Friendly labels from `BOARDING_VALUES` (BB → "Bed & Breakfast", AI → "All inclusive", etc.) |
| Room type | `room_base` | Friendly labels: chambre → "Room", suite → "Suite", studio → "Studio", appartement → "Apartment", bungalow → "Bungalow", villa → "Villa" |
| Nights | `nights` | 1 / 2 / 3 / 5 / 7 |
| Adults | `adults` | 1 / 2 / 3 / 4 |

### Holiday toggles (always visible — condensed to 2)
- **Public holidays** → `is_tunisia_public_holiday` overlay
- **School holidays** → `is_school_holiday_france` + `is_tunisia_school_holiday` overlays together

### "More options ▾" (collapsed by default)
Contains: `room_view`, `room_tier`, `room_occupancy`, `scrape_date`, `best_peer_granularity_used`, Germany/UK school holiday toggles.

All query params sent to `GET /manager/calendar` are identical to today — the simplification is display-only.

### Default state
On load, the filter bar initialises from `GET /manager/calendar/options` defaults, same as today. "More options" starts collapsed.

---

## Section 5 — Frontend: disable dark mode

**File:** `frontend/src/styles.css`

Remove the block:
```css
@media (prefers-color-scheme: dark) {
  :root { … }
}
```

The light-mode tokens at `:root` are unchanged:
- `--color-background: #F8FAFC`
- `--color-surface: #FFFFFF`
- `--color-primary: #2563EB`
- `--color-accent: #059669`

No component changes required.

---

## Demo configuration

| Item | Value |
|---|---|
| Manager hotel | **Concorde Marco Polo** — Hammamet, 4★ |
| Manager login | `manager@revway.tn` / `REDACTED_DEV_PASSWORD` |
| Data available | 114,480 rows · 200 check-in dates · 28 scrape runs |
| Competitor 1 | Iberostar Averroes (148K rows) |
| Competitor 2 | El Mouradi Hammamet (124K rows) |
| Competitor 3 | Bel Azur Thalasso & Bungalows (122K rows) |
| Admin login | (from admin seed — `admin@revway.tn`) |

No DB seed changes required. The existing `feat/admin-platform` seed covers all platform hotels and the manager assignment.

---

## What is NOT changing

- Manager dashboard, competitors, recommendations, alerts, settings — already live-wired, untouched
- Admin pages (hotels, managers, assignments, scrapers, competitor selection) — already live-wired on `feat/admin-platform`, this branch inherits them
- Backend calendar query logic — additive only (`competitor_avg_per_night` is a new field)
- Route structure, auth, JWT handling — unchanged
- `core/data/mock.ts` taxonomy value lists — still used as the single source of truth for filter dropdowns

---

## Out of scope

- Redesigning the recommendations or alerts pages
- Adding new backend endpoints beyond the `competitor_avg_per_night` addition
- Any ML model changes
- Mobile responsiveness improvements
