# RevWay — Frontend

Static Angular 19 frontend for **RevWay (Satisfy Insight)** — a multi-tenant SaaS
giving Tunisian hotel revenue managers AI-driven competitor price intelligence.

This build is **fully static**: all data is mocked in `src/app/core/data/mock.ts`.
No backend calls. It is meant as a defense-ready demo of the UX surface that the
forecaster, anomaly module, and recommender will plug into.

## Design system

Generated with the `ui-ux-pro-max` skill.

- **Pattern**: Real-Time / Operations
- **Style**: Data-Dense Dashboard · light + dark
- **Primary**: `#2563EB` · **Accent**: `#059669` · **Destructive**: `#DC2626`
- **Type**: Fira Sans (body) · Fira Code (numbers / mono)
- **Anti-patterns avoided**: ornate decoration, no filtering, emojis as icons

All tokens live as CSS custom properties in `src/styles.css` and adapt to
`prefers-color-scheme: dark` automatically.

## Run

```bash
cd frontend
npm install
npm start
# → http://localhost:4200
```

## Routes

| Path                            | Persona  | Purpose                                        |
| ------------------------------- | -------- | ---------------------------------------------- |
| `/`                             | public   | Landing — product hero, metrics, how it works  |
| `/login`                        | public   | Sign-in (with demo shortcuts)                  |
| `/admin/dashboard`              | admin    | Platform overview · scrape health · coverage   |
| `/admin/hotels`                 | admin    | Hotel pool, search + filter                    |
| `/admin/managers`               | admin    | Manager accounts                               |
| `/admin/assignments`            | admin    | Map managers ↔ hotels                          |
| `/admin/scrapers`               | admin    | Scrapy run history + KPIs                      |
| `/manager/dashboard`            | manager  | KPIs · 14-day price chart · top recs · alerts  |
| `/manager/calendar`             | manager  | 30-day price calendar with rec overlay         |
| `/manager/competitors`          | manager  | The 3–4 self-picked competitors                |
| `/manager/recommendations`      | manager  | Full rec list with rationale + accept/dismiss  |
| `/manager/alerts`               | manager  | Anomaly + market-event feed                    |
| `/manager/settings`             | manager  | Profile, hotel info, alert preferences         |

## Structure

```
src/app/
  app.routes.ts         lazy-loaded standalone routes
  app.config.ts         router + zone change detection
  core/
    models/domain.ts    Hotel, ManagerUser, PricePoint, Recommendation, Alert…
    data/mock.ts        all seed data (anchored to 2026-05-23)
  shared/components/    sidebar · topbar · kpi-card · sparkline · status-pill
  features/
    landing/  login/
    admin/    shell · dashboard · hotels · managers · assignments · scrapers
    manager/  shell · dashboard · calendar · competitors · recommendations · alerts · settings
```

## Notes for the backend integration (later)

- Manager authorization must validate every `hotel_id` against
  `user_competitor_selections` — the frontend never enforces this.
- `competitor_avg` shown to a manager is scoped to that manager's personal pick,
  not the whole market.
- The 30-day calendar is the surface the LightGBM quantile forecaster feeds.
- The alerts feed is the surface the D3 anomaly module feeds.
