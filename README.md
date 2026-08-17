# RevWay

AI-driven competitor price intelligence for independent Tunisian hotels — turns public OTA listings into a daily-refreshed price forecast, anomaly flag, and repricing recommendation for each hotel's chosen competitor set.

## Overview

Tunisian hotel revenue managers set prices with little visibility into what comparable hotels are actually charging on OTAs. RevWay scrapes public listings from two Tunisian booking sites, builds a feature table describing each hotel's competitive position, and serves two personas through a multi-tenant web app:

- **Admins** manage hotel/manager records, assign managers to hotels, define each hotel's competitor set (max 4), and monitor the scraping pipeline (execution stats, error reports, collection alerts).
- **Managers** get a dashboard of pricing KPIs, a competitive price calendar, price recommendations with confidence scores, anomaly flags, and configurable alerts — scoped strictly to their assigned hotel and personal competitor selection.

Predictions model **public market pricing at scrape time** — a market-aligned recommendation given the observed competitive landscape, not a booking/occupancy forecast.

## Key features

- Scrapy pipeline collecting ~1M price observations/day from two independent OTA sources into MongoDB
- Leakage-safe feature engineering (temporal self-exclusion on peer aggregates) producing a ~52-column PostgreSQL serving table + calendar/segment dimensions
- Quantile forecaster (LightGBM/CatBoost/XGBoost bake-off) predicting P10/P50/P90 price bands per hotel/date
- Conformalized Quantile Regression (CQR) anomaly detection on top of the forecaster's calibrated intervals
- Rule-based repricing recommendations (raise/hold/lower) with confidence scores and peer-context reasoning
- Multi-tenant FastAPI backend with JWT auth — every request validates the caller's `hotel_id` against server-side assignment/competitor-selection tables, never trusting the frontend
- Angular admin + manager web apps

## Architecture

```
Scrapy spiders (promohotel.tn, tunisiepromo.tn)
        │
        ▼
MongoDB  hotel_scraper.hotel_prices   (~24M+ rows, ~1M/day growth)
        │
        ▼
ml/feature_engineering  (load → clean → taxonomy → calendar/segment → competitive → demand)
        │
        ▼
PostgreSQL  hotel_features + calendar_dim + segment_dim
        │
        ▼
ml/models  (quantile forecaster → CQR anomaly detector → rule-based recommender)
        │
        ▼
FastAPI backend (JWT auth, per-tenant authorization)  ──►  Angular frontend (Admin / Manager)
```

## Technology stack

| Layer              | Technology                                              |
| ------------------ | -------------------------------------------------------- |
| Scraping            | Scrapy, MongoDB (raw store)                              |
| Data Engineering    | pandas, pymongoarrow, pyarrow, PostgreSQL, SQLAlchemy     |
| Machine Learning    | LightGBM, CatBoost, XGBoost, Optuna, conformal prediction |
| Backend             | FastAPI, PyJWT, bcrypt, asyncpg, Pydantic                 |
| Frontend            | Angular 19, TypeScript                                   |
| Testing             | pytest, pytest-asyncio                                   |

## Project structure

```
revway/
├── scraper/     Scrapy spiders → MongoDB hotel_prices (see scraper/README.md)
├── ml/          feature engineering + forecaster/anomaly/recommender models (see ml/CLAUDE.md)
├── backend/     FastAPI services, JWT auth, admin + manager APIs
├── frontend/    Angular app (Admin and Manager modules)
├── database/postgres/migrations/  platform + feature table schema
└── assets/      diagrams / figures referenced from this README
```

## Installation

Each module keeps its own virtual environment and dependency list — never share venvs across modules.

```bash
# Scraper
cd scraper && python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt

# ML
cd ml && python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt

# Backend
cd backend && python -m venv .venv && .venv/Scripts/activate && pip install -r requirements.txt

# Frontend
cd frontend && npm install
```

Requires a running MongoDB instance (raw scrape data) and PostgreSQL instance (platform + feature tables); apply migrations from `database/postgres/migrations/` in order.

## Configuration

Each module has its own `.env`, gitignored, based on its `.env.example` template (`backend/.env.example`, `ml/.env.example`, `scraper/.envexemple`). Copy the template, fill in real values — never commit a real `.env`. Key variables: `MONGO_URI`, `POSTGRES_URI` (or the discrete `POSTGRES_*` vars), and JWT signing secret for the backend.

## Running the project

```bash
# Backend API (from backend/, venv active)
.venv/Scripts/python.exe -m uvicorn main:app --reload --port 8000

# Frontend (from frontend/)
ng serve

# Scraper (from scraper/, venv active)
python run_scrape.py

# Feature engineering (from ml/, venv active)
python -m feature_engineering.assemble
```

There is no Docker setup in this repo — each service runs directly against local MongoDB/PostgreSQL instances.

## ML / data pipeline

1. **Collection** — two Scrapy spiders (`promohotel_spider.py`, `tunisiepromo_spider.py`) poll listings on a tiered daily schedule, writing normalized items to MongoDB.
2. **Feature engineering** (`ml/feature_engineering/`) — loads raw Mongo rows, cleans, expands view-upgrade supplements into synthetic rows, canonicalizes taxonomy (boarding, city), builds calendar/segment dimensions, computes leakage-safe competitive/demand aggregates, and writes a ~52-column serving table to PostgreSQL plus a frozen training Parquet snapshot.
3. **Modeling** (`ml/models/`) — a quantile forecaster (LightGBM/CatBoost/XGBoost, selected via an Optuna bake-off) predicts P10/P50/P90 log-price; a conformal calibrator widens the raw quantiles to hit nominal interval coverage; an anomaly detector flags observations outside the calibrated band; a rule-based recommender turns the calibrated interval + peer context into raise/hold/lower guidance.
4. **Evaluation** — every trained model is compared against four baselines (global median, group median, competitor median, linear hedonic OLS on log-price) on a hotel-wise hold-out split; a model ships only if it beats all four.

See `ml/CLAUDE.md` for the full pipeline design, locked modeling decisions, and measured baseline/coverage numbers.

## API

The backend exposes JWT-authenticated REST endpoints under two role scopes:

**Admin** (`/admin/...`): hotel CRUD, manager CRUD, hotel↔manager assignments, competitor management, scraper run monitoring, collection alerts.

**Manager** (scoped to the caller's assigned hotel): `/auth/login`, `/me` (profile), `/dashboard` (KPIs + panels), `/calendar` (competitive price calendar), `/recommendations` + decision persistence (accept/dismiss), `/anomalies`, `/competitors`.

Every manager-scoped endpoint validates the requested `hotel_id` against the caller's server-side assignment and competitor-selection records — the JWT carries role and hotel access, and the frontend's requested scope is never trusted directly.

## Testing

```bash
cd backend && .venv/Scripts/python.exe -m pytest
cd ml && .venv/Scripts/python.exe -m pytest
```

## Future improvements

- Temporal/longitudinal features (`temporal_features.py`) once enough daily snapshots have accumulated
- Docker Compose for one-command local bring-up
- Vectorized recommendation scoring (current row-by-row scoring takes ~5–15 min on the full dataset, acceptable for offline batch use only)

## Academic context

RevWay is a final-year engineering project (PFE). The scraper and platform layers are running against live data; the ML layer (feature engineering, forecasting, anomaly detection, recommendation) and the full-stack admin/manager web app are complete and documented per-module in each subdirectory.
