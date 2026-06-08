# backend/main.py
from __future__ import annotations
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
from core.exceptions import register_exception_handlers
from core.dependencies import set_ml_store
from ml_store.store import load_ml_store

log = logging.getLogger("revway")

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading ML models from %s", settings.model_dir)
    try:
        store = load_ml_store(Path(settings.model_dir))
        set_ml_store(store)
        log.info("ML models loaded: %d forecaster features", len(store.forecaster.feature_names_))
    except Exception as exc:
        log.error("ML model load failed: %s — /health will report not_loaded", exc)
    yield
    log.info("Shutting down")
    from db.mongo import close_mongo_client
    close_mongo_client()

app = FastAPI(title="RevWay API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

register_exception_handlers(app)

@app.get("/health")
async def health():
    from core.dependencies import _ml_store
    return {
        "status": "ok",
        "ml_store": "loaded" if _ml_store is not None else "not_loaded",
    }

from routers import auth, calendar, competitors, recommendations, recommendation_decisions, anomalies, profile
from routers.admin import hotels as admin_hotels
from routers.admin import managers as admin_managers
from routers.admin import assignments as admin_assignments
from routers.admin import competitors as admin_competitors
from routers.admin import monitoring as admin_monitoring
from routers.admin import alerts as admin_alerts
app.include_router(auth.router)
app.include_router(calendar.router)
app.include_router(competitors.router)
app.include_router(recommendations.router)
app.include_router(recommendation_decisions.router)
app.include_router(anomalies.router)
app.include_router(profile.router)
app.include_router(admin_hotels.router)
app.include_router(admin_managers.router)
app.include_router(admin_assignments.router)
app.include_router(admin_competitors.router)
app.include_router(admin_monitoring.router)
app.include_router(admin_alerts.router)
