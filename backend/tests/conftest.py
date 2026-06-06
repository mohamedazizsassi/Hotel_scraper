# backend/tests/conftest.py
import uuid
import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy import text
from unittest.mock import MagicMock
import numpy as np
import pandas as pd

from core.config import settings
from core.security import hash_password
from db.models import Base
from main import app
from core.dependencies import get_db, get_ml_store
from ml_store.store import MLStore

# Each async fixture owns an engine created in (and disposed within) its own
# event loop. A single shared module-level engine would bind asyncpg
# connections to the session loop and then be reused from the per-test function
# loop, raising "another operation is in progress". NullPool keeps no
# cross-loop connections alive.

@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def setup_test_db():
    engine = create_async_engine(settings.test_db_url, echo=False, poolclass=NullPool)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
        # Seed city
        await conn.execute(text(
            "INSERT INTO cities (name_normalized, name_display) "
            "VALUES ('hammamet', 'Hammamet')"
        ))
        # Seed hotels — city_id via subquery so it works regardless of auto-generated id
        await conn.execute(text("""
            INSERT INTO platform_hotels (hotel_name_normalized, hotel_name_display, city_id, stars_int, is_active)
            VALUES
              ('hotel_manager_test', 'Hotel Manager Test',
               (SELECT id FROM cities WHERE name_normalized = 'hammamet'), 4, true),
              ('hotel_comp_1', 'Hotel Comp 1',
               (SELECT id FROM cities WHERE name_normalized = 'hammamet'), 4, true),
              ('hotel_comp_2', 'Hotel Comp 2',
               (SELECT id FROM cities WHERE name_normalized = 'hammamet'), 4, true)
        """))
        # Seed manager user
        uid = str(uuid.uuid4())
        pw = hash_password("testpass")
        await conn.execute(text(
            f"INSERT INTO users (id, email, password_hash, full_name, role, is_active) "
            f"VALUES ('{uid}', 'manager@test.com', '{pw}', 'Test Manager', 'manager', true)"
        ))
        # Seed admin user (for /admin/* tests)
        admin_id = str(uuid.uuid4())
        admin_pw = hash_password("adminpass")
        await conn.execute(text(
            f"INSERT INTO users (id, email, password_hash, full_name, role, is_active) "
            f"VALUES ('{admin_id}', 'admin@test.com', '{admin_pw}', 'Test Admin', 'admin', true)"
        ))
        # Seed assignment: manager → hotel_manager_test
        await conn.execute(text(
            f"INSERT INTO user_hotel_assignments (user_id, hotel_id, max_competitors, is_active) "
            f"VALUES ('{uid}', "
            f"(SELECT id FROM platform_hotels WHERE hotel_name_normalized = 'hotel_manager_test'), "
            f"4, true)"
        ))
        # Seed 2 competitor selections
        await conn.execute(text(
            f"INSERT INTO user_competitor_selections (user_id, hotel_id, display_order) VALUES "
            f"('{uid}', (SELECT id FROM platform_hotels WHERE hotel_name_normalized = 'hotel_comp_1'), 1), "
            f"('{uid}', (SELECT id FROM platform_hotels WHERE hotel_name_normalized = 'hotel_comp_2'), 2)"
        ))
        # Minimal hotel_features table + hotel_features_full view. The real table
        # is created by the ML pipeline (not the ORM), so the recommendation /
        # anomaly services have nothing to query against in a fresh test DB. The
        # mock ML store ignores the row content; we only need >=1 row for the
        # manager's hotel so the service reaches the scoring/mapping path.
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS hotel_features (
                hotel_name_normalized text,
                city_name             text,
                stars_int             int,
                check_in              date,
                nights                int,
                adults                int,
                boarding_canonical    text,
                room_base             text,
                room_view             text,
                room_tier             text,
                room_occupancy        text,
                price                 double precision,
                price_per_night       double precision,
                scraped_at            text,
                peer_medium_median    double precision,
                peer_medium_count     int
            )
        """))
        await conn.execute(text("""
            INSERT INTO hotel_features
              (hotel_name_normalized, city_name, stars_int, check_in, nights, adults,
               boarding_canonical, room_base, room_view, room_tier, room_occupancy,
               price, price_per_night, scraped_at,
               peer_medium_median, peer_medium_count)
            VALUES
              ('hotel_manager_test', 'hammamet', 4, DATE '2026-07-01', 3, 2,
               'BB', 'chambre', 'mer', '', 'double', 1350.0, 450.0, '2026-05-18T10:00:00', 480.0, 8)
        """))
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
        # A hotel present in features but NOT registered -> appears in /discoverable
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
        await conn.execute(text(
            "CREATE OR REPLACE VIEW hotel_features_full AS SELECT * FROM hotel_features"
        ))
    yield
    async with engine.begin() as conn:
        await conn.execute(text("DROP VIEW IF EXISTS hotel_features_full"))
        await conn.execute(text("DROP TABLE IF EXISTS hotel_features"))
        await conn.execute(text("DROP TABLE IF EXISTS segment_dim"))
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest_asyncio.fixture(loop_scope="function")
async def db_session():
    engine = create_async_engine(settings.test_db_url, echo=False, poolclass=NullPool)
    SessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with SessionLocal() as session:
        yield session
    await engine.dispose()

def _make_mock_ml_store() -> MLStore:
    # No spec= : MLStore's attributes are dataclass fields set in __init__,
    # so MagicMock(spec=MLStore) would not expose .recommender/.detector/etc.
    store = MagicMock()
    rec_df = pd.DataFrame({
        "hotel_name_normalized": ["hotel_manager_test"],
        "city_name":             ["hammamet"],
        "stars_int":             [4],
        "macro_region":          ["cap_bon"],
        "stars_band":            ["4-5"],
        "scraped_at":            ["2026-05-18T10:00:00"],
        "check_in":              [pd.Timestamp("2026-07-01").date()],
        "nights":                [3],
        "adults":                [2],
        "boarding_canonical":    ["BB"],
        "current_price_tnd":     [450.0],
        "q10_cal_tnd":           [300.0],
        "q50_tnd":               [500.0],
        "q90_cal_tnd":           [700.0],
        "interval_status":       ["in_band"],
        "direction":             ["hold"],
        "recommended_price_tnd": [450.0],
        "delta_pct_vs_current":  [0.0],
        "peer_medium_median":    [480.0],
        "peer_medium_count":     [8],
        "reasons":               [["price within calibrated band"]],
    })
    store.recommender.score.return_value = rec_df
    anom_df = pd.DataFrame({
        "q10_log": [5.7], "q50_log": [6.1], "q90_log": [6.5],
        "q10_cal_log": [5.5], "q90_cal_log": [6.7],
        "anomaly_score": [0.8], "is_anomaly": [True],
    })
    store.detector.score.return_value = anom_df
    store.forecaster.feature_names_ = ["nights", "stars_int"]
    return store

@pytest_asyncio.fixture()
async def client(db_session):
    async def override_get_db():
        yield db_session
    async def override_get_ml_store():
        return _make_mock_ml_store()
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_ml_store] = override_get_ml_store
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac
    app.dependency_overrides.clear()
