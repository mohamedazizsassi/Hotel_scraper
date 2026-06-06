from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    db_url: str = "postgresql+asyncpg://localhost/revway"
    jwt_secret: str = "dev_secret_change_in_production"
    jwt_expire_minutes: int = 30
    model_dir: Path = Path("../ml/models/forecasting/lgbm_quantile_2026-05-23/hotel_wise")
    cors_origins: str = "http://localhost:4200"
    test_db_url: str = ""
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "hotel_scraper"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

settings = Settings()
