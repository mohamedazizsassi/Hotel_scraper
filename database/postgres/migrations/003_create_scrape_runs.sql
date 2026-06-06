-- =========================================================
-- 003_create_scrape_runs.sql
-- =========================================================
-- Per-run scrape statistics parsed read-only from scraper/logs/*.log.
-- One row per log file (= one scheduled run). Populated by
-- backend/scripts/load_scrape_runs.py. The scraper is NOT modified.

CREATE TABLE scrape_runs (
    id            INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_ts        TIMESTAMPTZ NOT NULL,        -- parsed from the filename
    log_filename  TEXT NOT NULL UNIQUE,        -- idempotency key
    source        TEXT,                         -- promohotel / tunisiepromo / mixed
    spiders_count INTEGER NOT NULL DEFAULT 0,
    items_total   INTEGER NOT NULL DEFAULT 0,   -- = rows inserted into hotel_prices
    errors_total  INTEGER NOT NULL DEFAULT 0,
    duration_s    NUMERIC,
    status        TEXT NOT NULL,                -- finished / partial / failed
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_scrape_runs_ts ON scrape_runs (run_ts);

INSERT INTO schema_migrations (version, description)
VALUES ('003', 'scrape_runs: per-run scrape stats parsed from logs');
