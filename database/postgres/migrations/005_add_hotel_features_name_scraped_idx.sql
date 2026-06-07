-- =========================================================
-- 005_add_hotel_features_name_scraped_idx.sql
-- =========================================================
-- Lets the admin hotels list (services/admin_hotels.py) find each hotel's
-- latest_scraped_at via a per-hotel index seek instead of a 29M-row scan.
-- CONCURRENTLY: builds without locking writers; cannot run inside a
-- transaction block, so this file must be applied with autocommit on
-- (plain `psql -f`, not wrapped in BEGIN/COMMIT).

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_hf_name_scraped_at
    ON hotel_features (hotel_name_normalized, scraped_at DESC);

INSERT INTO schema_migrations (version, description)
VALUES ('005', 'hotel_features: (hotel_name_normalized, scraped_at DESC) index for admin hotel list latest_scraped_at lookup');
