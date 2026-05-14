-- =========================================================
-- 001_create_platform_tables.sql
-- =========================================================
-- Platform schema for RevWay SaaS: users, hotels, assignments, selections.
-- Includes schema_migrations tracking, reference tables, and enforcing triggers.

-- Required extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS citext;     -- case-insensitive email

-- Migration tracking
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    description TEXT
);

-- ---------------------------------------------------------
-- 1. cities  (small reference dimension)
-- ---------------------------------------------------------
CREATE TABLE cities (
    id SMALLINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name_normalized TEXT NOT NULL UNIQUE,
    name_display TEXT NOT NULL,
    country CHAR(2) NOT NULL DEFAULT 'TN',
    is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- ---------------------------------------------------------
-- 2. users
-- ---------------------------------------------------------
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email CITEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    full_name TEXT,
    role TEXT NOT NULL CHECK (role IN ('admin', 'manager')),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    last_login_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------
-- 3. platform_hotels
-- ---------------------------------------------------------
CREATE TABLE platform_hotels (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    hotel_name_normalized TEXT NOT NULL,
    hotel_name_display TEXT NOT NULL,
    city_id SMALLINT NOT NULL REFERENCES cities(id) ON DELETE RESTRICT,
    stars_int SMALLINT CHECK (stars_int BETWEEN 1 AND 5),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (hotel_name_normalized, city_id)
);

CREATE INDEX idx_platform_hotels_active_filter
    ON platform_hotels (city_id, stars_int)
    WHERE is_active = TRUE;

-- ---------------------------------------------------------
-- 4. platform_hotel_sources  (which sources a hotel is in)
-- ---------------------------------------------------------
CREATE TABLE platform_hotel_sources (
    platform_hotel_id INTEGER NOT NULL
        REFERENCES platform_hotels(id) ON DELETE CASCADE,
    source TEXT NOT NULL
        CHECK (source IN ('promohotel', 'tunisiepromo')),
    source_hotel_name TEXT NOT NULL,
    source_city_id INTEGER,
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (platform_hotel_id, source)
);

CREATE INDEX idx_phs_source ON platform_hotel_sources(source);

-- ---------------------------------------------------------
-- 5. user_hotel_assignments  (manager-to-hotel 1:1)
-- ---------------------------------------------------------
CREATE TABLE user_hotel_assignments (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    hotel_id INTEGER NOT NULL REFERENCES platform_hotels(id) ON DELETE RESTRICT,
    max_competitors SMALLINT NOT NULL DEFAULT 4
        CHECK (max_competitors BETWEEN 1 AND 10),
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id)
);

CREATE INDEX idx_uha_hotel ON user_hotel_assignments(hotel_id);

-- ---------------------------------------------------------
-- 6. user_competitor_selections
-- ---------------------------------------------------------
CREATE TABLE user_competitor_selections (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    hotel_id INTEGER NOT NULL REFERENCES platform_hotels(id) ON DELETE RESTRICT,
    display_order SMALLINT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, hotel_id),
    UNIQUE (user_id, display_order)
);

CREATE INDEX idx_ucs_hotel ON user_competitor_selections(hotel_id);

-- ---------------------------------------------------------
-- 7. updated_at trigger (applied uniformly)
-- ---------------------------------------------------------
CREATE OR REPLACE FUNCTION set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_users_updated_at
    BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_hotels_updated_at
    BEFORE UPDATE ON platform_hotels FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_assignments_updated_at
    BEFORE UPDATE ON user_hotel_assignments FOR EACH ROW EXECUTE FUNCTION set_updated_at();

CREATE TRIGGER trg_selections_updated_at
    BEFORE UPDATE ON user_competitor_selections FOR EACH ROW EXECUTE FUNCTION set_updated_at();

-- ---------------------------------------------------------
-- 8. Domain-rule triggers
-- ---------------------------------------------------------

-- Manager cannot pick own hotel as competitor
CREATE OR REPLACE FUNCTION check_competitor_not_self()
RETURNS TRIGGER AS $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM user_hotel_assignments
        WHERE user_id = NEW.user_id AND hotel_id = NEW.hotel_id
    ) THEN
        RAISE EXCEPTION
            'Manager % cannot select their own hotel (id=%) as a competitor',
            NEW.user_id, NEW.hotel_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_competitor_not_self
    BEFORE INSERT OR UPDATE ON user_competitor_selections
    FOR EACH ROW EXECUTE FUNCTION check_competitor_not_self();

-- Enforce max_competitors cap from assignments
CREATE OR REPLACE FUNCTION check_competitor_count()
RETURNS TRIGGER AS $$
DECLARE
    current_count INTEGER;
    cap INTEGER;
BEGIN
    SELECT COUNT(*) INTO current_count
        FROM user_competitor_selections WHERE user_id = NEW.user_id;
    SELECT max_competitors INTO cap
        FROM user_hotel_assignments WHERE user_id = NEW.user_id;
    IF cap IS NULL THEN
        RAISE EXCEPTION
            'User % has no active assignment; cannot select competitors',
            NEW.user_id;
    END IF;
    IF current_count >= cap THEN
        RAISE EXCEPTION
            'User % has reached competitor limit (%); cannot add more',
            NEW.user_id, cap;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_competitor_count
    BEFORE INSERT ON user_competitor_selections
    FOR EACH ROW EXECUTE FUNCTION check_competitor_count();

-- ---------------------------------------------------------
-- 9. Mark migration as applied
-- ---------------------------------------------------------
INSERT INTO schema_migrations (version, description)
VALUES ('001', 'platform tables: users, hotels, sources, assignments, selections');
