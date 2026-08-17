-- =========================================================
-- seeds/03_seed_manager.sql
-- =========================================================
-- Development seed data: one Manager user for local testing.
-- **DEVELOPMENT ONLY** — Do not use in production.
--
-- Creates a manager assigned to a real hotel that has data in hotel_features,
-- so the /manager/* endpoints return live forecaster/recommender output:
--   email:    manager@revway.tn
--   password: not stored in this file — see your local notes / secrets
--             manager for the dev password, or generate a new one below
--   hotel:    iberostar averroes (Hammamet, 4*)
--   competitors: el mouradi hammamet, bel azur thalasso bungalows,
--                aziza beach golf spa adults only 16 ans, concorde marco polo
--
-- To regenerate the bcrypt hash for a new password:
--   python -c "import bcrypt; print(bcrypt.hashpw(b'your_password', bcrypt.gensalt()).decode())"
--
-- Idempotent: ON CONFLICT guards make this safe to re-run.

BEGIN;

INSERT INTO cities (name_normalized, name_display)
VALUES ('hammamet', 'Hammamet')
ON CONFLICT (name_normalized) DO NOTHING;

INSERT INTO platform_hotels (hotel_name_normalized, hotel_name_display, city_id, stars_int)
SELECT v.norm, v.disp, c.id, 4
FROM (VALUES
    ('iberostar averroes',                      'Iberostar Averroes'),
    ('el mouradi hammamet',                     'El Mouradi Hammamet'),
    ('bel azur thalasso bungalows',             'Bel Azur Thalasso & Bungalows'),
    ('aziza beach golf spa adults only 16 ans', 'Aziza Beach Golf & Spa (Adults Only)'),
    ('concorde marco polo',                     'Concorde Marco Polo')
) AS v(norm, disp)
CROSS JOIN cities c
WHERE c.name_normalized = 'hammamet'
ON CONFLICT (hotel_name_normalized, city_id) DO NOTHING;

INSERT INTO users (email, password_hash, full_name, role)
VALUES (
    'manager@revway.tn',
    '$2b$12$dvz4G361AKFYL57PYGpxiOka6kt42WostwdLA6yeQ8KSEPD/KPDVW',  -- bcrypt hash of a rotated dev-only password
    'Iberostar Averroes Manager',
    'manager'
)
ON CONFLICT (email) DO NOTHING;

-- Assignment: manager -> iberostar averroes
INSERT INTO user_hotel_assignments (user_id, hotel_id, max_competitors)
SELECT u.id, ph.id, 4
FROM users u, platform_hotels ph JOIN cities c ON c.id = ph.city_id
WHERE u.email = 'manager@revway.tn'
  AND c.name_normalized = 'hammamet'
  AND ph.hotel_name_normalized = 'iberostar averroes'
ON CONFLICT (user_id) DO NOTHING;

-- 4 competitor selections
INSERT INTO user_competitor_selections (user_id, hotel_id, display_order)
SELECT u.id, ph.id, v.ord
FROM users u
CROSS JOIN (VALUES
    ('el mouradi hammamet', 1),
    ('bel azur thalasso bungalows', 2),
    ('aziza beach golf spa adults only 16 ans', 3),
    ('concorde marco polo', 4)
) AS v(norm, ord)
JOIN platform_hotels ph ON ph.hotel_name_normalized = v.norm
JOIN cities c ON c.id = ph.city_id AND c.name_normalized = 'hammamet'
WHERE u.email = 'manager@revway.tn'
ON CONFLICT (user_id, hotel_id) DO NOTHING;

COMMIT;

\echo '--- Seeded manager ---'
SELECT u.email, u.role, ph.hotel_name_normalized AS assigned_hotel, a.max_competitors
FROM users u
JOIN user_hotel_assignments a ON a.user_id = u.id
JOIN platform_hotels ph ON ph.id = a.hotel_id
WHERE u.email = 'manager@revway.tn';
\echo '--- Competitors ---'
SELECT s.display_order, ph.hotel_name_normalized
FROM user_competitor_selections s
JOIN users u ON u.id = s.user_id
JOIN platform_hotels ph ON ph.id = s.hotel_id
WHERE u.email = 'manager@revway.tn'
ORDER BY s.display_order;
