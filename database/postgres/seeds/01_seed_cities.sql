-- =========================================================
-- seeds/01_seed_cities.sql
-- =========================================================
-- Development seed data: Tunisian cities reference table.
-- Run after 001_create_platform_tables.sql.
--
-- Cities extracted from scraper EDA. These are the canonical normalized
-- and display names; all hotel records must reference one of these cities
-- via platform_hotels.city_id.

INSERT INTO cities (name_normalized, name_display, country, is_active) VALUES
    ('tunis', 'Tunis', 'TN', TRUE),
    ('hammamet', 'Hammamet', 'TN', TRUE),
    ('sousse', 'Sousse', 'TN', TRUE),
    ('sfax', 'Sfax', 'TN', TRUE),
    ('kairouan', 'Kairouan', 'TN', TRUE),
    ('gafsa', 'Gafsa', 'TN', TRUE),
    ('medenine', 'Médenine', 'TN', TRUE),
    ('tozeur', 'Tozeur', 'TN', TRUE),
    ('monastir', 'Monastir', 'TN', TRUE),
    ('nabeul', 'Nabeul', 'TN', TRUE),
    ('bizerte', 'Bizerte', 'TN', TRUE),
    ('douz', 'Douz', 'TN', TRUE),
    ('djerba', 'Djerba', 'TN', TRUE),
    ('kerkennah', 'Kerkennah', 'TN', TRUE),
    ('tabarka', 'Tabarka', 'TN', TRUE)
ON CONFLICT (name_normalized) DO NOTHING;