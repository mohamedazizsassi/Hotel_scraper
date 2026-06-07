-- =========================================================
-- 004_add_hotel_contact_columns.sql
-- =========================================================
-- Admin-editable contact info on platform_hotels (not scraped).

ALTER TABLE platform_hotels
    ADD COLUMN contact_email TEXT,
    ADD COLUMN contact_phone TEXT;

INSERT INTO schema_migrations (version, description)
VALUES ('004', 'platform_hotels: contact_email, contact_phone');
