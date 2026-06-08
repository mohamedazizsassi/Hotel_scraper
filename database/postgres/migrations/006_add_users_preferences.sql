-- =========================================================
-- 006_add_users_preferences.sql
-- =========================================================
-- Per-user UI preferences (language + alert toggles) for the manager Settings
-- page. Stored as JSONB so Settings is functional without a per-setting table.
-- Honest scope: preferences only — nothing consumes them to send notifications.

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS preferences JSONB NOT NULL DEFAULT '{}'::jsonb;

INSERT INTO schema_migrations (version, description)
VALUES ('006', 'users.preferences JSONB: manager language + alert toggles')
ON CONFLICT (version) DO NOTHING;
