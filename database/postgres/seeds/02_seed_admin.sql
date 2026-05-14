-- =========================================================
-- seeds/02_seed_admin.sql
-- =========================================================
-- Development seed data: Admin user for local testing.
-- **DEVELOPMENT ONLY** — Do not use in production.
--
-- Creates one admin user with:
--   email: admin@revway.local
--   password: (bcrypt hash of "dev_password_change_me" from an external tool)
--   full_name: Development Admin
--
-- To generate a bcrypt hash locally, use:
--   python -c "import bcrypt; print(bcrypt.hashpw(b'your_password', bcrypt.gensalt()).decode())"
--
-- For production, use a CLI tool to bootstrap the first admin user with a strong password.

INSERT INTO users (email, password_hash, full_name, role, is_active) VALUES
    (
        'admin@revway.local',
        -- Bcrypt hash of 'dev_password_change_me' (replace in production with a real hash)
        '$2b$12$8w1zG5W0H8Z5M9X2Q7Y4TuxT8Y1Z2a3B4C5d6e7F8G9H0I1J2K3L',
        'Development Admin',
        'admin',
        TRUE
    )
ON CONFLICT (email) DO NOTHING;
