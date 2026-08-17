-- =========================================================
-- seeds/02_seed_admin.sql
-- =========================================================
-- Development seed data: Admin user for local testing.
-- **DEVELOPMENT ONLY** — Do not use in production.
--
-- Creates one admin user with:
--   email: admin@revway.tn
--   password: not stored in this file — see your local notes / secrets
--     manager for the dev password, or generate a new one and re-hash below
--   full_name: Development Admin
--
-- NOTE: '@revway.local' was the original placeholder domain here, but the
-- '.local' TLD is rejected by pydantic's EmailStr as a "special-use or
-- reserved name" — that admin could never actually log in through POST
-- /auth/login. Switched to '.tn' to match the working manager seed
-- (manager@revway.tn in 03_seed_manager.sql), which Pydantic accepts.
--
-- To generate a bcrypt hash locally, use:
--   python -c "import bcrypt; print(bcrypt.hashpw(b'your_password', bcrypt.gensalt()).decode())"
--
-- For production, use a CLI tool to bootstrap the first admin user with a strong password.

INSERT INTO users (email, password_hash, full_name, role, is_active) VALUES
    (
        'admin@revway.tn',
        -- bcrypt hash of a rotated dev-only password (plaintext not stored in this repo)
        '$2b$12$W5c1j8TzmPlDClVOhHHkku.YTmDYWefNVYv.N2wj3k9yufWWFZ0KC',
        'Development Admin',
        'admin',
        TRUE
    )
ON CONFLICT (email) DO NOTHING;
