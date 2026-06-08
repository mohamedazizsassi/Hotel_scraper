-- =========================================================
-- 007_create_recommendation_decisions.sql
-- =========================================================
-- Persists a manager's Accept/Dismiss decision on a recommendation. Decisions
-- are decision-support tracking only (no PMS / price execution). Key matches the
-- fields the recommendation row exposes and the frontend's recommendation id
-- (check_in + nights + adults + boarding_canonical).

CREATE TABLE IF NOT EXISTS manager_recommendation_decisions (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    hotel_id INTEGER NOT NULL REFERENCES platform_hotels(id) ON DELETE CASCADE,
    check_in DATE NOT NULL,
    nights SMALLINT NOT NULL,
    adults SMALLINT NOT NULL,
    boarding_canonical TEXT NOT NULL,
    recommended_price_tnd NUMERIC,
    status TEXT NOT NULL CHECK (status IN ('accepted', 'dismissed')),
    decided_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (user_id, check_in, nights, adults, boarding_canonical)
);

CREATE INDEX IF NOT EXISTS idx_mrd_user_checkin
    ON manager_recommendation_decisions (user_id, check_in);

DROP TRIGGER IF EXISTS trg_mrd_updated_at ON manager_recommendation_decisions;
CREATE TRIGGER trg_mrd_updated_at
    BEFORE UPDATE ON manager_recommendation_decisions
    FOR EACH ROW EXECUTE FUNCTION set_updated_at();

INSERT INTO schema_migrations (version, description)
VALUES ('007', 'manager_recommendation_decisions: accept/dismiss tracking')
ON CONFLICT (version) DO NOTHING;
