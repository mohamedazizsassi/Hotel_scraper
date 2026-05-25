-- =========================================================
-- 002_create_hotel_features_view.sql
-- =========================================================
-- Read-side VIEW: hotel_features ⋈ calendar_dim ⋈ segment_dim.
-- Also recomputes observed_delta_vs_peer_{g}_median_pct (excluded from
-- hotel_features to avoid duplication across 24M rows; derivable inline).
-- hotel_features is DROP+CREATE'd each pipeline run — the VIEW survives
-- because CREATE OR REPLACE re-resolves on the new table OID.

CREATE OR REPLACE VIEW hotel_features_full AS
SELECT
    hf.*,
    -- 13 calendar features from calendar_dim (joined on check_in)
    cd.check_in_dow,
    cd.check_in_month,
    cd.check_in_week_of_year,
    cd.check_in_day_of_month,
    cd.check_in_quarter,
    cd.is_weekend_checkin,
    cd.is_ramadan,
    cd.is_tunisia_public_holiday,
    cd.is_tunisia_school_holiday,
    cd.is_school_holiday_france,
    cd.is_school_holiday_germany,
    cd.is_school_holiday_uk,
    cd.days_to_nearest_european_holiday,
    -- 3 segment features from segment_dim (joined on city_name + stars_int)
    sd.macro_region,
    sd.stars_band,
    sd.market_segment_id,
    -- Leaky observed deltas (safe for anomaly/recommender; not for forecaster)
    CASE WHEN hf.peer_tight_median  IS NOT NULL AND hf.peer_tight_median  <> 0
         THEN (hf.price_per_night - hf.peer_tight_median)  / hf.peer_tight_median  * 100.0
         ELSE NULL END AS observed_delta_vs_peer_tight_median_pct,
    CASE WHEN hf.peer_medium_median IS NOT NULL AND hf.peer_medium_median <> 0
         THEN (hf.price_per_night - hf.peer_medium_median) / hf.peer_medium_median * 100.0
         ELSE NULL END AS observed_delta_vs_peer_medium_median_pct,
    CASE WHEN hf.peer_loose_median  IS NOT NULL AND hf.peer_loose_median  <> 0
         THEN (hf.price_per_night - hf.peer_loose_median)  / hf.peer_loose_median  * 100.0
         ELSE NULL END AS observed_delta_vs_peer_loose_median_pct
FROM hotel_features hf
LEFT JOIN calendar_dim cd ON hf.check_in = cd.check_in
LEFT JOIN segment_dim  sd ON hf.city_name = sd.city_name
                          AND hf.stars_int  = sd.stars_int;

INSERT INTO schema_migrations (version, description)
VALUES ('002', 'hotel_features_full: join view over hotel_features + calendar_dim + segment_dim')
ON CONFLICT (version) DO NOTHING;
