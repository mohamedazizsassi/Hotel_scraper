"""
Create the hotel_features_full VIEW: JOIN hotel_features + calendar_dim +
segment_dim, and recompute observed_delta_vs_peer_*_median_pct inline.

Why a VIEW (not a materialized view):
- hotel_features is rebuilt via DROP+SWAP every full reprocess. A VIEW
  re-resolves at query time, so it survives table swaps automatically.
- calendar_dim (213 rows) and segment_dim (50 rows) are tiny — the JOIN
  cost is negligible compared to the hotel_features scan.

observed_rank_in_peer_* is NOT recomputable from peer aggregates alone
(it's a percentile over raw peer prices, not derivable from median/p25/p75).
The notebooks that referenced it will need to be patched separately, OR
that column can be recomputed in pandas after pulling a sample.
"""
from sqlalchemy import create_engine, text

VIEW_SQL = """
CREATE OR REPLACE VIEW hotel_features_full AS
SELECT
    h.*,
    -- Calendar features (joined on check_in)
    c.check_in_dow,
    c.check_in_month,
    c.check_in_week_of_year,
    c.check_in_day_of_month,
    c.check_in_quarter,
    c.is_weekend_checkin,
    c.is_ramadan,
    c.is_tunisia_public_holiday,
    c.is_tunisia_school_holiday,
    c.is_school_holiday_france,
    c.is_school_holiday_germany,
    c.is_school_holiday_uk,
    c.days_to_nearest_european_holiday,
    -- Segment features (joined on city_name + stars_int)
    s.macro_region,
    s.stars_band,
    s.market_segment_id,
    -- Recomputed observed-deltas (derivable from price_per_night + peer_{g}_median)
    (h.price_per_night - h.peer_tight_median)
        / NULLIF(h.peer_tight_median, 0) * 100  AS observed_delta_vs_peer_tight_median_pct,
    (h.price_per_night - h.peer_medium_median)
        / NULLIF(h.peer_medium_median, 0) * 100 AS observed_delta_vs_peer_medium_median_pct,
    (h.price_per_night - h.peer_loose_median)
        / NULLIF(h.peer_loose_median, 0) * 100  AS observed_delta_vs_peer_loose_median_pct
FROM hotel_features h
LEFT JOIN calendar_dim c USING (check_in)
LEFT JOIN segment_dim  s USING (city_name, stars_int);
"""

engine = create_engine("postgresql://revway:CHANGE_ME@localhost:5432/revway")
with engine.begin() as conn:
    conn.execute(text(VIEW_SQL))
    print("VIEW hotel_features_full created.")

    # Verify: column count, total rows, a few key stats
    cols = conn.execute(
        text(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='hotel_features_full' "
            "ORDER BY ordinal_position"
        )
    ).fetchall()
    print(f"VIEW columns: {len(cols)}")

    # Quick row-count + JOIN-coverage sanity check
    sanity = conn.execute(
        text(
            """
            SELECT
                COUNT(*)                                         AS total_rows,
                COUNT(*) FILTER (WHERE is_ramadan IS NULL)       AS calendar_miss,
                COUNT(*) FILTER (WHERE market_segment_id IS NULL) AS segment_miss,
                MIN(check_in)::text                              AS min_check_in,
                MAX(check_in)::text                              AS max_check_in,
                COUNT(DISTINCT city_name)                        AS n_cities,
                COUNT(DISTINCT hotel_name_normalized)            AS n_hotels
            FROM hotel_features_full
            """
        )
    ).mappings().first()
    print("\nSanity check:")
    for k, v in sanity.items():
        print(f"  {k:24s} {v}")
