"""
One-shot builder for 06_understand_data.ipynb.

Why this exists: generating a 30-cell notebook as JSON by hand is error-prone
(escaping, cell IDs, source-as-list-of-strings). Building it via nbformat
guarantees structural validity. After running this, the .ipynb file is the
artifact; this script can be deleted or kept for reproducibility.

Run from ml/ with:
    .\.venv\Scripts\python.exe notebooks\_build_06.py
"""
from __future__ import annotations

import uuid
from pathlib import Path

import nbformat


def md(src: str) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_markdown_cell(src.strip())
    cell["id"] = uuid.uuid4().hex[:12]
    return cell


def code(src: str) -> nbformat.NotebookNode:
    cell = nbformat.v4.new_code_cell(src.strip())
    cell["id"] = uuid.uuid4().hex[:12]
    return cell


CELLS: list[nbformat.NotebookNode] = []

# ---------------------------------------------------------------------------
# Title + setup
# ---------------------------------------------------------------------------

CELLS.append(md("""
# 06 — Understand Your Data

A guided tour of the freshly rebuilt feature table.

**Last rebuild:** 2026-05-19 (full reprocess, 29.4M rows, 28 scrape days, 539 hotels, 20 cities)

This notebook answers what you need to know before training:

- What's in the dataset? (scale, schema, coverage)
- Are the targets well-behaved? (price distribution, sur_demande rate)
- What's the peer-group structure? (sparsity, granularity selection)
- Are the calendar / segment dims aligned with the observation rows?
- Any leakage red flags?

**Source:** PostgreSQL VIEW `hotel_features_full` (JOINs hotel_features + calendar_dim + segment_dim; recomputes `observed_delta_vs_peer_*_median_pct` on the fly).
"""))

CELLS.append(code("""
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", 200)
warnings.filterwarnings("ignore", category=UserWarning)

ENGINE = create_engine(
    os.environ.get("POSTGRES_URI", "postgresql://revway:CHANGE_ME@localhost:5432/revway")
)

def q(sql: str, **params) -> pd.DataFrame:
    \"\"\"Run a SQL query against ENGINE and return a DataFrame.\"\"\"
    return pd.read_sql(text(sql), ENGINE, params=params)

print("connected:", ENGINE.url.database)
"""))

# ---------------------------------------------------------------------------
# § 1. Architecture
# ---------------------------------------------------------------------------

CELLS.append(md("""
## 1. The 3-table architecture

The pipeline writes three Postgres tables, not one:

| Table              | Rows         | Cols | Join key                 | Holds                                                                  |
|--------------------|--------------|------|--------------------------|------------------------------------------------------------------------|
| `hotel_features`   | 29,424,066   | 52   | —                        | observation rows: identifiers, targets, peer aggregates, demand        |
| `calendar_dim`     | 213          | 14   | `check_in`               | deterministic calendar features (Ramadan, EU/TN holidays, day-of-week) |
| `segment_dim`      | 50           | 5    | `(city_name, stars_int)` | `macro_region`, `stars_band`, `market_segment_id`                      |

This is the **offline / online feature-store pattern**:

- **Offline (Parquet)** keeps the full 77-col frame — training reads this for reproducibility.
- **Online (Postgres)** splits deterministic features into dim tables. Calendar (213 distinct check_in dates) and segment (50 distinct city×stars pairs) are not duplicated across 29M observation rows.

The VIEW `hotel_features_full` joins all three and recomputes the three `observed_delta_vs_peer_*_median_pct` columns (leaky for the forecaster — they include the row's own price — but safe for anomaly / recommender).
"""))

CELLS.append(code("""
# Row + column counts per table
rows = []
for table in ("hotel_features", "calendar_dim", "segment_dim", "hotel_features_full"):
    cnt = q(f'SELECT COUNT(*) AS n FROM "{table}"')["n"].iloc[0]
    ncols = q(
        "SELECT COUNT(*) AS n FROM information_schema.columns "
        "WHERE table_schema='public' AND table_name = :t",
        t=table,
    )["n"].iloc[0]
    rows.append({"table": table, "rows": cnt, "cols": ncols})
pd.DataFrame(rows)
"""))

# ---------------------------------------------------------------------------
# § 2. Loading patterns
# ---------------------------------------------------------------------------

CELLS.append(md("""
## 2. Loading patterns

Three idiomatic ways to load this dataset:

1. **Aggregates** — `GROUP BY` against `hotel_features_full`. Preferred for 29M-row analyses; Postgres is fastest at this.
2. **Sample for ad-hoc work** — `TABLESAMPLE BERNOULLI(N)` on `hotel_features`, then `LEFT JOIN` dims. Use `REPEATABLE (42)` for reproducibility.
3. **Full dim tables** — `calendar_dim` (213 rows) and `segment_dim` (50 rows) are tiny; pull them whole.

Never `SELECT * FROM hotel_features_full` without a `LIMIT` or sample — it's 29M × 71 cols.
"""))

CELLS.append(code("""
# Pull the small dim tables whole — useful as join references in pandas
calendar = q("SELECT * FROM calendar_dim ORDER BY check_in")
segment  = q("SELECT * FROM segment_dim ORDER BY market_segment_id, stars_int")

print(f"calendar_dim: {len(calendar)} rows × {calendar.shape[1]} cols")
print(f"segment_dim : {len(segment)} rows × {segment.shape[1]} cols")
calendar.head(3)
"""))

CELLS.append(code("""
segment.head(10)
"""))

CELLS.append(code("""
# Sample ~0.1% (~30K rows) for ad-hoc exploration
sample = q(\"\"\"
    SELECT *
    FROM hotel_features TABLESAMPLE BERNOULLI(0.1) REPEATABLE (42)
    LEFT JOIN calendar_dim USING (check_in)
    LEFT JOIN segment_dim  USING (city_name, stars_int)
\"\"\")
print(f"sample: {len(sample):,} rows × {sample.shape[1]} cols")
sample.dtypes.value_counts()
"""))

# ---------------------------------------------------------------------------
# § 3. Scale at a glance
# ---------------------------------------------------------------------------

CELLS.append(md("""
## 3. Scale at a glance

How many distinct things does this dataset see?
"""))

CELLS.append(code("""
overview = q(\"\"\"
    SELECT
        COUNT(*)                              AS observations,
        COUNT(DISTINCT hotel_name_normalized) AS hotels,
        COUNT(DISTINCT city_name)             AS cities,
        COUNT(DISTINCT scrape_date)           AS scrape_days,
        COUNT(DISTINCT check_in)              AS check_in_dates,
        MIN(scrape_date)::text                AS first_scrape,
        MAX(scrape_date)::text                AS last_scrape,
        MIN(check_in)::text                   AS first_check_in,
        MAX(check_in)::text                   AS last_check_in
    FROM hotel_features
\"\"\")
overview.T.rename(columns={0: "value"})
"""))

CELLS.append(code("""
by_source = q(\"\"\"
    SELECT
        source,
        COUNT(*)                              AS observations,
        COUNT(DISTINCT hotel_name_normalized) AS hotels,
        COUNT(DISTINCT city_name)             AS cities
    FROM hotel_features
    GROUP BY source
    ORDER BY source
\"\"\")
by_source
"""))

# ---------------------------------------------------------------------------
# § 4. Cross-source overlap
# ---------------------------------------------------------------------------

CELLS.append(md("""
## 4. Source representativeness — promohotel vs tunisiepromo

The two scrapers cover overlapping but not identical inventory. The overlap (hotels seen by both) is the only cross-source comparable set and matters for cross-source peer aggregates.
"""))

CELLS.append(code("""
overlap = q(\"\"\"
    WITH hotels_per_source AS (
        SELECT DISTINCT source, hotel_name_normalized
        FROM hotel_features
    ),
    pivot AS (
        SELECT
            hotel_name_normalized,
            BOOL_OR(source = 'promohotel')   AS in_promohotel,
            BOOL_OR(source = 'tunisiepromo') AS in_tunisiepromo
        FROM hotels_per_source
        GROUP BY hotel_name_normalized
    )
    SELECT
        SUM(CASE WHEN in_promohotel     AND in_tunisiepromo     THEN 1 ELSE 0 END) AS in_both,
        SUM(CASE WHEN in_promohotel     AND NOT in_tunisiepromo THEN 1 ELSE 0 END) AS promohotel_only,
        SUM(CASE WHEN NOT in_promohotel AND in_tunisiepromo     THEN 1 ELSE 0 END) AS tunisiepromo_only,
        COUNT(*)                                                                  AS total_distinct_hotels
    FROM pivot
\"\"\")
overlap.T.rename(columns={0: "hotels"})
"""))

# ---------------------------------------------------------------------------
# § 5. price_per_night distribution
# ---------------------------------------------------------------------------

CELLS.append(md("""
## 5. Targets — `price_per_night`

Primary forecaster target. Cleaners enforce `[30, 20000]` TND; outliers are already dropped.
"""))

CELLS.append(code("""
price_stats = q(\"\"\"
    SELECT
        COALESCE(source, 'ALL')                                                       AS source,
        COUNT(*)                                                                      AS n,
        ROUND(AVG(price_per_night)::numeric, 2)                                       AS mean,
        ROUND(STDDEV(price_per_night)::numeric, 2)                                    AS std,
        ROUND(MIN(price_per_night)::numeric, 2)                                       AS min,
        ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price_per_night)::numeric, 2) AS p25,
        ROUND(PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY price_per_night)::numeric, 2) AS median,
        ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price_per_night)::numeric, 2) AS p75,
        ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY price_per_night)::numeric, 2) AS p95,
        ROUND(PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY price_per_night)::numeric, 2) AS p99,
        ROUND(MAX(price_per_night)::numeric, 2)                                       AS max
    FROM hotel_features
    WHERE NOT sur_demande
    GROUP BY ROLLUP(source)
    ORDER BY source
\"\"\")
price_stats
"""))

CELLS.append(code("""
# Histogram from the sample, log-scaled because the price distribution is heavy-tailed
fig, ax = plt.subplots(figsize=(11, 4))
prices = sample.loc[~sample["sur_demande"].fillna(False), "price_per_night"].dropna()
ax.hist(np.log10(prices), bins=80, edgecolor="black", alpha=0.75)
ax.set_xlabel("log10(price_per_night) [TND]")
ax.set_ylabel("count")
ax.set_title(f"price_per_night — sample n={len(prices):,} (sur_demande excluded)")
ax.grid(alpha=0.3)
plt.show()
"""))

# ---------------------------------------------------------------------------
# § 6. sur_demande
# ---------------------------------------------------------------------------

CELLS.append(md("""
## 6. Targets — `sur_demande` prevalence

`sur_demande = True` means the offer is "on demand" — no firm price quote. It's our only demand-side signal (no bookings data). Tracks both as a forecaster filter (these rows have no price target) and as a demand proxy.
"""))

CELLS.append(code("""
sd_rate = q(\"\"\"
    SELECT
        COALESCE(source, 'ALL')               AS source,
        COUNT(*)                              AS n,
        ROUND(AVG(sur_demande::int)::numeric, 4) AS sur_demande_rate,
        COUNT(*) FILTER (WHERE sur_demande)   AS sur_demande_count
    FROM hotel_features
    GROUP BY ROLLUP(source)
    ORDER BY source
\"\"\")
sd_rate
"""))

CELLS.append(code("""
sd_window = q(\"\"\"
    SELECT
        CASE
            WHEN days_until_checkin <= 1  THEN 'D0-1'
            WHEN days_until_checkin <= 7  THEN 'D2-7'
            WHEN days_until_checkin <= 14 THEN 'D8-14'
            WHEN days_until_checkin <= 30 THEN 'D15-30'
            WHEN days_until_checkin <= 60 THEN 'D31-60'
            WHEN days_until_checkin <= 90 THEN 'D61-90'
            ELSE 'D90+'
        END                              AS booking_window_bucket,
        COUNT(*)                         AS n,
        ROUND(AVG(sur_demande::int)::numeric, 4) AS sur_demande_rate
    FROM hotel_features
    GROUP BY 1
    ORDER BY MIN(days_until_checkin)
\"\"\")
sd_window
"""))

# ---------------------------------------------------------------------------
# § 7. Booking window
# ---------------------------------------------------------------------------

CELLS.append(md("""
## 7. Booking window — `days_until_checkin`

CLAUDE.md notes price inversion in Tunisia: last-minute offers are sometimes *cheaper* than advance. `days_until_checkin` must be a non-linear feature (use bucketed groups + the raw integer; let the tree model fit the curve).
"""))

CELLS.append(code("""
window = q(\"\"\"
    SELECT
        days_until_checkin,
        COUNT(*)                                                                       AS n,
        ROUND(AVG(price_per_night)::numeric, 2)                                        AS mean_price,
        ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY price_per_night)::numeric, 2) AS median_price
    FROM hotel_features
    WHERE NOT sur_demande
      AND days_until_checkin BETWEEN 0 AND 180
    GROUP BY days_until_checkin
    ORDER BY days_until_checkin
\"\"\")
window.head()
"""))

CELLS.append(code("""
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax1.bar(window["days_until_checkin"], window["n"], width=1.0, color="steelblue")
ax1.set_ylabel("row count")
ax1.set_title("days_until_checkin distribution")
ax1.grid(alpha=0.3)

ax2.plot(window["days_until_checkin"], window["median_price"], label="median", color="darkgreen")
ax2.plot(window["days_until_checkin"], window["mean_price"],   label="mean",   color="darkorange", alpha=0.7)
ax2.set_xlabel("days_until_checkin")
ax2.set_ylabel("price_per_night [TND]")
ax2.set_title("median + mean price by booking window")
ax2.legend()
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.show()
"""))

# ---------------------------------------------------------------------------
# § 8. Peer sparsity
# ---------------------------------------------------------------------------

CELLS.append(md("""
## 8. Peer-group sparsity

For each granularity, what fraction of rows have **fewer than 5 peers**? Rows with `peer_count < 5` are statistically unreliable — the pipeline's `best_peer_granularity_used` falls back tight→medium→loose when the tighter level is too sparse.
"""))

CELLS.append(code("""
sparsity = q(\"\"\"
    SELECT
        ROUND((COUNT(*) FILTER (WHERE peer_tight_count  IS NULL OR peer_tight_count  < 5)::float / COUNT(*))::numeric, 4) AS tight_sparse_rate,
        ROUND((COUNT(*) FILTER (WHERE peer_medium_count IS NULL OR peer_medium_count < 5)::float / COUNT(*))::numeric, 4) AS medium_sparse_rate,
        ROUND((COUNT(*) FILTER (WHERE peer_loose_count  IS NULL OR peer_loose_count  < 5)::float / COUNT(*))::numeric, 4) AS loose_sparse_rate,
        ROUND(AVG(peer_tight_count) ::numeric, 1) AS tight_avg_peers,
        ROUND(AVG(peer_medium_count)::numeric, 1) AS medium_avg_peers,
        ROUND(AVG(peer_loose_count) ::numeric, 1) AS loose_avg_peers
    FROM hotel_features
\"\"\")
sparsity.T.rename(columns={0: "value"})
"""))

# ---------------------------------------------------------------------------
# § 9. Best-granularity
# ---------------------------------------------------------------------------

CELLS.append(md("""
## 9. Best-granularity distribution

`best_peer_granularity_used` is the tightest level meeting the ≥5-peers threshold per row. Fallback ladder: tight → medium → loose → NULL (no usable peers).
"""))

CELLS.append(code("""
best = q(\"\"\"
    SELECT
        COALESCE(best_peer_granularity_used, 'NONE') AS granularity,
        COUNT(*)                                     AS n,
        ROUND(COUNT(*)::numeric / SUM(COUNT(*)) OVER () * 100, 2) AS pct
    FROM hotel_features
    GROUP BY granularity
    ORDER BY n DESC
\"\"\")
best
"""))

# ---------------------------------------------------------------------------
# § 10. Calendar firing rates
# ---------------------------------------------------------------------------

CELLS.append(md("""
## 10. Calendar feature firing rates

Each `is_*` is a binary flag; here's the fraction of observation rows that hit each. Low rates aren't necessarily bad (Ramadan is just rare), but a near-zero rate means the model effectively can't learn from that feature.
"""))

CELLS.append(code("""
cal_rates = q(\"\"\"
    SELECT
        ROUND(AVG(is_weekend_checkin::int)       ::numeric, 4) AS weekend_checkin_rate,
        ROUND(AVG(is_ramadan::int)               ::numeric, 4) AS ramadan_rate,
        ROUND(AVG(is_tunisia_public_holiday::int)::numeric, 4) AS tn_public_holiday_rate,
        ROUND(AVG(is_tunisia_school_holiday::int)::numeric, 4) AS tn_school_holiday_rate,
        ROUND(AVG(is_school_holiday_france::int) ::numeric, 4) AS fr_school_holiday_rate,
        ROUND(AVG(is_school_holiday_germany::int)::numeric, 4) AS de_school_holiday_rate,
        ROUND(AVG(is_school_holiday_uk::int)     ::numeric, 4) AS uk_school_holiday_rate
    FROM hotel_features_full
\"\"\")
cal_rates.T.rename(columns={0: "fraction_of_rows"})
"""))

# ---------------------------------------------------------------------------
# § 11. Segment density
# ---------------------------------------------------------------------------

CELLS.append(md("""
## 11. Segment-level density

How many observation rows land in each `(macro_region, stars_band)` segment? Sparse segments are where peer aggregates and demand proxies will be noisy.
"""))

CELLS.append(code("""
seg_density = q(\"\"\"
    SELECT
        s.macro_region,
        s.stars_band,
        COUNT(DISTINCT h.hotel_name_normalized)  AS hotels,
        COUNT(*)                                  AS observations,
        ROUND(AVG(h.price_per_night)::numeric, 2) AS mean_price
    FROM hotel_features h
    JOIN segment_dim s USING (city_name, stars_int)
    GROUP BY s.macro_region, s.stars_band
    ORDER BY observations DESC
\"\"\")
seg_density
"""))

# ---------------------------------------------------------------------------
# § 12. Coverage matrix
# ---------------------------------------------------------------------------

CELLS.append(md("""
## 12. Column coverage — non-null rates

Same coverage check the pipeline validator runs. Helps spot features that may need imputation (or aren't reliable at all).
"""))

CELLS.append(code("""
cov = q(\"\"\"
    SELECT
        COUNT(*)                                                        AS n_sampled,
        ROUND(AVG((source                                       IS NOT NULL)::int)::numeric, 4) AS source,
        ROUND(AVG((boarding_canonical                           IS NOT NULL)::int)::numeric, 4) AS boarding_canonical,
        ROUND(AVG((room_base                                    IS NOT NULL)::int)::numeric, 4) AS room_base,
        ROUND(AVG((room_view                                    IS NOT NULL)::int)::numeric, 4) AS room_view,
        ROUND(AVG((room_tier                                    IS NOT NULL)::int)::numeric, 4) AS room_tier,
        ROUND(AVG((room_occupancy                               IS NOT NULL)::int)::numeric, 4) AS room_occupancy,
        ROUND(AVG((peer_tight_median                            IS NOT NULL)::int)::numeric, 4) AS peer_tight_median,
        ROUND(AVG((peer_medium_median                           IS NOT NULL)::int)::numeric, 4) AS peer_medium_median,
        ROUND(AVG((peer_loose_median                            IS NOT NULL)::int)::numeric, 4) AS peer_loose_median,
        ROUND(AVG((best_peer_granularity_used                   IS NOT NULL)::int)::numeric, 4) AS best_peer_granularity_used,
        ROUND(AVG((sur_demande_rate_city_stars_boarding_checkin IS NOT NULL)::int)::numeric, 4) AS sur_demande_rate_csb
    FROM hotel_features TABLESAMPLE BERNOULLI(1) REPEATABLE (42)
\"\"\")
cov.T.rename(columns={0: "non_null_rate"})
"""))

# ---------------------------------------------------------------------------
# § 13. Leakage gate
# ---------------------------------------------------------------------------

CELLS.append(md("""
## 13. Leakage spot-check

The pipeline's validator already ran a full leakage check (`leakage_mismatches = 0` in the final report). Here we verify the feature-manifest API enforces it: trying to slip an `observed_*` column into the forecaster set must raise.
"""))

CELLS.append(code("""
# Import the feature manifest from the pipeline package
NB_DIR = os.path.dirname(os.path.abspath("__file__")) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.abspath(os.path.join(NB_DIR, "..")))

from feature_engineering.model_feature_sets import (
    FORECASTER_FEATURES,
    LEAKY_OBSERVED_FEATURES,
    TARGETS,
    assert_no_leakage,
)

print(f"FORECASTER_FEATURES : {len(FORECASTER_FEATURES)} columns")
print(f"  first 5            : {FORECASTER_FEATURES[:5]}")
print(f"LEAKY_OBSERVED      : {LEAKY_OBSERVED_FEATURES}")
print(f"TARGETS              : {TARGETS}")

try:
    assert_no_leakage(["observed_delta_vs_peer_tight_median_pct"], "forecaster")
    print("\\n[!] gate did NOT fire — investigate")
except ValueError as e:
    print(f"\\n[ok] gate fired: {e}")
"""))

# ---------------------------------------------------------------------------
# § 14. Next
# ---------------------------------------------------------------------------

CELLS.append(md("""
## 14. What now?

You now know the dataset's shape. Next moves toward modeling:

1. **Baselines first.** From `ml/CLAUDE.md`: no trained model ships without beating (a) global median, (b) segment median, (c) competitor median, (d) linear hedonic OLS on `log(price)`. Build these in `ml/models/forecasting/baseline.py`.

2. **Hold-out split — *hotel-wise*, not row-wise.** Same hotel must not appear in both train and validation; otherwise peer-aggregate leakage will inflate scores.

3. **Filter `sur_demande = True` rows out of the forecaster target** — those have no firm price. Keep them for the anomaly model.

4. **Features the forecaster gets**: `FORECASTER_FEATURES` from `model_feature_sets`. No `observed_*`, no `TARGETS`.

5. **Don't lean on `peer_tight_*` alone** — see §8 (sparsity is high). Use `best_peer_granularity_used` as a categorical AND keep all 3 granularity blocks; let the tree-based model pick.
"""))

# ---------------------------------------------------------------------------
# Build + write
# ---------------------------------------------------------------------------

nb = nbformat.v4.new_notebook()
nb["cells"] = CELLS
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (.venv)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.11",
    },
}

OUT = Path(__file__).parent / "06_understand_data.ipynb"
nbformat.write(nb, OUT)
print(f"wrote {OUT}  ({len(CELLS)} cells)")
