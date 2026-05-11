# EDA Notebooks — Design Spec

**Date:** 2026-05-11  
**Status:** Approved  
**Context:** RevWay / ml/ — feature engineering complete (1.74M rows, 73 columns, validation passed). EDA notebooks serve two equal goals: figures for the PFE report chapters and modeling conclusions that feed the model spec.

---

## Data source

All notebooks read from PostgreSQL `hotel_features` table via `POSTGRES_URI` from `ml/.env`.  
Two Parquet snapshots also exist in `ml/artifacts/` as fallback.

**Sampling strategy:** SQL `GROUP BY` aggregations are preferred over full-table pandas loads. Row-level analyses use `TABLESAMPLE BERNOULLI(30)` (~500K rows). Notebook 05 uses the full table for mutual information scoring and exposes a `FULL_TABLE` flag.

---

## Shared bootstrap (all notebooks)

```python
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sqlalchemy import create_engine
import sys; sys.path.insert(0, "..")
from feature_engineering.config import POSTGRES_URI

engine = create_engine(POSTGRES_URI)
SAMPLE_N = 500_000  # approximate; BERNOULLI(30) gives ~30% of table
df = pd.read_sql("SELECT * FROM hotel_features TABLESAMPLE BERNOULLI(30)", engine)
```

Figure style: `seaborn-v0_8-whitegrid`, French-language axis labels and titles (PFE report language), saved to `ml/notebooks/figures/{notebook_stem}/` as 300 dpi PNG.

---

## Notebook 02 — `02_market_structure.ipynb`

**PFE chapter:** Analyse du marché  
**Question answered:** What does the hotel dataset actually cover, and where are the gaps?

### Sections

1. **Hotel & city landscape**  
   Bar charts: hotels per city, rows per city, star-rating distribution overall and by city.  
   Table: top-20 hotels by row count.

2. **Source balance**  
   Promohotel vs tunisiepromo: share of rows, share of distinct hotels, share per city.  
   Stacked bar by city showing source mix.

3. **Cross-source hotel matching**  
   Venn diagram: hotels in promohotel only / tunisiepromo only / both.  
   The 66/490 overlap is the join key for cross-source peer features — flag the 424 single-source hotels as modeling risk (their peer groups are source-homogeneous).

4. **Room taxonomy coverage**  
   Bar chart of non-null rates: room_base (95.5%), room_view (32.4%), room_tier (37.0%), room_occupancy (49.6%).  
   Top-20 unmatched room_name strings (for taxonomy gap triage).

5. **Data density heatmap**  
   SQL: `GROUP BY city_name, check_in_month` → row count.  
   Heatmap: city × month. Identifies sparse (city, season) cells that will produce unreliable peer aggregates.

### Conclusions cell

- Which cities have sufficient data for reliable modeling.
- Cross-source gap: single-source hotels' peer aggregates are source-homogeneous — model should be aware.
- room_view/tier/occupancy sparsity is expected (not all room names encode this); tight peer group falls back to medium for these rows (86.8% tight coverage confirmed).

---

## Notebook 03 — `03_price_dynamics.ipynb`

**PFE chapter:** Structure des prix  
**Question answered:** How are prices distributed, what drives price variation, and what does the quantile spread say about model output choice?

### Sections

1. **Overall price distribution**  
   Histogram of `price_per_night` (log scale). Summary stats table. IQR, median, P5/P95.

2. **Price by segment**  
   Box plots: price_per_night by (a) city, (b) stars_int, (c) boarding_canonical.  
   Grouped median table: (city × stars × boarding) — the primary modelling slice.

3. **Multi-night discount structure**  
   Median price_per_night by `nights` (1, 2, 3, 5, 7) per boarding.  
   Validates that `nights` must be a feature, not a neutral normalizer.

4. **Booking-window curves**  
   Line plot: median price_per_night vs `days_until_checkin` (grouped into the 7 buckets from config).  
   Faceted by boarding_canonical (BB / HDP / AI / AI_ULTRA).  
   Validates the non-linear feature design and price-inversion hypothesis.

5. **Quantile spread (P10 / P50 / P90)**  
   For each (city, stars, boarding) cell: P10, P50, P90 of price_per_night.  
   Spread ratio `P90/P10` distribution — justifies probabilistic output (D2) if spread is wide.

6. **Extreme-delta outlier anatomy**  
   The ~12K rows with `|delta_vs_peer_medium_median_pct| > 500%`.  
   Profile: what cities, hotels, boardings, booking windows do they come from?  
   Distinguish genuine outliers from data artefacts.

### Conclusions cell

- Booking-window nonlinearity: confirmed/quantified.
- Price inversion: present/absent (last-minute cheaper than advance?).
- Quantile spread (P90/P10 ratio): mean ≈ X → justifies probabilistic model (D2).
- Multi-night discount: estimated per-night discount for 7-night vs 1-night.
- Outlier anatomy: artefact vs market signal, implications for anomaly detection (D3).

---

## Notebook 04 — `04_temporal_demand.ipynb`

**PFE chapter:** Dynamiques temporelles et demande  
**Question answered:** Which calendar features carry real price signal, and is `sur_demande` a reliable demand proxy?

### Sections

1. **Monthly and seasonal price patterns**  
   Line plot: median price_per_night by check_in_month.  
   Bar: median by check_in_quarter. Seasonal high/low quantified.

2. **Ramadan effect**  
   Box plot: price_per_night inside vs outside Ramadan period, per city.  
   Expected: lower occupancy-driven prices during Ramadan for leisure travel.

3. **EU holiday correlation**  
   For each of France / Germany / UK school holiday flags:  
   Median price when flag=True vs flag=False, by city.  
   Bar chart showing effect size (TND/night premium or discount).

4. **Day-of-week departure patterns**  
   Bar chart: row count by check_in_dow (0=Mon … 6=Sun) per city and per source.  
   Median price by check_in_dow — goes beyond the boolean weekend flag in notebook 01 to show whether mid-week or specific days carry price structure.

5. **`sur_demande` semantic audit** *(open risk from ml/CLAUDE.md)*  
   - Overall rate: what fraction of rows have sur_demande=True?  
   - Rate by booking window: does sur_demande spike close to check-in (last-minute unavailability) or far out (pre-sold)?  
   - Rate by city and stars_int.  
   - Price when sur_demande=True vs False: does True mean "expensive" or "unavailable/0-price"?  
   - Hotels always/never sur_demande: flag degenerate hotels.  
   - Correlation: sur_demande_rate_city_checkin vs median price in the slice.

### Conclusions cell

- Calendar flags with measurable price effect (ranked by effect size).
- Weekend effect verdict: near-zero confirmed / refuted.
- Ramadan effect direction and magnitude.
- sur_demande verdict: reliable demand proxy / noise. Specific semantic interpretation.
- Recommendation: which calendar and demand features to include/exclude in model.

---

## Notebook 05 — `05_feature_readiness.ipynb`

**PFE chapter:** Préparation des données pour la modélisation  
**Question answered:** Which features have signal, which are redundant, and what do the answers imply for model choice?

### Sections

1. **Correlation with price_per_night**  
   Pearson and Spearman correlation for all numeric features vs `price_per_night`.  
   Horizontal bar chart, ranked by |Spearman r|. Highlight top 10 and bottom 10.

2. **Mutual information scores**  
   `sklearn.feature_selection.mutual_info_regression` on numeric features vs `price_per_night`.  
   For boolean/categorical features: binarize `price_per_night` into quartile labels via `pd.qcut(..., q=4, labels=False)` then apply `mutual_info_classif`.  
   Ranked bar chart combining both. Uses full table (`FULL_TABLE = True`); NaN in peer features imputed to column median for MI scoring only.

3. **Multicollinearity check (VIF)**  
   Variance Inflation Factor for the competitive feature block (peer_tight/medium/loose medians).  
   Flag any VIF > 10 as a collinearity risk.

4. **Feature group summary table**

   | Group | Features | Avg |Spearman r| | Coverage | Recommendation |
   |---|---|---|---|---|
   | Competitive (medium) | peer_medium_median, delta_vs_peer_medium_median_pct, rank_in_peer_medium | ? | 95.6% | Keep |
   | Booking window | days_until_checkin, booking_window_bucket | ? | 100% | Keep (non-linear) |
   | Taxonomy | boarding_canonical, stars_int, room_base | ? | 100% / 95.5% | Keep |
   | Calendar | check_in_month, is_ramadan, EU holiday flags | ? | 100% | Keep top-N |
   | Demand proxy | sur_demande_rate_city_checkin | ? | 99.96% | Conditional on audit |
   | Room detail | room_view, room_tier, room_occupancy | ? | 32–50% | Keep for tight group only |

   (Values filled from computed results.)

5. **Model-readiness conclusions and open decisions**  
   Structured answer to:
   - **D1** (model type): does the feature profile (mixed types, nonlinearity, sparse categoricals) favour CatBoost or LightGBM?
   - **D2** (probabilistic): P90/P10 spread — does it justify quantile-loss output?
   - **D3** (anomaly): do outlier patterns suggest quantile-interval or Isolation Forest?

### Conclusions cell

Final feature list recommended for the model spec, grouped by keep / transform / drop, with one-line justification per group.

---

## File layout

```
ml/notebooks/
├── 01_data_quality.ipynb          ← existing
├── 02_market_structure.ipynb
├── 03_price_dynamics.ipynb
├── 04_temporal_demand.ipynb
├── 05_feature_readiness.ipynb
└── figures/
    ├── 02_market_structure/
    ├── 03_price_dynamics/
    ├── 04_temporal_demand/
    └── 05_feature_readiness/
```

Figures are saved as `{notebook_stem}/{section_slug}.png` at 300 dpi. All figure-save calls go through a shared helper `save_fig(name)` defined in the bootstrap cell.

---

## Constraints

- No external API calls. All data from PostgreSQL `hotel_features`.
- French-language titles and axis labels (PFE report language).
- Each notebook must be runnable top-to-bottom in a fresh kernel with only `ml/.env` set.
- No cell should take > 2 minutes (use SQL aggregations or TABLESAMPLE where needed).
- Each notebook ends with a "Conclusions pour la modélisation" markdown cell summarizing findings and implications for the model spec.
- Do not reproduce analyses already in `01_data_quality.ipynb` (coverage, basic price histogram, booking-window overview, boolean weekend effect, temporal coverage). Deeper cuts on the same topics (e.g. per-DOW price structure) are fine.
