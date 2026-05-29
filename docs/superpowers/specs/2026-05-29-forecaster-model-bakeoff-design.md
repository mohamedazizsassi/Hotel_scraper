# Forecaster Model Bake-Off + Calibration Showcase — Design

**Date:** 2026-05-29
**Scope:** A rigorous, reproducible comparison of three gradient-boosting quantile forecasters on the full feature snapshot, with full hyperparameter optimization, conformal-calibration deep-dive, and ablations. Methods-showcase / portfolio framing for the PFE defense.
**Defense deadline:** ~2026-06-15 (~17 days)
**Compute:** Kaggle Notebooks, free GPU tier (T4×2 or P100 16GB VRAM, ~29GB RAM high-RAM, ≤12h/session, 30h GPU/week).
**Stack:** LightGBM · CatBoost · XGBoost (≥2.0) · Optuna · scikit-learn · pandas · pyarrow · matplotlib

---

## 1. Goal & Non-Goals

### Goal
Produce a defensible, side-by-side evaluation of the three dominant tabular learners as **quantile price forecasters** for the RevWay market-pricing problem, under one identical leakage-safe protocol, then:
1. Pick a winner on the **hotel-wise** hold-out (the deployment-realistic split).
2. Improve interval quality via a **per-segment (Mondrian) conformal** calibration on top of the global CQR already in place.
3. Tell a complete methods story with ablations (data-scaling, feature-group, tail-hyperparameter, quantile-crossing).

### Non-Goals (explicit)
- **Not** rebuilding the anomaly detector or recommender. They consume calibrated quantiles; they inherit the winning model by re-running their existing scripts. No rule changes.
- **Not** building temporal/sequence features (D5 — out of scope for the defense window).
- **Not** swapping the backend's served model in this effort (separate, later step once a winner is confirmed).
- **Not** introducing deep-learning or time-series models — ruled out per SUPERVISOR list (justification preserved in `ml/CLAUDE.md` "Models explicitly ruled out": tabular data favors GBTs; ~1 month of snapshots gives no longitudinal depth for sequence models; no bookings signal for RL/elasticity).

---

## 2. Data

| Dataset | Rows | Use | Kaggle form |
| --- | --- | --- | --- |
| `features_2026-05-18.parquet` | ~29.4M (frozen 77-col snapshot) | **Final fit** of each winner | Kaggle Dataset (~1.6GB upload) |
| `sample_5M_seed42.parquet` | 5,000,000 (stratified reservoir) | **Optuna HPO** search | Kaggle Dataset (~385MB) |

> Note: the "24M" figure is the 2026-05-15 count; the frozen training snapshot is 29,424,066 rows (2026-05-19). We train final models on the full frozen snapshot.

- **Target:** `log(price)` with `nights` as a feature (locked D-decision; never `log1p(price_per_night)`).
- **Features:** `FORECASTER_FEATURES` from `feature_engineering/model_feature_sets.py` (single source of truth). Categoricals per `data.categorical_feature_names()`.
- **Leakage discipline preserved:** peer aggregates are already self-excluded and `scraped_at`-bounded by the feature pipeline. This effort adds no new aggregating features, so no new leakage surface.
- **Memory on Kaggle:** load with `category` dtype + downcast numerics to `float32`; project to `FORECASTER_FEATURES + target + segment cols` only. ~29M × ~55 cols at float32/category ≈ 6–10GB, fits the 29GB high-RAM kernel. GPU training copies a float32 matrix (~5–6GB) into 16GB VRAM — fits.

---

## 3. Directory Structure

Reusable, testable code extends the existing `ml/models/forecasting/` package. A thin `kaggle/` layer holds notebook runners that import the package and handle Kaggle I/O. Rationale: a folder named `kaggle` full of logic is not portable or unit-testable; models belong under `ml/models/` per `ml/CLAUDE.md`.

```
ml/models/forecasting/
├── lgbm_quantile.py            # EXISTS — LGBMQuantileForecaster (3 boosters)
├── conformal.py                # EXISTS — ConformalQuantileCalibrator (global CQR)
├── data.py                     # EXISTS — prepare_xy, sample/load frame, categoricals
├── splits.py                   # EXISTS — hotel_wise_split / time_wise_split (+ cal bucket)
├── metrics.py                  # EXISTS — mae, mape, wape, pinball_loss, coverage
├── base.py                     # NEW — QuantileForecaster Protocol (fit/predict/save/load)
├── catboost_quantile.py        # NEW — CatBoostMultiQuantileForecaster (one model, GPU)
├── xgboost_quantile.py         # NEW — XGBoostQuantileForecaster (reg:quantileerror, GPU)
├── mondrian_conformal.py       # NEW — per-segment conformal calibrator (+ global fallback)
├── optuna_study.py             # NEW — search spaces + objective + resumable study
├── run_bakeoff.py              # NEW — CLI: HPO → full-fit → calibrate → eval → report
├── run_ablations.py            # NEW — CLI: data-scaling / feature-group / tail sweeps
└── tests/
    ├── test_catboost_quantile.py   # NEW — fit/predict shape, save/load round-trip
    ├── test_xgboost_quantile.py    # NEW — same + categorical handling
    ├── test_mondrian_conformal.py  # NEW — per-group c, small-group fallback, coverage
    └── test_optuna_study.py        # NEW — objective smoke, study resume

kaggle/
├── 00_upload_datasets.md       # how to publish the two parquet datasets
├── 10_hpo_lightgbm.ipynb       # import package, run Optuna on 5M, persist study
├── 11_hpo_catboost.ipynb
├── 12_hpo_xgboost.ipynb
├── 20_fit_full_and_eval.ipynb  # retrain 3 winners on 29M, calibrate, eval, plots
└── 30_ablations.ipynb
```

---

## 4. Model Interface (one Protocol, three backends)

All three models conform to a common `QuantileForecaster` Protocol so eval, conformal, and the recommender are model-agnostic. The existing `LGBMQuantileForecaster` already matches it.

```python
class QuantileForecaster(Protocol):
    feature_names_: list[str]
    def fit(self, X_tr, y_tr, X_val, y_val, categorical_features) -> Self: ...
    def predict(self, X) -> dict[str, np.ndarray]:   # {"q10","q50","q90"} in LOG space
    def save(self, out_dir) -> None: ...
    @classmethod
    def load(cls, model_dir) -> Self: ...
```

| Model | How quantiles are produced | Categoricals | GPU | Crossing risk |
| --- | --- | --- | --- | --- |
| **LightGBM** (incumbent) | 3 separate boosters (q10/q50/q90) | native | weak (OpenCL) | yes (~5% measured) |
| **CatBoost** | **one** model, `MultiQuantile:alpha=0.1,0.5,0.9` | native (`cat_features`) | strong (`task_type=GPU`) | none (shared tree) |
| **XGBoost ≥2.0** | **one** model, `reg:quantileerror`, `quantile_alpha=[.1,.5,.9]` | native (`enable_categorical=True`) | strong (`device=cuda`) | low (shared tree) |

- Predictions are always returned in **log space** (callers `np.exp`), matching `LGBMQuantileForecaster`, so `metrics.py`, `conformal.py`, and the recommender need no change.
- **XGBoost categorical** is the one finicky point: use native `enable_categorical=True` for a fair comparison; fallback to ordinal encoding if GPU + categorical proves unstable (documented in the wrapper).

---

## 5. Hyperparameter Optimization (Optuna)

- **Search on `sample_5M_seed42.parquet`**, hotel-wise split (`train` fits, `val` scores). 40–60 trials per model.
- **Objective (minimize):** summed pinball loss on `val` across the three quantiles (log space) — the loss the models actually optimize. Report WAPE(q50) and raw coverage80 alongside for transparency.
- **Pruning:** `MedianPruner` on the q50 booster's intermediate val score.
- **Resumability across 12h sessions:** `optuna.create_study(storage="sqlite:///<study>.db", load_if_exists=True)`. The SQLite file is round-tripped as a Kaggle Dataset between sessions.
- **Why sample, not 29M:** Optuna = 40–60 fits/model; on 29M that is tens of hours/model, exceeding the 30h/week + 12h/session caps. Hyperparameters transfer across data size (more data reduces variance and helps sparse segments; it does not change the *ranking* of good knobs). We **validate the transfer** by confirming the 29M-fit model's metrics match or beat its 5M-tuned expectation; if it regresses, run a small (≤10-trial) confirmation tune near the best point.

**Final fit:** each model's best config retrained **once on the full 29M snapshot**, hotel-wise split, artifacts saved.

---

## 6. Conformal Calibration Deep-Dive

Two calibrators fit on the `cal` split (already carved by `splits.py`), compared on `test`:

1. **Global CQR** (`conformal.py`, exists): single scalar `c_`. Marginal coverage only.
2. **Mondrian / group-conditional CQR** (`mondrian_conformal.py`, new): a separate `c_g` per segment `g`. **Primary grouping: `macro_region`** (5 groups); a `stars_band` variant is run as part of the calibration ablation. Groups with fewer than `MIN_CAL_PER_GROUP` (default 1000) calibration points fall back to the global `c_` (protects sparse Sud). Cite Mondrian conformal (Vovk et al.; Romano et al. 2019 for the CQR base).

**Hypothesis to demonstrate:** global `c_` over-widens well-modeled segments (Djerba ~21% WAPE) to cover badly-modeled ones (Sud ~53%). Per-segment `c_g` should **narrow median interval width** on strong segments while holding their coverage ≥ 0.78, and widen Sud honestly.

---

## 7. Evaluation Protocol

Identical for all three models — reuse `metrics.py` verbatim.

- **Primary split:** hotel-wise (unseen hotels). **Secondary:** time-wise (temporal-stress diagnostic).
- **Point:** MAE, MAPE, WAPE on q50 (TND scale).
- **Probabilistic:** pinball loss (log + TND), raw vs calibrated coverage80, quantile-crossing rate.
- **Gate (unchanged):** must beat linear-hedonic WAPE 37.6% (hotel-wise). Current LightGBM = 28.65%.
- **Per-segment:** WAPE by `macro_region` and `stars_band`.
- **Outputs:** `artifacts/reports/bakeoff_<run-date>/{model}_{split}.json` + a combined `comparison.json` + plots (see §9).

---

## 8. Ablations (the methods-story content)

1. **Data-scaling curve:** retrain the winner at n ∈ {1M, 5M, 12M, 29M}; plot WAPE & coverage vs n. Shows the value of full-data training and where returns flatten.
2. **Feature-group drop:** drop `peer_*`, then `calendar_*`, then `sur_demande_*`; report ΔWAPE per group (group-level importance, complements SHAP).
3. **Tail-hyperparameter sensitivity:** sweep the q10/q90 tail controls (e.g., `min_data_in_leaf` / `min_child_weight` analogues); plot coverage & interval width — directly explains the current q90-head pathology (early-stopped at 57 rounds).
4. **Quantile-crossing comparison:** crossing rate across the 3 models — quantifies the coherence advantage of single-model CatBoost/XGBoost over the 3-booster LightGBM.

---

## 9. Plots (report-ready)

Extend the `_eval_fresh.py` plotting style: (1) WAPE-by-model grouped bars; (2) calibrated-coverage-by-model; (3) data-scaling curve; (4) interval-width distribution per model; (5) crossing-rate bars; (6) global-vs-Mondrian coverage & width by segment; (7) predicted-vs-actual for the winner. Saved under `artifacts/reports/bakeoff_<run-date>/`.

---

## 10. Testing

- New wrappers: fit→predict shape, log-space output, save/load round-trip, monotonic-sort option, categorical handling. Mirror `tests/test_lgbm_quantile.py`.
- Mondrian conformal: per-group `c_g` correctness on a toy frame, small-group fallback to global, coverage sanity.
- Optuna: objective returns finite loss; study resumes from a persisted SQLite.
- All tests run on tiny in-memory frames, <60s total, no real Mongo/Postgres, no GPU required (CPU `task_type`/`device` in tests).

---

## 11. Kaggle Session Plan (fits 30h/week)

| Session | Work | ~GPU h |
| --- | --- | --- |
| 1 | Publish datasets; wrapper smoke on 5M; start LightGBM Optuna | 3–4 |
| 2 | CatBoost Optuna (resume study) | 3–4 |
| 3 | XGBoost Optuna (resume study) | 3–4 |
| 4 | Retrain 3 winners on 29M; global + Mondrian conformal; eval + plots | 4–6 |
| 5 | Ablations (data-scaling, feature-group, tail, crossing) | 3–5 |

---

## 12. Risks & Mitigations

| Risk | Mitigation |
| --- | --- |
| HPO transfer (5M→29M) doesn't hold | Validate 29M metrics vs 5M expectation; ≤10-trial confirmation tune if regressed |
| LightGBM GPU slow / not faster | Tune LightGBM with capped rounds or accept CPU; it's the incumbent baseline, not the favorite |
| XGBoost GPU + categorical unstable | Fallback to ordinal encoding; document in wrapper |
| 29M OOM on Kaggle | float32 + category dtypes, column projection, GPU streaming |
| Weekly quota (30h) | Resumable Optuna study, aggressive pruning, capped trials |
| Winner ≠ current served model | Backend swap is out of scope here; only swap after a confirmed, validated win |

---

## 13. Success Criteria

- All 3 models beat the OLS gate (WAPE < 37.6% hotel-wise).
- Winner's WAPE ≤ 28.65% (current), with calibrated coverage within ±1.5pp of 0.80 hotel-wise.
- CatBoost/XGBoost quantile-crossing ≈ 0 vs LightGBM ~5% (coherence win demonstrated).
- Mondrian conformal narrows median interval width on strong segments without their coverage dropping below 0.78.
- Reproducible: fixed seeds, persisted studies, JSON reports + plots committed.
