"""
Four mandatory baselines (per ml/CLAUDE.md non-negotiable principle #3).

A trained GBT model must beat ALL FOUR on the hotel-wise hold-out — otherwise
it does not ship. So the numbers these produce are the gate, not decoration.

All baselines predict in **log-price space** (same scale as y = log(price)).
Callers must np.exp() before computing MAE/MAPE/WAPE on the raw TND scale.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import booking_window_bucket

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Baseline 1 — global median
# ---------------------------------------------------------------------------

class GlobalMedianBaseline:
    """Floor: predict log(median(price_train)) for every row."""

    def __init__(self) -> None:
        self.median_logp_: float | None = None

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "GlobalMedianBaseline":
        self.median_logp_ = float(np.median(y))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.median_logp_ is not None, "call fit first"
        return np.full(len(df), self.median_logp_, dtype=np.float64)


# ---------------------------------------------------------------------------
# Baseline 2 — hierarchical group median (the realistic non-ML floor)
# ---------------------------------------------------------------------------

_GROUP_HIERARCHY: tuple[tuple[str, ...], ...] = (
    ("city_name", "stars_int", "boarding_canonical", "nights", "booking_window_bucket"),
    ("city_name", "stars_int", "boarding_canonical", "nights"),
    ("city_name", "stars_int", "nights"),
    ("city_name", "stars_int"),
    ("stars_int",),
)


class GroupMedianBaseline:
    """
    Median log-price per group, narrowest → broadest fallback.

    On unseen group combos at predict time, walks up the hierarchy until a
    group with training data is found. Final fallback: global median.
    """

    def __init__(self) -> None:
        self.tables_: list[pd.Series] | None = None
        self.global_median_: float | None = None

    def _augment(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["booking_window_bucket"] = booking_window_bucket(out["days_until_checkin"])
        return out

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "GroupMedianBaseline":
        d = self._augment(df).assign(_y=y)
        self.tables_ = [
            d.groupby(list(keys), dropna=False)["_y"].median()
            for keys in _GROUP_HIERARCHY
        ]
        self.global_median_ = float(np.median(y))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.tables_ is not None, "call fit first"
        d = self._augment(df)
        out = np.full(len(d), np.nan, dtype=np.float64)
        unresolved = np.ones(len(d), dtype=bool)

        for keys, table in zip(_GROUP_HIERARCHY, self.tables_):
            if not unresolved.any():
                break
            sub = d.loc[unresolved, list(keys)]
            idx = pd.MultiIndex.from_frame(sub) if len(keys) > 1 else sub.iloc[:, 0]
            vals = table.reindex(idx).to_numpy()
            positions = np.flatnonzero(unresolved)
            hit = ~np.isnan(vals)
            out[positions[hit]] = vals[hit]
            unresolved[positions[hit]] = False

        out[unresolved] = self.global_median_  # type: ignore[arg-type]
        return out


# ---------------------------------------------------------------------------
# Baseline 3 — competitor median (predict the peer median directly)
# ---------------------------------------------------------------------------

class CompetitorMedianBaseline:
    """
    Predict log(peer_medium_median). On NaN (peer count too low), fall back
    loose → tight → global. Tests whether ML beats "just use peers".
    """

    def __init__(self) -> None:
        self.global_median_: float | None = None

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "CompetitorMedianBaseline":
        self.global_median_ = float(np.median(y))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.global_median_ is not None, "call fit first"
        out = np.full(len(df), np.nan, dtype=np.float64)
        for col in ("peer_medium_median", "peer_loose_median", "peer_tight_median"):
            if col not in df.columns:
                continue
            mask = np.isnan(out)
            if not mask.any():
                break
            vals = df[col].to_numpy(dtype=np.float64)
            log_vals = np.full_like(vals, np.nan)
            positive = vals > 0
            log_vals[positive] = np.log(vals[positive])
            update_mask = mask & ~np.isnan(log_vals)
            out[update_mask] = log_vals[update_mask]
        out[np.isnan(out)] = self.global_median_  # type: ignore[arg-type]
        return out


# ---------------------------------------------------------------------------
# Baseline 4 — linear hedonic OLS (Ridge) with sparse one-hot
# ---------------------------------------------------------------------------

_OLS_CATEGORICAL: tuple[str, ...] = (
    "boarding_canonical",
    "room_base",
    "room_view",
    "room_tier",
    "room_occupancy",
    "best_peer_granularity_used",
    "macro_region",
    "stars_band",
    "market_segment_id",
)

_OLS_NUMERIC: tuple[str, ...] = (
    "nights",
    "days_until_checkin",
    "adults",
    "children",
    "stars_int",
    "is_supplement_variant",
    "view_upgrade_count_offered",
    "has_free_view_upgrade",
    "check_in_dow",
    "check_in_month",
    "check_in_week_of_year",
    "check_in_day_of_month",
    "check_in_quarter",
    "is_weekend_checkin",
    "is_ramadan",
    "is_tunisia_public_holiday",
    "is_tunisia_school_holiday",
    "is_school_holiday_france",
    "is_school_holiday_germany",
    "is_school_holiday_uk",
    "days_to_nearest_european_holiday",
    "peer_medium_count",
    "peer_medium_median",
    "peer_medium_p25",
    "peer_medium_p75",
    "peer_medium_std",
    "peer_loose_count",
    "peer_loose_median",
    "peer_tight_count",
    "peer_tight_median",
    "sur_demande_rate_city_checkin",
    "sur_demande_rate_city_stars_checkin",
    "sur_demande_rate_city_stars_boarding_checkin",
    "city_activity_count_checkin",
)


class LinearHedonicBaseline:
    """
    Ridge regression on log(price): one-hot categoricals + standardised numerics.

    Sparse pipeline — fits 5M × ~120 features in <2 GB with sparse one-hot.

    Parameters
    ----------
    max_fit_rows:
        If train size exceeds this, fit on a random subsample. None = no cap.
        Default 1M is comfortable on 5 GB RAM.
    alpha:
        Ridge regularization strength.
    """

    def __init__(self, max_fit_rows: int | None = 1_000_000, alpha: float = 1.0) -> None:
        self.max_fit_rows = max_fit_rows
        self.alpha = alpha
        self.pipeline_: Pipeline | None = None

    def _make_pipeline(self) -> Pipeline:
        # Categorical NaN already replaced with "_missing" in _prepare,
        # so OneHotEncoder sees a clean string column.
        cat = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        num = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),  # keep sparse-compatible
        ])
        pre = ColumnTransformer([
            ("cat", cat, list(_OLS_CATEGORICAL)),
            ("num", num, list(_OLS_NUMERIC)),
        ])
        return Pipeline([
            ("pre", pre),
            ("ridge", Ridge(alpha=self.alpha, solver="sparse_cg")),
        ])

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Cast categoricals to object str with NaN → '_missing'."""
        cols = list(_OLS_CATEGORICAL) + list(_OLS_NUMERIC)
        out = df[cols].copy()
        for c in _OLS_CATEGORICAL:
            # astype(object) avoids pd.NA semantics; fillna handles None and NaN.
            out[c] = out[c].astype(object).where(out[c].notna(), "_missing").astype(str)
        return out

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "LinearHedonicBaseline":
        if self.max_fit_rows is not None and len(df) > self.max_fit_rows:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(df), size=self.max_fit_rows, replace=False)
            X_fit = self._prepare(df.iloc[idx])
            y_fit = y[idx]
            logger.info(
                "LinearHedonic: subsampled %d → %d rows for fit",
                len(df), self.max_fit_rows,
            )
        else:
            X_fit = self._prepare(df)
            y_fit = y
        self.pipeline_ = self._make_pipeline().fit(X_fit, y_fit)
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        assert self.pipeline_ is not None, "call fit first"
        return self.pipeline_.predict(self._prepare(df))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def all_baselines() -> dict[str, "Baseline"]:
    """Construct one of each baseline. Keys are stable identifiers for reports."""
    return {
        "global_median": GlobalMedianBaseline(),
        "group_median": GroupMedianBaseline(),
        "competitor_median": CompetitorMedianBaseline(),
        "linear_hedonic": LinearHedonicBaseline(),
    }


# Type alias for the Baseline protocol (duck-typed: fit + predict).
Baseline = GlobalMedianBaseline  # for type hints — any of the four works
