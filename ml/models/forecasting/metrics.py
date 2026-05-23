"""
Forecaster evaluation metrics.

All point metrics (MAE, MAPE, WAPE) expect inputs on the **raw price scale**.
Callers training on log(price) must np.exp() the predictions before passing
them in. MAE in log units is meaningless to a hotel manager.

Pinball loss operates on whatever scale you pass — typically also raw price.
"""
from __future__ import annotations

import numpy as np


def _as_arrays(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"shape mismatch: {yt.shape} vs {yp.shape}")
    if yt.size == 0:
        raise ValueError("empty input")
    return yt, yp


def mae(y_true, y_pred) -> float:
    """Mean absolute error on the raw price scale (TND)."""
    yt, yp = _as_arrays(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def mape(y_true, y_pred) -> float:
    """
    Mean absolute percentage error (%).
    Asymmetric and unstable when y_true is small — pair with WAPE.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    if (yt <= 0).any():
        raise ValueError("MAPE undefined when y_true has non-positive values")
    return float(np.mean(np.abs((yt - yp) / yt)) * 100.0)


def wape(y_true, y_pred) -> float:
    """
    Weighted absolute percentage error (%). Sum |error| / Sum |y_true|.
    More stable than MAPE for skewed targets — preferred summary metric.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    denom = float(np.sum(np.abs(yt)))
    if denom == 0.0:
        raise ValueError("WAPE undefined when sum(|y_true|) == 0")
    return float(np.sum(np.abs(yt - yp)) / denom * 100.0)


def pinball_loss(y_true, y_pred, quantile: float) -> float:
    """
    Pinball (quantile) loss at the given quantile in (0, 1).

    Lower is better. Symmetric at q=0.5 (== 0.5 * MAE).
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    yt, yp = _as_arrays(y_true, y_pred)
    diff = yt - yp
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1.0) * diff)))


def coverage(y_true, q_lo, q_hi) -> float:
    """
    Fraction of `y_true` falling within [min(q_lo,q_hi), max(q_lo,q_hi)].
    Calibrated q10/q90 intervals should yield coverage ~0.80.

    Per-row min/max makes this robust to quantile crossing (q_lo > q_hi for
    some rows). Callers should report the crossing rate as a separate
    diagnostic — coverage alone hides it.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    lo = np.asarray(q_lo, dtype=np.float64)
    hi = np.asarray(q_hi, dtype=np.float64)
    if not (yt.shape == lo.shape == hi.shape):
        raise ValueError("y_true, q_lo, q_hi must have the same shape")
    lo_eff = np.minimum(lo, hi)
    hi_eff = np.maximum(lo, hi)
    return float(np.mean((yt >= lo_eff) & (yt <= hi_eff)))
