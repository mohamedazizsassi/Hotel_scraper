"""
Conformalized Quantile Regression (CQR) calibration for the forecaster.

Reference
---------
Romano, Patterson, Candès (2019), "Conformalized Quantile Regression",
NeurIPS. arXiv:1905.03222

Method
------
Given raw quantile predictions q_lo, q_hi (the lower and upper quantiles of
the trained forecaster, here q10 and q90), compute per-point conformity
scores on a held-out calibration set:

    s_i = max(q_lo_i - y_i, y_i - q_hi_i)

Positive when y_i is outside the raw interval; negative when y_i is
strictly inside. The calibration constant `c` is the (1 - alpha) empirical
quantile of {s_i}, with finite-sample correction (1 + 1/n_cal):

    c = quantile(s_i, level = min(1, (1 - alpha) * (1 + 1/n_cal)))

Calibrated interval is [q_lo - c, q_hi + c]. Marginal coverage of this
interval on exchangeable test data is ≥ 1 - alpha (Theorem 1 of Romano).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class ConformalQuantileCalibrator:
    """
    Single-c conformal calibrator over a quantile-regression interval.

    Attributes
    ----------
    alpha:
        Miscoverage rate. alpha=0.20 → nominal 80% interval.
    c_:
        Fitted scalar. Added to q_hi, subtracted from q_lo at apply time.
    alpha_fit_, n_cal_:
        Recorded at fit time for the metadata.json.
    """

    alpha: float = 0.20
    c_: float = field(default=float("nan"))
    alpha_fit_: float = field(default=float("nan"))
    n_cal_: int = 0

    # ----------------------------------------------------------------- fit
    def fit(
        self,
        q_lo: np.ndarray,
        q_hi: np.ndarray,
        y_true: np.ndarray,
    ) -> "ConformalQuantileCalibrator":
        """Compute and store the scalar widening c from a calibration set."""
        q_lo = np.asarray(q_lo, dtype=np.float64)
        q_hi = np.asarray(q_hi, dtype=np.float64)
        yt   = np.asarray(y_true, dtype=np.float64)
        if not (q_lo.shape == q_hi.shape == yt.shape):
            raise ValueError(
                f"fit: shape mismatch lo={q_lo.shape} hi={q_hi.shape} y={yt.shape}"
            )
        if q_lo.size == 0:
            raise ValueError("fit: empty calibration set")

        scores = np.maximum(q_lo - yt, yt - q_hi)
        n = scores.size
        level = min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / n))
        # np.quantile with 'higher' interpolation matches the conservative
        # finite-sample CQR construction.
        self.c_ = float(np.quantile(scores, level, method="higher"))
        self.alpha_fit_ = self.alpha
        self.n_cal_ = int(n)
        return self

    # --------------------------------------------------------------- apply
    def apply(
        self, q_lo: np.ndarray, q_hi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return calibrated (q_lo - c, q_hi + c)."""
        if not np.isfinite(self.c_):
            raise RuntimeError("apply: calibrator not fitted (c_ is NaN)")
        q_lo = np.asarray(q_lo, dtype=np.float64)
        q_hi = np.asarray(q_hi, dtype=np.float64)
        return q_lo - self.c_, q_hi + self.c_

    # ------------------------------------------------------------------ io
    def save(self, path: str | Path) -> None:
        """Persist the fitted calibrator (alpha, c_, alpha_fit_, n_cal_) to JSON."""
        Path(path).write_text(
            json.dumps(
                {
                    "alpha": self.alpha,
                    "c_": self.c_,
                    "alpha_fit_": self.alpha_fit_,
                    "n_cal_": self.n_cal_,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ConformalQuantileCalibrator":
        """Reconstruct a calibrator from a JSON file previously written by save()."""
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            alpha=d["alpha"],
            c_=d["c_"],
            alpha_fit_=d["alpha_fit_"],
            n_cal_=d["n_cal_"],
        )
