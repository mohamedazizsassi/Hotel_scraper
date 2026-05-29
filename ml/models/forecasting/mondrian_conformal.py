"""
Mondrian (group-conditional) Conformalized Quantile Regression.

Generalizes the single-scalar CQR in conformal.py: fit one widening constant
c_g per segment g (e.g. macro_region), so well-modeled segments get tight
intervals and poorly-modeled ones get wide ones - instead of one global c that
over-widens the good segments to cover the bad. Segments with fewer than
`min_cal_per_group` calibration points fall back to the global c.

References: Romano, Patterson, Candes (2019) CQR; Vovk et al. Mondrian
conformal prediction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _conformal_c(scores: np.ndarray, alpha: float) -> float:
    n = scores.size
    level = min(1.0, (1.0 - alpha) * (1.0 + 1.0 / n))
    return float(np.quantile(scores, level, method="higher"))


@dataclass
class MondrianConformalCalibrator:
    alpha: float = 0.20
    min_cal_per_group: int = 1000
    c_global_: float = field(default=float("nan"))
    c_by_group_: dict[str, float] = field(default_factory=dict)

    def fit(self, q_lo, q_hi, y_true, groups) -> "MondrianConformalCalibrator":
        q_lo = np.asarray(q_lo, dtype=np.float64)
        q_hi = np.asarray(q_hi, dtype=np.float64)
        yt = np.asarray(y_true, dtype=np.float64)
        groups = np.asarray(groups)
        if not (q_lo.shape == q_hi.shape == yt.shape == groups.shape):
            raise ValueError("fit: q_lo, q_hi, y_true, groups must share shape")
        if q_lo.size == 0:
            raise ValueError("fit: empty calibration set")

        scores = np.maximum(q_lo - yt, yt - q_hi)
        self.c_global_ = _conformal_c(scores, self.alpha)
        self.c_by_group_ = {}
        for g in np.unique(groups):
            mask = groups == g
            if int(mask.sum()) >= self.min_cal_per_group:
                self.c_by_group_[str(g)] = _conformal_c(scores[mask], self.alpha)
            else:
                self.c_by_group_[str(g)] = self.c_global_
        return self

    def apply(self, q_lo, q_hi, groups) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite(self.c_global_):
            raise RuntimeError("apply: calibrator not fitted")
        q_lo = np.asarray(q_lo, dtype=np.float64)
        q_hi = np.asarray(q_hi, dtype=np.float64)
        groups = np.asarray(groups)
        c = np.array([self.c_by_group_.get(str(g), self.c_global_) for g in groups],
                     dtype=np.float64)
        return q_lo - c, q_hi + c

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "alpha": self.alpha,
            "min_cal_per_group": self.min_cal_per_group,
            "c_global_": self.c_global_,
            "c_by_group_": self.c_by_group_,
        }, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "MondrianConformalCalibrator":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        inst = cls(alpha=d["alpha"], min_cal_per_group=d["min_cal_per_group"])
        inst.c_global_ = d["c_global_"]
        inst.c_by_group_ = d["c_by_group_"]
        return inst
