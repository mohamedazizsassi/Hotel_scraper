# backend/ml_store/store.py
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

_ML_ROOT = Path(__file__).parent.parent.parent / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from models.forecasting.lgbm_quantile import LGBMQuantileForecaster    # noqa: E402
from models.forecasting.conformal import ConformalQuantileCalibrator    # noqa: E402
from models.anomaly.detector import IntervalAnomalyDetector              # noqa: E402
from models.recommendation.recommender import Recommender               # noqa: E402

_CATEGORICAL_COLS: tuple[str, ...] = (
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


def prepare_serve_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical string columns to 'category' dtype for LightGBM inference."""
    df = df.copy()
    for col in _CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


@dataclass
class MLStore:
    forecaster: LGBMQuantileForecaster
    calibrator: ConformalQuantileCalibrator
    recommender: Recommender
    detector: IntervalAnomalyDetector

    def __init__(self, model_dir: Path) -> None:
        self.forecaster  = LGBMQuantileForecaster.load(model_dir)
        self.calibrator  = ConformalQuantileCalibrator.load(model_dir / "conformal.json")
        self.recommender = Recommender(self.forecaster, self.calibrator)
        self.detector    = IntervalAnomalyDetector(self.forecaster, self.calibrator)


def load_ml_store(model_dir: Path) -> MLStore:
    """Factory called once from FastAPI lifespan."""
    return MLStore(model_dir)
