import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Ensure ml/ is importable for the fixture
_ML_ROOT = Path(__file__).parent.parent.parent / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from ml_store.store import MLStore, prepare_serve_frame

@pytest.fixture()
def mock_model_dir(tmp_path):
    """Train a 2-feature toy LightGBM so the test doesn't need real model files."""
    from models.forecasting.lgbm_quantile import LGBMQuantileForecaster
    from models.forecasting.conformal import ConformalQuantileCalibrator

    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({
        "nights":    rng.integers(1, 8, n).astype(float),
        "stars_int": rng.integers(1, 6, n).astype(float),
    })
    y = np.log(rng.uniform(100, 2000, n))

    model = LGBMQuantileForecaster(num_boost_round=5, early_stopping_rounds=3, seed=0)
    model.fit(X.iloc[:240], y[:240], X.iloc[240:], y[240:])
    model.save(tmp_path)

    preds = model.predict(X.iloc[240:260])
    cal = ConformalQuantileCalibrator(alpha=0.20).fit(preds["q10"], preds["q90"], y[240:260])
    cal.save(tmp_path / "conformal.json")
    return tmp_path

def test_ml_store_loads(mock_model_dir):
    store = MLStore(mock_model_dir)
    assert store.forecaster is not None
    assert store.calibrator is not None
    assert store.recommender is not None
    assert store.detector is not None
    assert len(store.forecaster.feature_names_) == 2  # nights, stars_int

def test_prepare_serve_frame_casts_categoricals():
    df = pd.DataFrame({
        "boarding_canonical":      ["BB", "AI"],
        "room_base":               ["chambre", "suite"],
        "room_view":               [None, "mer"],
        "room_tier":               [None, None],
        "room_occupancy":          ["double", "double"],
        "best_peer_granularity_used": ["medium", "loose"],
        "macro_region":            ["sahel", "cap_bon"],
        "stars_band":              ["4-5", "4-5"],
        "market_segment_id":       ["sahel_4", "cap_bon_4"],
        "nights":                  [3, 7],
    })
    result = prepare_serve_frame(df)
    assert result["boarding_canonical"].dtype.name == "category"
    assert result["nights"].dtype.name != "category"
