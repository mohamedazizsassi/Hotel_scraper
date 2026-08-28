"""
Generator script: produces the four Kaggle runner notebooks.
Run from repo root: ml/.venv/Scripts/python.exe kaggle/_gen_notebooks.py
"""
import json, os

OUT = os.path.join(os.path.dirname(__file__))

COMMON_HEADER = """\
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna>=3.6", "catboost>=1.2"])
sys.path.insert(0, "/kaggle/input/revway-forecasting-src/ml")

import pyarrow.parquet as pq
from models.forecasting.data import categorical_feature_names, prepare_xy
from models.forecasting.splits import hotel_wise_split
from models.forecasting.optuna_study import run_study

df = pq.read_table("/kaggle/input/revway-sample-5m/sample_5M_seed42.parquet").to_pandas()
X, y, _ = prepare_xy(df)
idx = hotel_wise_split(df["hotel_name_normalized"], seed=42)
"""

NOTEBOOKS = {
    "10_hpo_lightgbm.ipynb": COMMON_HEADER + """\
study = run_study("lightgbm", X, y, idx, categorical_feature_names(),
                  n_trials=50, storage="sqlite:////kaggle/working/study_lightgbm.db",
                  study_name="bakeoff_lightgbm")
print("BEST", study.best_value, study.best_params)
""",
    "11_hpo_catboost.ipynb": COMMON_HEADER + """\
study = run_study("catboost", X, y, idx, categorical_feature_names(),
                  n_trials=50, storage="sqlite:////kaggle/working/study_catboost.db",
                  study_name="bakeoff_catboost", task_type="GPU")
print("BEST", study.best_value, study.best_params)
""",
    "12_hpo_xgboost.ipynb": COMMON_HEADER + """\
study = run_study("xgboost", X, y, idx, categorical_feature_names(),
                  n_trials=50, storage="sqlite:////kaggle/working/study_xgboost.db",
                  study_name="bakeoff_xgboost", device="cuda")
print("BEST", study.best_value, study.best_params)
""",
    "20_fit_full_and_eval.ipynb": """\
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna>=3.6", "catboost>=1.2"])
sys.path.insert(0, "/kaggle/input/revway-forecasting-src/ml")

from models.forecasting.run_bakeoff import run_bakeoff
# Studies were uploaded as a dataset after the HPO notebooks; point storage there.
report = run_bakeoff(
    sample_path="/kaggle/input/revway-sample-5m/sample_5M_seed42.parquet",
    full_path="/kaggle/input/revway-features-full/features_2026-05-18.parquet",
    models=["lightgbm", "catboost", "xgboost"],
    n_trials=0,                       # 0 = reuse existing study best_params, no new trials
    out_dir="/kaggle/working/bakeoff",
    device="cuda", task_type="GPU",
    storage_dir="/kaggle/input/revway-studies",
)
import json; print(json.dumps(report, indent=2))
""",
    "30_ablations.ipynb": """\
import sys, subprocess
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "optuna>=3.6", "catboost>=1.2"])
sys.path.insert(0, "/kaggle/input/revway-forecasting-src/ml")

from models.forecasting.run_ablations import data_scaling, feature_group_drop
import pyarrow.parquet as pq, json
df = pq.read_table("/kaggle/input/revway-features-full/features_2026-05-18.parquet").to_pandas()
report = {
    "data_scaling": data_scaling(df, "catboost", [1_000_000, 5_000_000, 12_000_000, 29_000_000],
                                 {}, device="cuda", task_type="GPU"),
    "feature_group_drop": feature_group_drop(df, "catboost", {}, device="cuda", task_type="GPU"),
}
print(json.dumps(report, indent=2))
""",
}


def make_notebook(source: str) -> dict:
    return {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


for fname, source in NOTEBOOKS.items():
    path = os.path.join(OUT, fname)
    nb = make_notebook(source)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Written: {fname}")

print("Done.")
