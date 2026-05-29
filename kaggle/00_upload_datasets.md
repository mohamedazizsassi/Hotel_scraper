# Kaggle datasets for the forecaster bake-off

Upload two private Kaggle Datasets (Kaggle -> Datasets -> New Dataset):

1. `revway-features-full` - `ml/artifacts/features_2026-05-18.parquet` (~1.6GB, ~29.4M rows)
2. `revway-sample-5m`     - `ml/artifacts/cache/sample_5M_seed42.parquet` (~385MB)

Also upload the package code as a Dataset `revway-forecasting-src` containing the
`ml/models/` and `ml/feature_engineering/` trees (or attach the GitHub repo).

In each notebook, add all three as inputs. Mount paths will be:
  /kaggle/input/revway-features-full/features_2026-05-18.parquet
  /kaggle/input/revway-sample-5m/sample_5M_seed42.parquet
  /kaggle/input/revway-forecasting-src/ml

Enable GPU: Notebook -> Settings -> Accelerator -> GPU T4 x2 (or P100).
Studies persist to /kaggle/working; download study_*.db between sessions and
re-upload as a dataset to resume HPO.
