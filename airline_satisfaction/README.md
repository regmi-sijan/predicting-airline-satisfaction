# Predicting Airline Satisfaction

Code and Kaggle notebooks for the Playground Series S6E10 airline-satisfaction competition. The repository records the progression from pretrained tabular models and boosted-tree ensembles through neural models, fold-safe validation, teacher/route features, and a nested neural ensemble.

## Experiments

- `experiments/kaggle_v1`: TabICLv2 baseline.
- `experiments/kaggle_v2`–`kaggle_v4`: LightGBM, XGBoost, and CatBoost ensembles.
- `experiments/kaggle_v5`–`kaggle_v6`: neural GPU models and a consistent neural/tree fold blend.
- `experiments/kaggle_v7`: fold-safe RealMLP experiment.
- `experiments/kaggle_v8`: teacher-feature and route-feature ensemble.
- `experiments/kaggle_v9`: nested RealMLP, TabM, and LightGBM ensemble.
- `experiments/v1`: early local benchmark and notebook-packaging utilities.

Each Kaggle experiment directory contains its notebook, Python source, run notes, and—in most cases—a short README describing its validation design.

## Running on Kaggle

1. Open the notebook from the desired `experiments/kaggle_v*` directory in Kaggle.
2. Attach the Playground Series S6E10 competition data and any external datasets named in that experiment's README.
3. Select the accelerator specified in the experiment README (GPU for the neural experiments).
4. Use **Save Version** and **Run All**.

The later experiments are intentionally guarded or designed to run on Kaggle rather than on a local laptop. Their outputs are written beneath `/kaggle/working/`.

## Repository contents

Training/test data, pretrained weights, caches, and generated prediction CSVs are excluded. Download or attach those resources separately before running an experiment.

This repository is a local snapshot. Any changes made manually in the Kaggle editor after the latest local download are not automatically reflected here.
