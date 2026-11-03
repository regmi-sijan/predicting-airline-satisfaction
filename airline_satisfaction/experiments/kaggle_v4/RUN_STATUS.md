# Experiment 4 — Extended GPU ensemble

## Completed and scored

Run 363344721 completed successfully in 7512.5 seconds (2h 5m 12.5s); script runtime 7189.13 seconds.
Selection kept E3 anchor: selection AUC 0.95915270 vs best new ensemble 0.95912801. Selected.csv therefore reproduces the E3 baseline; it was not redundantly submitted.
Submitted experiment4/submission_ensemble.csv on October 29, 2026, with user authorization to submit and compare. Ensemble: log-odds weights 90% five-fold LightGBM + 10% three-seed XGBoost, CatBoost weight zero. Report AUC 0.95775708 vs E3 0.95774440; reused holdout.
Kaggle submission Complete / Success: public AUC 0.95856. Improvement over E3 0.95847: +0.00009. New personal best; public rank at observation 1974 of 3464 teams. Screenshot: submission-leaderboard.png.

## Launch record

Launched October 28, 2026 at 9:16 PM America/Chicago using Kaggle Save & Run All.
Notebook: https://www.kaggle.com/code/sijanregmi/airline-satisfaction-e4-extended-gpu-ensemble
Run: 363344721. Private, GPU T4 x2 runtime. All training/inference on Kaggle; local source preparation and AST syntax checks only.

Twelve recipes: four LightGBM, four CUDA XGBoost, four GPU CatBoost. Five outer folds each have an inner 12% stopping split; models refit on all outer training rows before validation. Recipe choices use OOF AUC. Compare fold averaging with three-seed pool refits; choose probability/logit blends using a fixed 10% weight grid on the reused selection holdout. Evaluate reused report holdout after choices. Full-data three-seed refits create final CSVs. Save a best blend with at least two families separately, and retain the E3 baseline comparator if it validates better.

Expected outputs: experiment4/metrics.json, search.json, blend_candidates.json, submission_ensemble.csv, submission_selected.csv, per-family CSVs, models, predictions and checkpoints.
Public baseline: E3 0.95847. No automatic competition submission. Improvement not yet established.

Status: running. At 323 seconds, logs confirm setup completed, Tesla T4 x2 available, competition data loaded, and lightgbm_00_original fold 1 started. Setup took about five minutes. No final validation results or leaderboard submission yet.
