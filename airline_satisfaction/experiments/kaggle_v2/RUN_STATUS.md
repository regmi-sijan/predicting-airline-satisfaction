# Experiment 2 cloud run

## Final result

Completed successfully in 514.1 seconds. Submitted experiment2/submission_selected.csv from run 363279354 on October 28, 2026. Kaggle confirms Success / Complete, public ROC AUC 0.95831. Selected blend: 75% LightGBM + 25% XGBoost. E1 remains better at 0.95835 (difference 0.00004).

## Launch record

- Launched October 28, 2026 on Kaggle with Save & Run All (Commit).
- Notebook: https://www.kaggle.com/code/sijanregmi/airline-satisfaction-e2-tree-ensemble
- Run ID: 363279354; version name: E2 trees three-fold CV.
- Private notebook, accelerator GPU T4 x2. Logs confirm Tesla T4, two devices, competition data loaded, and LightGBM fold 1 started at 32 seconds.
- Status at last observation: running; no final metrics or competition submission yet.
- Compares LightGBM, CUDA XGBoost, and GPU CatBoost using 38 features, three folds, separate blend selection/report partitions, and full-data refits.
- All training and prediction run on Kaggle. Local work was source preparation and syntax validation only.
- Previous public AUC baseline: 0.95835. Improvement remains unverified until completion and scoring.
- Expected output: /kaggle/working/experiment2/submission_selected.csv and metrics.json.
