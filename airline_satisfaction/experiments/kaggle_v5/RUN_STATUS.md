# Experiment 5 — Neural GPU models

## Submitted

Submitted experiment5/submission_selected.csv from run 363493587 on October 29, 2026. Kaggle confirms Success / Complete: public ROC AUC 0.95849. This is 0.00007 below E4's 0.95856; E4 remains the best. Current team rank at submission check: 1976 of 3471, based on E4. Screenshot: submission-leaderboard.png.

## Completed

Kaggle confirms successful run in 1194.9 seconds (19m 55s), script runtime 1166.53 seconds. Selected architecture: width 512, four residual blocks, dropout .20; final three-seed models trained for 12 epochs.
Selected blend: 20% ResNet + 80% LightGBM, TabICL zero weight. Reused report AUC: blend 0.9578097492, LightGBM 0.9577444003, ResNet 0.9559505085, TabICL 0.9551914419. Selection blend AUC 0.9592482147.
TabICL 65,536-row context exceeded GPU memory; logged fallback to 32,768 rows / eight estimators completed held and test inference successfully.
Ready output: experiment5/submission_selected.csv. Not submitted during this status-only check; no E5 leaderboard score yet. E4 public baseline remains 0.95856.

## Launch record

Launched October 29, 2026 at 7:39 AM America/Chicago.
Notebook: https://www.kaggle.com/code/sijanregmi/airline-satisfaction-e5-neural-gpu-models
Kaggle run 363493587, private, GPU T4 x2 runtime. All training/inference on Kaggle; local preparation and AST syntax checks only.

Status: running. GPU Tesla T4 count 2 and competition data loaded confirmed. First residual network fold began at 22 seconds; epoch 8 validation AUC 0.95688853 at 42 seconds. This is a partial fold score, not the final validation or leaderboard score.

Two residual neural-network recipes: width 256 / 3 blocks / dropout .15, and width 512 / 4 blocks / dropout .20. Categorical embeddings, training-fold-only quantile preprocessing, mixed precision, AdamW, early stopping up to 100 epochs, three-fold CV. Chosen architecture refit with three seeds on all training rows. Frozen TabICLv2 2.2.0 uses official jingang/TabICL v2 checkpoint, 65,536 context rows / eight ensemble members; fallback 32,768 logged on worker failure. Each held/test phase worker has 90-minute timeout. Failures recorded explicitly without discarding neural work. Blend with LightGBM control; no XGBoost. Reused selection/report partitions, not fresh independent evaluation. NN CV uses early stopping on fold labels and is developmental.

Expected outputs: experiment5/metrics.json, search.json, held_predictions.csv, neural models and preprocessing, submission_selected.csv, submission_neural_blend.csv, submission_resnet.csv, submission_resnet_fold.csv, submission_tabicl.csv if pretrained inference succeeds, submission_lightgbm.csv.
Current public baseline E4 0.95856. No automatic competition submission; no improvement established yet.
