# E9 — Nested Neural Ensemble

Notebook: https://www.kaggle.com/code/sijanregmi/airline-satisfaction-e9-nested-neural-ensemble

Prepared 2026-10-30. All model fitting and inference are restricted to Kaggle. Local work was source assembly, review and Python syntax compilation only.

Seven candidates: three RealMLP recipes (three seeds each, eight networks per seed), TabM (32 members, 12 fixed epochs), E8 distance-tree control, smoothed leave-one-out distance statistics with variability/deviations, and the same without original-data teacher features. External teacher matches E8 and excludes exact feature overlaps.

Five outer folds (seed 42). A 20% holdout inside each outer-training partition produces independent base predictions for blend selection. Three-fold meta validation inside that holdout chooses anchor / bounded fine logit blend / regularized logistic stack. Bases refit on all outer-training rows before outer evaluation and test prediction. Scored outer labels do not select models or weights. Preserve seed42-only E8 control and match E8's base-first test averaging.

Primary artifact after completion: experiment9/submission_nested_selected.csv. No automatic competition submission. E8 public reference: 0.96126.

Source SHA256: d3833da536aa7f237f616b04c1f20e15b25be14970fdae99446836fc47fb9072. Independent source/API review and syntax compilation passed. Uploaded 54.31 KB notebook; launch verification pending.
