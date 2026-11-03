Experiment 7 launched October 29, 2026 on Kaggle GPU T4 x2.

Notebook: https://www.kaggle.com/code/sijanregmi/airline-satisfaction-e7-fold-safe-realmlp/log
Run: 363604085, version 1 (E7 fixed RealMLP 5-fold GPU). Private. Internet enabled for pytabkit==1.7.3. Competition data inherited from credited original notebook. No local training or inference.

Fixed 5-fold seed42; 3 epochs; 8 networks/fold; raw numeric values plus categorical twins, counts and interactions; inner-cross-fitted target encoding; no outer-validation labels passed to model.fit. Matched LightGBM fixed626-tree reference. Results pending; public best to beat0.95856. No automatic competition submission.

## Version 2 repair
Original run363604085 failed after173.6seconds: joblib PicklingError for PyTabKit get_schedule lambda, after first fold trained and predicted. Replaced model joblib export with cloudpickle, reload-and-predict verification (16rows), and nonfatal export warnings. Saves ID-aligned validation predictions before optional serialization. Training recipe and folds unchanged.

Replacement run363606271, version2, launched October29on Kaggle GPU T4x2. Link: https://www.kaggle.com/code/sijanregmi/airline-satisfaction-e7-fold-safe-realmlp?scriptVersionId=363606271
Verified logs: neural fold1 AUC0.9614857858024882, logloss0.22174539190602066; MODEL EXPORT VERIFIED at189.4seconds. Running at230.5seconds, matched LightGBM baseline pending. This is one validation fold, not full OOF or public leaderboard score. No competition submission. Screenshot:kaggle-fixed-running.png.

## Completed successfully
Kaggle version2/run363606271 completed successfully in978.2seconds (16m18s), all five folds and all five model exports verified. RealMLP full OOF AUC0.9612316545901254, logloss0.2222311403525724. Matched LightGBM full OOF AUC0.9590037194952841, logloss0.22672735762742507. AUC improvement0.0022279350948413157; RealMLP better on every fold. 31 output files. Submission:experiment7/submission_realmlp.csv, sha2564779a1443492c883b4f354a6ad074e3dff6d75110f5b40bd16bf2a00a7cbf857. Not submitted to competition; no E7 public score yet.

## Competition submission confirmed
Submitted experiment7/submission_realmlp.csv from version2/run363606271. Kaggle status Complete. Public ROC AUC0.96115, new personal best versus E4 0.95856 (delta+0.00259). Verified leaderboard position1135of3513teams on October29,2026. Seven total competition submissions. Screenshot:submission-leaderboard.png. No duplicate submission during resumed verification.
