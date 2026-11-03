# Experiment 6 completed

Resumed and completed October 29, 2026 on Kaggle CPU. Notebook: https://www.kaggle.com/code/sijanregmi/airline-satisfaction-e6-neural-fold-blend
Run 363586057, version 1. Kaggle runtime 31 seconds, blending script 6.58 seconds. E4 and E5 version 1 outputs attached as inputs. No local training/inference.

Selected log-odds blend: 75% E4 ensemble + 25% E5 saved three-fold ResNet ensemble. Selection AUC 0.9592654041 vs E4 0.9591280054. Reused report AUC 0.9578360100 vs E4 0.9577570802. Report log loss 0.23038138 vs E4 0.23075610. Model predictions are consistent between validation and test for the neural component. Reused holdouts are not fresh independent evaluation.

Submitted experiment6/submission_selected.csv. Kaggle confirms Complete / Success, public ROC AUC 0.95855. E4 remains best at 0.95856; E6 is lower by 0.00001. Team rank at observation: 1987 of 3493. Screenshot: submission-leaderboard.png.
