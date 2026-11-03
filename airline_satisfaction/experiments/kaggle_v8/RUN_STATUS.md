# Experiment 8 — launched

Kaggle private notebook: https://www.kaggle.com/code/sijanregmi/airline-satisfaction-e8-teacher-route-ensemble/log
Version1/run363647448, GPU T4x2. All training/inference on Kaggle; local syntax/source preparation only.

Three matched candidates: E7 RealMLP control retrained; same RealMLP plus one continuous original-data teacher feature; original-teacher plus outer-training-only route statistics and zero-aware ratings in LightGBM. Teacher uses original-data labels only and excludes exact competition-feature matches. Original dataset inherited from credited public source notebook, not downloaded locally.

Five folds seed42, neural3epochs/8networks. 30 fixed blend candidates (15quarter-step simplex weights x probability/logit); report conditional meta-CV, clearly distinguished from fully nested retraining. Raw models/OOF/predictions preserved under experiment8. No automatic competition submission. E7 public reference0.96115.

At63.8seconds startup was running; no errors observed, teacher completion pending.

Startup verified at105.3seconds: EXTERNAL TEACHER READY at71.7seconds; 129880originalrows,21exactcompetition-feature matches excluded,129859retained. Both originalCSVinputs attached. FOLD START1at72.3seconds; GPU available True/used True at101.2seconds. Runtime image uses Python3.13 (E7was3.12), so E7is retrained as same-environment matched control. Screenshot:kaggle-running.png. Results pending.

## Completed successfully
Version1/run363647448 completed on Kaggle GPU T4x2 in1478.9seconds (24m39s). E7 control OOF AUC0.9612271603576075/logloss0.22224844634686028; teacher RealMLP AUC0.9609748915288829/logloss0.22294854188398713; route tree AUC0.960627050600269/logloss0.22372689509370886. Teacher addition worsened AUC by0.0002522688287245334 and was worse on all5folds.

Selected logit blend75%E7control +25%route tree,0%teacherRealMLP. Selection OOF AUC0.9614087774449693. Conditional meta-CV AUC0.9614014933391792/logloss0.22178579763754644, conditional on fixed OOF library, not fully nested. Submission experiment8/submission_selected.csv sha256d60cd8b90cf3245cbc3bbb0c4568edaec1e65a83c55ad9f70a50985e65b88a68. Four submission artifacts produced. No E8 competition submission yet; no public E8score.

## Competition submission confirmed
Submitted experiment8/submission_selected.csv from version1/run363647448. Kaggle status Complete, public ROC AUC0.96126. New personal best vs E7 0.96115, delta+0.00011. Leaderboard rank1077of3515teams at verification,8totalentries. Screenshot:submission-leaderboard.png.
