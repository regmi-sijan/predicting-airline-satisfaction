# S6E10 public-method audit — 29 October 2026

Read-only review of signed-in Kaggle leaderboard, entrant profiles, public discussions, rendered source cells and saved notebook outputs. No model was trained or submitted during this review. Public authors' measurements have not been independently reproduced.

## Top ten snapshot

[Leaderboard](https://www.kaggle.com/competitions/playground-series-s6e10/leaderboard), checked on 29 October. Public split is approximately 20%; private split is the other 80%.

| Rank | Team | Public ROC AUC | Verified method disclosure found |
|---:|---|---:|---|
| 1 | Ahmet Bugrahan ALTUNOK | 0.96446 | No complete S6E10 solution found; public notebook inspected was an S6E9 solution. |
| 2 | Masaya Kawamata | 0.96443 | Explicitly described public-LB probing after a plateau around 0.9618; expects poor private transfer. No complete S6E10 training solution found. |
| 3 | nammovuivui | 0.96432 | No S6E10 solution found in public code/discussion inspected. |
| 4 | Token it easy - LAB7 | 0.96363 | No complete public solution found on inspected member profiles. |
| 5 | Gray Robertson | 0.96276 | Public comment corrects an alleged reversed-rating correlation; no complete solution found. |
| 6 | Aurrog | 0.96262 | No complete S6E10 solution found. |
| 7 | Fritz Cremer | 0.96254 | No S6E10 code match or Airline discussion match found. |
| 8 | Prior | 0.96252 | No complete S6E10 solution found on inspected member profiles. Team members' affiliation is not evidence of the model used. |
| 9 | srikalyan | 0.96238 | No public notebook found. |
| 10 | Ole-Jakob | 0.96236 | Public discussion describes experiment-record management in a previous competition; no current complete solution found. |

Absence from these searches is not proof that no public material exists elsewhere. Do not assign architectures or probing to teams without evidence.

Masaya's direct comment: [discussion 753288](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/753288). He separates probing from CV-based modeling. This is a statement about his own experiment, not an allegation about other entrants.

Our best score remains E4 **0.95856**; latest E6 **0.95855**. User rank at latest check: **1988**. Ranks change during the competition.

## Stronger reproducible candidates outside the top ten

### RealMLP: highest priority for a new independently trained neural baseline

[Vladimir Demidov's original notebook](https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit), saved run approximately 597 seconds, public score **0.96115**, printed OOF **0.96126**. These are different metrics/splits, not directly comparable to our held-out report AUC.

Inspected source uses categorical copies of numeric columns, counts and selected category interactions. TargetEncoder.fit_transform runs inside each outer training fold, with transform on outer validation/test. RealMLP uses 8 networks, 3 epochs, batch 288, flat_anneal schedule, widths 512/256/128, learning rate 0.053 and weight decay 0.015. This is substantially different from our generic quantile-transformed ResNet. Editor-source inspection subsequently confirmed params is a module-level variable; the rendered snapshot had lost indentation. There is no configuration-name bug in the original source.

### Distance-group statistics and an original-data teacher

[Executable LightGBM notebook](https://www.kaggle.com/code/mathsonchain/s6e10-lgbm-trasy): saved public **0.96052**, printed five-fold CV **0.96068**, 326 seconds. Source groups by exact Flight Distance for label-free means/counts, trains a teacher only on 129,880 original-data rows, and adds teacher probabilities plus a smoothed original distance satisfaction rate. Both original CSVs are visibly attached in this saved run.

Its label-free aggregation uses train, test and original features together: transductive feature construction, not competition-label leakage. Our clean comparison should also test fitting aggregates on outer-train only and document the choice. This notebook early-stops on the scored fold, so its OOF estimate has selection optimism.

[Author's detailed ablations](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/752304) report route features as a major gain; concatenating original rows was less useful than teacher transfer. They also report that extra route features can hurt an already strong RealMLP, so gains should not be assumed additive. Their linked realmlp-v4 notebook was unavailable when opened during this audit; the original yekenot source above was accessible.

### Foundation models and stacking

[Chris Deotte's benchmark](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/746363) reports full-context 5-fold OOF: TabPFN3.5 0.961145, Kumo large 0.960036, TabICLv2 0.958088. Full context is approximately 560k rows per fold, using an A100 offline; this is not a promise that the same recipe fits Kaggle T4 memory. Our TabICL experiment used a small context and is not evidence against all pretrained tabular models.

[Fine-tuning results](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/749010) report TabPFN3.5 0.961248 and Kumo large 0.960378. The author notes that improved standalone scores did not necessarily improve an already diverse stack.

[38-member comparison](https://www.kaggle.com/competitions/playground-series-s6e10/discussion/752852) reports meta-OOF 0.962021 and public 0.96169 using regularized logistic regression on logits. These reuse public predictions; training cost/provenance and inherited base-model selection bias remain relevant.

## Concrete cautions found in public material

[15-fold pipeline, saved version 4](https://www.kaggle.com/code/jiatianzhang2/s6e10-15-fold-gpu-stacking-pipeline):

- Narrative promotes 0.962376 and a teacher prior, but saved execution says original data were missing and uses Teacher_Score=0.5. The printed final score is 0.961729, not the advertised number.
- Logistic regression is fitted on S_oof and scored on those same rows. This is in-sample meta evaluation, not held-out meta-OOF.
- Target encoding is precomputed across all outer folds before base-model training. Although each row's encoding excludes its own fold, training-row encodings can incorporate the current outer validation fold's labels. Proper outer validation requires rebuilding these encodings wholly inside outer-train, with inner cross-fitting for its training rows.
- Some printed deltas are hard-coded and do not match the displayed scores. A larger fold count does not establish causality for the headline gain.

These are code-review findings, not accusations of misconduct. They show why a public title or narrative cannot substitute for execution evidence.

## Corrections needed in our own workflow

- We spent several experiments tuning/blending a baseline without testing distance-group features, original-data teachers or the public RealMLP recipe.
- E4 XGBoost fixes max_bin=256. Testing finer bins is sensible for distance resolution, but public ablations show this can be redundant with route features.
- Our engineered rating means include zero as a low value. Test separate zero indicators and nonzero-only summaries; preserve raw ratings and validate rather than impose survey semantics globally.
- E5 used a validation-driven learning-rate scheduler for fold training but fixed-epoch full refits without that scheduler. E6 used the consistent fold predictions; nevertheless this mismatch should be eliminated in future training code.
- E1–E6 reuse the same selection/report partitions. The report is no longer a fresh independent confirmation set. Freeze a full five-fold protocol, inner stopping and separate meta evaluation before the next search. Public-OOF stack estimates remain conditional unless all base learners are retrained inside the meta protocol.

## Recommended order, all compute on Kaggle

1. Reproduce the original RealMLP recipe, save ID-aligned raw OOF and fold-average test predictions, and establish a clean score.
2. Independently test LightGBM with distance-group features and an original-data teacher; include zero-aware aggregates as a separate ablation.
3. Add TabPFN/Kumo only with an explicit Kaggle memory plan, or use clearly credited, audited public raw OOF/test pairs as a separately labeled experiment.
4. Combine only validated, complementary models with a regularized logit stack and held-out meta evaluation. Avoid public-LB weight searching.

There is evidence for substantially better baselines than ours. There is no evidence that merely increasing depth, epochs or GPU hours will reproduce the top-ten leaderboard scores.
