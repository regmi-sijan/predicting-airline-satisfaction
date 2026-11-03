# Airline Satisfaction — Experiment 1

This experiment uses the official frozen **TabICLv2** pretrained classifier directly. It does not copy public submissions, fit transformer weights, or use external passenger labels. A LightGBM classifier is the control and potential blend partner.

## Sources

- [Official TabICLv2 implementation](https://github.com/soda-inria/tabicl)
- [Official checkpoint and BSD-3-Clause model license](https://huggingface.co/jingang/TabICL)
- [TabICLv2 paper](https://arxiv.org/abs/2602.11139)
- [Competition overview](https://www.kaggle.com/competitions/playground-series-s6e10/overview)
- [Competition data](https://www.kaggle.com/competitions/playground-series-s6e10/data)

Public TabPFN notebooks were inspected for feasibility. One direct run (https://www.kaggle.com/code/mehrankazeminia/1-airline-s6e10-tabpfn) reports over 11 hours on two T4 GPUs. No code, features, fitted estimators, or prediction files were copied from it. Latest TabPFN weight licensing is more restrictive than TabICLv2; this experiment uses the openly licensed alternative. TabICLv2 is a strong relevant candidate, not a claim that it is the best model on this competition.

## Data and validation

Official files: 699,635 training passengers, 299,844 test passengers, 21 predictors plus ID and target. Training satisfaction is 44.3573% positive. Arrival delay has 292 missing training values. `id` is excluded from both models.

Fixed seed-42 stratified split:

- 599,635 rows: base-model training pool.
- 50,000 rows: select among TabICL weights 0, .25, .5, .75, 1.
- 50,000 rows: report performance only after selecting the blend.

LightGBM early stopping uses a further 15% split inside the training pool. It then refits on the full pool with the selected iteration count. TabICLv2 uses a stratified 16,384-row context sampled only from that pool, with two fixed ensemble permutations, KV caching, and the official `tabicl-classifier-v2-20260212.ckpt` weights. The pretrained model handles its categorical/missing-value preprocessing internally. LightGBM category levels are learned from training rows; unseen categories become missing.

No reported holdout label trains base models or chooses early stopping. No test label is used. Reported performance is a single holdout estimate, not cross-validation or a Kaggle public score. The frozen recipe refits on the full labeled dataset for deployment (TabICL resamples its context); this changes the final fitted models.

## Reproduce locally

From the Kaggle workspace root, with the checkpoint and official CSVs downloaded:

```sh
.venv/bin/python airline_satisfaction/experiments/v1/run_experiment.py
```

`environment-lock.txt` records the local runtime. The self-contained `airline_experiment1.ipynb` downloads the official checkpoint and runs on Kaggle CPU after attaching competition data and enabling Internet. It creates submission files and does not submit automatically.

Outputs are under `results/`: data audit and file/checkpoint/source hashes, partition IDs, untouched reporting metrics, validation predictions, context IDs, fitted LightGBM, and three submission CSVs. Cached transformer batches use a fingerprint of source, data, model weights, parameters, and library versions; invalid-length/nonfinite batches fail rather than silently continue.

## Limitations

The context subset is a deliberate compute budget on this CPU laptop. It does not use all 699,635 passengers as transformer context. A larger GPU context and more ensemble members may help but have not been established by this experiment. Public leaderboard evaluation covers about 20% of test passengers; final ranking uses the remaining 80%. Avoid selecting future recipes solely on tiny public-score changes.
