"""E7: fold-safe RealMLP recipe; execute only on Kaggle GPU.

Recipe credit: Vladimir Demidov (yekenot), Apache-2.0 Kaggle notebook
https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit
Changes: outer-train-only preprocessing, fixed epochs without scored-fold
labels, model artifacts, complete OOF/test predictions, matched LGB baseline.
"""
import gc
import hashlib
import json
import os
import platform
import random
import time
from importlib.metadata import version
from pathlib import Path

import joblib
import cloudpickle
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from pytabkit import RealMLP_TD_Classifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

SEED = 42
OUT = Path('/kaggle/working/experiment7')
COMBOS = [('Class', 'Type of Travel', 'Gender'),
          ('Class', 'Type of Travel', 'Age'),
          ('Food and drink', 'Cleanliness'),
          ('Flight Distance', 'Departure/Arrival time convenient')]
PARAMS = dict(random_state=42, verbosity=1, device='cuda', n_threads=4,
              val_metric_name='1-auc_ovr', n_ens=8, n_epochs=3,
              batch_size=288, use_early_stopping=False, lr=0.053, wd=0.015,
              sq_mom=0.988, lr_sched='flat_anneal', wd_sched='cos_log_15',
              first_layer_lr_factor=0.25, embedding_size=5,
              max_one_hot_cat_size=2, hidden_sizes=[512, 256, 128],
              act='silu', p_drop=0.05, p_drop_sched='pow5',
              plr_hidden_1=16, plr_hidden_2=8, plr_act_name='gelu',
              plr_lr_factor=0.1151, plr_sigma=0.75,
              ls_eps=0.01, ls_eps_sched='sqrt_cos', add_front_scale=False,
              bias_init_mode='neg-uniform-dynamic-2',
              tfms=['one_hot', 'median_center', 'robust_scale',
                    'smooth_clip', 'embedding', 'l2_normalize'])


def metrics(y, p):
    return dict(auc=float(roc_auc_score(y, p)),
                log_loss=float(log_loss(y, p)))


def prepare(train, others, y):
    """Fit all category/count/target maps within this outer training fold."""
    cats = train.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    nums = [c for c in train if c not in cats]
    frames = [train.copy().reset_index(drop=True)] + [a.copy().reset_index(drop=True) for a in others]
    for a in frames:
        for c in cats:
            a[c] = a[c].astype('string').fillna('__missing__').astype(str)
        for c in nums:
            a[c] = a[c].fillna(0.)
    maps = {}
    for c in cats:
        levels = pd.Index(frames[0][c].unique())
        maps[c] = levels
        for a in frames:
            a[c] = pd.Categorical(a[c], categories=levels)
    for c in nums:
        levels = pd.Index(frames[0][c].unique())
        name = c + '_cat_'
        maps[name] = levels
        for a in frames:
            a[name] = pd.Categorical(a[c], categories=levels)
    for c in cats + ['Leg room service', 'Flight Distance']:
        counts = frames[0][c].value_counts()
        name = '_' + c + '_count'
        maps[name] = counts
        for a in frames:
            a[name] = a[c].astype(object).map(counts).fillna(0).astype('float32')
    names = []
    for cols in COMBOS:
        name = '_'.join(cols) + '_'
        names.append(name)
        values = [a[list(cols)].astype(str).agg('_'.join, axis=1) for a in frames]
        levels = pd.Index(values[0].unique())
        maps[name] = levels
        for a, v in zip(frames, values):
            a[name] = pd.Categorical(v, categories=levels)
    # Strings avoid inconsistent ordinal codes between train and unknown categories.
    te = TargetEncoder(cv=5, smooth='auto', shuffle=True, random_state=SEED)
    te_input = [a[names].astype(str) for a in frames]
    encoded = [te.fit_transform(te_input[0], y)] + [te.transform(a) for a in te_input[1:]]
    drop = ['Class_Type of Travel_Age_', 'Flight Distance_Departure/Arrival time convenient_']
    for a, enc in zip(frames, encoded):
        a[[f'_{c}TE' for c in names]] = enc
        a.drop(columns=drop, inplace=True)
    return frames, dict(maps=maps, target_encoder=te, combos=COMBOS,
                        columns=frames[0].columns.tolist())


def predict_batched(model, x):
    return np.concatenate([model.predict_proba(x.iloc[i:i+20000])[:, 1]
                           for i in range(0, len(x), 20000)])


def main():
    assert Path('/kaggle/working').exists(), 'Run this on Kaggle, not locally.'
    assert torch.cuda.is_available(), 'Enable Kaggle GPU before running.'
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    files = list(Path('/kaggle/input').rglob('sample_submission.csv'))
    comp = next(p.parent for p in files if 's6e10' in str(p))
    train = pd.read_csv(comp/'train.csv'); test = pd.read_csv(comp/'test.csv')
    sample = pd.read_csv(comp/'sample_submission.csv')
    assert train.shape == (699635, 23) and test.shape == (299844, 22)
    assert np.array_equal(test.id, sample.id)
    y = train.satisfaction.astype(int).to_numpy()
    x = train.drop(columns=['id', 'satisfaction']); xt = test.drop(columns='id')
    oof = np.full(len(y), np.nan); baseline = np.full(len(y), np.nan)
    folds = np.full(len(y), -1); tp = np.zeros(len(test)); bp = np.zeros(len(test))
    rows = []; start = time.time()
    manifest = dict(recipe=PARAMS, source=__doc__, platform=platform.platform(),
                    versions={p: version(p) for p in ['torch','pytabkit','scikit-learn','lightgbm']},
                    gpu=torch.cuda.get_device_name(0),
                    protocol='5-fold seed42; fixed 3 epochs; no outer validation labels passed to training; inner-cross-fitted TE; outer-train preprocessing',
                    prior_public_best=0.95856,
                    evaluation_note='Protocol fixed before E7; not an untouched test after previous experiments on these data.')
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2))
    for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(x, y)):
        tic = time.time(); folds[va] = fold
        print(f'FOLD {fold+1}/5 START | train={len(tr)} valid={len(va)}', flush=True)
        frames, prep = prepare(x.iloc[tr], [x.iloc[va], xt], y[tr])
        a, v, t = frames
        joblib.dump(prep, OUT/f'preprocess_fold{fold}.joblib')
        model = RealMLP_TD_Classifier(**PARAMS)
        # Empty validation indices train on all outer-train at a fixed epoch budget.
        model.fit(a, y[tr], val_idxs=np.array([], dtype=np.int64))
        oof[va] = predict_batched(model, v)
        test_fold = predict_batched(model, t); tp += test_fold / 5
        np.save(OUT/f'test_realmlp_fold{fold}.npy', test_fold)
        np.savez_compressed(OUT/f'validation_realmlp_fold{fold}.npz',
                            rows=va, ids=train.id.to_numpy()[va],
                            labels=y[va], predictions=oof[va])
        print('NEURAL FOLD', fold+1, metrics(y[va],oof[va]), flush=True)
        # Standard pickle/joblib cannot serialize PyTabKit's schedule lambdas.
        # Export is supplementary: preserve predictions even if export fails.
        model.to('cpu')
        export_path=OUT/f'realmlp_fold{fold}.cloudpickle'
        try:
            with export_path.open('wb') as stream:
                cloudpickle.dump(model, stream)
            with export_path.open('rb') as stream:
                restored=cloudpickle.load(stream)
            restored_pred=restored.predict_proba(v.iloc[:16])[:,1]
            np.testing.assert_allclose(restored_pred,oof[va][:16],rtol=1e-4,atol=1e-5)
            del restored
            print('MODEL EXPORT VERIFIED',str(export_path),flush=True)
        except Exception as exc:
            warning=dict(fold=fold,error=repr(exc),
                         note='Prediction artifacts saved; model export not verified.')
            (OUT/f'model_export_warning_fold{fold}.json').write_text(json.dumps(warning,indent=2))
            print('MODEL EXPORT WARNING',warning,flush=True)
        del model, a, v, t, frames, prep; gc.collect(); torch.cuda.empty_cache()
        # Matched-fold reference: E3's fixed LightGBM recipe, no validation tuning.
        aa=x.iloc[tr].copy(); vv=x.iloc[va].copy(); tt=xt.copy()
        for c in aa.select_dtypes(include='object').columns:
            levels=pd.Index(aa[c].dropna().unique())
            for df in [aa,vv,tt]: df[c]=pd.Categorical(df[c],categories=levels)
        ref=lgb.LGBMClassifier(n_estimators=626, learning_rate=.035,
            num_leaves=63, min_child_samples=120, reg_lambda=8,
            colsample_bytree=.85, subsample=.8, subsample_freq=1,
            n_jobs=4, random_state=SEED, verbosity=-1, deterministic=True,
            force_col_wise=True)
        ref.fit(aa,y[tr]); baseline[va]=ref.predict_proba(vv)[:,1]
        bp += ref.predict_proba(tt)[:,1]/5
        ref.booster_.save_model(str(OUT/f'lightgbm_fold{fold}.txt'))
        row=dict(fold=fold, realmlp=metrics(y[va],oof[va]),
                 lightgbm=metrics(y[va],baseline[va]), seconds=time.time()-tic)
        rows.append(row); print('FOLD COMPLETE',json.dumps(row),flush=True)
        pd.DataFrame(dict(id=train.id,label=y,fold=folds,realmlp=oof,lightgbm=baseline)).to_csv(OUT/'oof_predictions.csv',index=False)
        (OUT/'fold_metrics.json').write_text(json.dumps(rows,indent=2))
        del aa,vv,tt,ref; gc.collect()
    assert np.isfinite(oof).all() and np.isfinite(tp).all()
    assert ((tp>=0)&(tp<=1)).all() and (folds>=0).all()
    for name,p in [('realmlp',tp),('lightgbm',bp)]:
        sub=sample.copy(); sub['satisfaction']=p
        sub.to_csv(OUT/f'submission_{name}.csv',index=False)
    result=dict(realmlp=metrics(y,oof), lightgbm=metrics(y,baseline),
                auc_delta=float(roc_auc_score(y,oof)-roc_auc_score(y,baseline)),
                fold_metrics=rows, seconds=time.time()-start,
                selected_submission='submission_realmlp.csv',
                selection_rule='Predeclared neural experiment; no weight search or leaderboard tuning.',
                caveat='OOF scores differ from E1-E6 held-out report protocol; compare matched reference above. No automatic competition submission.')
    result['submission_sha256']=hashlib.sha256((OUT/'submission_realmlp.csv').read_bytes()).hexdigest()
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print('EXPERIMENT 7 COMPLETE',json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
