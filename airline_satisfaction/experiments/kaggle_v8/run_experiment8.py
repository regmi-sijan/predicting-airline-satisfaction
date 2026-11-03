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


"""E8 additions to the E7 RealMLP recipe. Kaggle GPU only.
Credits: yekenot's RealMLP and mathsonchain's original-teacher/route ideas.
https://www.kaggle.com/code/yekenot/ps-s6-e10-realmlp-pytabkit
https://www.kaggle.com/code/mathsonchain/s6e10-lgbm-trasy
"""
from scipy.special import expit, logit

CATS=['Gender','Customer Type','Type of Travel','Class']
RATINGS=['Inflight wifi service','Departure/Arrival time convenient',
 'Ease of Online booking','Gate location','Food and drink','Online boarding',
 'Seat comfort','Inflight entertainment','On-board service','Leg room service',
 'Baggage handling','Checkin service','Cleanliness']
OUT=Path('/kaggle/working/experiment8')


def category_align(a, others):
    frames=[a.copy()]+[b.copy() for b in others]
    for c in CATS:
        levels=pd.Index(frames[0][c].dropna().unique())
        for b in frames: b[c]=pd.Categorical(b[c],categories=levels)
    return frames


def external_teacher(x,xt,comp):
    paths=[p for p in Path('/kaggle/input').rglob('*.csv')
           if 'airline-passenger-satisfaction' in str(p) and p.parent!=comp]
    assert paths, 'Attach teejmahal20/airline-passenger-satisfaction input.'
    pieces=[]
    for p in paths:
        df=pd.read_csv(p)
        if set(x.columns).union({'satisfaction'}).issubset(df.columns):
            pieces.append(df[list(x.columns)+['satisfaction']])
    assert pieces, 'No original-data CSVs with required shared features.'
    orig=pd.concat(pieces,ignore_index=True)
    assert orig.satisfaction.notna().all()
    orig['satisfaction']=orig.satisfaction.map({'satisfied':1,'neutral or dissatisfied':0})
    assert orig.satisfaction.notna().all(), 'Unexpected original label encoding.'
    # Exclude exact feature matches with competition rows, without using their labels.
    full=pd.concat([x,xt],ignore_index=True)
    blocked=pd.util.hash_pandas_object(full,index=False)
    keys=pd.util.hash_pandas_object(orig[x.columns],index=False)
    keep=~keys.isin(blocked)
    audit=dict(original_rows=len(orig),exact_overlap_removed=int((~keep).sum()),
               sources=[str(p) for p in paths],
               source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    orig=orig.loc[keep].drop_duplicates().reset_index(drop=True)
    assert len(orig)>100000
    ox,xx,tx=category_align(orig[x.columns],[x,xt])
    teacher=lgb.LGBMClassifier(n_estimators=800,learning_rate=.04,num_leaves=31,
        min_child_samples=30,reg_lambda=3,random_state=42,n_jobs=4,verbosity=-1,
        deterministic=True,force_col_wise=True)
    teacher.fit(ox,orig.satisfaction.astype(int))
    pp=teacher.predict_proba(xx)[:,1];pt=teacher.predict_proba(tx)[:,1]
    teacher.booster_.save_model(str(OUT/'original_teacher.txt'))
    rate=orig.groupby('Flight Distance').satisfaction.agg(['sum','count'])
    prior=float(orig.satisfaction.mean())
    rate=(rate['sum']+20*prior)/(rate['count']+20)
    joblib.dump(dict(rate=rate,prior=prior),OUT/'original_distance_rate.joblib')
    audit['retained_rows']=len(orig)
    (OUT/'teacher_audit.json').write_text(json.dumps(audit,indent=2))
    print('EXTERNAL TEACHER READY',json.dumps(audit),flush=True)
    return pp,pt,rate,prior


def route_view(train,others,teacher_probs,rate,prior):
    g=train.groupby('Flight Distance')
    agg=g[RATINGS+['Age']].mean().add_prefix('route_mean_')
    agg['route_count']=g.size()
    fallback=agg.mean();fallback['route_count']=0
    joblib.dump(dict(aggregates=agg,fallback=fallback),OUT/'current_route_maps.joblib')
    frames=[]
    for df,p in zip([train]+others,teacher_probs):
        a=df.copy().reset_index(drop=True)
        features=agg.reindex(a['Flight Distance']).reset_index(drop=True).fillna(fallback)
        a=pd.concat([a,features],axis=1)
        a['teacher_probability']=p
        a['original_route_rate']=a['Flight Distance'].map(rate).fillna(prior)
        nz=a[RATINGS].replace(0,np.nan)
        a['rating_zero_count']=(a[RATINGS]==0).sum(axis=1)
        a['rating_nonzero_mean']=nz.mean(axis=1)
        a['rating_nonzero_std']=nz.std(axis=1)
        frames.append(a)
    return category_align(frames[0],frames[1:])


def export_neural(model,v,expected,name):
    model.to('cpu')
    try:
        path=OUT/(name+'.cloudpickle')
        with path.open('wb') as f:cloudpickle.dump(model,f)
        with path.open('rb') as f:restored=cloudpickle.load(f)
        np.testing.assert_allclose(restored.predict_proba(v.iloc[:16])[:,1],expected[:16],rtol=1e-4,atol=1e-5)
        del restored
        print('MODEL EXPORT VERIFIED',name,flush=True)
    except Exception as exc:
        (OUT/(name+'_export_warning.json')).write_text(json.dumps(dict(error=repr(exc))))
        print('MODEL EXPORT WARNING',name,repr(exc),flush=True)


def blend(p,w,space):
    return p@w if space=='probability' else expit(logit(np.clip(p,1e-6,1-1e-6))@w)


def choose(p,y):
    trials=[]
    for i in range(5):
        for j in range(5-i):
            w=np.array([i,j,4-i-j])/4
            for space in ['probability','logit']:
                trials.append(dict(weights=w.tolist(),space=space,
                                   auc=float(roc_auc_score(y,blend(p,w,space)))))
    return max(trials,key=lambda t:t['auc']),trials


def main8():
    assert Path('/kaggle/working').exists() and torch.cuda.is_available()
    OUT.mkdir(exist_ok=True);start=time.time()
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    comp=next(p.parent for p in Path('/kaggle/input').rglob('sample_submission.csv') if 's6e10' in str(p))
    tr=pd.read_csv(comp/'train.csv');te=pd.read_csv(comp/'test.csv');sample=pd.read_csv(comp/'sample_submission.csv')
    assert tr.shape==(699635,23) and te.shape==(299844,22)
    assert np.array_equal(sample.id,te.id)
    y=tr.satisfaction.astype(int).to_numpy();x=tr.drop(columns=['id','satisfaction']);xt=te.drop(columns='id')
    pp,pt,rate,prior=external_teacher(x,xt,comp)
    np.savez_compressed(OUT/'teacher_predictions.npz',train=pp,test=pt)
    names=['realmlp_e7','realmlp_teacher','route_tree'];oof=np.full((len(x),3),np.nan);tp=np.zeros((len(xt),3))
    folds=np.full(len(x),-1);rows=[]
    (OUT/'manifest.json').write_text(json.dumps(dict(realmlp=PARAMS,
        versions={p:version(p) for p in ['torch','pytabkit','scikit-learn','lightgbm','cloudpickle']},
        protocol='E7 same 5 folds seed42; 3 epochs; teacher only original-data labels; outer-train-only routes and encoders; no public-LB weight tuning',
        reference_public_auc=.96115,credits=__doc__),indent=2))
    for fold,(ti,vi) in enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(x,y)):
        tic=time.time();folds[vi]=fold
        print('FOLD START',fold+1,flush=True)
        frames,prep=prepare(x.iloc[ti],[x.iloc[vi],xt],y[ti])
        joblib.dump(prep,OUT/f'preprocess_fold{fold}.joblib')
        for col,name in enumerate(names[:2]):
            a,v,t=[f.copy() for f in frames]
            if col==1:
                # Continuous teacher score only: no enormous categorical twin.
                for df,p in zip([a,v,t],[pp[ti],pp[vi],pt]):df['teacher_probability']=p
            model=RealMLP_TD_Classifier(**PARAMS)
            model.fit(a,y[ti],val_idxs=np.array([],dtype=np.int64))
            oof[vi,col]=predict_batched(model,v);testp=predict_batched(model,t);tp[:,col]+=testp/5
            np.savez_compressed(OUT/f'{name}_fold{fold}_predictions.npz',rows=vi,ids=tr.id.to_numpy()[vi],validation=oof[vi,col],test=testp)
            print('NEURAL COMPLETE',fold+1,name,metrics(y[vi],oof[vi,col]),flush=True)
            export_neural(model,v,oof[vi,col],f'{name}_fold{fold}')
            del model,a,v,t;gc.collect();torch.cuda.empty_cache()
        del frames,prep;gc.collect()
        a,v,t=route_view(x.iloc[ti],[x.iloc[vi],xt],[pp[ti],pp[vi],pt],rate,prior)
        # Fixed recipe: no scored-fold labels used for stopping or feature selection.
        tree=lgb.LGBMClassifier(n_estimators=700,learning_rate=.035,num_leaves=63,
            min_child_samples=80,reg_lambda=5,colsample_bytree=.85,
            subsample=.8,subsample_freq=1,random_state=42,n_jobs=4,
            verbosity=-1,deterministic=True,force_col_wise=True)
        tree.fit(a,y[ti]);oof[vi,2]=tree.predict_proba(v)[:,1];testp=tree.predict_proba(t)[:,1];tp[:,2]+=testp/5
        tree.booster_.save_model(str(OUT/f'route_tree_fold{fold}.txt'))
        # Preserve the fold-specific route maps for exact reconstruction.
        (OUT/f'route_maps_fold{fold}.joblib').write_bytes((OUT/'current_route_maps.joblib').read_bytes())
        np.savez_compressed(OUT/f'route_tree_fold{fold}_predictions.npz',rows=vi,validation=oof[vi,2],test=testp)
        row=dict(fold=fold,seconds=time.time()-tic,**{n:metrics(y[vi],oof[vi,c]) for c,n in enumerate(names)})
        rows.append(row);print('FOLD COMPLETE',json.dumps(row),flush=True)
        frame=pd.DataFrame(oof,columns=names);frame.insert(0,'fold',folds);frame.insert(0,'label',y);frame.insert(0,'id',tr.id)
        frame.to_csv(OUT/'oof_predictions.csv',index=False)
        (OUT/'fold_metrics.json').write_text(json.dumps(rows,indent=2))
        del a,v,t,tree;gc.collect()
    assert np.isfinite(oof).all() and np.isfinite(tp).all() and ((tp>=0)&(tp<=1)).all()
    best,trials=choose(oof,y);meta=np.zeros(len(y));meta_choices=[]
    for a,b in StratifiedKFold(5,shuffle=True,random_state=20261030).split(oof,y):
        choice,_=choose(oof[a],y[a]);meta[b]=blend(oof[b],choice['weights'],choice['space']);meta_choices.append(choice)
    # Conditional meta evaluation: base OOF library fixed, not fully nested retraining.
    selected=blend(tp,best['weights'],best['space'])
    for name,p in list(zip(names,tp.T))+[('selected',selected)]:
        sub=sample.copy();sub['satisfaction']=p;assert sub.satisfaction.notna().all()
        sub.to_csv(OUT/f'submission_{name}.csv',index=False)
    result=dict(models={n:metrics(y,oof[:,c]) for c,n in enumerate(names)},
        teacher_delta=float(roc_auc_score(y,oof[:,1])-roc_auc_score(y,oof[:,0])),
        blend_selection=best,conditional_meta_metrics=metrics(y,meta),
        conditional_meta_choices=meta_choices,fold_metrics=rows,seconds=time.time()-start,
        caveat='Blend selection AUC is tuned on these OOF labels; conditional meta estimate reuses a fixed base library and is not fully nested. Prior competition experiments used these data. No automatic submission.',
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('submission_*.csv')})
    (OUT/'blend_trials.json').write_text(json.dumps(trials,indent=2))
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print('EXPERIMENT 8 COMPLETE',json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main8()
