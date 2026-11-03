"""E4: nested early stopping, five-fold model search, GPU ensembles. Kaggle only."""
from pathlib import Path
import os,json,time,gc,hashlib,importlib.metadata,itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import roc_auc_score,log_loss
def feature_frame(d):
    x=d.drop(columns=['id','satisfaction'],errors='ignore').copy()
    ratings=['Inflight wifi service','Departure/Arrival time convenient','Ease of Online booking','Gate location','Food and drink','Online boarding','Seat comfort','Inflight entertainment','On-board service','Leg room service','Baggage handling','Checkin service','Cleanliness']
    r=x[ratings]
    x['rating_mean']=r.mean(axis=1);x['rating_std']=r.std(axis=1)
    x['rating_min']=r.min(axis=1);x['rating_max']=r.max(axis=1)
    x['rating_low_count']=(r<=2).sum(axis=1);x['rating_high_count']=(r>=4).sum(axis=1);x['rating_zero_count']=(r==0).sum(axis=1)
    x['digital_mean']=x[['Inflight wifi service','Ease of Online booking','Online boarding']].mean(axis=1)
    x['cabin_mean']=x[['Seat comfort','Inflight entertainment','Leg room service','Cleanliness']].mean(axis=1)
    x['wifi_boarding_gap']=x['Inflight wifi service']-x['Online boarding']
    x['seat_entertainment_gap']=x['Seat comfort']-x['Inflight entertainment']
    dep=x['Departure Delay in Minutes'];arr=x['Arrival Delay in Minutes']
    x['arrival_missing']=arr.isna().astype('int8');x['delay_change']=arr-dep
    x['departure_log']=np.log1p(dep.clip(lower=0));x['arrival_log']=np.log1p(arr.clip(lower=0))
    x['distance_log']=np.log1p(x['Flight Distance'])
    x['travel_segment']=x['Class'].astype(str)+'|'+x['Type of Travel'].astype(str)+'|'+x['Customer Type'].astype(str)
    return x

def encode(a,others,name):
    a=a.copy();others=[b.copy() for b in others]
    cats=list(a.select_dtypes(include=['object','string']).columns)
    if name=='catboost':
        for c in cats:
            a[c]=a[c].fillna('Missing').astype(str)
            for b in others:b[c]=b[c].fillna('Missing').astype(str)
    else:
        for c in cats:
            levels=sorted(a[c].dropna().unique().tolist())
            a[c]=pd.Categorical(a[c],categories=levels)
            for b in others:b[c]=pd.Categorical(b[c],categories=levels)
    # XGBoost prohibits these characters in feature names; original names currently have none.
    safe={c:c.replace('[','_').replace(']','_').replace('<','_') for c in a.columns}
    a=a.rename(columns=safe);others=[b.rename(columns=safe) for b in others]
    return a,others,[safe[c] for c in cats]


SEEDS=[42,2026,20261028]
FAMILIES=['lightgbm','xgboost','catboost']
def score(y,p):return dict(auc=float(roc_auc_score(y,p)),log_loss=float(log_loss(y,p)))
def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def specs():
    rows=[]
    for feat,leaves,minimum,reg in [('original',31,180,15),('original',63,120,8),('original',127,100,12),('engineered',127,150,20)]:
        rows.append(dict(family='lightgbm',features=feat,depth=leaves,minimum=minimum,reg=reg))
    for feat,depth,minimum,reg in [('original',4,20,12),('original',6,25,12),('original',8,35,20),('engineered',8,50,25)]:
        rows.append(dict(family='xgboost',features=feat,depth=depth,minimum=minimum,reg=reg))
    for feat,depth,reg in [('original',6,5),('original',8,10),('original',10,20),('engineered',9,20)]:
        rows.append(dict(family='catboost',features=feat,depth=depth,minimum=0,reg=reg))
    for i,c in enumerate(rows):c['name']=f"{c['family']}_{i:02d}_{c['features']}"
    return rows
def make(c,seed,trees=6000,early=False):
    f=c['family']
    if f=='lightgbm':
        import lightgbm as lgb
        return lgb.LGBMClassifier(objective='binary',metric='auc',n_estimators=trees,learning_rate=.02,
            num_leaves=c['depth'],min_child_samples=c['minimum'],reg_lambda=c['reg'],
            colsample_bytree=.85,subsample=.8,subsample_freq=1,n_jobs=4,random_state=seed,
            verbosity=-1,deterministic=True,force_col_wise=True)
    if f=='xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(objective='binary:logistic',eval_metric='auc',n_estimators=trees,
            learning_rate=.025,max_depth=c['depth'],min_child_weight=c['minimum'],reg_lambda=c['reg'],
            reg_alpha=.1,max_bin=256,subsample=.85,colsample_bytree=.85,tree_method='hist',
            device='cuda:0',enable_categorical=True,n_jobs=4,random_state=seed,
            **({'early_stopping_rounds':200} if early else {}))
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=trees,depth=c['depth'],learning_rate=.03,l2_leaf_reg=c['reg'],
        loss_function='Logloss',eval_metric='AUC',task_type='GPU',devices='0',random_seed=seed,
        thread_count=4,allow_writing_files=False,verbose=1000,**({'od_type':'Iter','od_wait':200} if early else {}))
def fit(m,c,x,y,v=None,vy=None):
    f=c['family']
    if f=='lightgbm':
        import lightgbm as lgb
        kw={} if v is None else dict(eval_set=[(v,vy)],eval_metric='auc',callbacks=[lgb.early_stopping(200,first_metric_only=True,verbose=False),lgb.log_evaluation(1000)])
        m.fit(x,y,**kw)
    elif f=='xgboost':m.fit(x,y,**({} if v is None else dict(eval_set=[(v,vy)],verbose=1000)))
    else:m.fit(x,y,cat_features=list(x.select_dtypes(include=['object','string']).columns),**({} if v is None else dict(eval_set=(v,vy),use_best_model=True)))
    if f=='xgboost' and 'cuda' not in m.get_booster().save_config():raise RuntimeError('XGBoost GPU execution not confirmed')
    return m

def main():
    if not Path('/kaggle/working').is_dir():raise RuntimeError('Kaggle only: no laptop training')
    import torch
    assert torch.cuda.is_available(),'Enable Kaggle GPU T4 x2'
    start=time.time();out=Path('/kaggle/working/experiment4');out.mkdir(exist_ok=True)
    paths=[p.parent for p in Path('/kaggle/input').rglob('train.csv') if 'playground-series-s6e10' in str(p) and (p.parent/'test.csv').exists()];assert len(paths)==1,paths
    root=paths[0];tr=pd.read_csv(root/'train.csv');te=pd.read_csv(root/'test.csv');sample=pd.read_csv(root/'sample_submission.csv')
    y=tr.satisfaction.astype(int).to_numpy();assert tr.id.is_unique and te.id.is_unique and np.array_equal(te.id,sample.id) and not set(tr.id)&set(te.id)
    pool,held=train_test_split(np.arange(len(y)),test_size=100000,random_state=42,stratify=y)
    selection,report=train_test_split(held,test_size=.5,random_state=2026,stratify=y[held]);sm=np.isin(held,selection)
    frames={'original':tr.drop(columns=['id','satisfaction']),'engineered':feature_frame(tr)}
    tests={'original':te.drop(columns=['id']),'engineered':feature_frame(te)};candidates=specs()
    cfg=dict(source=digest(__file__),train=digest(root/'train.csv'),test=digest(root/'test.csv'),specs=candidates,
        versions={p:importlib.metadata.version(p) for p in ['lightgbm','xgboost','catboost','numpy','pandas','scikit-learn']},folds=5,seeds=SEEDS)
    fingerprint=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()[:16];cache=out/f'cache_{fingerprint}';cache.mkdir(exist_ok=True)
    (out/'config.json').write_text(json.dumps(cfg,indent=2))
    print('HOST Kaggle; GPU',torch.cuda.get_device_name(0),'count',torch.cuda.device_count(),'DATA',tr.shape,te.shape,'configs',len(candidates),'folds',5,flush=True)
    folds=list(StratifiedKFold(5,shuffle=True,random_state=20261029).split(pool,y[pool]));search=[];preds={};tree_counts={}
    for c in candidates:
        name=c['name'];f=c['family'];x=frames[c['features']];xt=tests[c['features']];oof=np.zeros(len(pool));hp=np.zeros(len(held));tp=np.zeros(len(te));trees=[];foldrows=[]
        for k,(a,b) in enumerate(folds):
            ti,vi=pool[a],pool[b];path=cache/f'{name}_fold{k}.npz';print('START',name,'fold',k+1,flush=True)
            if path.exists():
                z=np.load(path);assert np.array_equal(z['ids'],tr.id.iloc[vi]);pv,ph,pt,n=z['val'],z['held'],z['test'],int(z['trees'])
            else:
                # Inner early stopping never sees this fold's outer validation labels.
                inner,stop=train_test_split(ti,test_size=.12,random_state=100+k,stratify=y[ti])
                aa,bb,_=encode(x.iloc[inner],[x.iloc[stop]],f);m=fit(make(c,42+k,early=True),c,aa,y[inner],bb[0],y[stop])
                n=int(m.best_iteration_ if f=='lightgbm' else m.best_iteration+1 if f=='xgboost' else m.tree_count_)
                del m,aa,bb;gc.collect();torch.cuda.empty_cache()
                aa,bb,_=encode(x.iloc[ti],[x.iloc[vi],x.iloc[held],xt],f);m=fit(make(c,42+k,trees=n),c,aa,y[ti])
                pv,ph,pt=[m.predict_proba(v)[:,1] for v in bb]
                np.savez_compressed(path,val=pv,held=ph,test=pt,trees=n,ids=tr.id.iloc[vi].to_numpy())
                del m,aa,bb;gc.collect();torch.cuda.empty_cache()
            for v in [pv,ph,pt]:assert np.isfinite(v).all() and ((v>=0)&(v<=1)).all()
            oof[b]=pv;hp+=ph/5;tp+=pt/5;trees.append(n);foldrows.append(score(y[vi],pv))
            print('DONE',name,'fold',k+1,'trees',n,foldrows[-1],'elapsed',round(time.time()-start),flush=True)
        row=dict(config=c,cv=score(y[pool],oof),folds=foldrows,iterations=trees);search.append(row);preds[name]=(hp,tp);tree_counts[name]=trees
        pd.DataFrame({'id':tr.id.iloc[pool].to_numpy(),'label':y[pool],'prediction':oof}).to_csv(out/f'{name}_oof.csv',index=False)
        (out/'search.json').write_text(json.dumps(search,indent=2));print('CANDIDATE',row,flush=True)
    # Recipe choice uses OOF CV; separate selection holdout chooses prediction strategy and blend.
    winners={f:max([r for r in search if r['config']['family']==f],key=lambda r:r['cv']['auc']) for f in FAMILIES}
    hp_options={};tp_options={};fullplans={}
    for f,row in winners.items():
        c=row['config'];name=c['name'];n=int(np.median(tree_counts[name]));x=frames[c['features']];hp_options[f+'_fold']=preds[name][0];tp_options[f+'_fold']=preds[name][1];seedh=[]
        for seed in SEEDS:
            path=cache/f'{name}_pool_seed{seed}.npy';print('POOL REFIT',name,seed,n,flush=True)
            if path.exists():ph=np.load(path)
            else:
                aa,bb,_=encode(x.iloc[pool],[x.iloc[held]],f);m=fit(make(c,seed,n),c,aa,y[pool]);ph=m.predict_proba(bb[0])[:,1];np.save(path,ph);del m,aa,bb;gc.collect();torch.cuda.empty_cache()
            seedh.append(ph)
        hp_options[f+'_seed']=np.mean(seedh,axis=0);fullplans[f]=(c,n)
    # Reproduce E3's selected recipe as the explicit baseline comparator.
    anchor=dict(family='lightgbm',features='original',depth=63,minimum=120,reg=8);anchorh=[]
    import lightgbm as lgb
    def anchor_model(seed):
        return lgb.LGBMClassifier(objective='binary',metric='auc',n_estimators=626,learning_rate=.035,num_leaves=63,min_child_samples=120,reg_lambda=8,colsample_bytree=.85,subsample=.8,subsample_freq=1,n_jobs=4,random_state=seed,verbosity=-1,deterministic=True,force_col_wise=True)
    for seed in SEEDS:
        aa,bb,_=encode(frames['original'].iloc[pool],[frames['original'].iloc[held]],'lightgbm');m=anchor_model(seed);m.fit(aa,y[pool]);anchorh.append(m.predict_proba(bb[0])[:,1]);del m,aa,bb;gc.collect()
    hp_options['e3_anchor']=np.mean(anchorh,axis=0)
    def combine(p,cols,w,method):
        a=np.column_stack([p[k] for k in cols])
        if method=='logit':
            a=np.log(np.clip(a,1e-7,1-1e-7)/np.clip(1-a,1e-7,1));z=np.clip(a@w,-30,30);return 1/(1+np.exp(-z))
        return a@w
    grid=[np.array([a,b,10-a-b])/10 for a in range(11) for b in range(11-a)]
    scored=[];small={k:v[sm] for k,v in hp_options.items()};ys=y[held][sm]
    for modes in itertools.product(['fold','seed'],repeat=3):
        cols=[f+'_'+m for f,m in zip(FAMILIES,modes)]
        for method in ['probability','logit']:
            for w in grid:
                auc=float(roc_auc_score(ys,combine(small,cols,w,method)));scored.append(dict(columns=cols,weights=w.tolist(),method=method,auc=auc))
    ensemble=max([r for r in scored if np.count_nonzero(r['weights'])>=2],key=lambda r:r['auc'])
    best=max(scored,key=lambda r:r['auc']);baseline=score(ys,small['e3_anchor']);use_anchor=baseline['auc']>best['auc']
    def prediction(p,r):return combine(p,r['columns'],np.array(r['weights']),r['method'])
    selectedh=hp_options['e3_anchor'] if use_anchor else prediction(hp_options,best)
    metric=dict(search=search,winners=winners,best_ensemble=ensemble,best_unconstrained=best,selected_is_e3_anchor=use_anchor,
        baseline_selection=baseline,selected_selection=score(ys,selectedh[sm]),
        report={k:score(y[held][~sm],v[~sm]) for k,v in hp_options.items()},
        selected_report=score(y[held][~sm],selectedh[~sm]),ensemble_report=score(y[held][~sm],prediction(hp_options,ensemble)[~sm]),
        note='Inner stopping splits exclude outer fold labels. Recipes selected by CV; strategies and weights by selection holdout. Report holdout reused from E1-E3, not a fresh independent estimate. No leaderboard tuning; no guarantee of improvement.')
    (out/'metrics.json').write_text(json.dumps(metric,indent=2));(out/'blend_candidates.json').write_text(json.dumps(scored,indent=2));print('VALIDATION COMPLETE',json.dumps({k:v for k,v in metric.items() if k not in ['search','winners']}),flush=True)
    pd.DataFrame({'id':tr.id.iloc[held].to_numpy(),'label':y[held],'partition':np.where(sm,'selection','report'),**hp_options}).to_csv(out/'held_predictions.csv',index=False)
    for f,(c,n) in fullplans.items():
        seedt=[];x=frames[c['features']];xt=tests[c['features']]
        for seed in SEEDS:
            path=cache/f'{f}_full_seed{seed}.npy';print('FULL REFIT',f,seed,n,flush=True)
            if path.exists():pt=np.load(path)
            else:
                aa,bb,_=encode(x,[xt],f);m=fit(make(c,seed,n),c,aa,y);pt=m.predict_proba(bb[0])[:,1];np.save(path,pt)
                if f=='lightgbm':m.booster_.save_model(str(out/f'{f}_seed{seed}.txt'))
                else:m.save_model(str(out/(f'{f}_seed{seed}.'+('ubj' if f=='xgboost' else 'cbm'))))
                (out/f'{f}_preprocessing.json').write_text(json.dumps({'features':list(aa),'categories':{k:list(aa[k].cat.categories) for k in aa.select_dtypes('category')}},indent=2))
                del m,aa,bb;gc.collect();torch.cuda.empty_cache()
            seedt.append(pt)
        tp_options[f+'_seed']=np.mean(seedt,axis=0)
    anchort=[]
    for seed in SEEDS:
        aa,bb,_=encode(frames['original'],[tests['original']],'lightgbm');m=anchor_model(seed);m.fit(aa,y);anchort.append(m.predict_proba(bb[0])[:,1]);del m,aa,bb;gc.collect()
    tp_options['e3_anchor']=np.mean(anchort,axis=0)
    final=tp_options['e3_anchor'] if use_anchor else prediction(tp_options,best)
    outputs={**tp_options,'ensemble':prediction(tp_options,ensemble),'selected':final}
    for name,p in outputs.items():
        assert len(p)==len(sample) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        s=sample.copy();s.satisfaction=p;s.to_csv(out/f'submission_{name}.csv',index=False)
    metric['runtime_seconds']=time.time()-start;metric['submission_sha256']=digest(out/'submission_selected.csv');(out/'metrics.json').write_text(json.dumps(metric,indent=2));print('COMPLETED',metric['runtime_seconds'],flush=True)
if __name__=='__main__':main()
