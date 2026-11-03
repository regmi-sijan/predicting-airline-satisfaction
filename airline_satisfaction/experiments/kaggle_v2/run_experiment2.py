"""Kaggle-only E2: three tree families, three-fold CV, fixed features and blends."""
from pathlib import Path
import os,sys,json,time,hashlib,importlib.metadata,gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold,train_test_split
from sklearn.metrics import roc_auc_score,log_loss

NAMES=['lightgbm','xgboost','catboost']

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

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

def make_model(name,trees=3000,early=False,seed=42):
    if name=='lightgbm':
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=trees,objective='binary',learning_rate=.035,
            num_leaves=63,max_depth=-1,min_child_samples=120,colsample_bytree=.85,
            subsample=.8,subsample_freq=1,reg_lambda=8,n_jobs=4,random_state=seed,
            verbosity=-1,deterministic=True,force_col_wise=True)
    if name=='xgboost':
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=trees,objective='binary:logistic',eval_metric='auc',
            learning_rate=.035,max_depth=6,min_child_weight=25,subsample=.85,
            colsample_bytree=.85,reg_lambda=12,reg_alpha=.1,max_bin=256,
            tree_method='hist',device='cuda',enable_categorical=True,n_jobs=4,
            random_state=seed,**({'early_stopping_rounds':120} if early else {}))
    from catboost import CatBoostClassifier
    return CatBoostClassifier(iterations=trees,depth=7,learning_rate=.05,l2_leaf_reg=8,
        loss_function='Logloss',eval_metric='AUC',task_type='GPU',devices='0',
        random_seed=seed,thread_count=4,allow_writing_files=False,verbose=300,
        **({'od_type':'Iter','od_wait':120} if early else {}))

def metrics(y,p):return {'auc':float(roc_auc_score(y,p)),'log_loss':float(log_loss(y,p))}

def run(data_dir,out):
    if not Path('/kaggle/working').is_dir():raise RuntimeError('This experiment must run on Kaggle, never on the laptop.')
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Enable Kaggle GPU accelerator.')
    print('HOST: Kaggle; GPU:',torch.cuda.get_device_name(0),'device count:',torch.cuda.device_count(),flush=True)
    start=time.time();out=Path(out);out.mkdir(parents=True,exist_ok=True);data_dir=Path(data_dir)
    tr=pd.read_csv(data_dir/'train.csv');te=pd.read_csv(data_dir/'test.csv');sample=pd.read_csv(data_dir/'sample_submission.csv')
    y=tr.satisfaction.astype(int).to_numpy();x=feature_frame(tr);xt=feature_frame(te)
    assert tr.id.is_unique and te.id.is_unique and np.array_equal(te.id,sample.id)
    assert list(x)==list(xt) and not set(tr.id)&set(te.id)
    # Preserve E1's partitions. This is reused housing-free airline data, not a new independent dataset.
    pool,held=train_test_split(np.arange(len(y)),test_size=100000,random_state=42,stratify=y)
    selection,report=train_test_split(held,test_size=.5,random_state=2026,stratify=y[held])
    mask=np.isin(held,selection);assert not set(pool)&set(held)
    versions={p:importlib.metadata.version(p) for p in ['lightgbm','xgboost','catboost','pandas','numpy','scikit-learn','torch']}
    config=dict(source_sha256=sha(__file__),train_sha256=sha(data_dir/'train.csv'),test_sha256=sha(data_dir/'test.csv'),versions=versions,
                folds=3,seed=20261028,feature_columns=list(x),models=NAMES,blend_grid_step=.25,
                host='Kaggle',gpu=torch.cuda.get_device_name(0),baseline_public_auc=.95835)
    fp=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest();cache=out/f'cache_{fp[:16]}';cache.mkdir(exist_ok=True)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    print('DATA:',tr.shape,te.shape,'features:',x.shape[1],'training pool:',len(pool),'held selection/report:',len(selection),len(report),flush=True)
    folds=list(StratifiedKFold(3,shuffle=True,random_state=20261028).split(pool,y[pool]))
    oof=np.zeros((len(pool),3));hp=np.zeros((len(held),3));fold_records=[];best={n:[] for n in NAMES}
    for fold,(a,b) in enumerate(folds):
        ti,vi=pool[a],pool[b]
        for j,name in enumerate(NAMES):
            f=cache/f'{name}_fold{fold}.npz'
            print('START',name,'fold',fold+1,flush=True)
            if f.exists():
                z=np.load(f);pv=z['validation'];ph=z['held'];iterations=int(z['iterations'])
                assert np.array_equal(z['validation_ids'],tr.id.iloc[vi])
            else:
                aa,bb,cats=encode(x.iloc[ti],[x.iloc[vi],x.iloc[held]],name);av,ah=bb
                model=make_model(name,early=True,seed=42+fold)
                if name=='lightgbm':
                    import lightgbm as lgb
                    model.fit(aa,y[ti],eval_set=[(av,y[vi])],eval_metric='auc',callbacks=[lgb.early_stopping(120,verbose=False),lgb.log_evaluation(300)])
                    iterations=int(model.best_iteration_)
                elif name=='xgboost':
                    model.fit(aa,y[ti],eval_set=[(av,y[vi])],verbose=300);iterations=int(model.best_iteration)+1
                    if model.get_booster().save_config().find('cuda')<0:raise RuntimeError('XGBoost did not use GPU')
                else:
                    model.fit(aa,y[ti],cat_features=cats,eval_set=(av,y[vi]),use_best_model=True);iterations=int(model.tree_count_)
                pv=model.predict_proba(av)[:,1];ph=model.predict_proba(ah)[:,1]
                np.savez_compressed(f,validation=pv,held=ph,iterations=iterations,validation_ids=tr.id.iloc[vi].to_numpy())
                del model,aa,av,ah,bb;gc.collect();torch.cuda.empty_cache()
            assert np.isfinite(pv).all() and np.isfinite(ph).all()
            oof[b,j]=pv;hp[:,j]+=ph/3;best[name].append(iterations)
            record=dict(model=name,fold=fold+1,iterations=iterations,**metrics(y[vi],pv));fold_records.append(record)
            print('DONE',record,'elapsed seconds:',round(time.time()-start),flush=True)
            (out/'fold_metrics.json').write_text(json.dumps(fold_records,indent=2))
    # Small fixed grid selected on selection holdout only; report labels do not choose weights.
    weights=[np.array([a,b,4-a-b])/4 for a in range(5) for b in range(5-a)]
    candidates=[{'weights':w.tolist(),**metrics(y[held][mask],(hp@w)[mask])} for w in weights]
    win=max(candidates,key=lambda c:c['auc']);w=np.asarray(win['weights'])
    result=dict(config=config,fingerprint=fp,folds=fold_records,
        cv_selection_note='Fold labels choose early stopping, so CV is developmental. Blends selected on separate 50k selection holdout. Report holdout reused from E1, not fresh independent data.',
        cv={name:metrics(y[pool],oof[:,j]) for j,name in enumerate(NAMES)},selection_candidates=candidates,
        selected_weights=dict(zip(NAMES,w.tolist())),report={name:metrics(y[held][~mask],hp[~mask,j]) for j,name in enumerate(NAMES)})
    result['report']['selected']=metrics(y[held][~mask],(hp@w)[~mask]);(out/'metrics.json').write_text(json.dumps(result,indent=2))
    print('REPORT',json.dumps(result['report']),'SELECTED WEIGHTS',result['selected_weights'],flush=True)
    pd.DataFrame({'id':tr.id.iloc[pool].to_numpy(),'y':y[pool],**{n:oof[:,j] for j,n in enumerate(NAMES)}}).to_csv(out/'oof_predictions.csv',index=False)
    pd.DataFrame({'id':tr.id.iloc[held].to_numpy(),'y':y[held],'partition':np.where(mask,'selection','report'),**{n:hp[:,j] for j,n in enumerate(NAMES)}}).to_csv(out/'held_predictions.csv',index=False)
    tp=np.zeros((len(te),3))
    for j,name in enumerate(NAMES):
        f=cache/f'{name}_full.npy';iterations=int(np.median(best[name]));print('FULL REFIT',name,iterations,flush=True)
        if f.exists():tp[:,j]=np.load(f)
        else:
            aa,bb,cats=encode(x,[xt],name);at=bb[0];model=make_model(name,trees=iterations)
            if name=='catboost':model.fit(aa,y,cat_features=cats);model.save_model(str(out/'catboost.cbm'))
            else:
                model.fit(aa,y)
                if name=='xgboost':model.save_model(out/'xgboost.ubj')
                else:model.booster_.save_model(str(out/'lightgbm.txt'))
            tp[:,j]=model.predict_proba(at)[:,1];np.save(f,tp[:,j])
            (out/f'{name}_preprocessing.json').write_text(json.dumps({'features':list(aa),'categories':{c:list(aa[c].cat.categories) for c in aa.select_dtypes('category')}},indent=2))
            del model,aa,at,bb;gc.collect();torch.cuda.empty_cache()
        assert np.isfinite(tp[:,j]).all() and ((tp[:,j]>=0)&(tp[:,j]<=1)).all()
        s=sample.copy();s.satisfaction=tp[:,j];s.to_csv(out/f'submission_{name}.csv',index=False)
    s=sample.copy();s.satisfaction=tp@w;assert s.satisfaction.between(0,1).all();s.to_csv(out/'submission_selected.csv',index=False)
    result.update(runtime_seconds=time.time()-start,submission_sha256=sha(out/'submission_selected.csv'))
    (out/'metrics.json').write_text(json.dumps(result,indent=2));print('COMPLETED',result['runtime_seconds'],flush=True)
    return result

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--output-dir',default='/kaggle/working/experiment2');a=p.parse_args();run(a.data_dir,a.output_dir)
