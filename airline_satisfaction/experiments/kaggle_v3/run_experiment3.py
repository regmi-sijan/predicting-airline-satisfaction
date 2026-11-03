from pathlib import Path
import os,time,json,gc
import numpy as np
import pandas as pd
import lightgbm as lgb
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


def score(y,p):return dict(auc=float(roc_auc_score(y,p)),log_loss=float(log_loss(y,p)))
def model(cfg,seed,trees=2500):
    return lgb.LGBMClassifier(objective='binary',metric='auc',n_estimators=trees,learning_rate=.035,
        num_leaves=cfg['leaves'],min_child_samples=cfg['minimum'],reg_lambda=cfg['lambda'],
        colsample_bytree=.85,subsample=.8,subsample_freq=1,n_jobs=4,random_state=seed,
        verbosity=-1,deterministic=True,force_col_wise=True)
def run():
    if not Path('/kaggle/working').is_dir():raise RuntimeError('Kaggle execution only')
    start=time.time();out=Path('/kaggle/working/experiment3');out.mkdir(exist_ok=True)
    paths=[p.parent for p in Path('/kaggle/input').rglob('train.csv') if 'playground-series-s6e10' in str(p) and (p.parent/'test.csv').exists()]
    assert len(paths)==1,paths
    tr=pd.read_csv(paths[0]/'train.csv');te=pd.read_csv(paths[0]/'test.csv');sample=pd.read_csv(paths[0]/'sample_submission.csv')
    y=tr.satisfaction.astype(int).to_numpy();assert tr.id.is_unique and te.id.is_unique and np.array_equal(te.id,sample.id)
    pool,held=train_test_split(np.arange(len(y)),test_size=100000,random_state=42,stratify=y)
    selection,report=train_test_split(held,test_size=.5,random_state=2026,stratify=y[held]);sm=np.isin(held,selection)
    print('HOST: Kaggle CPU; data:',tr.shape,te.shape,'LightGBM:',lgb.__version__,flush=True)
    frames={'original':tr.drop(columns=['id','satisfaction']),'engineered':feature_frame(tr)}
    tests={'original':te.drop(columns=['id']),'engineered':feature_frame(te)}
    configs=[dict(name=f'{f}_{v}',features=f,leaves=l,minimum=m,**{'lambda':r}) for f in frames for v,l,m,r in [('regularized',31,200,15),('e2',63,120,8),('flexible',127,80,5)]]
    folds=list(StratifiedKFold(3,shuffle=True,random_state=20261028).split(pool,y[pool]))
    results=[];heldpred={};testpred={};iterations={}
    for cfg in configs:
        name=cfg['name'];x=frames[cfg['features']];xt=tests[cfg['features']];oof=np.zeros(len(pool));hp=np.zeros(len(held));tp=np.zeros(len(te));iters=[]
        for fold,(a,b) in enumerate(folds):
            ti,vi=pool[a],pool[b];print('START',name,'fold',fold+1,flush=True)
            aa,bb,_=encode(x.iloc[ti],[x.iloc[vi],x.iloc[held],xt],'lightgbm');av,ah,at=bb
            m=model(cfg,42+fold);m.fit(aa,y[ti],eval_set=[(av,y[vi])],eval_metric='auc',callbacks=[lgb.early_stopping(120,first_metric_only=True,verbose=False),lgb.log_evaluation(500)])
            oof[b]=m.predict_proba(av)[:,1];hp+=m.predict_proba(ah)[:,1]/3;tp+=m.predict_proba(at)[:,1]/3;iters.append(int(m.best_iteration_))
            print('DONE',name,fold+1,score(y[vi],oof[b]),'trees',iters[-1],flush=True)
            del m,aa,bb,av,ah,at;gc.collect()
        heldpred[name]=hp;testpred[name]=tp;iterations[name]=iters
        row=dict(config=cfg,cv=score(y[pool],oof),selection=score(y[held][sm],hp[sm]),iterations=iters);results.append(row)
        (out/'search.json').write_text(json.dumps(results,indent=2));np.savez_compressed(out/f'{name}_predictions.npz',held=hp,test=tp,oof=oof)
        print('CANDIDATE',row,flush=True)
    # Select recipe using selection partition only. Reused report labels are read only after all choices.
    winner=max(results,key=lambda r:r['selection']['auc']);cfg=winner['config'];name=cfg['name'];trees=int(np.median(iterations[name]))
    x=frames[cfg['features']];xt=tests[cfg['features']];hseeds=[];tseeds=[]
    for seed in [42,2026,20261028]:
        print('POOL SEED REFIT',seed,name,trees,flush=True)
        aa,bb,_=encode(x.iloc[pool],[x.iloc[held]],'lightgbm');m=model(cfg,seed,trees);m.fit(aa,y[pool]);hseeds.append(m.predict_proba(bb[0])[:,1]);del m,aa,bb;gc.collect()
    options={'fold_average':heldpred[name],'single_refit':hseeds[0],'three_seed_refit':np.mean(hseeds,axis=0)}
    selection_scores={k:score(y[held][sm],v[sm]) for k,v in options.items()};mode=max(selection_scores,key=lambda k:selection_scores[k]['auc'])
    metrics=dict(search=results,selected_config=cfg,selected_mode=mode,selection_options=selection_scores,
        report_options={k:score(y[held][~sm],v[~sm]) for k,v in options.items()},
        note='CV labels select early stopping; CV is developmental. Selection/report partitions reused from E1/E2, not fresh independent evaluation.',baseline_public_auc=.95835)
    print('SELECTED',json.dumps(metrics),flush=True);(out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    for seed in [42,2026,20261028]:
        print('FULL REFIT',seed,flush=True);aa,bb,_=encode(x,[xt],'lightgbm');m=model(cfg,seed,trees);m.fit(aa,y);tseeds.append(m.predict_proba(bb[0])[:,1]);m.booster_.save_model(str(out/f'lightgbm_seed{seed}.txt'))
        if seed==42:(out/'preprocessing.json').write_text(json.dumps({'features':list(aa),'categories':{c:list(aa[c].cat.categories) for c in aa.select_dtypes('category')}},indent=2))
        del m,aa,bb;gc.collect()
    outputs={'fold_average':testpred[name],'single_refit':tseeds[0],'three_seed_refit':np.mean(tseeds,axis=0)}
    for k,v in {**outputs,'selected':outputs[mode]}.items():
        assert len(v)==len(sample) and np.isfinite(v).all() and ((v>=0)&(v<=1)).all()
        submission=sample.copy();submission.satisfaction=v;submission.to_csv(out/f'submission_{k}.csv',index=False)
    metrics['runtime_seconds']=time.time()-start;(out/'metrics.json').write_text(json.dumps(metrics,indent=2));print('COMPLETED',metrics['runtime_seconds'],flush=True)
if __name__=='__main__':run()
