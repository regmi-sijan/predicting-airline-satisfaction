"""Experiment 1: direct pretrained TabICLv2 inference and a LightGBM control.

The transformer weights stay frozen. Only labeled context/preprocessing is fitted.
Selection and reporting partitions are disjoint; neither trains base models.
"""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, time, os, subprocess, sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
import joblib

CHECKPOINT = 'tabicl-classifier-v2-20260212.ckpt'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def scores(y,p):
    return dict(auc=float(roc_auc_score(y,p)),log_loss=float(log_loss(y,p)),
                brier=float(brier_score_loss(y,p)))

def lgb_features(x, levels=None):
    x=x.copy()
    if levels is None:
        levels={c:sorted(x[c].dropna().unique().tolist()) for c in x.select_dtypes(include=['object','string'])}
    for c,lv in levels.items(): x[c]=pd.Categorical(x[c],categories=lv)
    return x,levels

def predict_batches(model,x,path,batch=2048):
    """Resume only within a fingerprint-specific run directory."""
    path.mkdir(parents=True,exist_ok=True)
    out=[]
    positive=list(model.classes_).index(1)
    for start in range(0,len(x),batch):
        f=path/f'{start:07d}.npy'; end=min(start+batch,len(x))
        if f.exists(): p=np.load(f,allow_pickle=False)
        else:
            p=model.predict_proba(x.iloc[start:end])[:,positive]
            np.save(f,p)
        if p.shape!=(end-start,) or not np.isfinite(p).all(): raise ValueError(f'Invalid batch {f}')
        out.append(p)
        if start//batch%20==0 or end==len(x): print(f'TabICL {path.name}: {end}/{len(x)}',flush=True)
    return np.concatenate(out)

def transformer_worker(config_file):
    from tabicl import TabICLClassifier
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    c=json.loads(Path(config_file).read_text())
    train=pd.read_csv(c['train']); y=train.satisfaction.astype(int).to_numpy()
    x=train.drop(columns=['id','satisfaction']);ctx=np.load(c['context'])
    if c['phase']=='held': xp=x.iloc[np.load(c['rows'])]
    else: xp=pd.read_csv(c['test']).drop(columns=['id'])
    model=TabICLClassifier(n_estimators=c['estimators'],n_jobs=4,device='cpu',kv_cache=True,batch_size=1,
        model_path=c['checkpoint'],allow_auto_download=False,random_state=42)
    print('Building pretrained context',c['phase'],len(ctx),flush=True)
    model.fit(x.iloc[ctx],y[ctx]);print('Pretrained context ready',c['phase'],flush=True)
    pred=predict_batches(model,xp,Path(c['batches']))
    np.save(c['output'],pred)


def transformer_process(data_dir,checkpoint,ctx,rows,cache,estimators,phase):
    ctxfile=(cache/f'{phase}_context.npy').resolve();np.save(ctxfile,ctx)
    rowfile=(cache/f'{phase}_rows.npy').resolve();np.save(rowfile,rows)
    outfile=(cache/f'{phase}_pred.npy').resolve()
    if not outfile.exists():
        c=dict(train=str((data_dir/'train.csv').resolve()),test=str((data_dir/'test.csv').resolve()),
               context=str(ctxfile),rows=str(rowfile),phase=phase,checkpoint=str(checkpoint.resolve()),
               estimators=estimators,batches=str((cache/phase).resolve()),output=str(outfile))
        configfile=(cache/f'{phase}_worker.json').resolve();configfile.write_text(json.dumps(c))
        env=os.environ.copy();env.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker-config',str(configfile)],check=True,env=env)
    return np.load(outfile,allow_pickle=False)


def run(data_dir,output_dir,checkpoint,context=16384,estimators=2):
    import lightgbm as lgb
    t0=time.time(); data_dir=Path(data_dir); output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(checkpoint)
    train=pd.read_csv(data_dir/'train.csv');test=pd.read_csv(data_dir/'test.csv');sample=pd.read_csv(data_dir/'sample_submission.csv')
    assert train.id.is_unique and test.id.is_unique and sample.id.is_unique
    assert np.array_equal(test.id,sample.id) and not set(train.id)&set(test.id)
    assert set(train.satisfaction.unique())=={False,True}
    y=train.satisfaction.astype(int).to_numpy();x=train.drop(columns=['id','satisfaction']);xt=test.drop(columns=['id'])
    assert list(x)==list(xt)
    packages={p:importlib.metadata.version(p) for p in ['tabicl','torch','lightgbm','scikit-learn','pandas','numpy']}
    config=dict(seed=42,context=context,estimators=estimators,checkpoint_sha256=sha(checkpoint),
                train_sha256=sha(data_dir/'train.csv'),test_sha256=sha(data_dir/'test.csv'),
                source_sha256=sha(__file__),packages=packages)
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    cache=output_dir/f'cache_{fingerprint[:16]}';cache.mkdir(exist_ok=True)
    pool,held=train_test_split(np.arange(len(y)),test_size=100000,random_state=42,stratify=y)
    selection,report=train_test_split(held,test_size=.5,random_state=2026,stratify=y[held])
    inner,stop=train_test_split(pool,test_size=.15,random_state=314159,stratify=y[pool])
    ctx,_=train_test_split(pool,train_size=context,random_state=77,stratify=y[pool])
    assert not set(pool)&set(held) and not set(selection)&set(report)
    pd.DataFrame({'id':train.id,'partition':np.select([np.isin(np.arange(len(y)),selection),np.isin(np.arange(len(y)),report)],['selection','report'],default='train_pool')}).to_csv(output_dir/'partitions.csv',index=False)
    audit=dict(train_shape=list(train.shape),test_shape=list(test.shape),positive_fraction=float(y.mean()),
               missing_train=train.isna().sum().to_dict(),missing_test=test.isna().sum().to_dict(),
               duplicate_feature_rows=int(x.duplicated().sum()),features=list(x),config=config,fingerprint=fingerprint,
               n_train_pool=len(pool),n_selection=len(selection),n_report=len(report))
    (output_dir/'audit.json').write_text(json.dumps(audit,indent=2))
    print('Data audit',train.shape,test.shape,'positive',y.mean(),flush=True)
    pfile=cache/'validation.npz'
    params=dict(objective='binary',learning_rate=.04,num_leaves=31,min_child_samples=100,
                colsample_bytree=.9,subsample=.8,subsample_freq=1,reg_lambda=5,
                n_jobs=4,random_state=42,verbosity=-1,deterministic=True,force_col_wise=True)
    if pfile.exists():
        d=np.load(pfile); tab=d['tab'];gb=d['gb'];best=int(d['best'])
    else:
        a,levels=lgb_features(x.iloc[inner]);b,_=lgb_features(x.iloc[stop],levels)
        early=lgb.LGBMClassifier(**params,n_estimators=2000)
        early.fit(a,y[inner],eval_set=[(b,y[stop])],eval_metric='auc',callbacks=[lgb.early_stopping(100,verbose=False)])
        best=int(early.best_iteration_);print('LightGBM iterations',best,flush=True)
        a,levels=lgb_features(x.iloc[pool]);b,_=lgb_features(x.iloc[held],levels)
        control=lgb.LGBMClassifier(**params,n_estimators=best).fit(a,y[pool]);gb=control.predict_proba(b)[:,1]
        print('LightGBM validation predictions complete',flush=True)
        tab=transformer_process(data_dir,checkpoint,ctx,held,cache,estimators,'held')
        np.savez_compressed(pfile,tab=tab,gb=gb,best=best)
    selmask=np.isin(held,selection);repmask=np.isin(held,report)
    weights=[0.,.25,.5,.75,1.]
    candidates=[dict(tabicl_weight=w,**scores(y[held][selmask],(w*tab+(1-w)*gb)[selmask])) for w in weights]
    weight=max(candidates,key=lambda d:d['auc'])['tabicl_weight']
    metrics=dict(config=config,fingerprint=fingerprint,context_rows=context,best_lgb_iteration=best,
                 selection_candidates=candidates,selected_tabicl_weight=weight,
                 report={n:scores(y[held][repmask],p[repmask]) for n,p in {'tabicl':tab,'lightgbm':gb,'selected':weight*tab+(1-weight)*gb}.items()},
                 notes=['Report partition never selects weights, context, preprocessing, or stopping iteration.',
                        'One fixed stratified holdout, not cross-validation. Public test score is separate.',
                        'TabICLv2 frozen pretrained weights; 16384 labeled context rows by default; no gradient fine-tuning.',
                        'Final deployment refits context and LightGBM using all available labeled rows.'])
    (output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2));print(json.dumps(metrics['report'],indent=2),flush=True)
    pd.DataFrame({'id':train.id.iloc[held].to_numpy(),'y':y[held],'partition':np.where(selmask,'selection','report'),'tabicl':tab,'lightgbm':gb}).to_csv(output_dir/'validation_predictions.csv',index=False)
    final=cache/'test.npz'
    if final.exists():
        d=np.load(final);pt=d['tab'];pg=d['gb']
    else:
        allctx,_=train_test_split(np.arange(len(y)),train_size=context,random_state=77,stratify=y)
        pt=transformer_process(data_dir,checkpoint,allctx,np.arange(len(test)),cache,estimators,'test')
        np.save(output_dir/'tabicl_context_ids.npy',train.id.iloc[allctx].to_numpy())
        a,levels=lgb_features(x);b,_=lgb_features(xt,levels)
        control=lgb.LGBMClassifier(**params,n_estimators=best).fit(a,y);pg=control.predict_proba(b)[:,1]
        joblib.dump({'model':control,'categories':levels,'features':list(x)},output_dir/'lightgbm.joblib')
        np.savez_compressed(final,tab=pt,gb=pg)
    for name,p in {'tabicl':pt,'lightgbm':pg,'selected':weight*pt+(1-weight)*pg}.items():
        assert p.shape==(len(test),) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
        out=sample.copy();out.satisfaction=p;out.to_csv(output_dir/f'submission_{name}.csv',index=False)
    metrics['runtime_seconds']=time.time()-t0;metrics['submission_sha256']=sha(output_dir/'submission_selected.csv')
    (output_dir/'metrics.json').write_text(json.dumps(metrics,indent=2));print('Experiment completed',metrics['runtime_seconds'],flush=True)
    return metrics

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--worker-config':
        transformer_worker(sys.argv[2]);sys.exit(0)
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='airline_satisfaction/data');p.add_argument('--output-dir',default='airline_satisfaction/experiments/v1/results');p.add_argument('--checkpoint',default=f'airline_satisfaction/models/{CHECKPOINT}');p.add_argument('--context',type=int,default=16384);p.add_argument('--estimators',type=int,default=2)
    args=p.parse_args();run(args.data_dir,args.output_dir,args.checkpoint,args.context,args.estimators)
