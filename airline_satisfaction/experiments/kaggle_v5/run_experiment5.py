from pathlib import Path
import os,sys,json,time,gc,random,subprocess,hashlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score,log_loss
import joblib

SEEDS=[42,2026,20261028]
def score(y,p):return dict(auc=float(roc_auc_score(y,p)),log_loss=float(log_loss(y,p)))
def seed_all(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
def release():gc.collect();torch.cuda.empty_cache()
def preprocess(a,others):
    cats=list(a.select_dtypes(include=['object','string']).columns);nums=[c for c in a if c not in cats]
    levels={c:{v:i+1 for i,v in enumerate(sorted(a[c].dropna().astype(str).unique()))} for c in cats}
    med=a[nums].median().fillna(0);qt=QuantileTransformer(n_quantiles=1000,output_distribution='normal',subsample=100000,random_state=42)
    qt.fit(a[nums].fillna(med));state=dict(cats=cats,nums=nums,levels=levels,median=med,qt=qt)
    def transform(d):
        num=np.concatenate([qt.transform(d[nums].fillna(med)),d[nums].isna().to_numpy()],axis=1).astype('float32')
        cat=np.column_stack([d[c].astype(str).map(levels[c]).fillna(0).astype('int64') for c in cats])
        return num,cat
    return transform(a),[transform(d) for d in others],state
class Block(nn.Module):
    def __init__(self,w,drop):
        super().__init__();self.layers=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,w*2),nn.GELU(),nn.Dropout(drop),nn.Linear(w*2,w),nn.Dropout(drop))
    def forward(self,x):return x+self.layers(x)
class Net(nn.Module):
    def __init__(self,nnum,cards,width,depth,drop):
        super().__init__();self.emb=nn.ModuleList([nn.Embedding(n,min(32,max(4,int(n**.5)*2))) for n in cards])
        self.input=nn.Linear(nnum+sum(e.embedding_dim for e in self.emb),width)
        self.body=nn.Sequential(*[Block(width,drop) for _ in range(depth)]);self.head=nn.Sequential(nn.LayerNorm(width),nn.GELU(),nn.Linear(width,1))
    def forward(self,n,c):
        z=torch.cat([n]+[e(c[:,j]) for j,e in enumerate(self.emb)],dim=1)
        return self.head(self.body(self.input(z))).squeeze(1)
def predict(m,arrays):
    m.eval();ans=[]
    with torch.inference_mode():
        for i in range(0,len(arrays[0]),8192):
            n=torch.as_tensor(arrays[0][i:i+8192],device='cuda');c=torch.as_tensor(arrays[1][i:i+8192],device='cuda')
            with torch.autocast('cuda',dtype=torch.float16):p=torch.sigmoid(m(n,c))
            ans.append(p.float().cpu().numpy())
    p=np.concatenate(ans);assert np.isfinite(p).all();return p
def train_net(a,y,state,cfg,seed,path,validation=None,epochs=100):
    seed_all(seed);cards=[len(state['levels'][c])+1 for c in state['cats']];m=Net(a[0].shape[1],cards,**cfg).cuda()
    opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.01);scaler=torch.amp.GradScaler('cuda')
    n=torch.as_tensor(a[0],device='cuda');c=torch.as_tensor(a[1],device='cuda');t=torch.as_tensor(y,dtype=torch.float32,device='cuda')
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=3,min_lr=1e-5) if validation else None
    best=-1;bestepoch=0;stale=0;history=[]
    for epoch in range(1,epochs+1):
        m.train();order=torch.randperm(len(y),device='cuda');losses=[]
        for batch in order.split(2048):
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):loss=nn.functional.binary_cross_entropy_with_logits(m(n[batch],c[batch]),t[batch])
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite neural loss')
            scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(m.parameters(),1.0);scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
        if validation:
            auc=roc_auc_score(validation[1],predict(m,validation[0]));scheduler.step(auc);history.append(dict(epoch=epoch,auc=float(auc),loss=float(np.mean(losses))))
            print('NN epoch',epoch,'AUC',auc,'loss',np.mean(losses),flush=True)
            if auc>best+1e-6:best=auc;bestepoch=epoch;stale=0;torch.save(m.state_dict(),path)
            else:stale+=1
            if stale>=12:break
        elif epoch==1 or epoch%10==0:print('NN full epoch',epoch,'/',epochs,'loss',np.mean(losses),flush=True)
    if validation:m.load_state_dict(torch.load(path,weights_only=True))
    else:torch.save(m.state_dict(),path);bestepoch=epochs
    del n,c,t,opt,scaler;release();return m,bestepoch,history

def tab_worker(config):
    from tabicl import TabICLClassifier
    cfg=json.loads(Path(config).read_text());root=Path(cfg['root']);tr=pd.read_csv(root/'train.csv');x=tr.drop(columns=['id','satisfaction']);y=tr.satisfaction.astype(int).to_numpy()
    ctx=np.load(cfg['ctx']);xp=x.iloc[np.load(cfg['rows'])] if cfg['phase']=='held' else pd.read_csv(root/'test.csv').drop(columns='id')
    m=TabICLClassifier(n_estimators=8,n_jobs=4,device='cuda',kv_cache=True,batch_size=1,model_path=cfg['checkpoint'],allow_auto_download=False,random_state=42)
    print('TABICL CONTEXT',len(ctx),'estimators',8,flush=True);m.fit(x.iloc[ctx],y[ctx]);pred=[];positive=list(m.classes_).index(1)
    batchdir=Path(cfg['output']).with_suffix('');batchdir.mkdir(exist_ok=True)
    for i in range(0,len(xp),1024):
        f=batchdir/f'{i:07d}.npy'
        if f.exists():p=np.load(f)
        else:p=m.predict_proba(xp.iloc[i:i+1024])[:,positive];np.save(f,p)
        assert np.isfinite(p).all();pred.append(p)
        if i%20480==0:print('TABICL',cfg['phase'],i,'/',len(xp),flush=True)
    np.save(cfg['output'],np.concatenate(pred))
def run_tab(root,out,pool,held,checkpoint):
    tr=pd.read_csv(root/'train.csv');y=tr.satisfaction.astype(int).to_numpy()
    attempts=[]
    for size in [65536,32768]:
        cache=out/f'tabicl_{size}';cache.mkdir(exist_ok=True)
        try:
            for phase,available in [('held',pool),('test',np.arange(len(y)))]:
                ctx,_=train_test_split(available,train_size=size,stratify=y[available],random_state=77)
                cf=cache/f'{phase}_context.npy';rf=cache/'held_rows.npy';np.save(cf,ctx);np.save(rf,held);dest=cache/f'{phase}_pred.npy'
                if not dest.exists():
                    config=dict(root=str(root),ctx=str(cf),rows=str(rf),phase=phase,output=str(dest),checkpoint=checkpoint)
                    fp=cache/f'{phase}.json';fp.write_text(json.dumps(config));print('START TABICL',size,phase,flush=True)
                    subprocess.run([sys.executable,'-u',__file__,'--tab-worker',str(fp)],check=True,timeout=5400)
            return np.load(cache/'held_pred.npy'),np.load(cache/'test_pred.npy'),dict(context=size,estimators=8,attempts=attempts)
        except (subprocess.CalledProcessError,subprocess.TimeoutExpired) as e:
            attempts.append(dict(context=size,error=str(e)));print('TABICL ATTEMPT FAILED',attempts[-1],flush=True)
    (out/'tabicl_failure.json').write_text(json.dumps(attempts,indent=2));return None,None,dict(failed=True,attempts=attempts)

def main():
    if not Path('/kaggle/working').is_dir():raise RuntimeError('Kaggle execution only')
    assert torch.cuda.is_available(),'Kaggle GPU required';torch.set_num_threads(4)
    start=time.time();out=Path('/kaggle/working/experiment5');out.mkdir(exist_ok=True)
    paths=[p.parent for p in Path('/kaggle/input').rglob('train.csv') if 'playground-series-s6e10' in str(p) and (p.parent/'test.csv').exists()];assert len(paths)==1;root=paths[0]
    tr=pd.read_csv(root/'train.csv');te=pd.read_csv(root/'test.csv');sample=pd.read_csv(root/'sample_submission.csv');x=tr.drop(columns=['id','satisfaction']);xt=te.drop(columns=['id']);y=tr.satisfaction.astype(int).to_numpy()
    assert np.array_equal(te.id,sample.id) and tr.id.is_unique and te.id.is_unique
    pool,held=train_test_split(np.arange(len(y)),test_size=100000,stratify=y,random_state=42);sel,rep=train_test_split(held,test_size=.5,stratify=y[held],random_state=2026);sm=np.isin(held,sel)
    print('HOST Kaggle; GPU',torch.cuda.get_device_name(0),'count',torch.cuda.device_count(),'DATA',tr.shape,te.shape,flush=True)
    configs=[dict(width=256,depth=3,drop=.15),dict(width=512,depth=4,drop=.2)];search=[];foldpred={};epochcounts={}
    folds=list(StratifiedKFold(3,shuffle=True,random_state=20261029).split(pool,y[pool]))
    for j,cfg in enumerate(configs):
        oof=np.zeros(len(pool));hp=np.zeros(len(held));tp=np.zeros(len(te));epochs=[]
        for k,(a,b) in enumerate(folds):
            ti,vi=pool[a],pool[b];print('START RESNET',j,'fold',k+1,cfg,flush=True)
            aa,bb,state=preprocess(x.iloc[ti],[x.iloc[vi],x.iloc[held],xt]);av,ah,at=bb
            m,ep,hist=train_net(aa,y[ti],state,cfg,42+k,out/f'nn_{j}_fold{k}.pt',validation=(av,y[vi]))
            oof[b]=predict(m,av);hp+=predict(m,ah)/3;tp+=predict(m,at)/3;epochs.append(ep)
            joblib.dump(state,out/f'nn_{j}_fold{k}_preprocess.joblib');(out/f'nn_{j}_fold{k}_history.json').write_text(json.dumps(hist))
            print('DONE RESNET',j,k+1,score(y[vi],oof[b]),'epoch',ep,flush=True);del m,aa,bb,av,ah,at,state;release()
        row=dict(index=j,config=cfg,cv=score(y[pool],oof),epochs=epochs);search.append(row);foldpred[j]=(hp,tp);epochcounts[j]=epochs
        np.savez_compressed(out/f'nn_{j}_predictions.npz',held=hp,test=tp,oof=oof);(out/'search.json').write_text(json.dumps(search,indent=2))
    win=max(search,key=lambda r:r['cv']['auc']);j=win['index'];cfg=win['config'];epochs=int(np.median(epochcounts[j]));hp={'resnet':foldpred[j][0]};tp={'resnet_fold':foldpred[j][1]}
    # Control uses E3's validated LightGBM recipe. No XGBoost is used.
    import lightgbm as lgb
    def lgbfit(a,b,seed):
        levels={c:sorted(a[c].dropna().unique()) for c in a.select_dtypes(include=['object','string'])};a=a.copy();b=b.copy()
        for c,lv in levels.items():a[c]=pd.Categorical(a[c],categories=lv);b[c]=pd.Categorical(b[c],categories=lv)
        m=lgb.LGBMClassifier(n_estimators=626,learning_rate=.035,num_leaves=63,min_child_samples=120,reg_lambda=8,colsample_bytree=.85,subsample=.8,subsample_freq=1,n_jobs=4,random_state=seed,verbosity=-1,deterministic=True,force_col_wise=True)
        m.fit(a,y[pool] if len(a)==len(pool) else y);return m.predict_proba(b)[:,1]
    hp['lightgbm']=np.mean([lgbfit(x.iloc[pool],x.iloc[held],s) for s in SEEDS],axis=0)
    from huggingface_hub import hf_hub_download
    try:
        checkpoint=hf_hub_download(repo_id='jingang/TabICL',filename='tabicl-classifier-v2-20260212.ckpt')
        th,tt,tabstatus=run_tab(root,out,pool,held,checkpoint)
        if th is not None:hp['tabicl']=th;tp['tabicl']=tt
    except Exception as e:
        tabstatus=dict(failed=True,error=repr(e));print('TABICL UNAVAILABLE',tabstatus,flush=True)
    cols=list(hp);candidates=[]
    weights=[np.array([a,b,10-a-b])/10 for a in range(11) for b in range(11-a)] if len(cols)==3 else [np.array([a,10-a])/10 for a in range(11)]
    mat=np.column_stack([hp[c] for c in cols])
    for w in weights:candidates.append(dict(weights=w.tolist(),**score(y[held][sm],(mat@w)[sm])))
    best=max(candidates,key=lambda r:r['auc']);w=np.array(best['weights']);neural=max([r for r in candidates if sum(v for c,v in zip(cols,r['weights']) if c!='lightgbm')>0],key=lambda r:r['auc'])
    metrics=dict(configs=search,selected_resnet=win,tabicl=tabstatus,columns=cols,selected_weights=dict(zip(cols,w.tolist())),selection_candidates=candidates,
        report={c:score(y[held][~sm],hp[c][~sm]) for c in cols},selected_report=score(y[held][~sm],(mat@w)[~sm]),best_neural_weights=neural['weights'],
        note='Outer NN folds choose early stopping so CV is developmental. Recipe selected by CV, blends on separate selection partition. Report partition reused from E1-E4 and is not fresh independent evaluation. Final NN refits use median fold epochs, no validation labels.',baseline_public_auc=.95856)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2));print('VALIDATION',json.dumps(metrics),flush=True)
    pd.DataFrame({'id':tr.id.iloc[held].to_numpy(),'label':y[held],'partition':np.where(sm,'selection','report'),**hp}).to_csv(out/'held_predictions.csv',index=False)
    print('FULL DATA RESNET',cfg,'epochs',epochs,flush=True);aa,bb,state=preprocess(x,[xt]);at=bb[0];npreds=[];joblib.dump(state,out/'nn_full_preprocess.joblib')
    for seed in SEEDS:
        m,_,_=train_net(aa,y,state,cfg,seed,out/f'nn_full_seed{seed}.pt',epochs=epochs);npreds.append(predict(m,at));del m;release()
    tp['resnet']=np.mean(npreds,axis=0);del aa,bb,at,state;release();tp['lightgbm']=np.mean([lgbfit(x,xt,s) for s in SEEDS],axis=0)
    testmat=np.column_stack([tp[c] for c in cols]);outputs={**tp,'selected':testmat@w,'neural_blend':testmat@np.array(neural['weights'])}
    for name,p in outputs.items():
        assert len(p)==len(sample) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all();s=sample.copy();s.satisfaction=p;s.to_csv(out/f'submission_{name}.csv',index=False)
    metrics['runtime_seconds']=time.time()-start;(out/'metrics.json').write_text(json.dumps(metrics,indent=2));print('COMPLETED',metrics['runtime_seconds'],flush=True)
if __name__=='__main__':
    if '--tab-worker' in sys.argv:tab_worker(sys.argv[sys.argv.index('--tab-worker')+1])
    else:main()
