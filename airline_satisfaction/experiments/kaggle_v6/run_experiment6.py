from pathlib import Path
import json,hashlib,time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def score(y,p):return dict(auc=float(roc_auc_score(y,p)),log_loss=float(log_loss(y,p)))
def logit(p):
    p=np.clip(p,1e-7,1-1e-7);return np.log(p/(1-p))
def sigmoid(p):return 1/(1+np.exp(-np.clip(p,-30,30)))
def blend(a,b,w,method):return (1-w)*a+w*b if method=='probability' else sigmoid((1-w)*logit(a)+w*logit(b))
def main():
    if not Path('/kaggle/working').is_dir():raise RuntimeError('Run only on Kaggle')
    started=time.time();out=Path('/kaggle/working/experiment6');out.mkdir(exist_ok=True)
    inputs=Path('/kaggle/input');e4files=[];e5files=[]
    for p in inputs.rglob('held_predictions.csv'):
        cols=set(pd.read_csv(p,nrows=0).columns)
        if {'lightgbm_fold','xgboost_seed','label','partition','id'}<=cols:e4files.append(p)
        if {'resnet','lightgbm','tabicl','label','partition','id'}<=cols:e5files.append(p)
    assert len(e4files)==1 and len(e5files)==1,f'Attach exactly E4 and E5 output notebooks. Found E4={e4files}, E5={e5files}'
    e4,e5=e4files[0],e5files[0];root4,root5=e4.parent,e5.parent
    h4=pd.read_csv(e4);h5=pd.read_csv(e5);assert h4.id.is_unique and h5.id.is_unique and set(h4.id)==set(h5.id)
    h5=h5.set_index('id').loc[h4.id].reset_index();assert np.array_equal(h4.label,h5.label) and np.array_equal(h4.partition,h5.partition)
    assert len(h4)==100000 and set(h4.partition)=={'selection','report'}
    a=sigmoid(.9*logit(h4.lightgbm_fold.to_numpy())+.1*logit(h4.xgboost_seed.to_numpy()));b=h5.resnet.to_numpy()
    sel=h4.partition.eq('selection').to_numpy();report=~sel;y=h4.label.to_numpy();assert sel.sum()==report.sum()==50000
    s4=pd.read_csv(root4/'submission_ensemble.csv');s5=pd.read_csv(root5/'submission_resnet_fold.csv');assert s4.id.is_unique and s5.id.is_unique and set(s4.id)==set(s5.id)
    s5=s5.set_index('id').loc[s4.id].reset_index();assert len(s4)==299844
    pa,pb=s4.satisfaction.to_numpy(),s5.satisfaction.to_numpy()
    # Verify the upstream E4 blend specification and selected E5 architecture before combining.
    m4=json.loads((root4/'metrics.json').read_text());m5=json.loads((root5/'metrics.json').read_text())
    assert m4['best_ensemble']['weights']==[.9,.1,0.] and m4['best_ensemble']['method']=='logit'
    assert m4['best_ensemble']['columns'][:2]==['lightgbm_fold','xgboost_seed']
    assert m5['selected_resnet']['index']==1
    for p in [a,b,pa,pb]:assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    choices=[]
    for method in ['probability','logit']:
        for weight in [0.,.025,.05,.075,.1,.15,.2,.25,.3,.4]:
            choices.append(dict(method=method,neural_weight=weight,**score(y[sel],blend(a[sel],b[sel],weight,method))))
    # Only selection labels choose weights. Reused report labels are inspected after this decision.
    best=max(choices,key=lambda r:r['auc']);best_neural=max([r for r in choices if r['neural_weight']>0],key=lambda r:r['auc'])
    selected=blend(a,b,best['neural_weight'],best['method']);neural_blend=blend(a,b,best_neural['neural_weight'],best_neural['method'])
    metrics=dict(selected=best,best_neural=best_neural,selection_candidates=choices,
        report={name:score(y[report],p[report]) for name,p in {'e4_baseline':a,'resnet_fold':b,'selected':selected,'best_neural_blend':neural_blend}.items()},
        sources={str(p):sha(p) for p in [e4,e5,root4/'submission_ensemble.csv',root5/'submission_resnet_fold.csv',root4/'metrics.json',root5/'metrics.json']},
        note='No new training. E4 baseline uses 5-fold LightGBM + 3-seed XGBoost. E5 neural component uses the same 3 fold models for held and test predictions, avoiding the prior NN refit/scheduler mismatch. Selection and report sets reused from E1-E5; report is not independent new data. Blend weights chosen using selection only. Baseline public AUC 0.95856; these metrics are validation, not leaderboard scores.')
    outputs={'baseline_e4':pa,'resnet_fold':pb,'selected':blend(pa,pb,best['neural_weight'],best['method']),'neural_blend':blend(pa,pb,best_neural['neural_weight'],best_neural['method'])}
    for name,p in outputs.items():
        assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all();pd.DataFrame({'id':s4.id,'satisfaction':p}).to_csv(out/f'submission_{name}.csv',index=False)
    pd.DataFrame({'id':h4.id,'label':y,'partition':h4.partition,'e4_baseline':a,'resnet_fold':b,'selected':selected}).to_csv(out/'held_predictions.csv',index=False)
    metrics['runtime_seconds']=time.time()-started;metrics['submission_sha256']=sha(out/'submission_selected.csv');(out/'metrics.json').write_text(json.dumps(metrics,indent=2));print('COMPLETED',json.dumps(metrics),flush=True)
if __name__=='__main__':main()
