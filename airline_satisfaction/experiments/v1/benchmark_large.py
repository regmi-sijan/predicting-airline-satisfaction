import time
import pandas as pd
from sklearn.model_selection import train_test_split
from tabicl import TabICLClassifier
p='airline_satisfaction/data/train.csv'
d=pd.read_csv(p); X=d.drop(columns=['id','satisfaction']); y=d.satisfaction.astype(int)
a,b=train_test_split(range(len(d)),train_size=16384,test_size=1024,random_state=42,stratify=y)
m=TabICLClassifier(n_estimators=1,n_jobs=4,device='cpu',kv_cache=True,batch_size=1,model_path='airline_satisfaction/models/tabicl-classifier-v2-20260212.ckpt',allow_auto_download=False,verbose=False)
t=time.time();m.fit(X.iloc[a],y.iloc[a]); print('fit seconds',time.time()-t,flush=True)
t=time.time();pr=m.predict_proba(X.iloc[b])[:,1];print('predict seconds',time.time()-t,pr[:5],flush=True)
