"""Post-outcome pair-grouped split check (methods note). Not preregistered; descriptive only. Run from repo root."""
import json,sys,random
sys.path.insert(0,'src')
import numpy as np,pandas as pd,torch
from flygnn.data import *
from flygnn.models import *
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
ann=load_annotations();A,ids=load_graph();inp,out=load_morphology()
A,X,ys,classes,ids=align(A,ids,ann,inp,out,min_class=40)
y=np.array([classes.index(v) for v in ys]);Ah=normalize_adj(A)
pos={int(b):i for i,b in enumerate(ids)};mate={}
for _,r in ann.iterrows():
  l,rr=r['left_id'],r['right_id']
  if l!='no pair' and rr!='no pair' and pd.notna(l) and pd.notna(rr):
    l,rr=int(l),int(rr)
    if l in pos and rr in pos: mate[pos[l]]=pos[rr];mate[pos[rr]]=pos[l]
print('nodes',len(y),'in pairs',len(mate))
def gsplit(seed):
  rng=np.random.default_rng(seed);tr=[];te=[]
  for c in np.unique(y):
    idx=[i for i in np.where(y==c)[0]]
    groups=[];seen=set()
    for i in idx:
      if i in seen:continue
      g=[i]+([mate[i]] if i in mate and y[mate[i]]==c and mate[i] not in seen else [])
      seen.update(g);groups.append(g)
    order=rng.permutation(len(groups));k=int(len(idx)*.6);n=0
    for j in order:
      (tr if n<k else te).extend(groups[j]);n+=len(groups[j])
  return np.array(tr),np.array(te)
tr,te=gsplit(11)
trs=set(tr.tolist());leak=sum(1 for i in te if i in mate and mate[i] in trs)
print('train',len(tr),'test',len(te),'test nodes with mate in train',leak)
# baseline leakage in locked split
sp=json.load(open('validation/split.json'));trb={pos[b] for b in sp['train_body_ids']}
print('locked split test nodes with mate in train',sum(1 for b in sp['test_body_ids'] if pos[b] in mate and mate[pos[b]] in trb),'of',len(sp['test_body_ids']))
res={k:[] for k in['gcn','cnn','linear']}
for seed in [11,22,33]:
  for n,c in [('gcn',lambda:GCN(X.shape[1],32,len(classes))),('cnn',lambda:MorphCNN(X.shape[1],len(classes))),('linear',lambda:LinearBaseline(X.shape[1],len(classes)))]:
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    m=c();adj=Ah if n=='gcn' else None
    train_node_classifier(m,X,adj,y,tr,epochs=250,lr=.02,weight_decay=5e-4,seed=seed)
    res[n].append(evaluate(m,X,adj,y,te)[0])
for n,v in res.items():print(n,round(np.mean(v),4),round(np.std(v),4))
g=np.mean(res['gcn']);print('gate (gcn>cnn+sd & >lin+sd):',all(g>np.mean(res[k])+np.std(res[k]) for k in['cnn','linear']))

json.dump({"note":"Post-outcome, single split seed (11), descriptive only. Mates kept on one side of a class-stratified ~60/40 split. Not part of the frozen protocol.","n_nodes":int(len(y)),"nodes_with_mate":len(mate),"n_train":int(len(tr)),"n_test":int(len(te)),"test_with_mate_in_train":int(leak),"seeds":[11,22,33],"accuracy":{k:{"mean":float(np.mean(v)),"sd":float(np.std(v)),"runs":[float(x) for x in v]} for k,v in res.items()},"baseline_gate_pass":bool(all(g>np.mean(res[k])+np.std(res[k]) for k in ["cnn","linear"]))},open("validation/pair_grouped.json","w"),indent=2)
