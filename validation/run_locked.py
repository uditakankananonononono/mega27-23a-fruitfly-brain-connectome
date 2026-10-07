"""Post-outcome, locked rerun. No claim of preregistration or new validation."""
import hashlib,json,os,sys,random
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from flygnn.data import load_graph,load_annotations,load_morphology,align
from flygnn.models import normalize_adj,GCN,MorphCNN,LinearBaseline,train_node_classifier,evaluate,stratified_split

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 p=json.loads((ROOT/'validation/protocol.json').read_text())
 lock=sha(ROOT/'validation/protocol.json')
 assert lock==(ROOT/'validation/protocol.sha256').read_text().split()[0], 'protocol changed'
 for name,h in p['input_sha256'].items(): assert sha(ROOT/name)==h,name
 torch.set_num_threads(p['threads']); torch.use_deterministic_algorithms(True)
 A,ids=load_graph(); inp,out=load_morphology()
 A,X,ys,classes,ids=align(A,ids,load_annotations(),inp,out,min_class=p['min_class'])
 y=np.array([classes.index(v) for v in ys]); Ah=normalize_adj(A)
 tr,te=stratified_split(y,p['train_fraction'],seed=p['split_seed'])
 split={'train_body_ids':[int(ids[i]) for i in tr],'test_body_ids':[int(ids[i]) for i in te]}
 assert hashlib.sha256(json.dumps(split,sort_keys=True,separators=(',',':')).encode()).hexdigest()==p['split_sha256']
 runs=[]
 for seed in p['seeds']:
  row={'seed':seed}
  for name,constructor in [('gcn',lambda:GCN(X.shape[1],32,len(classes))),('cnn',lambda:MorphCNN(X.shape[1],len(classes))),('linear',lambda:LinearBaseline(X.shape[1],len(classes)))]:
   random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
   model=constructor(); adj=Ah if name=='gcn' else None
   train_node_classifier(model,X,adj,y,tr,epochs=p['epochs'],lr=p['learning_rate'],weight_decay=p['weight_decay'],seed=seed)
   acc,f1=evaluate(model,X,adj,y,te); row[name]={'accuracy':acc,'macro_f1':f1}
  runs.append(row); print(row,flush=True)
 agg={k:{'mean':float(np.mean([r[k]['accuracy'] for r in runs])),'sd':float(np.std([r[k]['accuracy'] for r in runs],ddof=0))} for k in ['gcn','cnn','linear']}
 beat=all(agg['gcn']['mean']>agg[k]['mean']+agg[k]['sd'] for k in ['cnn','linear'])
 seed=p['nomination_seed']; torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
 m=GCN(X.shape[1],32,len(classes));train_node_classifier(m,X,Ah,y,np.arange(len(y)),epochs=p['epochs'],lr=p['learning_rate'],weight_decay=p['weight_decay'],seed=seed)
 m.eval()
 with torch.no_grad(): P=torch.softmax(m(torch.tensor(X),Ah),1).numpy()
 pt=P[np.arange(len(y)),y];pm=P.max(1);pred=P.argmax(1)
 chosen=np.where((pt<p['p_annotated_lt'])&((pm-pt)>p['margin_gt']))[0]
 candidates=[{'body_id':int(ids[i]),'annotated':str(ys[i]),'predicted':classes[pred[i]],'p_annotated':float(pt[i]),'margin':float(pm[i]-pt[i])} for i in chosen]
 result={'protocol_sha256':lock,'source_commit':p['source_commit'],'classes':classes,'n_nodes':len(y),'n_train':len(tr),'n_test':len(te),'runs':runs,'accuracy':agg,'baseline_gate_pass':beat,'nomination_gate_pass':bool(candidates),'frozen_gate_pass':bool(beat and candidates),'n_candidates':len(candidates),'candidates':candidates,'strict_promotion':False,'caveat':p['caveat']}
 dest=ROOT/'validation/rerun.json';dest.write_text(json.dumps(result,indent=2)+'\n');print('GATE',result['frozen_gate_pass'],'candidates',len(candidates),flush=True)
if __name__=='__main__':main()
