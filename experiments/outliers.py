"""DISCOVERY arm: find annotated neurons whose wiring contradicts their
assigned cell type - candidate misannotations or graph-unusual neurons.

Method: train the GCN on ALL annotated nodes; compute per-node class
probability for the annotated label; outliers = low p(annotated class)
with high confident prediction of another class (margin > 0.5).
Output: named candidate list (body IDs), quantified, falsifiable against
independent annotation evidence (morphology, hemibrain/FlyWire labels)."""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flygnn.data import load_graph, load_annotations, load_morphology, align
from flygnn.models import normalize_adj, GCN, train_node_classifier

A, ids = load_graph()
ann = load_annotations()
inp, out = load_morphology()
A, X, y_str, classes, ids = align(A, ids, ann, inp, out, min_class=40)
cls2i = {c: i for i, c in enumerate(classes)}
y = np.array([cls2i[c] for c in y_str])
A_hat = normalize_adj(A)
all_idx = np.arange(len(y))
gcn = GCN(X.shape[1], 32, len(classes))
train_node_classifier(gcn, X, A_hat, y, all_idx, epochs=250, lr=0.02, seed=11)
gcn.eval()
with torch.no_grad():
    logits = gcn(torch.tensor(X), A_hat)
    P = torch.softmax(logits, dim=1).numpy()
p_true = P[np.arange(len(y)), y]
p_max = P.max(1)
pred = P.argmax(1)
margin = p_max - p_true
# outlier: model confidently prefers another class
out = np.where((p_true < 0.2) & (margin > 0.5))[0]
cands = []
for i in out:
    cands.append(dict(body_id=int(ids[i]), annotated=y_str[i],
                      predicted=classes[pred[i]],
                      p_annotated=float(p_true[i]),
                      p_predicted=float(p_max[i]),
                      margin=float(margin[i])))
cands.sort(key=lambda c: -c["margin"])
res = dict(n_nodes=int(len(y)), n_outliers=len(cands),
           frac=float(len(cands) / len(y)),
           method="GCN trained on all annotated nodes; outlier = p(annotated)<0.2 and margin>0.5",
           falsification="each candidate checkable against independent evidence (morphology in CATMAID/hemibrain, FlyWire cell typing); systematic enrichment of one (annotated->predicted) pair suggests a real class-boundary issue rather than noise",
           candidates=cands)
with open(os.path.join(os.path.dirname(__file__), "..", "results",
                       "misannotation_candidates.json"), "w") as f:
    json.dump(res, f, indent=1)
print("outliers:", len(cands), "of", len(y))
for c in cands[:12]:
    print(c["body_id"], c["annotated"], "->", c["predicted"],
          round(c["p_annotated"], 3), round(c["p_predicted"], 3), flush=True)
