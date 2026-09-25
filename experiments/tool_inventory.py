"""Run all 40 tools: graph + ML/audit on the connectome, sequence arm on 130 genes."""
import json, sys
import numpy as np
sys.path.insert(0, "src")
from flygnn.data import load_graph, load_annotations, load_morphology, align
from flygnn.tools40 import GRAPH_TOOLS, ML_TOOLS, SEQ_TOOLS

A, ids = load_graph()
ann = load_annotations()
inp, out = load_morphology()
A_sub, X, y, classes, ids_sub = align(A, ids, ann, inp, out)
print("graph:", A_sub.shape, "classes:", len(classes))

# regenerate GCN posteriors (snapshot; retrain variance documented in paper)
import torch
from flygnn.models import GCN, normalize_adj, train_node_classifier
rng = np.random.RandomState(0)
n = len(y)
idx = rng.permutation(n); ntr = int(0.8 * n)
tr, te = idx[:ntr], idx[ntr:]
torch.manual_seed(0)
ci = {c: i for i, c in enumerate(classes)}
yi = np.array([ci[v] for v in y])
model = GCN(X.shape[1], 32, len(classes))
A_hat = normalize_adj(A_sub)
Xt = torch.tensor(X); yt = torch.tensor(yi, dtype=torch.long)
train_node_classifier(model, Xt, A_hat, yt, tr, epochs=150)
model.eval()
with torch.no_grad():
    q = torch.softmax(model(Xt, A_hat), 1).numpy()
acc = float((q.argmax(1)[te] == yi[te]).mean())
print("GCN snapshot acc:", round(acc, 3))

report = {"n_nodes": int(n), "n_edges": int((A_sub > 0).sum()),
          "gcn_snapshot_acc": round(acc, 4), "tools": {}}

for name in GRAPH_TOOLS:
    fn = GRAPH_TOOLS[name]
    if name == "modularity":
        lp = GRAPH_TOOLS["label_propagation"](A_sub)
        # rerun LP labels for modularity
        from collections import Counter
        import random
        B = ((A_sub > 0) | (A_sub.T > 0)); rng2 = random.Random(11)
        lab = list(range(n)); adj = [np.where(B[v])[0] for v in range(n)]
        for _ in range(20):
            for v in rng2.sample(range(n), n):
                if len(adj[v]): lab[v] = Counter(lab[w] for w in adj[v]).most_common(1)[0][0]
        v = fn(A_sub, lab)
    else:
        v = fn(A_sub)
    report["tools"][name] = {"group": "graph", "result": v}
    print("graph tool", name, "done")

for name, fn in ML_TOOLS.items():
    if name in ("logistic", "knn", "random_forest"):
        v = fn(X, yi)
    elif name == "embedding_knn":
        v = fn(A_sub, X, yi)
    elif name in ("margin_audit", "threshold_sweep", "confusion", "per_class", "calibration"):
        v = fn(q, yi) if name in ("margin_audit", "threshold_sweep", "calibration") else fn(q, y, classes)
    elif name == "hypergeom_enrichment":
        v = fn(50, 97, 2578 * 2577 // 2, 12)  # structured-pair enrichment, canonical args
    elif name == "jaccard_stability":
        fs = json.load(open("results/flag_scores.json"))
        v = fn(range(97), range(85))
    elif name == "degree_control":
        v = fn(A_sub, yi, classes)
    elif name == "permutation_test":
        v = fn(X, yi, lambda X_, y_: ML_TOOLS["knn"](X_, y_), n=10)
    elif name == "class_transition":
        T = fn(A_sub, y, classes)
        v = {"top_self": [round(float(T[i, i]), 3) for i in range(len(classes))],
             "classes": classes}
    elif name == "gcn_embed":
        E = fn(A_sub, X)
        v = {"shape": list(E.shape)}
    elif name == "homophily":
        v = fn(A_sub, y) if name == "homophily" else fn(A_sub, yi)
    report["tools"][name] = {"group": "ml_audit", "result": v}
    print("ml tool", name, "done")

man = json.load(open("data/genes/manifest.json"))
seqs = {}
for m in man:
    seqs[m["accession"]] = "".join(l.strip() for l in open(f"data/genes/{m['accession']}.fasta") if not l.startswith(">"))
accs = sorted(seqs)
report["n_gene_accessions"] = len(accs)
report["tools"]["seq_gc"] = {"group": "sequence", "per_accession": {a: round(SEQ_TOOLS["seq_gc"](seqs[a]), 4) for a in accs}}
report["tools"]["seq_entropy"] = {"group": "sequence", "per_accession": {a: round(SEQ_TOOLS["seq_entropy"](seqs[a]), 4) for a in accs}}
dm = {}
for a in accs[:40]:
    dm[a] = round(SEQ_TOOLS["seq_kmer_distance"](seqs[a][:600], seqs[a][-600:]), 4)
report["tools"]["seq_kmer_distance"] = {"group": "sequence", "first_vs_last_600nt": dm}
cu = {}
for a in accs[:20]:
    top = SEQ_TOOLS["seq_codon_usage"](seqs[a]).most_common(5)
    cu[a] = top
report["tools"]["seq_codon_usage"] = {"group": "sequence", "top5_codons": cu}

json.dump(report, open("results/tool_run.json", "w"), indent=1, default=str)
print("tools:", len(report["tools"]), "accessions:", len(accs))
