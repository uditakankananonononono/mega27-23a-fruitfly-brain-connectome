"""GCN vs baselines on the real larval Drosophila connectome."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flygnn.data import load_graph, load_annotations, load_morphology, align
from flygnn.models import (normalize_adj, GCN, MorphCNN, LinearBaseline,
                           train_node_classifier, evaluate, stratified_split)

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)

A, ids = load_graph()
ann = load_annotations()
inp, out = load_morphology()
A, X, y_str, classes, ids = align(A, ids, ann, inp, out, min_class=40)
print("graph:", A.shape, "classes:", len(classes), "nodes/class min 40")
cls2i = {c: i for i, c in enumerate(classes)}
y = np.array([cls2i[c] for c in y_str])

A_hat = normalize_adj(A)
tr, te = stratified_split(y, 0.6, seed=11)
res = {"n_nodes": int(A.shape[0]), "n_edges": int((A > 0).sum()),
       "classes": classes, "n_train": int(len(tr)), "n_test": int(len(te))}

runs = {}
for seed in (11, 22, 33):
    gcn = GCN(X.shape[1], 32, len(classes))
    train_node_classifier(gcn, X, A_hat, y, tr, epochs=250, lr=0.02, seed=seed)
    acc_g, f1_g = evaluate(gcn, X, A_hat, y, te)
    cnn = MorphCNN(X.shape[1], len(classes))
    train_node_classifier(cnn, X, None, y, tr, epochs=250, lr=0.02, seed=seed)
    acc_c, f1_c = evaluate(cnn, X, None, y, te)
    lin = LinearBaseline(X.shape[1], len(classes))
    train_node_classifier(lin, X, None, y, tr, epochs=250, lr=0.02, seed=seed)
    acc_l, f1_l = evaluate(lin, X, None, y, te)
    runs[seed] = dict(gcn=(acc_g, f1_g), cnn=(acc_c, f1_c), linear=(acc_l, f1_l))
    print(seed, "GCN", round(acc_g, 3), "CNN", round(acc_c, 3), "LIN", round(acc_l, 3), flush=True)

def agg(key, i):
    return float(np.mean([runs[s][key][i] for s in runs])), \
           float(np.std([runs[s][key][i] for s in runs]))
res["accuracy"] = {k: agg(k, 0) for k in ("gcn", "cnn", "linear")}
res["macro_f1"] = {k: agg(k, 1) for k in ("gcn", "cnn", "linear")}
res["verdict_gcn_beats_baselines"] = bool(
    res["accuracy"]["gcn"][0] > res["accuracy"]["cnn"][0] + res["accuracy"]["cnn"][1]
    and res["accuracy"]["gcn"][0] > res["accuracy"]["linear"][0] + res["accuracy"]["linear"][1])

fig, ax = plt.subplots(1, 2, figsize=(8.5, 3.5))
names = ["GCN (graph)", "CNN (morph.)", "Linear (morph.)"]
accs = [res["accuracy"]["gcn"], res["accuracy"]["cnn"], res["accuracy"]["linear"]]
ax[0].bar(names, [a[0] for a in accs], yerr=[a[1] for a in accs], capsize=4,
          color=["#1f77b4", "#ff7f0e", "#7f7f7f"])
ax[0].set_ylabel("test accuracy"); ax[0].set_title("Cell-type classification (3 seeds)")
deg_in = (A > 0).sum(0); deg_out = (A > 0).sum(1)
ax[1].hist(np.log10(deg_in[deg_in > 0]), bins=40, alpha=0.6, label="in-degree")
ax[1].hist(np.log10(deg_out[deg_out > 0]), bins=40, alpha=0.6, label="out-degree")
ax[1].set_xlabel("log10 partners"); ax[1].legend(); ax[1].set_title("Connectome degree distribution")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "connectome_gnn.png"), dpi=150)

with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(res, f, indent=1)
print("VERDICT gcn_beats_baselines:", res["verdict_gcn_beats_baselines"])
print("acc:", {k: (round(v[0], 3), round(v[1], 3)) for k, v in res["accuracy"].items()})
