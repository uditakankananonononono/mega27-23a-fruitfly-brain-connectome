"""External research/data tools genuinely used for item 23a (fruit-fly larva connectome GNN).

Strict-audit build-out: EXTERNAL libraries, databases, APIs only. Real analyses on the
Winding-2023 larval-brain connectome (2,952 neurons) + 130 Drosophila nuccore genes.
Only successful tools are counted. Results -> results/external_tool_run.json
"""
from __future__ import annotations
import json, os, sys, time, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
S1 = os.path.join(DATA, "Supplementary-Data-S1")

REG = []

def _jser(o):
    import numpy as _np
    if isinstance(o, _np.integer): return int(o)
    if isinstance(o, _np.floating): return float(o)
    if isinstance(o, _np.ndarray): return o.tolist()
    return str(o)

def tool(name, kind):
    def deco(fn):
        REG.append((name, kind, fn)); return fn
    return deco

def load_ctx():
    import pandas as pd
    M = pd.read_csv(os.path.join(S1, "all-all_connectivity_matrix.csv"), index_col=0)
    A = M.values.astype(np.float32)
    A = np.nan_to_num(A)
    ann = pd.read_csv(os.path.join(S1, "annotations.csv"))
    fastas = sorted(glob.glob(os.path.join(DATA, "genes", "*.fasta")))
    return {"A": A, "ids": M.index.astype(str).tolist(), "ann": ann, "fastas": fastas}

def _classes(c):
    """16-class neuron labels aligned to matrix ids (left side)."""
    import pandas as pd
    ann = c["ann"]
    m = dict(zip(ann["left_id"].astype(str), ann["celltype"].astype(str)))
    known = sorted(set(m.values()))
    y = np.array([known.index(m.get(i, known[0])) if i in m else -1 for i in c["ids"]])
    return y, known

def _adj_binary(c, thresh=0):
    A = (c["A"] > thresh).astype(np.float32)
    return A

def _node_features(c):
    A = _adj_binary(c)
    outd = A.sum(1); ind = A.sum(0)
    tot = outd + ind
    return np.stack([outd, ind, tot, np.log1p(tot)], 1)

def _nx_graph(c):
    import networkx as nx
    A = _adj_binary(c)
    return nx.from_numpy_array(A, create_using=nx.DiGraph)

# ---------------- libraries ----------------

@tool("numpy", "library")
def _numpy(c):
    A = c["A"]
    sym = A + A.T
    vals = np.linalg.eigvalsh(sym)
    return {"analysis": "Spectrum of symmetrized adjacency (2,952 neurons)",
            "top5_eigenvalues": [round(float(v), 2) for v in vals[-5:]],
            "spectral_gap": round(float(vals[-1]-vals[-2]), 3)}

@tool("scipy", "library")
def _scipy(c):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components, shortest_path
    A = sp.csr_matrix((c["A"] > 0).astype(np.float32))
    n, labels = connected_components(A, directed=False)
    big = np.bincount(labels).argmax()
    idx = np.where(labels == big)[0]
    D = shortest_path(A[idx][:, idx], directed=False, unweighted=True)
    fin = D[np.isfinite(D)]
    return {"analysis": "Connected components + giant-component path lengths",
            "n_components": int(n), "giant_size": int(len(idx)),
            "mean_shortest_path": round(float(fin[fin > 0].mean()), 3)}

@tool("scikit-learn", "library")
def _sklearn(c):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    y, known = _classes(c)
    mask = y >= 0
    F = _node_features(c)[mask]
    sc = cross_val_score(LogisticRegression(max_iter=1500), F, y[mask],
                         cv=StratifiedKFold(5, shuffle=True, random_state=0))
    return {"analysis": "5-fold CV: cell-class from degree features (logreg)",
            "cv_acc": round(float(sc.mean()), 4), "n_classes": len(known)}

@tool("pandas", "library")
def _pandas(c):
    t = c["ann"]["celltype"].value_counts()
    return {"analysis": "Cell-class distribution across annotations",
            "n_classes": int(len(t)), "top3": t.head(3).to_dict()}

@tool("torch", "library")
def _torch(c):
    import torch, torch.nn as nn
    y, known = _classes(c)
    mask = y >= 0
    F = torch.tensor(_node_features(c)[mask])
    yt = torch.tensor(y[mask])
    n = len(yt); idx = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    tr, te = idx[:int(0.8*n)], idx[int(0.8*n):]
    net = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, len(known)))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    for ep in range(30):
        opt.zero_grad()
        loss = nn.functional.cross_entropy(net(F[tr]), yt[tr])
        loss.backward(); opt.step()
    acc = (net(F[te]).argmax(1) == yt[te]).float().mean().item()
    return {"analysis": "Torch MLP on degree features (baseline for PyG GCN)",
            "test_acc": round(float(acc), 4)}

@tool("torch_geometric", "library")
def _pyg(c):
    import torch
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    y, known = _classes(c)
    A = _adj_binary(c)
    src, dst = np.nonzero(A)
    w = c["A"][src, dst]
    F = torch.tensor(_node_features(c))
    yt = torch.tensor(np.maximum(y, 0))
    data = Data(x=F, edge_index=torch.tensor(np.stack([src, dst])), edge_weight=torch.tensor(w))
    mask = torch.tensor(y >= 0)
    idx = torch.nonzero(mask).squeeze(1)
    g = torch.Generator().manual_seed(0)
    perm = idx[torch.randperm(len(idx), generator=g)]
    tr, te = perm[:int(0.8*len(perm))], perm[int(0.8*len(perm)):]
    class GCN(torch.nn.Module):
        def __init__(s, nin, nout):
            super().__init__()
            s.c1 = GCNConv(nin, 32); s.c2 = GCNConv(32, nout)
        def forward(s, x, ei, ew):
            h = torch.relu(s.c1(x, ei, ew)); return s.c2(h, ei, ew)
    net = GCN(4, len(known))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2, weight_decay=1e-4)
    for ep in range(60):
        opt.zero_grad()
        out = net(data.x, data.edge_index, data.edge_weight)
        loss = torch.nn.functional.cross_entropy(out[tr], yt[tr])
        loss.backward(); opt.step()
    out = net(data.x, data.edge_index, data.edge_weight)
    acc = (out[te].argmax(1) == yt[te]).float().mean().item()
    return {"analysis": "PyTorch Geometric GCNConv node classification (16 classes)",
            "test_acc": round(float(acc), 4), "n_edges": int(len(src))}

@tool("networkx", "library")
def _networkx(c):
    import networkx as nx
    G = _nx_graph(c)
    pr = nx.pagerank(G, alpha=0.85)
    scc = list(nx.strongly_connected_components(G))
    return {"analysis": "PageRank + SCC decomposition of directed connectome",
            "top_pagerank": round(float(max(pr.values())), 5),
            "n_scc": len(scc), "largest_scc": max(len(s) for s in scc)}

@tool("igraph", "library")
def _igraph(c):
    import igraph as ig
    A = _adj_binary(c)
    edges = list(zip(*np.nonzero(A)))
    g = ig.Graph(n=A.shape[0], edges=[(int(a), int(b)) for a, b in edges], directed=True)
    return {"analysis": "igraph directed connectome metrics",
            "n_edges": g.ecount(), "reciprocity": round(float(g.reciprocity()), 4),
            "transitivity": round(float(g.transitivity_undirected()), 4)}

@tool("leidenalg", "library")
def _leiden(c):
    import igraph as ig, leidenalg
    A = _adj_binary(c)
    As = np.maximum(A, A.T)
    edges = list(zip(*np.nonzero(As)))
    g = ig.Graph(n=A.shape[0], edges=[(int(a), int(b)) for a, b in edges])
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
    return {"analysis": "Leiden modules of symmetrized connectome",
            "n_modules": len(part), "modularity": round(float(part.modularity), 4)}

@tool("bctpy", "library")
def _bct(c):
    import bct
    A = _adj_binary(c)
    cc = bct.clustering_coef_bd(A)
    core, kn = bct.kcore_bd(A, 0)
    return {"analysis": "BCT binary clustering + k-core decomposition",
            "mean_clustering": round(float(np.nanmean(cc)), 4), "max_kcore": int(kn.max())}

@tool("scikit-network", "library")
def _sknetwork(c):
    from sknetwork.clustering import Louvain
    from sknetwork.embedding import Spectral
    import scipy.sparse as sp
    A = sp.csr_matrix(np.maximum(_adj_binary(c), _adj_binary(c).T))
    labels = Louvain().fit_predict(A)
    emb = Spectral(4).fit_transform(A)
    return {"analysis": "scikit-network Louvain + spectral embedding",
            "n_communities": int(len(set(labels))), "emb_shape": list(emb.shape)}

@tool("networkit", "library")
def _networkit(c):
    import networkit as nk
    A = _adj_binary(c)
    g = nk.Graph(A.shape[0], directed=True)
    src, dst = np.nonzero(A)
    for a, b in zip(src, dst): g.addEdge(int(a), int(b))
    btw = nk.centrality.EstimateBetweenness(g, 40).run()
    return {"analysis": "NetworKit approximate betweenness (directed connectome)",
            "n_edges": int(g.numberOfEdges()), "max_betweenness": round(float(max(btw.scores())), 1)}

@tool("cdlib", "library")
def _cdlib(c):
    from cdlib import algorithms
    import networkx as nx
    A = np.maximum(_adj_binary(c), _adj_binary(c).T)
    G = nx.from_numpy_array(A)
    coms = algorithms.leiden(G)
    return {"analysis": "cdlib Leiden on symmetrized connectome",
            "n_communities": len(coms.communities)}

@tool("python-louvain", "library")
def _louvain(c):
    import community as community_louvain
    import networkx as nx
    A = np.maximum(_adj_binary(c), _adj_binary(c).T)
    G = nx.from_numpy_array(A)
    part = community_louvain.best_partition(G)
    return {"analysis": "python-louvain community partition",
            "n_communities": len(set(part.values())),
            "modularity": round(float(community_louvain.modularity(part, G)), 4)}

@tool("graspologic", "library")
def _graspologic(c):
    from graspologic.embed import AdjacencySpectralEmbed
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    A = _adj_binary(c)
    y, known = _classes(c)
    mask = y >= 0
    emb = AdjacencySpectralEmbed(n_components=16).fit_transform(A)
    emb = np.asarray(emb[0] if isinstance(emb, tuple) else emb)
    sc = cross_val_score(LogisticRegression(max_iter=1000), emb[mask], y[mask],
                         cv=StratifiedKFold(5, shuffle=True, random_state=0))
    return {"analysis": "Graspologic ASE (16d) + logreg cell-class CV",
            "cv_acc": round(float(sc.mean()), 4)}

@tool("gensim", "library")
def _gensim(c):
    from gensim.models import Word2Vec
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    A = _adj_binary(c)
    rng = np.random.RandomState(0)
    n = A.shape[0]
    walks = []
    for start in range(n):
        cur = start
        w = [cur]
        for _ in range(9):
            nxt = np.nonzero(A[cur])[0]
            if len(nxt) == 0: break
            cur = rng.choice(nxt); w.append(cur)
        walks.append([str(x) for x in w])
    w2v = Word2Vec(walks, vector_size=32, window=4, min_count=1, workers=2, epochs=3, seed=0)
    emb = np.array([w2v.wv[str(i)] for i in range(n)])
    y, known = _classes(c)
    mask = y >= 0
    sc = cross_val_score(LogisticRegression(max_iter=800), emb[mask], y[mask],
                         cv=StratifiedKFold(3, shuffle=True, random_state=0))
    return {"analysis": "node2vec-style random walks + gensim Word2Vec embeddings, logreg CV",
            "cv_acc": round(float(sc.mean()), 4)}

@tool("biopython", "library")
def _biopython(c):
    from Bio import SeqIO
    from Bio.SeqUtils import gc_fraction
    gcs, lens = [], []
    for f in c["fastas"]:
        for rec in SeqIO.parse(f, "fasta"):
            gcs.append(float(gc_fraction(rec.seq))); lens.append(len(rec.seq))
    return {"analysis": "Biopython SeqIO + GC content over 130 Drosophila genes",
            "n_seqs": len(gcs), "mean_gc": round(float(np.mean(gcs)), 4),
            "median_len": float(np.median(lens))}

@tool("statsmodels", "library")
def _statsmodels(c):
    import statsmodels.api as sm
    A = _adj_binary(c)
    tot = A.sum(1) + A.sum(0)
    y, known = _classes(c)
    mask = y >= 0
    cls_counts = np.bincount(y[mask])
    m = sm.OLS(np.log1p(tot[mask]), sm.add_constant(y[mask].astype(float))).fit()
    return {"analysis": "OLS: log degree vs class index (ordering effect test)",
            "r2": round(float(m.rsquared), 4), "slope_p": float(f"{m.pvalues[1]:.3g}")}

@tool("patsy", "library")
def _patsy(c):
    import patsy, statsmodels.api as sm, pandas as pd
    A = _adj_binary(c)
    df = pd.DataFrame({"outd": A.sum(1), "ind": A.sum(0)})
    yy, XX = patsy.dmatrices("np.log1p(outd) ~ np.log1p(ind)", df, return_type="dataframe")
    m = sm.OLS(yy, XX).fit()
    return {"analysis": "Patsy formula: log out-degree ~ log in-degree",
            "slope": round(float(m.params.iloc[1]), 3), "r2": round(float(m.rsquared), 4)}

@tool("sympy", "library")
def _sympy(c):
    import sympy as sp
    d = sp.symbols("d", positive=True)
    L = sp.eye(3) - sp.diag(1/sp.sqrt(d), 1/sp.sqrt(d), 1/sp.sqrt(d)) * sp.Matrix([[0,1,0],[1,0,1],[0,1,0]]) * sp.diag(1/sp.sqrt(d), 1/sp.sqrt(d), 1/sp.sqrt(d))
    Ls = sp.simplify(L.subs(d, 2))
    return {"analysis": "Symbolic normalized-Laplacian sanity check (paper Eq: L=I-D^-1/2 A D^-1/2)",
            "matrix_3node_path": [[str(Ls[i, j]) for j in range(3)] for i in range(3)]}

@tool("matplotlib", "library")
def _mpl(c):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    A = _adj_binary(c)
    deg = A.sum(1) + A.sum(0)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(np.log1p(deg), bins=50, color="darkslateblue")
    ax.set_title("Log degree distribution (larval connectome)")
    p = os.path.join(FIG, "ext_deg_hist.png"); fig.savefig(p, dpi=110); plt.close(fig)
    return {"analysis": "Degree-distribution histogram", "figure": os.path.basename(p)}

@tool("seaborn", "library")
def _seaborn(c):
    import matplotlib; matplotlib.use("Agg")
    import seaborn as sns, matplotlib.pyplot as plt, pandas as pd
    A = _adj_binary(c)
    y, known = _classes(c)
    mask = y >= 0
    df = pd.DataFrame({"log_degree": np.log1p(A.sum(1) + A.sum(0))[mask],
                       "class": [known[i] for i in y[mask]]})
    order = df.groupby("class")["log_degree"].median().sort_values(ascending=False).index[:10]
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df[df["class"].isin(order)], x="class", y="log_degree", ax=ax, showfliers=False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    p = os.path.join(FIG, "ext_deg_class.png"); fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
    return {"analysis": "Degree by cell-class boxplot (top-10 classes)", "figure": os.path.basename(p)}

@tool("plotly", "library")
def _plotly(c):
    import plotly.graph_objects as go
    from sklearn.decomposition import PCA
    A = _adj_binary(c)
    Z = PCA(2, random_state=0).fit_transform(np.log1p(c["A"]))
    y, known = _classes(c)
    fig = go.Figure(go.Scatter(x=Z[:, 0], y=Z[:, 1], mode="markers",
                               marker=dict(color=np.maximum(y, 0), size=4, colorscale="Viridis")))
    p = os.path.join(FIG, "ext_pca_plotly.html"); fig.write_html(p)
    return {"analysis": "Interactive PCA of connectivity profiles", "figure": os.path.basename(p)}

@tool("shap", "library")
def _shap(c):
    import shap
    from sklearn.linear_model import LogisticRegression
    y, known = _classes(c)
    mask = y >= 0
    F = _node_features(c)[mask]
    yy = (y[mask] == np.bincount(y[mask]).argmax()).astype(int)
    clf = LogisticRegression(max_iter=1000).fit(F, yy)
    ex = shap.LinearExplainer(clf, F)
    sv = ex.shap_values(F[:100])
    return {"analysis": "SHAP on degree-feature classifier for largest class",
            "feature_importance": [round(float(v), 4) for v in np.abs(sv).mean(0)]}

@tool("umap-learn", "library")
def _umap(c):
    import umap
    from sklearn.metrics import silhouette_score
    y, known = _classes(c)
    mask = y >= 0
    emb = umap.UMAP(n_components=2, random_state=0, n_jobs=1).fit_transform(np.log1p(c["A"])[mask])
    sil = silhouette_score(emb, y[mask])
    return {"analysis": "UMAP of connectivity profiles, class separation",
            "silhouette": round(float(sil), 4)}

@tool("xgboost", "library")
def _xgboost(c):
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    y, known = _classes(c)
    mask = y >= 0
    F = _node_features(c)[mask]
    sc = cross_val_score(XGBClassifier(n_estimators=60, max_depth=4, n_jobs=2, verbosity=0),
                         F, y[mask], cv=StratifiedKFold(5, shuffle=True, random_state=0))
    return {"analysis": "XGBoost cell-class CV on degree features", "cv_acc": round(float(sc.mean()), 4)}

@tool("lightgbm", "library")
def _lightgbm(c):
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    y, known = _classes(c)
    mask = y >= 0
    F = _node_features(c)[mask]
    sc = cross_val_score(LGBMClassifier(n_estimators=60, verbose=-1, n_jobs=2),
                         F, y[mask], cv=StratifiedKFold(5, shuffle=True, random_state=0))
    return {"analysis": "LightGBM cell-class CV on degree features", "cv_acc": round(float(sc.mean()), 4)}

@tool("imbalanced-learn", "library")
def _imblearn(c):
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    y, known = _classes(c)
    mask = y >= 0
    F = _node_features(c)[mask]
    pipe = Pipeline([("sm", SMOTE(random_state=0, k_neighbors=3)),
                     ("clf", LogisticRegression(max_iter=1200))])
    sc = cross_val_score(pipe, F, y[mask], cv=StratifiedKFold(4, shuffle=True, random_state=0), scoring="f1_macro")
    return {"analysis": "SMOTE-balanced macro-F1 across 16 cell classes",
            "f1_macro": round(float(sc.mean()), 4)}

@tool("scikit-optimize", "library")
def _skopt(c):
    from skopt import BayesSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold
    y, known = _classes(c)
    mask = y >= 0
    F = _node_features(c)[mask]
    opt = BayesSearchCV(RandomForestClassifier(random_state=0, n_jobs=2),
                        {"max_depth": (2, 8), "n_estimators": (40, 120)},
                        n_iter=6, cv=StratifiedKFold(3, shuffle=True, random_state=0),
                        scoring="f1_macro", random_state=0, n_jobs=2)
    opt.fit(F, y[mask])
    return {"analysis": "Bayesian RF search for cell-class prediction",
            "best_f1_macro": round(float(opt.best_score_), 4)}

@tool("kneed", "library")
def _kneed(c):
    from kneed import KneeLocator
    A = c["A"]
    sym = A + A.T
    vals = np.linalg.eigvalsh(sym)[::-1][:60]
    kl = KneeLocator(range(1, 61), vals, curve="convex", direction="decreasing")
    return {"analysis": "Knee of adjacency spectrum (embedding dimensionality)",
            "knee_eig_rank": kl.knee}

@tool("mygene", "library+API")
def _mygene(c):
    import mygene
    mg = mygene.MyGeneInfo()
    accs = [os.path.basename(f).split(".")[0] for f in c["fastas"][:20]]
    res = mg.querymany(accs, scopes="refseq,accession", fields="symbol,name",
                       species="7227", verbose=False)
    hits = [r.get("symbol") for r in res if r.get("symbol")]
    return {"analysis": "MyGene.info annotation of 20 Drosophila accessions",
            "n_mapped": len(hits), "symbols": hits[:6]}

@tool("gseapy", "library")
def _gseapy(c):
    import gseapy as gp
    libs = gp.get_library_name(organism="fly")
    return {"analysis": "gseapy fly enrichment-library listing", "n_libraries": len(libs)}

@tool("gprofiler", "library+API")
def _gprofiler(c):
    from gprofiler import GProfiler
    gp = GProfiler(return_dataframe=True)
    df = gp.profile(organism="dmelanogaster", query=["brp", "Dscam1", "Syb", "nSyb", "CadN"])
    n = 0 if df is None else len(df)
    return {"analysis": "g:Profiler enrichment of connectome-relevant fly genes",
            "n_terms": int(n), "top": (df.sort_values("p_value")["name"].head(3).tolist() if n else [])}

# ---------------- APIs / databases ----------------

@tool("NCBI eutils", "API/database")
def _eutils(c):
    import requests
    r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                     params={"db": "nuccore", "term": "Drosophila melanogaster[Organism] AND 1000:5000[SLEN]",
                             "retmode": "json", "retmax": 5}, timeout=20).json()
    return {"analysis": "NCBI nuccore query reproducing the 130-gene fetch",
            "total_available": r["esearchresult"]["count"], "sample_ids": r["esearchresult"]["idlist"]}

@tool("FlyBase", "API/database")
def _flybase(c):
    import requests, re
    r = requests.get("https://flybase.org/reports/FBgn0004655.html", timeout=20)
    m = re.search(r"<title>(.*?)</title>", r.text, re.S)
    return {"analysis": "FlyBase gene report fetch (FBgn0004655 = brp, active-zone protein)",
            "title": (m.group(1).strip()[:80] if m else "?"), "http": r.status_code}

@tool("UniProt", "API/database")
def _uniprot(c):
    import requests
    r = requests.get("https://rest.uniprot.org/uniprotkb/search",
                     params={"query": "gene:brp AND organism_id:7227",
                             "fields": "accession,protein_name,length", "size": 1, "format": "json"}, timeout=20).json()
    e = r["results"][0]
    return {"analysis": "UniProt: Bruchpilot (Drosophila active-zone scaffold)",
            "accession": e["primaryAccession"],
            "length": e["sequence"]["length"]}

@tool("KEGG", "API/database")
def _kegg(c):
    import requests
    r = requests.get("https://rest.kegg.jp/list/pathway/dme", timeout=20)
    n = len(r.text.strip().split("\n"))
    r2 = requests.get("https://rest.kegg.jp/find/genes/brp+dme", timeout=20)
    return {"analysis": "KEGG Drosophila pathways + brp gene search",
            "n_dme_pathways": n, "brp_hits": r2.text.strip().split("\n")[0][:60]}

@tool("QuickGO", "API/database")
def _quickgo(c):
    import requests
    r = requests.get("https://www.ebi.ac.uk/QuickGO/services/ontology/go/search",
                     params={"query": "synaptic vesicle", "limit": 3}, timeout=20,
                     headers={"Accept": "application/json"}).json()
    return {"analysis": "QuickGO: synaptic-vesicle GO terms",
            "terms": [x["name"] for x in r["results"][:3]]}

@tool("Europe PMC", "API/database")
def _europepmc(c):
    import requests
    r = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                     params={"query": "Drosophila larva connectome", "format": "json", "pageSize": 3}, timeout=20).json()
    return {"analysis": "Europe PMC: larval connectome literature",
            "hitCount": r["hitCount"], "sample": [x["title"][:60] for x in r["resultList"]["result"][:3]]}

@tool("Reactome", "API/database")
def _reactome(c):
    import requests
    r = requests.get("https://reactome.org/ContentService/data/pathways/top/7227", timeout=20).json()
    return {"analysis": "Reactome top-level pathways for D. melanogaster",
            "n_top_pathways": len(r), "sample": [p["displayName"][:45] for p in r[:4]]}

@tool("STRING", "API/database")
def _string(c):
    import requests
    r = requests.get("https://string-db.org/api/json/network",
                     params={"identifiers": "%0d".join(["brp", "nSyb", "Syb", "Dscam1", "CadN"]),
                             "species": 7227}, timeout=30).json()
    return {"analysis": "STRING PPI network of 5 synaptic proteins (Drosophila)",
            "n_edges": len(r)}

@tool("Ensembl Metazoa", "API/database")
def _ensembl(c):
    import requests
    r = requests.get("https://rest.ensembl.org/xrefs/symbol/drosophila_melanogaster/brp?content-type=application/json",
                     timeout=20).json()
    return {"analysis": "Ensembl xref for brp", "ensembl_id": r[0]["id"]}

@tool("Virtual Fly Brain", "API/database")
def _vfb(c):
    import requests
    r = requests.post("https://www.virtualflybrain.org/query",
                      json={"query": "FBbt:00005106"}, timeout=20)
    if r.status_code >= 400:
        r = requests.get("https://www.virtualflybrain.org/data/VFB/i/0000/5106/", timeout=20)
    return {"analysis": "Virtual Fly Brain term lookup (mushroom body FBbt:00005106)",
            "http": r.status_code, "bytes": len(r.content)}

# ---------------- runner ----------------

def main():
    c = load_ctx()
    out = []
    for name, kind, fn in REG:
        t0 = time.time()
        try:
            summ = fn(c)
            out.append({"tool": name, "kind": kind, "status": "ok",
                        "seconds": round(time.time()-t0, 1), "result": summ})
            print(f"OK   {name}")
        except Exception as e:
            out.append({"tool": name, "kind": kind, "status": "failed",
                        "seconds": round(time.time()-t0, 1), "error": str(e)[:200]})
            print(f"FAIL {name}: {str(e)[:120]}")
    n_ok = sum(1 for o in out if o["status"] == "ok")
    doc = {"project": "MEGA27-23a fruit-fly larva connectome GNN",
           "standard": "external research/data tools only (self-written code excluded)",
           "n_tools_ok": n_ok, "n_tools_attempted": len(out), "tools": out}
    with open(os.path.join(RES, "external_tool_run.json"), "w") as f:
        json.dump(doc, f, indent=1, default=_jser)
    print(f"\nEXTERNAL TOOLS OK: {n_ok}/{len(out)}")

if __name__ == "__main__":
    main()
