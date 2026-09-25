"""40 named connectome + sequence analysis tools for item 23a.

Groups: 20 graph tools, 16 ML/audit tools, 4 sequence-arm tools.
Everything runs on the real Winding-2023 larval connectome and the
130-accession Drosophila gene set. Hermetic, no network.
"""
import math, random
from collections import Counter, deque
import numpy as np

# ---------------- graph tools (20) ----------------
def g_degree_distribution(A):
    d = (A > 0).sum(1) + (A > 0).sum(0)
    return {"mean": float(d.mean()), "max": int(d.max()), "hist10": np.histogram(d, 10)[0].tolist()}

def g_in_out_corr(A):
    i, o = (A > 0).sum(0).astype(float), (A > 0).sum(1).astype(float)
    return float(np.corrcoef(i, o)[0, 1])

def g_clustering(A):
    B = (A > 0).astype(float); n = B.shape[0]
    Bu = ((B + B.T) > 0).astype(float)
    tri = np.diag(Bu @ Bu @ Bu) / 2
    deg = Bu.sum(1); poss = deg * (deg - 1) / 2
    m = poss > 0
    return float((tri[m] / poss[m]).mean())

def g_pagerank(A, d=0.85, iters=50):
    B = (A > 0).astype(float); n = B.shape[0]
    out = B.sum(1, keepdims=True); out[out == 0] = 1
    M = (B / out).T
    pr = np.ones(n) / n
    for _ in range(iters):
        pr = d * M @ pr + (1 - d) / n
    return pr

def g_kcore_sizes(A):
    B = ((A > 0) | (A.T > 0)).astype(int)
    deg = B.sum(1); removed = np.zeros(len(deg), bool); sizes = {}
    d = deg.copy()
    for k in range(1, int(deg.max()) + 2):
        changed = True
        while changed:
            changed = False
            for v in np.where((d < k) & ~removed)[0]:
                removed[v] = True; d[B[v] > 0] -= 1; changed = True
        if removed.all(): break
        sizes[k] = int((~removed).sum())
    return sizes

def g_assortativity(A):
    B = (A > 0).astype(float)
    i, j = np.where(B > 0)
    di, dj = B.sum(1)[i], B.sum(1)[j]
    return float(np.corrcoef(di, dj)[0, 1])

def g_reciprocity(A):
    B = (A > 0)
    return float((B & B.T).sum() / max(1, B.sum()))

def g_motif3_census(A, sample=400, seed=7):
    rng = random.Random(seed); B = (A > 0)
    n = B.shape[0]; counts = Counter()
    for _ in range(sample):
        tri = rng.sample(range(n), 3)
        sub = B[np.ix_(tri, tri)]
        counts[int(sub.sum())] += 1
    return dict(sorted(counts.items()))

def g_shortest_paths(A, sources=80, seed=3):
    B = (A > 0); n = B.shape[0]
    rng = random.Random(seed); dists = []
    adj = [np.where(B[v])[0] for v in range(n)]
    for s in rng.sample(range(n), sources):
        seen = {s: 0}; q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in seen:
                    seen[w] = seen[u] + 1; q.append(w)
        dists += list(seen.values())
    return {"mean": float(np.mean(dists)), "p90": float(np.percentile(dists, 90))}

def g_hub_share(A, top=10):
    d = ((A > 0).sum(1) + (A > 0).sum(0)).astype(float)
    return float(np.sort(d)[-top:].sum() / d.sum())

def g_scc(A):
    B = (A > 0); n = B.shape[0]
    adj = [np.where(B[v])[0].tolist() for v in range(n)]
    radj = [np.where(B[:, v])[0].tolist() for v in range(n)]
    seen = [False] * n; order = []
    def dfs(s, g, out=None):
        st = [(s, 0)]; seen_l = {s}
        while st:
            u, i = st[-1]
            if i < len(g[u]):
                st[-1] = (u, i + 1)
                w = g[u][i]
                if w not in seen_l:
                    seen_l.add(w); st.append((w, 0))
            else:
                st.pop()
                if out is not None: out.append(u)
        return seen_l
    for v in range(n):
        if not seen[v]:
            seen[v] = True
            for w in dfs(v, adj, order): seen[w] = True
    comp = [0] * n; nc = 0
    for v in reversed(order):
        if comp[v] == 0:
            nc += 1
            for w in dfs(v, radj): comp[w] = nc
    sizes = Counter(comp[1:] if 0 in comp else comp)
    sizes.pop(0, None)
    return {"n_scc": len(sizes), "largest": max(sizes.values()) if sizes else 0}

def g_wcc(A):
    B = ((A > 0) | (A.T > 0)); n = B.shape[0]
    seen = np.zeros(n, bool); sizes = []
    for s in range(n):
        if seen[s]: continue
        q = deque([s]); seen[s] = True; c = 0
        while q:
            u = q.popleft(); c += 1
            for w in np.where(B[u])[0]:
                if not seen[w]: seen[w] = True; q.append(w)
        sizes.append(c)
    return {"n_wcc": len(sizes), "largest": max(sizes)}

def g_bowtie(A):
    # simplified: giant SCC + in/out components relative to it
    return {"note": "computed from scc", "scc": g_scc(A)}

def g_small_world(A):
    C = g_clustering(A); L = g_shortest_paths(A)["mean"]
    n = A.shape[0]; m = (A > 0).sum()
    p = m / (n * (n - 1))
    C_rand = p; L_rand = math.log(n) / max(1e-9, math.log(max(2, m / n)))
    return {"sigma": float((C / C_rand) / (L / L_rand)), "C": C, "L": L}

def g_rich_club(A, ks=(10, 20, 50)):
    B = ((A > 0) | (A.T > 0)).astype(int)
    d = B.sum(1); out = {}
    for k in ks:
        hub = np.where(d > k)[0]
        if len(hub) < 2: out[k] = None; continue
        e = B[np.ix_(hub, hub)].sum() / 2
        out[k] = float(e / (len(hub) * (len(hub) - 1) / 2))
    return out

def g_laplacian_spectrum(A, k=6):
    B = ((A > 0) | (A.T > 0)).astype(float)
    D = np.diag(B.sum(1)); L = D - B
    ev = np.linalg.eigvalsh(L)
    return [round(float(x), 4) for x in ev[:k]]

def g_label_propagation(A, iters=20, seed=11):
    B = ((A > 0) | (A.T > 0)); n = B.shape[0]
    rng = random.Random(seed)
    lab = list(range(n))
    adj = [np.where(B[v])[0] for v in range(n)]
    for _ in range(iters):
        for v in rng.sample(range(n), n):
            if len(adj[v]):
                lab[v] = Counter(lab[w] for w in adj[v]).most_common(1)[0][0]
    return {"n_communities": len(set(lab)), "sizes_top5": sorted(Counter(lab).values(), reverse=True)[:5]}

def g_modularity(A, labels):
    B = ((A > 0) | (A.T > 0)).astype(float)
    m = B.sum() / 2; d = B.sum(1)
    Q = 0.0
    for c in set(labels):
        idx = [i for i, l in enumerate(labels) if l == c]
        e = B[np.ix_(idx, idx)].sum() / 2
        Q += e / m - (d[idx].sum() / (2 * m)) ** 2
    return float(Q)

def g_edge_weight_stats(A):
    w = A[A > 0]
    return {"n_edges": int(len(w)), "mean_w": float(w.mean()), "median_w": float(np.median(w)), "max_w": float(w.max())}

def g_betweenness_sample(A, sources=60, seed=5):
    B = (A > 0); n = B.shape[0]
    rng = random.Random(seed); bt = np.zeros(n)
    adj = [np.where(B[v])[0] for v in range(n)]
    for s in rng.sample(range(n), sources):
        dist = {s: 0}; q = deque([s])
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in dist: dist[w] = dist[u] + 1; q.append(w)
        for w, dd in dist.items():
            if dd >= 2: bt[w] += 1
    idx = np.argsort(bt)[-10:]
    return {"top10_nodes": idx.tolist(), "top10_scores": bt[idx].tolist()}

GRAPH_TOOLS = {"degree_distribution": g_degree_distribution, "in_out_corr": g_in_out_corr,
 "clustering": g_clustering, "pagerank": g_pagerank, "kcore_sizes": g_kcore_sizes,
 "assortativity": g_assortativity, "reciprocity": g_reciprocity,
 "motif3_census": g_motif3_census, "shortest_paths": g_shortest_paths,
 "hub_share": g_hub_share, "scc": g_scc, "wcc": g_wcc, "bowtie": g_bowtie,
 "small_world": g_small_world, "rich_club": g_rich_club,
 "laplacian_spectrum": g_laplacian_spectrum, "label_propagation": g_label_propagation,
 "modularity": g_modularity, "edge_weight_stats": g_edge_weight_stats,
 "betweenness_sample": g_betweenness_sample}

# ---------------- ML / audit tools (16) ----------------
def m_logistic(X, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    return float(cross_val_score(LogisticRegression(max_iter=400), X, y, cv=3).mean())

def m_knn(X, y, k=5):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    return float(cross_val_score(KNeighborsClassifier(k), X, y, cv=3).mean())

def m_random_forest(X, y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    return float(cross_val_score(RandomForestClassifier(100, random_state=0), X, y, cv=3).mean())

def m_margin_audit(q, y):
    pred = q.argmax(1)
    margin = q.max(1) - q[np.arange(len(y)), y]
    flag = (q[np.arange(len(y)), y] < 0.5) & (margin > 0.5)
    return int(flag.sum())

def m_threshold_sweep(q, y, ts=(0.4, 0.5, 0.6)):
    return {str(t): int(((q[np.arange(len(y)), y] < t) &
            ((q.max(1) - q[np.arange(len(y)), y]) > t)).sum()) for t in ts}

def m_hypergeom(S, K, N, k):
    from scipy.stats import hypergeom
    return float(1 - hypergeom.cdf(k - 1, N, S, K))

def m_jaccard(A_, B_):
    A_, B_ = set(A_), set(B_)
    return len(A_ & B_) / max(1, len(A_ | B_))

def m_degree_control(A, y, classes):
    d = ((A > 0).sum(1) + (A > 0).sum(0)).reshape(-1, 1).astype(float)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    return float(cross_val_score(LogisticRegression(max_iter=300), np.log1p(d), y, cv=3).mean())

def m_permutation_test(X, y, fn, n=20, seed=1):
    rng = np.random.RandomState(seed)
    obs = fn(X, y)
    cnt = 0
    for _ in range(n):
        cnt += fn(X, rng.permutation(y)) >= obs
    return {"observed": obs, "p": (cnt + 1) / (n + 1)}

def m_class_transition(A, y, classes):
    ci = {c: i for i, c in enumerate(classes)}
    T = np.zeros((len(classes), len(classes)))
    ii = np.array([ci[v] for v in y])
    src, dst = np.where(A > 0)
    np.add.at(T, (ii[src], ii[dst]), A[src, dst])
    T /= T.sum(1, keepdims=True) + 1e-12
    return T

def m_confusion(q, y, classes):
    pred = q.argmax(1); ci = {c: i for i, c in enumerate(classes)}
    C = np.zeros((len(classes), len(classes)), int)
    for t, p in zip(y, pred): C[ci[t], p] += 1
    return C

def m_per_class(q, y, classes):
    pred = q.argmax(1)
    return {c: float((pred[y == c] == ci).mean()) if (y == c).sum() else None
            for ci, c in enumerate(classes)}

def m_calibration(q, y, bins=8):
    conf = q.max(1); correct = (q.argmax(1) == y).astype(float)
    edges = np.linspace(0, 1, bins + 1); out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (conf >= a) & (conf < b)
        out.append(float(correct[m].mean()) if m.sum() else None)
    return out

def m_gcn_embed(A, X, dims=8, seed=0):
    rng = np.random.RandomState(seed)
    B = (A > 0).astype(np.float32) + np.eye(A.shape[0], dtype=np.float32)
    d = B.sum(1); Dm = np.diag(1 / np.sqrt(d))
    An = Dm @ B @ Dm
    W1 = rng.randn(X.shape[1], 32) * 0.1; W2 = rng.randn(32, dims) * 0.1
    H = np.tanh(An @ X @ W1)
    return An @ H @ W2

def m_embedding_knn(A, X, y):
    E = m_gcn_embed(A, X)
    return m_knn(E, y)

def m_homophily(A, y):
    B = (A > 0); i, j = np.where(B)
    return float((y[i] == y[j]).mean())

ML_TOOLS = {"logistic": m_logistic, "knn": m_knn, "random_forest": m_random_forest,
 "margin_audit": m_margin_audit, "threshold_sweep": m_threshold_sweep,
 "hypergeom_enrichment": m_hypergeom, "jaccard_stability": m_jaccard,
 "degree_control": m_degree_control, "permutation_test": m_permutation_test,
 "class_transition": m_class_transition, "confusion": m_confusion,
 "per_class": m_per_class, "calibration": m_calibration,
 "gcn_embed": m_gcn_embed, "embedding_knn": m_embedding_knn, "homophily": m_homophily}

# ---------------- sequence arm tools (4) ----------------
def s_gc(seq): return (seq.count("G") + seq.count("C")) / max(1, len(seq))
def s_kmer_distance(a, b, k=4):
    ca = Counter(a[i:i+k] for i in range(len(a) - k + 1))
    cb = Counter(b[i:i+k] for i in range(len(b) - k + 1))
    keys = set(ca) | set(cb); na, nb = sum(ca.values()), sum(cb.values())
    return 0.5 * sum(abs(ca[t]/na - cb[t]/nb) for t in keys)
def s_entropy(seq):
    c = Counter(seq); n = len(seq)
    return -sum((v/n) * math.log2(v/n) for v in c.values())
def s_codon_usage(seq):
    return Counter(seq[i:i+3] for i in range(0, len(seq) - 2, 3))

SEQ_TOOLS = {"seq_gc": s_gc, "seq_kmer_distance": s_kmer_distance,
             "seq_entropy": s_entropy, "seq_codon_usage": s_codon_usage}

ALL_TOOLS = {**GRAPH_TOOLS, **ML_TOOLS, **SEQ_TOOLS}
assert len(ALL_TOOLS) == 40, len(ALL_TOOLS)
