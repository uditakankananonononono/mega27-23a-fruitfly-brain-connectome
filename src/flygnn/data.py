"""Load and align the real larval-brain connectome release."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                    "Supplementary-Data-S1")


def load_graph(data_dir=DATA):
    """Returns adjacency (np.float32, n x n), node body_ids."""
    df = pd.read_csv(os.path.join(data_dir, "all-all_connectivity_matrix.csv"),
                     index_col=0)
    df = df.loc[:, [c for c in df.columns if c != ""]]
    A = df.to_numpy(dtype=np.float32)
    ids = df.index.to_numpy()
    return A, ids


def load_annotations(data_dir=DATA):
    ann = pd.read_csv(os.path.join(data_dir, "annotations.csv"))
    return ann


def load_morphology(data_dir=DATA):
    """Node features: axon/dendrite input+output totals (4 raw -> log1p)."""
    inp = pd.read_csv(os.path.join(data_dir, "inputs.csv"), index_col=0)
    out = pd.read_csv(os.path.join(data_dir, "outputs.csv"), index_col=0)
    return inp, out


def align(A, ids, ann, inp, out, min_class=40):
    """Keep nodes present in the graph with a known celltype of sufficient
    support; returns (A_sub, X, y, classes, ids_sub)."""
    labels = {}
    for _, row in ann.iterrows():
        for col in ("left_id", "right_id"):
            v = row[col]
            if v != "no pair" and pd.notna(v):
                labels[int(v)] = row["celltype"]
    vc = pd.Series(labels).value_counts()
    keep_classes = set(vc[vc >= min_class].index)
    idx = [i for i, b in enumerate(ids) if labels.get(int(b)) in keep_classes]
    ids_sub = ids[idx]
    A_sub = A[np.ix_(idx, idx)]
    y = np.array([labels[int(b)] for b in ids_sub])
    inp = inp.reindex(ids_sub).fillna(0.0)
    out = out.reindex(ids_sub).fillna(0.0)
    X = np.stack([inp["axon_input"], inp["dendrite_input"],
                  out["axon_output"], out["dendrite_output"]], axis=1)
    X = np.log1p(np.nan_to_num(X.astype(np.float32)))
    classes = sorted(keep_classes)
    return A_sub, X.astype(np.float32), y, classes, ids_sub
