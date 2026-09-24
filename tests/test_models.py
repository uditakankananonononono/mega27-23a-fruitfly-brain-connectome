import numpy as np
import torch
from flygnn.models import (normalize_adj, GCN, MorphCNN, LinearBaseline,
                           train_node_classifier, evaluate, stratified_split)


def toy_problem(seed=0):
    """Density-asymmetric SBM: class 0 is a dense community (p=0.25), class 1
    sparse (p=0.05), cross-links rare. X carries a constant channel, so
    propagation exposes neighborhood density; features alone are uninformative."""
    rng = np.random.default_rng(seed)
    n = 200
    y = np.repeat([0, 1], n // 2)
    P = np.where(y[:, None] == y[None, :],
                 np.where(y[:, None] == 0, 0.25, 0.05), 0.01)
    np.fill_diagonal(P, 0.0)
    A = (rng.random((n, n)) < P).astype(np.float32)
    X = rng.normal(0, 1, (n, 4)).astype(np.float32)
    X[:, 0] = 1.0  # constant channel: propagation reveals local density
    return A, X, y


def test_normalize_adj_properties():
    A = np.array([[0, 2], [2, 0]], dtype=np.float32)
    An = normalize_adj(A, self_loops=False).numpy()
    assert np.allclose(An, An.T)
    assert abs(An[0, 1] - 1.0) < 1e-6


def test_gcn_beats_feature_only_on_structure_problem():
    A, X, y = toy_problem()
    A_hat = normalize_adj(A)
    tr, te = stratified_split(y, 0.5, seed=1)
    gcn = GCN(4, 16, 2)
    train_node_classifier(gcn, X, A_hat, y, tr, epochs=200, lr=0.05)
    acc_gcn, _ = evaluate(gcn, X, A_hat, y, te)
    lin = LinearBaseline(4, 2)
    train_node_classifier(lin, X, None, y, tr, epochs=200, lr=0.05)
    acc_lin, _ = evaluate(lin, X, None, y, te)
    assert acc_gcn > 0.8
    assert acc_gcn > acc_lin + 0.15  # structure must carry the signal


def test_morph_cnn_trains():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (100, 4)).astype(np.float32)
    X[:50] += 2.0
    y = np.repeat([0, 1], 50)
    tr, te = stratified_split(y, 0.6, seed=3)
    cnn = MorphCNN(4, 2)
    train_node_classifier(cnn, X, None, y, tr, epochs=150, lr=0.02)
    acc, f1 = evaluate(cnn, X, None, y, te)
    assert acc > 0.9


def test_stratified_split_covers_all_classes():
    y = np.array([0] * 10 + [1] * 8 + [2] * 6)
    tr, te = stratified_split(y, 0.5, seed=4)
    assert set(np.unique(y[tr])) == {0, 1, 2}
    assert len(set(tr) & set(te)) == 0 and len(tr) + len(te) == len(y)
