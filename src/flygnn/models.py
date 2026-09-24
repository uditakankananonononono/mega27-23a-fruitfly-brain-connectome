"""GCN (pure-torch, no external geometric deps) + CNN and linear baselines."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


def normalize_adj(A: np.ndarray, self_loops: bool = True) -> torch.Tensor:
    """Symmetric D^-1/2 (A+I) D^-1/2 normalization (Kipf & Welling 2017)."""
    A = A.copy().astype(np.float32)
    A[A > 0] = 1.0  # binarize synapse counts for structure learning
    if self_loops:
        A += np.eye(A.shape[0], dtype=np.float32)
    d = A.sum(1)
    dinv = np.power(np.maximum(d, 1e-6), -0.5)
    An = A * dinv[:, None] * dinv[None, :]
    return torch.tensor(An)


class GCN(nn.Module):
    def __init__(self, n_in, n_hidden, n_out, dropout=0.3):
        super().__init__()
        self.w1 = nn.Linear(n_in, n_hidden)
        self.w2 = nn.Linear(n_hidden, n_out)
        self.drop = nn.Dropout(dropout)

    def forward(self, X, A_hat):
        h = torch.relu(A_hat @ self.w1(X))
        h = self.drop(h)
        return self.w2(A_hat @ h)


class MorphCNN(nn.Module):
    """1D CNN over the 4 morphological channels (CNN core-architecture arm)."""
    def __init__(self, n_channels, n_out, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, hidden, kernel_size=1), nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=1), nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_out)

    def forward(self, X):
        h = self.net(X.unsqueeze(-1)).squeeze(-1)
        return self.head(h)


class LinearBaseline(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.lin = nn.Linear(n_in, n_out)

    def forward(self, X, A_hat=None):
        return self.lin(X)


def train_node_classifier(model, X, A_hat, y, train_idx, epochs=300, lr=0.01,
                          weight_decay=5e-4, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lossf = nn.CrossEntropyLoss()
    Xt = torch.tensor(X); yt = torch.tensor(y)
    tr = torch.tensor(train_idx)
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(Xt, A_hat) if A_hat is not None else model(Xt)
        loss = lossf(out[tr], yt[tr])
        loss.backward()
        opt.step()
    return model


def evaluate(model, X, A_hat, y, idx):
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X)
        out = model(Xt, A_hat) if A_hat is not None else model(Xt)
        pred = out[torch.tensor(idx)].argmax(1).numpy()
    yt = y[idx]
    acc = float((pred == yt).mean())
    # macro F1
    f1s = []
    for c in np.unique(yt):
        tp = int(((pred == c) & (yt == c)).sum())
        fp = int(((pred == c) & (yt != c)).sum())
        fn = int(((pred != c) & (yt == c)).sum())
        p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        f1s.append(2 * p * r / max(p + r, 1e-9))
    return acc, float(np.mean(f1s))


def stratified_split(y, train_frac=0.6, seed=0):
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        k = max(1, int(len(idx) * train_frac))
        tr.extend(idx[:k]); te.extend(idx[k:])
    return np.array(tr), np.array(te)
