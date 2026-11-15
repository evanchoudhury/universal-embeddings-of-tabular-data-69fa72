"""Reduced entity graph: co-occurrence matrix A, targets T / S and propagation matrix P."""
import numpy as np


def build_graph(B):
    """A = B^T B with zero diagonal. A_ij = sum over rows of w_ri * w_rj."""
    A = np.asarray((B.T @ B).todense(), dtype=float)
    np.fill_diagonal(A, 0.0)
    return A


def _safe(d):
    return np.where(d > 0, d, 1.0)


def transition(A):
    """T = D^-1 A, row-stochastic (the paper's reconstruction target)."""
    return A / _safe(A.sum(axis=1))[:, None]


def symmetric(A):
    """S = D^-1/2 A D^-1/2 (ablation target)."""
    s = 1.0 / np.sqrt(_safe(A.sum(axis=1)))
    return A * s[:, None] * s[None, :]


def propagation(A):
    """P = D~^-1/2 (A + I) D~^-1/2, the standard GCN normalisation."""
    At = A + np.eye(A.shape[0])
    s = 1.0 / np.sqrt(At.sum(axis=1))
    return At * s[:, None] * s[None, :]


def make_target(A, kind):
    if kind == "T":
        return transition(A)
    if kind == "S":
        return symmetric(A)
    raise ValueError("unknown target kind: %s" % kind)


def graph_stats(A):
    n = A.shape[0]
    upper = np.triu(A, 1)
    n_edges = int((upper > 0).sum())
    return {"n_nodes": int(n), "n_edges": n_edges,
            "density": float(n_edges / max(n * (n - 1) / 2, 1)),
            "total_weight": float(upper.sum())}
