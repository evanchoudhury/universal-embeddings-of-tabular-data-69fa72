"""k-NN heads and metrics (classification, regression, outliers, neighbours). Plots: plots.py."""
import matplotlib
matplotlib.use("Agg")  # must come before pyplot (imported in src/plots.py)

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------- k-NN heads
def knn_search(E_ref, E_query, k):
    """Distances and indices of the k nearest reference rows (Euclidean, brute force)."""
    k = int(min(k, len(E_ref)))
    nn = NearestNeighbors(n_neighbors=k, algorithm="brute", metric="euclidean")
    nn.fit(np.asarray(E_ref, dtype=np.float64))
    return nn.kneighbors(np.asarray(E_query, dtype=np.float64))


def knn_eps(E_ref):
    """Distance floor (depends on the reference set only), so exact duplicates share weight."""
    scale = float(np.sqrt(np.asarray(E_ref, dtype=np.float64).var(axis=0).sum()))
    return 1e-4 * scale + 1e-12


def _as_list(weights):
    return [weights] if isinstance(weights, str) else list(weights)


def _weights(dist, eps, weights):
    if weights == "uniform":
        return np.ones_like(dist)
    return 1.0 / np.maximum(dist, eps)


def knn_proba(dist, idx, y_ref, k, eps, weights="distance"):
    """Distance-weighted share of positive neighbours among the first k."""
    w = _weights(dist[:, :k], eps, weights)
    return (w * np.asarray(y_ref)[idx[:, :k]]).sum(axis=1) / w.sum(axis=1)


knn_value = knn_proba  # the same weighted mean is the k-NN regression estimate


# ---------------------------------------------------------------- metrics
def classification_metrics(y_true, proba):
    y_true = np.asarray(y_true)
    pred = (np.asarray(proba) >= 0.5).astype(int)
    both = len(np.unique(y_true)) > 1
    return {"roc_auc": float(roc_auc_score(y_true, proba)) if both else float("nan"),
            "accuracy": float(accuracy_score(y_true, pred)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
            "average_precision": float(average_precision_score(y_true, proba)) if both
            else float("nan")}


def regression_metrics(y_true, pred):
    return {"mae": float(mean_absolute_error(y_true, pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
            "r2": float(r2_score(y_true, pred))}


def confusion(y_true, proba):
    return confusion_matrix(y_true, (np.asarray(proba) >= 0.5).astype(int), labels=[0, 1])


def evaluate_knn_classifier(E_tr, y_tr, E_va, y_va, E_te, y_te, k_grid, weights="distance"):
    """Pick k and the neighbour weighting on validation ROC-AUC, then score test once."""
    kmax = max(k_grid)
    eps = knn_eps(E_tr)
    dv, iv = knn_search(E_tr, E_va, kmax)
    val_auc = {(w, int(k)): float(roc_auc_score(y_va, knn_proba(dv, iv, y_tr, k, eps, w)))
               for w in _as_list(weights) for k in k_grid}
    best_w, best_k = max(val_auc, key=lambda c: (val_auc[c], -c[1]))
    dt, it = knn_search(E_tr, E_te, kmax)
    proba = knn_proba(dt, it, y_tr, best_k, eps, best_w)
    out = classification_metrics(y_te, proba)
    out.update({"k": int(best_k), "weights": best_w, "val_roc_auc": val_auc[(best_w, best_k)],
                "val_roc_auc_grid": {w: {str(k): val_auc[(w, int(k))] for k in k_grid}
                                     for w in _as_list(weights)},
                "dim": int(np.asarray(E_tr).shape[1])})
    return out, {"proba": proba, "dist": dt, "idx": it}


def evaluate_knn_regressor(E_tr, y_tr, E_va, y_va, E_te, y_te, k_grid, weights="distance"):
    """Pick k and the neighbour weighting on validation MAE, then score test once."""
    kmax = max(k_grid)
    eps = knn_eps(E_tr)
    dv, iv = knn_search(E_tr, E_va, kmax)
    val_mae = {(w, int(k)): float(mean_absolute_error(y_va, knn_value(dv, iv, y_tr, k, eps, w)))
               for w in _as_list(weights) for k in k_grid}
    best_w, best_k = min(val_mae, key=lambda c: (val_mae[c], c[1]))
    dt, it = knn_search(E_tr, E_te, kmax)
    pred = knn_value(dt, it, y_tr, best_k, eps, best_w)
    out = regression_metrics(y_te, pred)
    out.update({"k": int(best_k), "weights": best_w, "val_mae": val_mae[(best_w, best_k)]})
    return out, pred


def outlier_scores(E_tr, E_query, k):
    dist, _ = knn_search(E_tr, E_query, k)
    return dist.mean(axis=1)


def outlier_auc(score_original, score_corrupted):
    y = np.r_[np.zeros(len(score_original)), np.ones(len(score_corrupted))]
    return float(roc_auc_score(y, np.r_[score_original, score_corrupted]))


def neighbour_agreement(idx, y_ref, y_query, k):
    """Share of the k nearest training rows that have the same label as the query row."""
    same = (np.asarray(y_ref)[idx[:, :k]] == np.asarray(y_query)[:, None]).mean(axis=1)
    p = float(np.mean(y_ref))
    return {"k": int(k), "mean_same_label_share": float(same.mean()),
            "chance_level": float(p * p + (1 - p) * (1 - p)),
            "share_positive_queries": float(same[np.asarray(y_query) == 1].mean()),
            "share_negative_queries": float(same[np.asarray(y_query) == 0].mean())}
