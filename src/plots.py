"""Matplotlib figures written to artifacts/plots."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from . import config as C

BLUE, ORANGE, GREEN, GREY, RED = "#2b6cb0", "#dd6b20", "#2f855a", "#718096", "#c53030"


def _save(fig, name, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path.name


def data_overview(df, out_dir):
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    counts = df[C.TARGET].value_counts().sort_index()
    ax[0, 0].bar(["<=50K", ">50K"], counts.values, color=[BLUE, ORANGE])
    for i, v in enumerate(counts.values):
        ax[0, 0].text(i, v, "%d (%.1f%%)" % (v, 100.0 * v / len(df)), ha="center", va="bottom")
    ax[0, 0].set_title("Class balance (all rows)"); ax[0, 0].set_ylabel("rows")
    bins = np.arange(17, 92, 3)
    for lab, col, name in [(0, BLUE, "<=50K"), (1, ORANGE, ">50K")]:
        ax[0, 1].hist(df.loc[df[C.TARGET] == lab, "age"], bins=bins, alpha=0.6, color=col,
                      label=name, density=True)
    ax[0, 1].set_title("Age by income class"); ax[0, 1].set_xlabel("age"); ax[0, 1].legend()
    ax[1, 0].hist(df["hours-per-week"], bins=np.arange(0, 101, 5), color=GREEN)
    ax[1, 0].set_title("Hours per week"); ax[1, 0].set_xlabel("hours"); ax[1, 0].set_ylabel("rows")
    rate = df.groupby("education")[C.TARGET].mean().sort_values()
    ax[1, 1].barh(rate.index, rate.values, color=GREY)
    ax[1, 1].set_title("Share earning >50K by education"); ax[1, 1].set_xlabel("share >50K")
    return _save(fig, "01_data_overview.png", out_dir)


def loss_curves(histories, out_dir):
    """histories: {target_kind: {seed: [loss ...]}}"""
    fig, ax = plt.subplots(1, len(histories), figsize=(6 * len(histories), 4.2), squeeze=False)
    for a, (kind, runs) in zip(ax[0], histories.items()):
        for seed, h in runs.items():
            a.plot(np.arange(1, len(h) + 1), h, label="seed %s" % seed)
        a.set_yscale("log"); a.set_xscale("log")
        a.set_title("GAE training loss, target %s (d = %d)" % (kind, C.EMB_DIM))
        a.set_xlabel("epoch"); a.set_ylabel("L2 reconstruction loss"); a.legend()
        a.grid(alpha=0.3)
    return _save(fig, "02_gae_loss_curve.png", out_dir)


def entity_map(entities, xy, out_dir):
    cols = sorted({e["column"] for e in entities})
    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, col in enumerate(cols):
        ids = [j for j, e in enumerate(entities) if e["column"] == col]
        ax.scatter(xy[ids, 0], xy[ids, 1], s=28, color=cmap(i % 20), label=col)
        bins = [j for j in ids if entities[j]["kind"] == "bin"]
        if len(bins) > 1:
            ax.plot(xy[bins, 0], xy[bins, 1], color=cmap(i % 20), lw=1, alpha=0.7)
    for j, e in enumerate(entities):
        if e["column"] in ("marital-status", "relationship", "sex"):
            ax.annotate(e["name"].split("=", 1)[1][:18], xy[j], fontsize=7, alpha=0.8)
    ax.set_title("Entity embeddings (PCA to 2-D); numerical bins joined in order")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.legend(fontsize=8, ncol=2)
    return _save(fig, "03_entity_embedding_map.png", out_dir)


def row_map(xy, y, out_dir, max_points=4000, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.permutation(len(xy))[:max_points]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for lab, col, name in [(0, BLUE, "<=50K"), (1, ORANGE, ">50K")]:
        m = ids[y[ids] == lab]
        ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.5, color=col, label=name)
    ax.set_title("Test-row embeddings (PCA to 2-D), income never used in training")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2"); ax.legend(markerscale=3)
    return _save(fig, "04_row_embedding_map.png", out_dir)


def method_comparison(rows, out_dir):
    """rows: list of (label, test auc, group)"""
    colour = {"main": ORANGE, "embedding": BLUE, "raw": GREY, "supervised": GREEN}
    rows = sorted(rows, key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 1.8))
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[colour[r[2]] for r in rows])
    for i, r in enumerate(rows):
        ax.text(r[1] + 0.002, i, "%.4f" % r[1], va="center", fontsize=9)
    lo = max(0.5, min(r[1] for r in rows) - 0.03)
    ax.set_xlim(lo, min(1.0, max(r[1] for r in rows) + 0.03))
    ax.set_xlabel("test ROC-AUC (income > 50K)")
    ax.set_title("Method comparison (orange: GAE, blue: other label-free embeddings,\n"
                 "grey: raw k-NN, green: supervised references)")
    return _save(fig, "05_method_comparison.png", out_dir)


def roc_curves(y, curves, out_dir):
    """curves: list of (label, proba, auc)"""
    fig, ax = plt.subplots(figsize=(6.8, 6))
    for (label, proba, auc), col in zip(curves, [ORANGE, GREY, GREEN, BLUE, RED]):
        fpr, tpr, _ = roc_curve(y, proba)
        ax.plot(fpr, tpr, color=col, label="%s (AUC %.4f)" % (label, auc))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves on the test split"); ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return _save(fig, "06_roc_curves.png", out_dir)


def dimension_sweep(sweep, refs, out_dir):
    """sweep: {method: {dim: auc}}, refs: {label: auc}"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, res), col in zip(sweep.items(), [ORANGE, BLUE, GREEN, RED]):
        dims = sorted(int(d) for d in res)
        ax.plot(dims, [res[str(d)] for d in dims], "o-", color=col, label=name)
    for (name, v), ls in zip(refs.items(), ["--", ":"]):
        ax.axhline(v, color=GREY, ls=ls, label=name)
    ax.set_xscale("log", base=2); ax.set_xlabel("embedding dimension d")
    ax.set_ylabel("test ROC-AUC"); ax.set_title("ROC-AUC vs embedding dimension")
    ax.legend(); ax.grid(alpha=0.3)
    return _save(fig, "07_dimension_sweep.png", out_dir)


def age_regression(y, pred, r2, mae, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 6))
    hb = ax.hexbin(y, pred, gridsize=35, cmap="Blues", mincnt=1)
    fig.colorbar(hb, ax=ax, label="test rows")
    lim = [min(y.min(), pred.min()), max(y.max(), pred.max())]
    ax.plot(lim, lim, "k--", lw=0.8)
    ax.set_xlabel("true age"); ax.set_ylabel("predicted age (k-NN on GAE embedding)")
    ax.set_title("Age regression without the age column (R2 %.3f, MAE %.2f)" % (r2, mae))
    return _save(fig, "08_age_regression.png", out_dir)


def outlier_hist(orig, corr, auc, out_dir):
    hi = float(np.quantile(np.r_[orig, corr], 0.99))
    bins = np.linspace(0, hi, 50)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.hist(np.clip(orig, 0, hi), bins=bins, alpha=0.6, color=BLUE, label="original test rows")
    ax.hist(np.clip(corr, 0, hi), bins=bins, alpha=0.6, color=RED, label="corrupted copies")
    ax.set_xlabel("outlier score: mean distance to %d nearest training rows" % C.OUTLIER_K)
    ax.set_ylabel("rows"); ax.legend()
    ax.set_title("Outlier scores in the GAE embedding (ROC-AUC %.4f)" % auc)
    return _save(fig, "09_outlier_scores.png", out_dir)


def ablation_seeds(ablation, chosen, out_dir):
    """ablation: {kind: {"test_roc_auc_by_seed": {seed: auc}, "val_roc_auc_by_seed": {...}}}"""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    width, x0 = 0.38, 0
    ticks, labels = [], []
    for kind, res in ablation.items():
        seeds = sorted(res["test_roc_auc_by_seed"])
        for j, s in enumerate(seeds):
            x = x0 + j
            ax.bar(x - width / 2, res["val_roc_auc_by_seed"][s], width, color=GREY,
                   label="validation" if (x0 == 0 and j == 0) else None)
            ax.bar(x + width / 2, res["test_roc_auc_by_seed"][s], width,
                   color=ORANGE if kind == chosen else BLUE,
                   label=("test, target %s%s" % (kind, " (chosen)" if kind == chosen else ""))
                   if j == 0 else None)
            ticks.append(x); labels.append("%s\nseed %s" % (kind, s))
        x0 += len(seeds) + 1
    vals = [v for r in ablation.values() for d in ("val_roc_auc_by_seed", "test_roc_auc_by_seed")
            for v in r[d].values()]
    ax.set_ylim(min(vals) - 0.01, max(vals) + 0.01)
    ax.set_xticks(ticks); ax.set_xticklabels(labels)
    ax.set_ylabel("ROC-AUC"); ax.legend(fontsize=8)
    ax.set_title("Reconstruction target ablation and seed stability (d = %d)" % C.EMB_DIM)
    return _save(fig, "10_target_ablation_seeds.png", out_dir)


def confusion_plot(cm, out_dir):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, "%d\n(%.1f%%)" % (cm[i, j], 100.0 * cm[i, j] / cm.sum()),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["<=50K", ">50K"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["<=50K", ">50K"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Confusion matrix, GAE + k-NN (test, threshold 0.5)")
    return _save(fig, "11_confusion_matrix.png", out_dir)
