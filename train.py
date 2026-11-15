"""Entry point: `python train.py` trains the baseline and the main model, writes artifacts/."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from src import config as C
from src import plots
from src.baselines import (RawFeatures, fit_logreg, fit_pca, fit_xgb_classifier,
                           fit_xgb_regressor, random_walk_pairs, skipgram_entities,
                           spectral_entities)
from src.data import dataset_stats, load_adult, split
from src.embed import UniversalEmbedder, row_embeddings
from src.evaluate import (classification_metrics, confusion, evaluate_knn_classifier,
                          evaluate_knn_regressor, neighbour_agreement, outlier_auc,
                          outlier_scores, regression_metrics)
from src.gae import train_gae
from src.graph import build_graph, graph_stats, make_target, propagation, transition
from src.tokeniser import TableTokeniser


def log(msg):
    print(msg, flush=True)


def corrupt(df, share, seed):
    """Replace a random share of the cells of every row with values from other rows."""
    rng = np.random.default_rng(seed)
    n, m = len(df), len(C.FEATURE_COLS)
    n_swap = int(round(share * m))
    mask = np.zeros((n, m), dtype=bool)
    for i in range(n):
        mask[i, rng.choice(m, n_swap, replace=False)] = True
    out = df[C.FEATURE_COLS].copy()
    for j, col in enumerate(C.FEATURE_COLS):
        shuffled = df[col].to_numpy()[rng.permutation(n)]
        out[col] = np.where(mask[:, j], shuffled, df[col].to_numpy())
    changed = (out.to_numpy() != df[C.FEATURE_COLS].to_numpy()).mean()
    return out, float(changed)


def main():
    t_start = time.time()
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    C.ARTIFACTS.mkdir(parents=True, exist_ok=True)
    C.PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    timings = {}

    # ------------------------------------------------------------ 1. data
    log("=== 1/7 data ===")
    t0 = time.time()
    df = load_adult()
    train, val, test = split(df)
    stats = dataset_stats(df, train, val, test)
    y_tr, y_va, y_te = (d[C.TARGET].to_numpy() for d in (train, val, test))
    X_tr, X_va, X_te = (d[C.FEATURE_COLS] for d in (train, val, test))
    log("rows %d | train %d  val %d  test %d | positive share %.4f"
        % (len(df), len(train), len(val), len(test), stats["positive_share"]))
    timings["data"] = time.time() - t0

    # ------------------------------------------------------------ 2. tokeniser + graph
    log("=== 2/7 entity graph ===")
    t0 = time.time()
    tok = TableTokeniser(C.CAT_COLS, C.NUM_COLS, C.ZERO_COLS, C.N_BINS, C.N_BINS_ZERO).fit(X_tr)
    B_tr = tok.transform(X_tr)
    B_va, info_va = tok.transform(X_va, return_info=True)
    B_te, info_te = tok.transform(X_te, return_info=True)
    A = build_graph(B_tr)
    P = propagation(A)
    targets = {k: make_target(A, k) for k in C.TARGET_KINDS}
    gstats = graph_stats(A)
    gstats["entities_by_column"] = {c: sum(e["column"] == c for e in tok.entities)
                                    for c in tok.columns}
    gstats["unseen_cells_val"] = info_va["unseen_cells"]
    gstats["unseen_cells_test"] = info_te["unseen_cells"]
    row_w = np.asarray(B_tr.sum(axis=1)).ravel()
    assert np.allclose(row_w, len(tok.columns)), "row weights must equal the column count"
    assert np.allclose(targets["T"].sum(axis=1), 1.0), "transition rows must sum to 1"
    log("entities %d | edges %d | density %.3f | unseen cells val %d test %d"
        % (gstats["n_nodes"], gstats["n_edges"], gstats["density"],
           gstats["unseen_cells_val"], gstats["unseen_cells_test"]))
    timings["graph"] = time.time() - t0

    def knn_eval(E_a, E_b, E_c):
        return evaluate_knn_classifier(E_a, y_tr, E_b, y_va, E_c, y_te, C.K_GRID,
                                       C.KNN_WEIGHTS)

    def gae_run(kind, dim, seed):
        Z, info = train_gae(P, targets[kind], dim, seed=seed, lr=C.GAE_LR, clip=C.GAE_CLIP,
                            max_epochs=C.GAE_MAX_EPOCHS, patience=C.GAE_PATIENCE,
                            tol=C.GAE_TOL, alpha=C.GAE_ALPHA, init_std=C.GAE_INIT_STD)
        E = [row_embeddings(B, Z) for B in (B_tr, B_va, B_te)]
        res, extra = knn_eval(*E)
        res.update({"epochs": info["epochs"], "final_loss": info["final_loss"],
                    "explained_share": info["explained_share"]})
        return {"Z": Z, "info": info, "E": E, "res": res, "extra": extra}

    # ------------------------------------------------------------ 3. GAE, both targets, seeds
    log("=== 3/7 graph auto-encoder (d = %d) ===" % C.EMB_DIM)
    t0 = time.time()
    runs, histories, ablation = {}, {}, {}
    for kind in C.TARGET_KINDS:
        histories[kind] = {}
        for seed in C.GAE_SEEDS:
            r = gae_run(kind, C.EMB_DIM, seed)
            runs[(kind, seed)] = r
            histories[kind][str(seed)] = r["info"]["history"]
            log("  target %s seed %d: epochs %4d  loss %.5f  k %3d %-8s val AUC %.4f  test AUC %.4f"
                % (kind, seed, r["res"]["epochs"], r["res"]["final_loss"], r["res"]["k"],
                   r["res"]["weights"], r["res"]["val_roc_auc"], r["res"]["roc_auc"]))
        va = [runs[(kind, s)]["res"]["val_roc_auc"] for s in C.GAE_SEEDS]
        te = [runs[(kind, s)]["res"]["roc_auc"] for s in C.GAE_SEEDS]
        ablation[kind] = {
            "val_roc_auc_mean": float(np.mean(va)), "val_roc_auc_std": float(np.std(va)),
            "test_roc_auc_mean": float(np.mean(te)), "test_roc_auc_std": float(np.std(te)),
            "val_roc_auc_by_seed": {str(s): v for s, v in zip(C.GAE_SEEDS, va)},
            "test_roc_auc_by_seed": {str(s): v for s, v in zip(C.GAE_SEEDS, te)}}
    chosen = max(C.TARGET_KINDS, key=lambda k: ablation[k]["val_roc_auc_mean"])
    main_seed = C.GAE_SEEDS[0]
    main_run = runs[(chosen, main_seed)]
    main_res = main_run["res"]
    log("chosen target (validation): %s | main model = seed %d | test AUC %.4f"
        % (chosen, main_seed, main_res["roc_auc"]))
    timings["gae_main"] = time.time() - t0

    # ------------------------------------------------------------ 4. baselines
    log("=== 4/7 baselines ===")
    t0 = time.time()
    raw = RawFeatures().fit(X_tr)
    ord_tr, ord_va, ord_te = (raw.ordinal(d) for d in (X_tr, X_va, X_te))
    oh_tr, oh_va, oh_te = (raw.onehot(d) for d in (X_tr, X_va, X_te))
    comparison, probas = {}, {}

    base_res, base_extra = knn_eval(ord_tr, ord_va, ord_te)
    comparison["raw_ordinal_knn"] = base_res
    probas["raw_ordinal_knn"] = base_extra["proba"]
    log("  raw ordinal k-NN   : k %3d %-8s val AUC %.4f  test AUC %.4f"
        % (base_res["k"], base_res["weights"], base_res["val_roc_auc"], base_res["roc_auc"]))
    oh_res, _ = knn_eval(oh_tr, oh_va, oh_te)
    comparison["raw_onehot_knn"] = oh_res
    log("  raw one-hot k-NN   : k %3d %-8s val AUC %.4f  test AUC %.4f"
        % (oh_res["k"], oh_res["weights"], oh_res["val_roc_auc"], oh_res["roc_auc"]))

    sg_pairs = random_walk_pairs(transition(A), seed=0)
    pca_models = {}

    def other_embeddings(dim):
        out = {}
        pca = fit_pca(oh_tr, dim)
        pca_models[dim] = pca
        out["pca"] = knn_eval(pca.transform(oh_tr), pca.transform(oh_va),
                              pca.transform(oh_te))[0]
        Zs = spectral_entities(targets[chosen], dim)
        out["spectral"] = knn_eval(*[row_embeddings(B, Zs) for B in (B_tr, B_va, B_te)])[0]
        Zg = skipgram_entities(sg_pairs, tok.n_entities, dim, seed=0)
        out["skipgram"] = knn_eval(*[row_embeddings(B, Zg) for B in (B_tr, B_va, B_te)])[0]
        return out

    at16 = other_embeddings(C.EMB_DIM)
    for name, res in at16.items():
        comparison["%s_d%d" % (name, C.EMB_DIM)] = res
        log("  %-9s d = %2d   : k %3d %-8s val AUC %.4f  test AUC %.4f"
            % (name, C.EMB_DIM, res["k"], res["weights"], res["val_roc_auc"], res["roc_auc"]))

    lr_model = fit_logreg(oh_tr, y_tr)
    p_lr = lr_model.predict_proba(oh_te)[:, 1]
    comparison["logistic_regression"] = classification_metrics(y_te, p_lr)
    xgb = fit_xgb_classifier(raw.ordinal_raw(X_tr), y_tr)
    p_xgb = xgb.predict_proba(raw.ordinal_raw(X_te))[:, 1]
    comparison["xgboost"] = classification_metrics(y_te, p_xgb)
    p_major = np.full(len(y_te), float(y_tr.mean()))
    comparison["majority_class"] = {
        "roc_auc": 0.5, "accuracy": float((y_te == int(y_tr.mean() >= 0.5)).mean()),
        "f1": 0.0, "average_precision": float(y_te.mean())}
    del p_major
    log("  logistic regression: test AUC %.4f | XGBoost: test AUC %.4f"
        % (comparison["logistic_regression"]["roc_auc"], comparison["xgboost"]["roc_auc"]))
    timings["baselines"] = time.time() - t0

    # ------------------------------------------------------------ 5. dimension sweep
    log("=== 5/7 dimension sweep ===")
    t0 = time.time()
    sweep = {"gae": {}, "pca": {}, "spectral": {}, "skipgram": {}}
    sweep_val = {k: {} for k in sweep}
    for dim in C.DIM_SWEEP:
        if dim == C.EMB_DIM:
            g, others = main_res, at16
        else:
            g, others = gae_run(chosen, dim, main_seed)["res"], other_embeddings(dim)
        sweep["gae"][str(dim)] = g["roc_auc"]
        sweep_val["gae"][str(dim)] = g["val_roc_auc"]
        for name in ("pca", "spectral", "skipgram"):
            sweep[name][str(dim)] = others[name]["roc_auc"]
            sweep_val[name][str(dim)] = others[name]["val_roc_auc"]
        log("  d = %2d: GAE %.4f  PCA %.4f  spectral %.4f  skip-gram %.4f"
            % (dim, sweep["gae"][str(dim)], sweep["pca"][str(dim)],
               sweep["spectral"][str(dim)], sweep["skipgram"][str(dim)]))
    timings["dimension_sweep"] = time.time() - t0

    # ------------------------------------------------------------ 6. universality
    log("=== 6/7 universality checks ===")
    t0 = time.time()
    E_tr, E_va, E_te = main_run["E"]
    # 6a. age regression with a graph that never saw the age column
    num_wo_age = [c for c in C.NUM_COLS if c != "age"]
    age_model = UniversalEmbedder(C.EMB_DIM, chosen, main_seed, num_cols=num_wo_age).fit(X_tr)
    a_tr, a_va, a_te = (d["age"].to_numpy(dtype=float) for d in (train, val, test))
    age_res, age_pred = evaluate_knn_regressor(
        age_model.transform(X_tr), a_tr, age_model.transform(X_va), a_va,
        age_model.transform(X_te), a_te, C.K_GRID, C.KNN_WEIGHTS)
    raw_age = RawFeatures(num_cols=num_wo_age + C.ZERO_COLS).fit(X_tr)
    age_raw_res, _ = evaluate_knn_regressor(
        raw_age.ordinal(X_tr), a_tr, raw_age.ordinal(X_va), a_va, raw_age.ordinal(X_te), a_te,
        C.K_GRID, C.KNN_WEIGHTS)
    xgb_age = fit_xgb_regressor(raw_age.ordinal_raw(X_tr), a_tr)
    age_regression = {
        "gae_knn": age_res, "raw_ordinal_knn": age_raw_res,
        "xgboost": regression_metrics(a_te, xgb_age.predict(raw_age.ordinal_raw(X_te))),
        "training_mean": regression_metrics(a_te, np.full(len(a_te), a_tr.mean())),
        "n_entities_without_age": age_model.tokeniser.n_entities}
    log("  age regression: GAE k-NN MAE %.3f R2 %.4f | raw k-NN R2 %.4f | XGBoost R2 %.4f | "
        "mean MAE %.3f" % (age_res["mae"], age_res["r2"], age_raw_res["r2"],
                           age_regression["xgboost"]["r2"],
                           age_regression["training_mean"]["mae"]))

    # 6b. outlier detection on corrupted copies of the test rows
    X_bad, changed = corrupt(test, C.CORRUPT_SHARE, C.CORRUPT_SEED)
    E_bad = row_embeddings(tok.transform(X_bad), main_run["Z"])
    s_orig = outlier_scores(E_tr, E_te, C.OUTLIER_K)
    s_bad = outlier_scores(E_tr, E_bad, C.OUTLIER_K)
    pca16 = pca_models[C.EMB_DIM]
    oh_bad = raw.onehot(X_bad)
    outlier = {
        "k": C.OUTLIER_K, "share_of_cells_actually_changed": changed,
        "gae_roc_auc": outlier_auc(s_orig, s_bad),
        "pca_roc_auc": outlier_auc(
            outlier_scores(pca16.transform(oh_tr), pca16.transform(oh_te), C.OUTLIER_K),
            outlier_scores(pca16.transform(oh_tr), pca16.transform(oh_bad), C.OUTLIER_K)),
        "onehot_roc_auc": outlier_auc(outlier_scores(oh_tr, oh_te, C.OUTLIER_K),
                                      outlier_scores(oh_tr, oh_bad, C.OUTLIER_K)),
        "raw_ordinal_roc_auc": outlier_auc(
            outlier_scores(ord_tr, ord_te, C.OUTLIER_K),
            outlier_scores(ord_tr, raw.ordinal(X_bad), C.OUTLIER_K))}
    log("  outlier ROC-AUC: GAE %.4f | PCA %.4f | one-hot %.4f | raw ordinal %.4f"
        % (outlier["gae_roc_auc"], outlier["pca_roc_auc"], outlier["onehot_roc_auc"],
           outlier["raw_ordinal_roc_auc"]))

    # 6c. neighbour label agreement
    agree = neighbour_agreement(main_run["extra"]["idx"], y_tr, y_te, C.NEIGHBOUR_K)
    agree["raw_ordinal_mean_same_label_share"] = neighbour_agreement(
        base_extra["idx"], y_tr, y_te, C.NEIGHBOUR_K)["mean_same_label_share"]
    log("  neighbour agreement: GAE %.4f | raw ordinal %.4f | chance %.4f"
        % (agree["mean_same_label_share"], agree["raw_ordinal_mean_same_label_share"],
           agree["chance_level"]))
    timings["universality"] = time.time() - t0

    # ------------------------------------------------------------ 7. save
    log("=== 7/7 saving artifacts ===")
    t0 = time.time()
    proba = main_run["extra"]["proba"]
    model = UniversalEmbedder(C.EMB_DIM, chosen, main_seed)
    model.tokeniser, model.Z = tok, main_run["Z"]
    model.info = main_run["info"]
    model.set_reference(E_tr, y_tr, main_res["k"], main_res["weights"])
    model_path = C.ARTIFACTS / "model.joblib"
    model.save(model_path)
    reloaded = UniversalEmbedder.load(model_path)
    check = reloaded.predict_proba(X_te)
    reload_gap = float(np.abs(check - proba).max())
    # ties between duplicate rows may be cut differently; anything larger is a real bug
    assert np.abs(check - proba).mean() < 1e-3, "saved model does not reproduce"
    size_mb = model_path.stat().st_size / 1e6
    assert size_mb < 20, "model file too large"

    proj = PCA(n_components=2, random_state=0).fit(E_tr)
    xy_te = proj.transform(E_te)
    ent_xy = PCA(n_components=2, random_state=0).fit_transform(main_run["Z"])
    np.savez_compressed(C.ARTIFACTS / "embeddings.npz",
                        E_train=E_tr.astype(np.float32), y_train=y_tr,
                        E_val=E_va.astype(np.float32), y_val=y_va,
                        E_test=E_te.astype(np.float32), y_test=y_te,
                        xy_test=xy_te.astype(np.float32),
                        train_row_id=train.index.to_numpy(), test_row_id=test.index.to_numpy())
    np.savez_compressed(C.ARTIFACTS / "graph.npz", A=A.astype(np.float32),
                        names=np.array(tok.names))
    ent = pd.DataFrame(tok.entities)
    ent["train_weight"] = np.asarray(B_tr.sum(axis=0)).ravel()
    ent["x"], ent["y"] = ent_xy[:, 0], ent_xy[:, 1]
    for j in range(main_run["Z"].shape[1]):
        ent["z%d" % j] = main_run["Z"][:, j]
    ent.to_csv(C.ARTIFACTS / "entities.csv", index=False)

    pred = (proba >= 0.5).astype(int)
    rows = test[C.FEATURE_COLS].copy()
    rows.insert(0, "row_id", test.index.to_numpy())
    rows["true_label"] = np.where(y_te == 1, ">50K", "<=50K")
    rows["predicted_label"] = np.where(pred == 1, ">50K", "<=50K")
    rows["prob_gt_50k"] = np.round(proba, 4)
    rows["confidence"] = np.round(np.maximum(proba, 1 - proba), 4)
    rows["correct"] = (pred == y_te)
    rows["outlier_score"] = np.round(s_orig, 5)
    rows["x"], rows["y"] = np.round(xy_te[:, 0], 5), np.round(xy_te[:, 1], 5)
    rows.to_csv(C.ARTIFACTS / "test_rows.csv", index=False)
    pick = np.sort(np.random.default_rng(0).permutation(len(rows))[:C.N_SAMPLE_PREDICTIONS])
    rows.iloc[pick].drop(columns=["x", "y", "outlier_score"]).to_csv(
        C.ARTIFACTS / "sample_predictions.csv", index=False)
    with open(C.ARTIFACTS / "training_history.json", "w") as f:
        json.dump({"dim": C.EMB_DIM, "loss": histories}, f)

    # plots
    made = [plots.data_overview(df, C.PLOTS_DIR),
            plots.loss_curves(histories, C.PLOTS_DIR),
            plots.entity_map(tok.entities, ent_xy, C.PLOTS_DIR),
            plots.row_map(xy_te, y_te, C.PLOTS_DIR)]
    labels = {"raw_ordinal_knn": "k-NN on raw ordinal (baseline)",
              "raw_onehot_knn": "k-NN on raw one-hot",
              "pca_d%d" % C.EMB_DIM: "PCA-%d + k-NN" % C.EMB_DIM,
              "spectral_d%d" % C.EMB_DIM: "Spectral-%d + k-NN" % C.EMB_DIM,
              "skipgram_d%d" % C.EMB_DIM: "Skip-gram-%d + k-NN" % C.EMB_DIM,
              "logistic_regression": "Logistic regression (labels)",
              "xgboost": "XGBoost (labels)"}
    group = {"raw_ordinal_knn": "raw", "raw_onehot_knn": "raw",
             "logistic_regression": "supervised", "xgboost": "supervised"}
    bars = [("GAE-%d + k-NN (main)" % C.EMB_DIM, main_res["roc_auc"], "main")]
    bars += [(labels[k], comparison[k]["roc_auc"], group.get(k, "embedding")) for k in labels]
    made.append(plots.method_comparison(bars, C.PLOTS_DIR))
    made.append(plots.roc_curves(y_te, [
        ("GAE-%d + k-NN" % C.EMB_DIM, proba, main_res["roc_auc"]),
        ("raw ordinal k-NN", probas["raw_ordinal_knn"], base_res["roc_auc"]),
        ("XGBoost (labels)", p_xgb, comparison["xgboost"]["roc_auc"])], C.PLOTS_DIR))
    made.append(plots.dimension_sweep(
        {"GAE": sweep["gae"], "PCA": sweep["pca"], "Spectral": sweep["spectral"],
         "Skip-gram": sweep["skipgram"]},
        {"raw ordinal k-NN": base_res["roc_auc"], "XGBoost (labels)":
         comparison["xgboost"]["roc_auc"]}, C.PLOTS_DIR))
    made.append(plots.age_regression(a_te, age_pred, age_res["r2"], age_res["mae"],
                                     C.PLOTS_DIR))
    made.append(plots.outlier_hist(s_orig, s_bad, outlier["gae_roc_auc"], C.PLOTS_DIR))
    made.append(plots.ablation_seeds(ablation, chosen, C.PLOTS_DIR))
    cm = confusion(y_te, proba)
    made.append(plots.confusion_plot(cm, C.PLOTS_DIR))
    timings["saving_and_plots"] = time.time() - t0

    total = time.time() - t_start
    timings["total"] = total
    main_metrics = {k: main_res[k] for k in ("roc_auc", "accuracy", "f1", "average_precision",
                                             "k", "weights", "val_roc_auc", "val_roc_auc_grid",
                                             "dim",
                                             "epochs", "final_loss", "explained_share")}
    main_metrics.update({
        "target_kind": chosen, "seed": main_seed,
        "n_parameters": main_run["info"]["n_parameters"],
        "model_file_mb": round(size_mb, 3), "reload_max_abs_proba_gap": reload_gap,
        "confusion_matrix": {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                             "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
        "seed_stability": {"test_roc_auc_mean": ablation[chosen]["test_roc_auc_mean"],
                           "test_roc_auc_std": ablation[chosen]["test_roc_auc_std"],
                           "n_seeds": len(C.GAE_SEEDS)},
        "target_ablation": ablation,
        "comparison": comparison,
        "dimension_sweep": {"test_roc_auc": sweep, "val_roc_auc": sweep_val},
        "universality": {"age_regression": age_regression, "outlier_detection": outlier,
                         "neighbour_agreement": agree},
        "dataset_stats": stats, "graph": gstats, "config": C.config_dict(),
        "timings_seconds": {k: round(v, 2) for k, v in timings.items()},
        "plots": made})
    metrics = {
        "task": "Label-free row embeddings (entity-graph auto-encoder) evaluated by k-NN "
                "binary classification of income > 50K",
        "dataset": C.DATASET_NAME,
        "primary_metric": "roc_auc",
        "higher_is_better": True,
        "value": main_res["roc_auc"],
        "baseline": base_res["roc_auc"],
        "baseline_name": "k-NN on ordinal-encoded, standardised raw columns (no embedding)",
        "metrics": main_metrics,
        "baseline_metrics": base_res,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_seconds": round(total, 2)}
    with open(C.ARTIFACTS / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    log("")
    log("+--------------------------------+----------+")
    log("| method                         | test AUC |")
    log("+--------------------------------+----------+")
    for name, auc, _ in sorted(bars, key=lambda r: -r[1]):
        log("| %-30s | %8.4f |" % (name[:30], auc))
    log("+--------------------------------+----------+")
    log("primary metric (test ROC-AUC, GAE d=%d, target %s, k=%d, %s weights): %.4f"
        % (C.EMB_DIM, chosen, main_res["k"], main_res["weights"], main_res["roc_auc"]))
    log("baseline (raw ordinal k-NN): %.4f | seeds mean %.4f std %.4f"
        % (base_res["roc_auc"], ablation[chosen]["test_roc_auc_mean"],
           ablation[chosen]["test_roc_auc_std"]))
    log("model file %.2f MB | plots %d | total time %.1f s" % (size_mb, len(made), total))


if __name__ == "__main__":
    main()
