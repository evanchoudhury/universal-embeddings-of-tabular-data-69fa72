"""Offline smoke test: synthetic table with the Adult schema, no network, under 60 seconds.

Prints `ok` and exits 0 on success; prints the failure and exits 1 otherwise.
"""
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def run():
    import numpy as np
    import pandas as pd

    from src import config as C
    from src.baselines import (RawFeatures, fit_pca, random_walk_pairs, skipgram_entities,
                               spectral_entities)
    from src.data import make_synthetic, split
    from src.embed import row_embeddings
    from src.evaluate import (evaluate_knn_classifier, evaluate_knn_regressor, outlier_auc,
                              outlier_scores)
    from src.graph import build_graph, make_target, propagation, transition
    from src.model import RawKNNBaseline, UniversalEmbedder
    from src.tokeniser import TableTokeniser

    df = make_synthetic(600, seed=0)
    assert list(df.columns) == C.FEATURE_COLS + [C.TARGET], "schema mismatch"
    train, val, test = split(df)
    assert len(train) + len(val) + len(test) == len(df)
    assert not (set(train.index) & set(test.index)) and not (set(val.index) & set(test.index))
    X_tr, X_va, X_te = (d[C.FEATURE_COLS] for d in (train, val, test))
    y_tr, y_va, y_te = (d[C.TARGET].to_numpy() for d in (train, val, test))

    # tokeniser and graph invariants
    tok = TableTokeniser(C.CAT_COLS, C.NUM_COLS, C.ZERO_COLS, C.N_BINS, C.N_BINS_ZERO).fit(X_tr)
    B = tok.transform(X_tr)
    assert B.shape == (len(X_tr), tok.n_entities)
    assert np.allclose(np.asarray(B.sum(axis=1)).ravel(), len(C.FEATURE_COLS)), "row weights"
    assert B.min() >= 0 and B.max() <= 1.0 + 1e-9
    A = build_graph(B)
    assert np.allclose(A, A.T) and np.allclose(np.diag(A), 0)
    assert np.allclose(transition(A).sum(axis=1), 1.0), "transition rows must sum to 1"
    assert np.allclose(make_target(A, "S"), make_target(A, "S").T)
    assert np.isfinite(propagation(A)).all()

    # main model: fit without labels, embed, k-NN head
    C.GAE_MAX_EPOCHS = 300
    model = UniversalEmbedder(dim=8, target_kind="T", seed=0, max_epochs=300).fit(X_tr)
    assert model.Z.shape == (tok.n_entities, 8) and np.isfinite(model.Z).all()
    hist = model.info["history"]
    assert hist[-1] < hist[0], "GAE loss did not go down"
    E_tr, E_va, E_te = (model.transform(d) for d in (X_tr, X_va, X_te))
    assert E_te.shape == (len(X_te), 8) and np.isfinite(E_te).all()
    same = UniversalEmbedder(dim=8, target_kind="T", seed=0, max_epochs=300).fit(X_tr)
    assert np.allclose(same.Z, model.Z), "training is not reproducible"
    res, extra = evaluate_knn_classifier(E_tr, y_tr, E_va, y_va, E_te, y_te, [3, 5, 15],
                                         C.KNN_WEIGHTS)
    assert extra["proba"].shape == (len(X_te),)
    assert 0.0 <= extra["proba"].min() and extra["proba"].max() <= 1.0
    assert 0.0 <= res["roc_auc"] <= 1.0 and res["k"] in (3, 5, 15)
    model.set_reference(E_tr, y_tr, res["k"], res["weights"])
    proba = model.predict_proba(X_te)
    assert proba.shape == (len(X_te),) and set(np.unique(model.predict(X_te))) <= {0, 1}
    assert model.outlier_score(X_te).shape == (len(X_te),)

    # edge cases: single row, unseen category, missing number, out-of-range number
    one = X_te.iloc[[0]].copy()
    assert model.transform(one).shape == (1, 8)
    assert model.predict_proba(one).shape == (1,)
    odd = X_te.iloc[:3].copy()
    odd.loc[odd.index[0], "occupation"] = "Never-seen-job"
    odd.loc[odd.index[1], "age"] = np.nan
    odd.loc[odd.index[2], "age"] = 1e6
    odd.loc[odd.index[2], "capital-gain"] = 9e9
    Bo, info = model.tokeniser.transform(odd, return_info=True)
    assert info["unseen_cells"] == 2
    w = np.asarray(Bo.sum(axis=1)).ravel()
    assert np.allclose(w, [11, 11, 12]), "unseen cells must simply be skipped"
    assert np.isfinite(model.predict_proba(odd)).all()
    blank = X_te.iloc[[0]].copy()
    for c in C.CAT_COLS:
        blank[c] = "zzz-unknown"
    for c in C.NUM_COLS + C.ZERO_COLS:
        blank[c] = np.nan
    assert np.allclose(model.transform(blank), 0.0), "row with no known entity -> zero vector"
    assert model.transform(X_te.iloc[:0]).shape == (0, 8), "empty frame"

    # save / load round trip
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.joblib"
        model.save(path)
        again = UniversalEmbedder.load(path)
        assert np.allclose(again.transform(X_te), E_te)
        assert np.abs(again.predict_proba(X_te) - proba).mean() < 1e-3

    # baseline model and the other embeddings
    base = RawKNNBaseline(k=5).fit(X_tr, y_tr)
    pb = base.predict_proba(X_te)
    assert pb.shape == (len(X_te),) and np.isfinite(pb).all()
    assert np.isfinite(base.predict_proba(odd)).all()
    raw = RawFeatures().fit(X_tr)
    oh = raw.onehot(X_tr)
    assert fit_pca(oh, 8).transform(raw.onehot(X_te)).shape == (len(X_te), 8)
    Zs = spectral_entities(make_target(A, "T"), 8)
    assert Zs.shape == (tok.n_entities, 8)
    pairs = random_walk_pairs(transition(A), walks_per_node=5, walk_len=5, window=2, seed=0)
    Zg = skipgram_entities(pairs, tok.n_entities, 8, seed=0, epochs=1)
    assert row_embeddings(tok.transform(X_te), Zg).shape == (len(X_te), 8)

    # universality heads: regression without the age column, outlier scores
    no_age = UniversalEmbedder(dim=8, seed=0, num_cols=["hours-per-week"], max_epochs=200)
    no_age.fit(X_tr)
    assert not any(e["column"] == "age" for e in no_age.tokeniser.entities)
    ages = [d["age"].to_numpy(dtype=float) for d in (train, val, test)]
    reg, pred = evaluate_knn_regressor(no_age.transform(X_tr), ages[0], no_age.transform(X_va),
                                       ages[1], no_age.transform(X_te), ages[2], [3, 5],
                                       C.KNN_WEIGHTS)
    assert pred.shape == (len(X_te),) and np.isfinite(reg["mae"])
    s = outlier_scores(E_tr, E_te, 5)
    far = outlier_scores(E_tr, E_te + 10.0 * E_tr.std(), 5)
    assert outlier_auc(s, far) > 0.9, "shifted rows must look like outliers"
    assert isinstance(pd.DataFrame(tok.entities), pd.DataFrame)


def main():
    t0 = time.time()
    try:
        run()
    except Exception as exc:  # report and fail
        import traceback
        traceback.print_exc()
        print("selfcheck FAILED: %s" % type(exc).__name__)
        return 1
    if time.time() - t0 > 60:
        print("selfcheck FAILED: took more than 60 seconds")
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
