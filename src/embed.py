"""Row embeddings and the fitted pipeline object (tokeniser + entity vectors + k-NN memory)."""
import joblib
import numpy as np

from . import config as C
from .evaluate import knn_search, knn_proba, knn_eps
from .gae import train_gae
from .graph import build_graph, make_target, propagation
from .tokeniser import TableTokeniser


def row_embeddings(B, Z):
    """E = diag(1 / rowsum(B)) B Z. Rows without any known entity get the zero vector."""
    rs = np.asarray(B.sum(axis=1)).ravel()
    return np.asarray(B @ Z) / np.where(rs > 0, rs, 1.0)[:, None]


class UniversalEmbedder:
    """Main model: label-free row embeddings from an entity-graph auto-encoder."""

    def __init__(self, dim=C.EMB_DIM, target_kind="T", seed=0, cat_cols=None, num_cols=None,
                 zero_cols=None, max_epochs=C.GAE_MAX_EPOCHS):
        self.dim, self.target_kind, self.seed = int(dim), target_kind, int(seed)
        self.max_epochs = int(max_epochs)
        self.tokeniser = TableTokeniser(
            C.CAT_COLS if cat_cols is None else cat_cols,
            C.NUM_COLS if num_cols is None else num_cols,
            C.ZERO_COLS if zero_cols is None else zero_cols, C.N_BINS, C.N_BINS_ZERO)
        self.Z = None
        self.info = {}
        self.E_ref = None
        self.y_ref = None
        self.k = 25
        self.weights = "uniform"

    # ----- fitting (no labels) -----
    def fit(self, df):
        B = self.tokeniser.fit_transform(df)
        A = build_graph(B)
        self.fit_graph(A)
        return self

    def fit_graph(self, A):
        """Train on an already built graph (lets train.py share A between runs)."""
        self.Z, self.info = train_gae(
            propagation(A), make_target(A, self.target_kind), self.dim, seed=self.seed,
            lr=C.GAE_LR, clip=C.GAE_CLIP, max_epochs=self.max_epochs,
            patience=C.GAE_PATIENCE, tol=C.GAE_TOL, alpha=C.GAE_ALPHA,
            init_std=C.GAE_INIT_STD)
        return self

    def transform(self, df):
        return row_embeddings(self.tokeniser.transform(df), self.Z)

    # ----- downstream heads (only store the training embeddings) -----
    def set_reference(self, E_ref, y_ref, k, weights="uniform"):
        self.E_ref = np.asarray(E_ref, dtype=np.float64)
        self.y_ref = np.asarray(y_ref)
        self.k = int(k)
        self.weights = weights
        return self

    def neighbours(self, df, k=None):
        # search with the same depth as the evaluation so tied neighbours are cut the same way
        k = self.k if k is None else int(k)
        dist, idx = knn_search(self.E_ref, self.transform(df), max(k, max(C.K_GRID)))
        return dist[:, :k], idx[:, :k]

    def predict_proba(self, df):
        dist, idx = self.neighbours(df)
        return knn_proba(dist, idx, self.y_ref, self.k, knn_eps(self.E_ref), self.weights)

    def predict(self, df):
        return (self.predict_proba(df) >= 0.5).astype(int)

    def outlier_score(self, df, k=C.OUTLIER_K):
        dist, _ = self.neighbours(df, k)
        return dist.mean(axis=1)

    # ----- persistence (plain python / numpy only) -----
    def save(self, path):
        joblib.dump({"dim": self.dim, "target_kind": self.target_kind, "seed": self.seed,
                     "tokeniser": self.tokeniser.to_dict(),
                     "Z": np.asarray(self.Z, dtype=np.float64),
                     "E_ref": self.E_ref, "y_ref": self.y_ref, "k": self.k,
                     "weights": self.weights,
                     "info": {k: v for k, v in self.info.items() if k != "history"}},
                    path, compress=3)

    @classmethod
    def load(cls, path):
        d = joblib.load(path)
        m = cls(d["dim"], d["target_kind"], d["seed"])
        m.tokeniser = TableTokeniser.from_dict(d["tokeniser"])
        m.Z, m.E_ref, m.y_ref, m.k = d["Z"], d["E_ref"], d["y_ref"], d["k"]
        m.weights, m.info = d.get("weights", "uniform"), d.get("info", {})
        return m
