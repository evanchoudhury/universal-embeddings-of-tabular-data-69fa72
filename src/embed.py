"""Training side of the fitted pipeline: adds graph auto-encoder fitting (needs torch).

Inference (transform, k-NN heads, save/load) is in src/inference.py, which is torch-free so
the dashboard can import it. Saved files are identical for both classes.
"""
from . import config as C
from .gae import train_gae
from .graph import build_graph, make_target, propagation
from .inference import EmbeddingPredictor, row_embeddings  # noqa: F401  (re-exported)


class UniversalEmbedder(EmbeddingPredictor):
    """Main model: label-free row embeddings from an entity-graph auto-encoder."""

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
