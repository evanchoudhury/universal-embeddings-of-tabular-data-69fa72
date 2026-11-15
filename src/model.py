"""Baseline and main model in one place.

Main model : UniversalEmbedder (src/embed.py) - entity graph + one-layer graph auto-encoder,
             row embedding = weighted mean of entity vectors, k-NN on top.
Baseline   : RawKNNBaseline - the same k-NN applied directly to ordinal-encoded, standardised
             raw columns (no embedding).
"""
import numpy as np

from .baselines import RawFeatures
from .embed import UniversalEmbedder, row_embeddings  # noqa: F401  (re-exported)
from .evaluate import knn_eps, knn_proba, knn_search
from .gae import EntityGAE, train_gae  # noqa: F401  (re-exported)


class RawKNNBaseline:
    """k-NN on ordinal codes + standardised numbers."""

    def __init__(self, k=25, weights="uniform", cat_cols=None, num_cols=None, log_cols=None):
        self.k = int(k)
        self.weights = weights
        self.features = RawFeatures(cat_cols, num_cols, log_cols)

    def fit(self, df, y):
        self.features.fit(df)
        self.X_ref = self.features.ordinal(df)
        self.y_ref = np.asarray(y)
        return self

    def transform(self, df):
        return self.features.ordinal(df)

    def predict_proba(self, df):
        dist, idx = knn_search(self.X_ref, self.transform(df), self.k)
        return knn_proba(dist, idx, self.y_ref, self.k, knn_eps(self.X_ref), self.weights)

    def predict(self, df):
        return (self.predict_proba(df) >= 0.5).astype(int)
