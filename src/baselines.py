"""Baselines: raw-feature encoders, PCA, spectral and skip-gram entity vectors, supervised refs."""
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from . import config as C


class RawFeatures:
    """Classic encodings of the raw columns, fitted on training rows only."""

    def __init__(self, cat_cols=None, num_cols=None, log_cols=None):
        self.cat_cols = list(C.CAT_COLS if cat_cols is None else cat_cols)
        self.num_cols = list(C.NUM_COLS + C.ZERO_COLS if num_cols is None else num_cols)
        self.log_cols = list(C.ZERO_COLS if log_cols is None else log_cols)

    def _num(self, df):
        X = df[self.num_cols].apply(pd.to_numeric, errors="coerce").astype(float).copy()
        for c in self.log_cols:
            if c in X:
                X[c] = np.log1p(X[c].clip(lower=0))
        if hasattr(self, "medians_"):
            X = X.fillna(self.medians_)
        return X

    def _cat(self, df):
        return df[self.cat_cols].astype(str)

    def fit(self, df):
        num = self._num(df)
        self.medians_ = num.median()
        num = num.fillna(self.medians_)
        self.ord_ = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.ohe_ = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        codes = self.ord_.fit_transform(self._cat(df))
        self.ohe_.fit(self._cat(df))
        self.num_scaler_ = StandardScaler().fit(num.to_numpy())
        self.ord_scaler_ = StandardScaler().fit(np.hstack([codes, num.to_numpy()]))
        return self

    def ordinal_raw(self, df):
        """Ordinal codes + numbers, unscaled (for tree models)."""
        return np.hstack([self.ord_.transform(self._cat(df)), self._num(df).to_numpy()])

    def ordinal(self, df):
        """Ordinal codes + numbers, standardised (the simple 'no embedding' baseline)."""
        return self.ord_scaler_.transform(self.ordinal_raw(df))

    def onehot(self, df):
        return np.hstack([self.ohe_.transform(self._cat(df)),
                          self.num_scaler_.transform(self._num(df).to_numpy())])


def fit_pca(X_train, dim, seed=0):
    dim = int(min(dim, X_train.shape[1], X_train.shape[0]))
    return PCA(n_components=dim, svd_solver="full", random_state=seed).fit(X_train)


def spectral_entities(target, dim):
    """Truncated SVD of the target matrix; entity vectors U sqrt(S)."""
    U, s, _ = np.linalg.svd(np.asarray(target, dtype=float))
    dim = int(min(dim, len(s)))
    Z = U[:, :dim] * np.sqrt(s[:dim])
    # fix the sign of every component so the result does not depend on the LAPACK build
    sign = np.sign(Z[np.abs(Z).argmax(axis=0), np.arange(dim)])
    return Z * np.where(sign == 0, 1.0, sign)


def random_walk_pairs(T, walks_per_node=C.SG_WALKS_PER_NODE, walk_len=C.SG_WALK_LEN,
                      window=C.SG_WINDOW, seed=0):
    """(centre, context) pairs from random walks on the entity graph."""
    rng = np.random.default_rng(seed)
    n = T.shape[0]
    cum = np.cumsum(T, axis=1)
    cur = np.repeat(np.arange(n), walks_per_node)
    walks = [cur]
    for _ in range(walk_len - 1):
        u = rng.random(len(cur))
        cur = np.minimum((cum[cur] < u[:, None]).sum(axis=1), n - 1)
        walks.append(cur)
    W = np.stack(walks, axis=1)
    centre, context = [], []
    for off in range(1, window + 1):
        a, b = W[:, :-off].ravel(), W[:, off:].ravel()
        centre.extend([a, b]); context.extend([b, a])
    return np.concatenate(centre), np.concatenate(context)


def skipgram_entities(pairs, n_nodes, dim, seed=0, n_neg=C.SG_NEG, epochs=C.SG_EPOCHS,
                      batch=C.SG_BATCH, lr=C.SG_LR):
    """Skip-gram with negative sampling (EmbDI-style approximation). Returns input vectors."""
    centre = torch.tensor(pairs[0], dtype=torch.long)
    context = torch.tensor(pairs[1], dtype=torch.long)
    g = torch.Generator().manual_seed(int(seed))
    # one thread + float64: multi-threaded scatter-add sums in a varying order, which made
    # the result differ in the 5th decimal between runs
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    f64 = torch.float64
    emb_in = torch.nn.Parameter(torch.randn(n_nodes, dim, generator=g, dtype=f64) * 0.1)
    emb_out = torch.nn.Parameter(torch.randn(n_nodes, dim, generator=g, dtype=f64) * 0.1)
    noise = torch.bincount(context, minlength=n_nodes).double() ** 0.75
    noise = noise / noise.sum()
    opt = torch.optim.Adam([emb_in, emb_out], lr=lr)
    logsig = torch.nn.functional.logsigmoid
    m = len(centre)
    for _ in range(int(epochs)):
        perm = torch.randperm(m, generator=g)
        for s in range(0, m, batch):
            b = perm[s:s + batch]
            vi, vo = emb_in[centre[b]], emb_out[context[b]]
            neg = torch.multinomial(noise, len(b) * n_neg, replacement=True, generator=g)
            vn = emb_out[neg].view(len(b), n_neg, -1)
            loss = -logsig((vi * vo).sum(1)).mean() \
                   - logsig(-(vn * vi[:, None, :]).sum(2)).sum(1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    torch.set_num_threads(n_threads)
    return emb_in.detach().numpy().astype(np.float64)


def fit_logreg(X, y):
    return LogisticRegression(C=1.0, max_iter=2000).fit(X, y)


def fit_xgb_classifier(X, y, seed=0):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, tree_method="hist",
                         random_state=seed, n_jobs=4, verbosity=0).fit(X, y)


def fit_xgb_regressor(X, y, seed=0):
    from xgboost import XGBRegressor
    return XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, tree_method="hist",
                        random_state=seed, n_jobs=4, verbosity=0).fit(X, y)
