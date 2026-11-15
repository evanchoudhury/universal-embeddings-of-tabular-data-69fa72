"""TableTokeniser: turns rows into weighted entities (categories and soft numerical bins)."""
import numpy as np
import pandas as pd
from scipy import sparse


def _bin_centres(x, n_bins):
    """Mean of the values in each quantile bin (duplicate edges merged)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.array([0.0])
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 2:
        return np.array([float(x.mean())])
    b = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, len(edges) - 2)
    centres = [x[b == i].mean() for i in range(len(edges) - 1) if np.any(b == i)]
    return np.unique(np.asarray(centres, dtype=float))


def _soft_weights(x, centres):
    """Share each value between the two neighbouring bin centres (linear interpolation).

    Returns (lo, w_lo, hi, w_hi). Values outside the outer centres go fully to the outer bin.
    """
    k = len(centres)
    if k == 1:
        z = np.zeros(len(x), dtype=int)
        return z, np.ones(len(x)), z, np.zeros(len(x))
    lo = np.clip(np.searchsorted(centres, x, side="right") - 1, 0, k - 2)
    t = np.clip((x - centres[lo]) / (centres[lo + 1] - centres[lo]), 0.0, 1.0)
    return lo, 1.0 - t, lo + 1, t


class TableTokeniser:
    """Fitted on training rows; maps any dataframe to a sparse row-entity matrix B."""

    def __init__(self, cat_cols, num_cols, zero_cols, n_bins=20, n_bins_zero=10):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.zero_cols = list(zero_cols)
        self.n_bins = int(n_bins)
        self.n_bins_zero = int(n_bins_zero)
        self.entities = []       # list of dicts: name, column, kind, value
        self.cat_index = {}      # column -> {value: entity id}
        self.centres = {}        # column -> bin centres (numpy)
        self.offset = {}         # column -> entity id of the first bin
        self.zero_index = {}     # column -> entity id of the exact-zero entity

    @property
    def columns(self):
        return self.cat_cols + self.num_cols + self.zero_cols

    @property
    def n_entities(self):
        return len(self.entities)

    @property
    def names(self):
        return [e["name"] for e in self.entities]

    def _add(self, column, kind, value, label):
        self.entities.append({"name": "%s=%s" % (column, label), "column": column,
                              "kind": kind, "value": value})
        return len(self.entities) - 1

    def fit(self, df):
        self.entities, self.cat_index, self.centres = [], {}, {}
        self.offset, self.zero_index = {}, {}
        for col in self.cat_cols:
            values = sorted(pd.unique(df[col].astype(str)))
            self.cat_index[col] = {v: self._add(col, "category", v, v) for v in values}
        for col in self.num_cols:
            self._fit_bins(col, df[col].to_numpy(dtype=float), self.n_bins)
        for col in self.zero_cols:
            x = df[col].to_numpy(dtype=float)
            self.zero_index[col] = self._add(col, "zero", 0.0, "0")
            self._fit_bins(col, x[np.isfinite(x) & (x != 0)], self.n_bins_zero)
        return self

    def _fit_bins(self, col, x, n_bins):
        centres = _bin_centres(x, n_bins)
        self.centres[col] = centres
        ids = [self._add(col, "bin", float(c), "~%.6g" % c) for c in centres]
        self.offset[col] = ids[0]

    def transform(self, df, return_info=False):
        n = len(df)
        rows, cols, vals = [], [], []
        unseen = 0
        ar = np.arange(n)
        for col in self.cat_cols:
            idx = df[col].astype(str).map(self.cat_index[col]).to_numpy(dtype=float)
            ok = np.isfinite(idx)
            unseen += int((~ok).sum())
            rows.append(ar[ok]); cols.append(idx[ok].astype(int)); vals.append(np.ones(ok.sum()))
        for col in self.num_cols + self.zero_cols:
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(x)
            unseen += int((~ok).sum())
            if col in self.zero_index:
                zero = ok & (x == 0)
                rows.append(ar[zero]); vals.append(np.ones(zero.sum()))
                cols.append(np.full(zero.sum(), self.zero_index[col], dtype=int))
                ok = ok & (x != 0)
            lo, w_lo, hi, w_hi = _soft_weights(x[ok], self.centres[col])
            off = self.offset[col]
            rows.extend([ar[ok], ar[ok]]); cols.extend([lo + off, hi + off])
            vals.extend([w_lo, w_hi])
        B = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows),
                                                      np.concatenate(cols))),
                              shape=(n, self.n_entities)).tocsr()
        B.sum_duplicates()
        B.eliminate_zeros()
        if return_info:
            return B, {"unseen_cells": unseen, "n_cells": n * len(self.columns)}
        return B

    def fit_transform(self, df):
        return self.fit(df).transform(df)

    def to_dict(self):
        return {"cat_cols": self.cat_cols, "num_cols": self.num_cols,
                "zero_cols": self.zero_cols, "n_bins": self.n_bins,
                "n_bins_zero": self.n_bins_zero, "entities": self.entities,
                "cat_index": self.cat_index, "offset": self.offset,
                "zero_index": self.zero_index,
                "centres": {k: [float(v) for v in c] for k, c in self.centres.items()}}

    @classmethod
    def from_dict(cls, d):
        t = cls(d["cat_cols"], d["num_cols"], d["zero_cols"], d["n_bins"], d["n_bins_zero"])
        t.entities = d["entities"]
        t.cat_index = d["cat_index"]
        t.offset = d["offset"]
        t.zero_index = d["zero_index"]
        t.centres = {k: np.asarray(v, dtype=float) for k, v in d["centres"].items()}
        return t
