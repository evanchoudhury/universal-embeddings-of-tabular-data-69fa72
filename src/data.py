"""Load, clean and split the Adult table. Also builds a synthetic table with the same schema."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


def _fetch_raw(verbose=True):
    """Return the raw OpenML frame. Downloads once, then reuses data/ (offline)."""
    if C.CACHE_CSV.exists():
        if verbose:
            print("[data] using cached dataset: data/%s" % C.CACHE_CSV.name)
        return pd.read_csv(C.CACHE_CSV, keep_default_na=False, na_values=[""])
    if verbose:
        print("[data] downloading Adult from OpenML (id %d), first run only ..." % C.OPENML_ID)
    from sklearn.datasets import fetch_openml
    C.DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        bunch = fetch_openml(data_id=C.OPENML_ID, as_frame=True, parser="auto",
                             data_home=str(C.DATA_DIR))
    except Exception as exc:  # network problems
        raise RuntimeError("Could not download the Adult dataset from OpenML. Check the "
                           "internet connection and run again. (%s)" % type(exc).__name__)
    frame = bunch.frame.copy()
    for col in frame.columns:
        if str(frame[col].dtype) in ("category", "object", "string"):
            frame[col] = frame[col].astype(object)
    frame.to_csv(C.CACHE_CSV, index=False)
    if verbose:
        print("[data] cached to data/%s" % C.CACHE_CSV.name)
    return frame


def clean(raw):
    """Strip strings, mark missing categories, binarise the target, drop unused columns."""
    df = raw.copy()
    for col in C.CAT_COLS + [C.RAW_TARGET]:
        s = df[col].astype(object)
        s = s.where(s.isna(), s.astype(str).str.strip())
        s = s.where(~s.isin(["?", "", "nan", "None"]), np.nan)
        df[col] = s
    df[C.TARGET] = df[C.RAW_TARGET].astype(str).str.rstrip(".").eq(">50K").astype(int)
    for col in C.CAT_COLS:
        df[col] = df[col].fillna(C.MISSING).astype(str)
    for col in C.NUM_COLS + C.ZERO_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    df = df[C.FEATURE_COLS + [C.TARGET]].reset_index(drop=True)
    df.index.name = "row_id"
    return df


def load_adult(verbose=True):
    """Cleaned Adult table: 12 feature columns + binary `income`."""
    return clean(_fetch_raw(verbose))


def split(df, seed=C.SPLIT_SEED):
    """Stratified 70 / 15 / 15 split. The original row id is kept as the index."""
    rest, test = train_test_split(df, test_size=C.TEST_SHARE, stratify=df[C.TARGET],
                                  random_state=seed)
    val_share = C.VAL_SHARE / (1.0 - C.TEST_SHARE)
    train, val = train_test_split(rest, test_size=val_share, stratify=rest[C.TARGET],
                                  random_state=seed)
    return train.sort_index(), val.sort_index(), test.sort_index()


def dataset_stats(df, train, val, test):
    raw_missing = {c: int((df[c] == C.MISSING).sum()) for c in C.CAT_COLS}
    return {
        "n_rows": int(len(df)),
        "n_feature_columns": len(C.FEATURE_COLS),
        "n_train": int(len(train)), "n_val": int(len(val)), "n_test": int(len(test)),
        "positive_share": float(df[C.TARGET].mean()),
        "positive_share_train": float(train[C.TARGET].mean()),
        "positive_share_val": float(val[C.TARGET].mean()),
        "positive_share_test": float(test[C.TARGET].mean()),
        "missing_cells_by_column": {k: v for k, v in raw_missing.items() if v > 0},
        "n_levels_by_column": {c: int(df[c].nunique()) for c in C.CAT_COLS},
        "zero_share": {c: float((df[c] == 0).mean()) for c in C.ZERO_COLS},
        "n_duplicate_feature_rows": int(df[C.FEATURE_COLS].duplicated().sum()),
    }


def make_synthetic(n=600, seed=0):
    """Small random table with the Adult schema (already cleaned). Used by selfcheck.py only."""
    rng = np.random.default_rng(seed)
    levels = {
        "workclass": ["Private", "Self-emp", "Gov", C.MISSING],
        "education": ["HS-grad", "Bachelors", "Masters", "Some-college"],
        "marital-status": ["Married", "Never-married", "Divorced"],
        "occupation": ["Sales", "Tech", "Craft", "Exec", C.MISSING],
        "relationship": ["Husband", "Wife", "Own-child", "Not-in-family"],
        "race": ["A", "B", "C"],
        "sex": ["Male", "Female"],
        "native-country": ["X", "Y", C.MISSING],
    }
    edu = rng.integers(0, 4, n)
    age = np.clip(rng.normal(38 + 3 * edu, 12, n).round(), 17, 90)
    hours = np.clip(rng.normal(40, 10, n).round(), 1, 99)
    gain = np.where(rng.random(n) < 0.1, rng.integers(500, 20000, n), 0).astype(float)
    loss = np.where(rng.random(n) < 0.05, rng.integers(200, 3000, n), 0).astype(float)
    data = {c: rng.choice(v, n) for c, v in levels.items()}
    data["education"] = np.array(levels["education"])[edu]
    data.update({"age": age, "hours-per-week": hours, "capital-gain": gain,
                 "capital-loss": loss})
    score = 0.8 * edu + 0.04 * (age - 38) + 0.03 * (hours - 40) + (gain > 0) * 1.5
    score = score + rng.normal(0, 1, n)
    df = pd.DataFrame(data)[C.FEATURE_COLS]
    df[C.TARGET] = (score > np.quantile(score, 0.75)).astype(int)
    df.index.name = "row_id"
    return df
