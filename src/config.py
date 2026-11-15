"""Paths (resolved from this file), seeds and hyper-parameters."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS / "plots"

OPENML_ID = 1590
DATASET_NAME = "Adult Census Income (OpenML id 1590)"
CACHE_CSV = DATA_DIR / "adult_openml_1590.csv"

RAW_TARGET = "class"
TARGET = "income"            # 1 if >50K else 0
DROP_COLS = ["fnlwgt", "education-num"]
CAT_COLS = ["workclass", "education", "marital-status", "occupation", "relationship",
            "race", "sex", "native-country"]
NUM_COLS = ["age", "hours-per-week"]              # dense numerical: soft quantile bins
ZERO_COLS = ["capital-gain", "capital-loss"]      # zero-inflated: zero bin + quantile bins
FEATURE_COLS = ["age", "workclass", "education", "marital-status", "occupation",
                "relationship", "race", "sex", "capital-gain", "capital-loss",
                "hours-per-week", "native-country"]
MISSING = "Missing"

SPLIT_SEED = 42
VAL_SHARE = 0.15
TEST_SHARE = 0.15

# tokeniser
N_BINS = 20
N_BINS_ZERO = 10

# graph auto-encoder
EMB_DIM = 16
DIM_SWEEP = [2, 4, 8, 16, 32, 64]
GAE_LR = 0.01
GAE_CLIP = 1.0
GAE_MAX_EPOCHS = 10000
GAE_PATIENCE = 200
GAE_TOL = 1e-3
GAE_ALPHA = 0.0
GAE_INIT_STD = 0.1
GAE_SEEDS = [0, 1, 2]
TARGET_KINDS = ["T", "S"]     # T: row-stochastic (paper), S: symmetric (ablation)

# k-NN heads
K_GRID = [5, 15, 25, 50, 100]
KNN_WEIGHTS = ["distance", "uniform"]   # chosen on validation together with k
OUTLIER_K = 10
NEIGHBOUR_K = 10
CORRUPT_SHARE = 0.5
CORRUPT_SEED = 7

# skip-gram baseline (EmbDI-style approximation)
SG_WALKS_PER_NODE = 100
SG_WALK_LEN = 10
SG_WINDOW = 3
SG_NEG = 5
SG_EPOCHS = 3
SG_BATCH = 4096
SG_LR = 0.01

N_SAMPLE_PREDICTIONS = 500


def config_dict():
    """Hyper-parameters written to metrics.json."""
    keys = ["SPLIT_SEED", "VAL_SHARE", "TEST_SHARE", "N_BINS", "N_BINS_ZERO", "EMB_DIM",
            "DIM_SWEEP", "GAE_LR", "GAE_CLIP", "GAE_MAX_EPOCHS", "GAE_PATIENCE", "GAE_TOL",
            "GAE_ALPHA", "GAE_INIT_STD", "GAE_SEEDS", "TARGET_KINDS", "K_GRID", "KNN_WEIGHTS",
            "OUTLIER_K", "NEIGHBOUR_K", "CORRUPT_SHARE", "CORRUPT_SEED", "SG_WALKS_PER_NODE",
            "SG_WALK_LEN", "SG_WINDOW", "SG_NEG", "SG_EPOCHS", "SG_BATCH", "SG_LR",
            "FEATURE_COLS", "DROP_COLS"]
    g = globals()
    return {k.lower(): g[k] for k in keys}
