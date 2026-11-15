"""Streamlit dashboard: `streamlit run app.py`.

Everything shown here is read from artifacts/ (written by `python train.py`) and docs/.
No number is typed in by hand.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ART = ROOT / "artifacts"
PLOTS = ART / "plots"
DOCS = ROOT / "docs"
REQUIRED = ["metrics.json", "model.joblib", "embeddings.npz", "graph.npz", "entities.csv",
            "test_rows.csv", "sample_predictions.csv", "training_history.json"]
POS, NEG = ">50K", "<=50K"
COLORS = {POS: "#d95f02", NEG: "#1b9e77"}
GROUP_COLORS = {"Main model": "#d95f02", "Raw features + k-NN": "#7570b3",
                "Other label-free embedding + k-NN": "#1b9e77",
                "Supervised (uses labels)": "#666666", "Trivial": "#bbbbbb"}

st.set_page_config(page_title="Universal Embeddings of Tabular Data", layout="wide")


# ------------------------------------------------------------------ loading (cached)
@st.cache_data(show_spinner=False)
def load_json(name):
    with open(ART / name, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_csv(name):
    return pd.read_csv(ART / name, keep_default_na=False, na_values=[""])


@st.cache_data(show_spinner=False)
def load_npz(name):
    with np.load(ART / name, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


@st.cache_resource(show_spinner="Loading the trained model ...")
def load_model():
    from src.embed import UniversalEmbedder
    return UniversalEmbedder.load(ART / "model.joblib")


@st.cache_resource(show_spinner=False)
def load_projection():
    """2-D PCA of the training row embeddings, used only for drawing maps."""
    from sklearn.decomposition import PCA
    emb = load_npz("embeddings.npz")
    proj = PCA(n_components=2, random_state=0).fit(emb["E_train"].astype(np.float64))
    return proj, proj.transform(emb["E_test"].astype(np.float64))


@st.cache_data(show_spinner=False)
def load_history(max_points=600):
    """Loss curves, thinned so the browser stays fast (first and last epoch are kept)."""
    hist = load_json("training_history.json")
    rows = []
    for kind, by_seed in hist["loss"].items():
        for seed, loss in by_seed.items():
            n = len(loss)
            step = max(1, n // max_points)
            keep = sorted(set(range(0, n, step)) | {n - 1})
            rows += [{"target": kind, "seed": "seed %s" % seed, "epoch": i + 1,
                      "loss": loss[i]} for i in keep]
    return pd.DataFrame(rows), {k: {s: len(v) for s, v in d.items()}
                                for k, d in hist["loss"].items()}


@st.cache_data(show_spinner=False)
def load_full_table():
    """The cleaned full table if train.py has cached it in data/ (never downloads)."""
    try:
        from src import config as C
        from src.data import load_adult
        if not C.CACHE_CSV.exists():
            return None
        df = load_adult(verbose=False)
        out = df[C.FEATURE_COLS].copy()
        out["income"] = np.where(df[C.TARGET] == 1, POS, NEG)
        return out
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def read_doc(path_str):
    p = Path(path_str)
    return p.read_text(encoding="utf-8") if p.exists() else None


def doc(name):
    return read_doc(str(DOCS / name))


def md_section(text, heading):
    """Body of the '## heading' section of a markdown document ('' if absent)."""
    if not text:
        return ""
    out, on = [], False
    for line in text.splitlines():
        if line.startswith("## "):
            if on:
                break
            on = line[3:].strip().lower().startswith(heading.lower())
            continue
        if on:
            out.append(line)
    return "\n".join(out).strip()


def show_markdown(text, base):
    """st.markdown, but local images written as ![alt](relative/path.png) are drawn too."""
    import re
    buf = []

    def flush():
        if "".join(buf).strip():
            st.markdown("\n".join(buf))
        buf.clear()

    for line in text.splitlines():
        hit = re.fullmatch(r"\s*!\[([^\]]*)\]\(([^)\s]+)\)\s*", line)
        img = None
        if hit and not hit.group(2).startswith(("http://", "https://")):
            img = (Path(base) / hit.group(2)).resolve()
            if not (img.is_file() and ROOT in img.parents):
                img = None
        if img is None:
            buf.append(line)
            continue
        flush()
        st.image(str(img), caption=hit.group(1) or None, width="stretch")
    flush()


def find_paper():
    seen, found = set(), []
    for base in (ROOT, DOCS, ROOT / "paper", ROOT / "research_paper"):
        if base.exists():
            for p in sorted(base.glob("*.md")):
                if "paper" in p.name.lower() and p not in seen:
                    seen.add(p)
                    found.append(p)
    return found


def f4(x):
    try:
        return "%.4f" % float(x)
    except (TypeError, ValueError):
        return "n/a"


def pct(x, digits=1):
    try:
        return ("%." + str(digits) + "f%%") % (100.0 * float(x))
    except (TypeError, ValueError):
        return "n/a"


def show(fig, height=None):
    if height:
        fig.update_layout(height=height)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


# ------------------------------------------------------------------ derived tables
def comparison_table(m):
    mm = m["metrics"]
    d = mm.get("dim", 16)
    comp = mm.get("comparison", {})
    spec = [("raw_ordinal_knn", "k-NN on raw ordinal columns (baseline)", "Raw features + k-NN"),
            ("raw_onehot_knn", "k-NN on raw one-hot columns", "Raw features + k-NN"),
            ("pca_d%d" % d, "PCA-%d + k-NN" % d, "Other label-free embedding + k-NN"),
            ("spectral_d%d" % d, "Spectral-%d + k-NN" % d, "Other label-free embedding + k-NN"),
            ("skipgram_d%d" % d, "Skip-gram-%d + k-NN" % d, "Other label-free embedding + k-NN"),
            ("logistic_regression", "Logistic regression", "Supervised (uses labels)"),
            ("xgboost", "XGBoost", "Supervised (uses labels)"),
            ("majority_class", "Always predict the majority class", "Trivial")]
    rows = [{"method": "GAE-%d + k-NN (main model)" % d, "group": "Main model",
             "ROC-AUC": mm["roc_auc"], "accuracy": mm["accuracy"], "F1": mm["f1"],
             "average precision": mm["average_precision"],
             "vector size": mm.get("dim"), "k": mm.get("k")}]
    for key, label, group in spec:
        if key in comp:
            r = comp[key]
            rows.append({"method": label, "group": group, "ROC-AUC": r.get("roc_auc"),
                         "accuracy": r.get("accuracy"), "F1": r.get("f1"),
                         "average precision": r.get("average_precision"),
                         "vector size": r.get("dim"), "k": r.get("k")})
    return pd.DataFrame(rows)


def standing_sentence(comp_df):
    """Plain, computed statement of where the main model ranks (no hand-typed numbers)."""
    real = comp_df[comp_df["group"] != "Trivial"].sort_values("ROC-AUC", ascending=False)
    real = real.reset_index(drop=True)
    main_auc = float(real.loc[real["group"] == "Main model", "ROC-AUC"].iloc[0])
    rank = int(real.index[real["group"] == "Main model"][0]) + 1
    better = real[real["ROC-AUC"] > main_auc]["method"].tolist()
    worse = real[real["ROC-AUC"] < main_auc]["method"].tolist()
    txt = "By test ROC-AUC the main model ranks **%d of %d** methods." % (rank, len(real))
    if worse:
        txt += " It is ahead of: " + "; ".join(worse) + "."
    if better:
        txt += " It is behind: " + "; ".join(better) + "."
    return txt


# ------------------------------------------------------------------ tabs
def tab_overview(m, comp_df):
    mm, bm = m["metrics"], m["baseline_metrics"]
    st.subheader("The idea in simple words")
    st.markdown(
        "A database table is usually filled long before anyone knows what questions will be "
        "asked about it. This project turns **every row of a table into a short list of "
        "numbers (an *embedding*) without looking at any target column**. Rows that describe "
        "similar people end up close together. After that, simple distance look-ups are enough "
        "to do several jobs with the same vectors: classify, predict a number, flag strange "
        "rows and find similar rows.\n\n"
        "The method follows the paper *Universal Embeddings of Tabular Data* "
        "(arXiv:2507.05904): each cell value becomes an *entity* (for example "
        "`education=Bachelors` or an age bin), entities that occur in the same rows are linked "
        "in a graph, a small graph auto-encoder learns a vector per entity, and a row's vector "
        "is the average of its entity vectors.")
    st.caption("Dataset: %s. Task used to score the embeddings: %s."
               % (m.get("dataset", "n/a"), m.get("task", "n/a")))

    st.subheader("Headline results on the held-out test set")
    d_auc = mm["roc_auc"] - bm["roc_auc"]
    c = st.columns(4)
    c[0].metric("Main model ROC-AUC", f4(mm["roc_auc"]), "%+.4f vs baseline" % d_auc)
    c[1].metric("Baseline ROC-AUC", f4(bm["roc_auc"]), help=m.get("baseline_name", ""))
    c[2].metric("Main model accuracy", pct(mm["accuracy"], 2),
                "%+.2f points vs baseline" % (100 * (mm["accuracy"] - bm["accuracy"])))
    c[3].metric("Main model F1", f4(mm["f1"]), "%+.4f vs baseline" % (mm["f1"] - bm["f1"]))
    c = st.columns(4)
    ss = mm.get("seed_stability", {})
    c[0].metric("ROC-AUC over %s seeds" % ss.get("n_seeds", "?"),
                "%s +/- %s" % (f4(ss.get("test_roc_auc_mean")), f4(ss.get("test_roc_auc_std"))),
                help="Mean and standard deviation when the auto-encoder is retrained with "
                     "different random seeds.")
    xgb = mm.get("comparison", {}).get("xgboost", {})
    c[1].metric("XGBoost reference ROC-AUC", f4(xgb.get("roc_auc")),
                help="Supervised model trained with the labels. An upper reference, not a "
                     "fair competitor: the embedding never sees the label.")
    c[2].metric("Test rows", "%d" % m.get("n_test", 0))
    c[3].metric("Full training run", "%.0f s" % m.get("train_seconds", 0))

    st.info(
        "**Main model**: 16 numbers per row from the graph auto-encoder (trained without "
        "labels), then a k-nearest-neighbour vote.  \n"
        "**Baseline**: the same k-nearest-neighbour vote applied directly to the raw columns "
        "(%s)." % m.get("baseline_name", ""))
    st.markdown("**Honest standing.** " + standing_sentence(comp_df) +
                " The full table and charts are in the *Results* tab.")

    with st.expander("How to read these numbers", expanded=True):
        maj = mm.get("comparison", {}).get("majority_class", {})
        st.markdown(
            "- **ROC-AUC** (primary metric): pick one person who earns more than 50K and one "
            "who does not; ROC-AUC is the chance that the model gives the first one the higher "
            "score. 0.5 is coin flipping, 1.0 is perfect. It does not depend on a threshold.\n"
            "- **Accuracy**: share of test rows labelled correctly at a 0.5 threshold. Compare "
            "it with **%s**, which is what you get by always answering `<=50K`, because only "
            "%s of the rows are `>50K`.\n"
            "- **F1**: balance of precision and recall for the `>50K` class; it punishes a "
            "model that ignores the smaller class.\n"
            "- **Average precision**: area under the precision-recall curve (main model: %s; "
            "a random ranking would give about %s)."
            % (pct(maj.get("accuracy")), pct(m["metrics"]["dataset_stats"]["positive_share"]),
               f4(mm["average_precision"]), f4(maj.get("average_precision"))))

    prob = doc("02_problem_definition.md")
    if prob:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Problem")
            st.markdown(md_section(prob, "Problem statement"))
            st.markdown(md_section(prob, "Motivation"))
        with c2:
            st.subheader("Objectives")
            st.markdown(md_section(prob, "Objectives"))
    else:
        st.warning("docs/02_problem_definition.md was not found.")


def tab_data(m, test_rows):
    ds = m["metrics"]["dataset_stats"]
    st.subheader("Dataset")
    st.markdown(
        "**%s.** Each row is one person from the 1994 US census: age, work, education, family "
        "situation and so on. The label says whether the yearly income is above 50,000 USD. "
        "The label is **never** used to build the embeddings; it is only used afterwards to "
        "score them." % m.get("dataset", ""))
    c = st.columns(5)
    c[0].metric("Rows", "%d" % ds["n_rows"])
    c[1].metric("Feature columns", "%d" % ds["n_feature_columns"])
    c[2].metric("Train / val / test", "%d / %d / %d" % (ds["n_train"], ds["n_val"], ds["n_test"]))
    c[3].metric("Share earning >50K", pct(ds["positive_share"]))
    c[4].metric("Duplicate feature rows", "%d" % ds["n_duplicate_feature_rows"])

    full = load_full_table()
    if full is not None:
        df, src = full, "the full cleaned table (%d rows, cached in data/)" % len(full)
    else:
        df = test_rows.rename(columns={"true_label": "income"})
        src = "the held-out test split (%d rows, from artifacts/test_rows.csv)" % len(df)
    feat = [c_ for c_ in m["metrics"]["config"]["feature_cols"] if c_ in df.columns]
    num_cols = [c_ for c_ in feat if pd.api.types.is_numeric_dtype(df[c_])]
    cat_cols = [c_ for c_ in feat if c_ not in num_cols]
    st.caption("Exploring " + src + ".")

    st.markdown("#### Filters")
    f = st.columns(4)
    labels = f[0].multiselect("Income label", [NEG, POS], default=[NEG, POS], key="d_lab")
    a_lo, a_hi = int(df["age"].min()), int(df["age"].max())
    ages = f[1].slider("Age range", a_lo, a_hi, (a_lo, a_hi), key="d_age")
    sexes = f[2].multiselect("Sex", sorted(df["sex"].unique()), key="d_sex",
                             placeholder="all")
    edus = f[3].multiselect("Education", sorted(df["education"].unique()), key="d_edu",
                            placeholder="all")
    mask = df["income"].isin(labels) & df["age"].between(ages[0], ages[1])
    if sexes:
        mask &= df["sex"].isin(sexes)
    if edus:
        mask &= df["education"].isin(edus)
    view = df[mask]
    c = st.columns(3)
    c[0].metric("Rows after filters", "%d" % len(view))
    c[1].metric("Share >50K in selection",
                pct((view["income"] == POS).mean()) if len(view) else "n/a")
    c[2].metric("Median age in selection", "%.0f" % view["age"].median() if len(view) else "n/a")
    if len(view) == 0:
        st.warning("No rows match the filters. Widen them to see the charts.")
        return

    st.markdown("#### Sample rows")
    n_show = st.slider("Rows to show", 5, 200, 20, key="d_n")
    st.dataframe(view[feat + ["income"]].head(n_show), width="stretch", hide_index=True)

    st.markdown("#### Distributions")
    c1, c2 = st.columns(2)
    with c1:
        col = st.selectbox("Column", feat, key="d_col")
        if col in num_cols:
            nz = st.checkbox("Hide zeros (useful for capital-gain / capital-loss)",
                             value=False, key="d_nz")
            v = view[view[col] != 0] if nz else view
            fig = px.histogram(v, x=col, color="income", nbins=40, barmode="overlay",
                               opacity=0.65, color_discrete_map=COLORS,
                               title="Distribution of %s by income" % col)
        else:
            cnt = view.groupby([col, "income"]).size().reset_index(name="rows")
            order = view[col].value_counts().index.tolist()
            fig = px.bar(cnt, y=col, x="rows", color="income", orientation="h",
                         color_discrete_map=COLORS, category_orders={col: order},
                         title="Rows per value of %s" % col)
        show(fig, 460)
        st.caption("How to read: bar height (or length) is the number of rows; colours split "
                   "them by income label.")
    with c2:
        col2 = st.selectbox("Share earning >50K by", cat_cols + num_cols, key="d_col2")
        tmp = view[[col2, "income"]].copy()
        if col2 in num_cols:
            try:
                tmp[col2] = pd.qcut(tmp[col2], 8, duplicates="drop").astype(str)
            except ValueError:
                tmp[col2] = tmp[col2].astype(str)
        g = tmp.groupby(col2)["income"].agg(share=lambda s: (s == POS).mean(), rows="size")
        g = g.reset_index().sort_values("share")
        fig = px.bar(g, y=col2, x="share", orientation="h", hover_data=["rows"],
                     title="Share of rows with income >50K, by %s" % col2,
                     color_discrete_sequence=[COLORS[POS]])
        fig.add_vline(x=float((view["income"] == POS).mean()), line_dash="dash",
                      annotation_text="selection average")
        fig.update_xaxes(tickformat=".0%")
        show(fig, 460)
        st.caption("How to read: longer bar = that group earns >50K more often. Hover to see "
                   "how many rows are behind each bar; small groups are noisy.")

    c1, c2 = st.columns(2)
    with c1:
        miss = ds.get("missing_cells_by_column", {})
        if miss:
            st.markdown("**Missing cells** (kept as their own `Missing` category)")
            st.dataframe(pd.DataFrame({"column": list(miss), "missing cells": list(miss.values())}),
                         hide_index=True, width="stretch")
    with c2:
        z = ds.get("zero_share", {})
        if z:
            st.markdown("**Zero-inflated columns** (get a dedicated 'exactly zero' entity)")
            st.dataframe(pd.DataFrame({"column": list(z),
                                       "share of zeros": [pct(v) for v in z.values()]}),
                         hide_index=True, width="stretch")
    d4 = doc("04_dataset.md")
    if d4:
        with st.expander("Full dataset notes (docs/04_dataset.md): source, license, "
                         "preprocessing, ethics"):
            st.markdown(d4)


PIPELINE_DOT = """
digraph G {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#f3f3f3", fontsize=11];
  A [label="Table rows\\n(no label column)"];
  B [label="Tokeniser\\ncategories + soft\\nquantile bins"];
  C [label="Entity graph\\nA = B^T B"];
  D [label="Graph auto-encoder\\nZ = P W, decode Z Z^T"];
  E [label="Entity vectors Z"];
  F [label="Row embedding\\nweighted mean of\\nits entity vectors", fillcolor="#fde3cf"];
  G1 [label="k-NN vote:\\nincome class"]; G2 [label="k-NN mean:\\nage regression"];
  G3 [label="mean k-NN distance:\\noutlier score"]; G4 [label="nearest rows:\\nsimilarity search"];
  A -> B -> C -> D -> E -> F; B -> F; F -> G1; F -> G2; F -> G3; F -> G4;
}
"""


def tab_model(m, entities, graph):
    mm = m["metrics"]
    g, cfg = mm["graph"], mm["config"]
    st.subheader("Pipeline")
    st.graphviz_chart(PIPELINE_DOT, width="stretch")
    st.markdown(
        "1. **Tokenise.** Every category value becomes an entity. Numbers are cut into "
        "quantile bins and each value is shared between its two nearest bins, so close "
        "numbers stay close.\n"
        "2. **Build the entity graph.** Two entities are linked when they occur in the same "
        "rows; the link weight counts how often.\n"
        "3. **Train a one-layer graph auto-encoder** to reproduce the graph from a small "
        "vector per entity. No label is involved.\n"
        "4. **Embed rows.** A row's vector is the weighted mean of its entity vectors. This "
        "also works for rows never seen in training.\n"
        "5. **Use distances.** A k-nearest-neighbour look-up over the stored training vectors "
        "gives classification, regression, outlier scores and similar-row search.")
    c = st.columns(5)
    c[0].metric("Entities (graph nodes)", "%d" % g["n_nodes"])
    c[1].metric("Graph edges", "%d" % g["n_edges"])
    c[2].metric("Embedding size", "%d" % mm["dim"])
    c[3].metric("Trainable parameters", "%d" % mm["n_parameters"])
    c[4].metric("Saved model size", "%.2f MB" % mm["model_file_mb"])

    st.subheader("Training curves of the graph auto-encoder")
    hist, lengths = load_history()
    kinds = sorted(hist["target"].unique(), key=lambda k: k != mm["target_kind"])
    c1, c2 = st.columns([1, 3])
    with c1:
        kind = st.radio("Reconstruction target", kinds, key="m_kind",
                        format_func=lambda k: "%s (%s)" % (
                            k, "row-stochastic, paper" if k == "T" else "symmetric, ablation"))
        logy = st.checkbox("Log scale for the loss", value=False, key="m_log")
        st.markdown("Epochs until early stopping:")
        st.dataframe(pd.DataFrame({"seed": list(lengths[kind]),
                                   "epochs": list(lengths[kind].values())}),
                     hide_index=True, width="stretch")
        st.caption("The target chosen on validation ROC-AUC is **%s**; the saved model is "
                   "seed %s." % (mm["target_kind"], mm["seed"]))
    with c2:
        fig = px.line(hist[hist["target"] == kind], x="epoch", y="loss", color="seed",
                      log_y=logy, title="Reconstruction loss per epoch, target %s" % kind)
        show(fig, 420)
        st.caption(
            "How to read: the loss measures how badly the entity vectors reproduce the entity "
            "graph; lower is better. Curves that flatten mean training has converged; curves "
            "for different seeds lying on top of each other mean the result does not depend "
            "on the random start. There is no validation loss because training uses no "
            "labels; the stop rule is 'improvement below %s over %s epochs'. The final model "
            "explains %s of the target matrix (it is a deliberate compression to %d numbers)."
            % (pct(cfg["gae_tol"]), cfg["gae_patience"], pct(mm["explained_share"]), mm["dim"]))

    st.subheader("What the model learned about the entities")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(entities, x="x", y="y", color="column", hover_name="name",
                         size=np.sqrt(entities["train_weight"].clip(lower=1)), size_max=22,
                         title="Entity vectors (2-D PCA view of %d dimensions)" % mm["dim"])
        show(fig, 520)
        st.caption("Each dot is one entity; size grows with how often it occurs. Entities "
                   "that occur with the same kinds of rows sit close together. It is a 2-D "
                   "shadow of the real vectors, so distances are only approximate.")
    with c2:
        zcols = [c_ for c_ in entities.columns if c_.startswith("z") and c_[1:].isdigit()]
        Z = entities[zcols].to_numpy(dtype=float)
        pick = st.selectbox("Find the entities most similar to", entities["name"].tolist(),
                            index=int(entities["train_weight"].to_numpy().argmax()),
                            key="m_ent")
        i = int(entities.index[entities["name"] == pick][0])
        Zn = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-12)
        sim = Zn @ Zn[i]
        other = entities["column"].to_numpy() != entities.loc[i, "column"]
        only_other = st.checkbox("Only show entities from other columns", value=True,
                                 key="m_other")
        cand = np.where(other if only_other else np.arange(len(sim)) != i)[0]
        top = cand[np.argsort(-sim[cand])[:12]]
        tab = pd.DataFrame({"entity": entities["name"].to_numpy()[top],
                            "cosine similarity": np.round(sim[top], 3),
                            "rows in training": entities["train_weight"].to_numpy()[top].round(0)})
        st.dataframe(tab, hide_index=True, width="stretch", height=455)
        st.caption("Cosine similarity of the learned vectors: 1 = same direction, 0 = "
                   "unrelated. Nothing here was taught with labels; it comes only from which "
                   "values occur together.")

    with st.expander("Entity graph as a heat map (%d x %d co-occurrence weights)"
                     % (g["n_nodes"], g["n_nodes"])):
        fig = go.Figure(go.Heatmap(z=np.log1p(graph["A"]), x=graph["names"], y=graph["names"],
                                   colorscale="Viridis", colorbar=dict(title="log(1+w)")))
        fig.update_layout(title="Link weight between every pair of entities (log scale)")
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False, autorange="reversed")
        show(fig, 650)
        st.caption("Bright = the two entities often appear in the same row. Dark blocks on "
                   "the diagonal are values of the same column, which can never co-occur. "
                   "Hover for the entity names.")
        ebc = g.get("entities_by_column", {})
        st.dataframe(pd.DataFrame({"column": list(ebc), "entities": list(ebc.values())}),
                     hide_index=True)

    st.subheader("Hyper-parameters")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Values used in this run** (artifacts/metrics.json, `config`)")
        st.dataframe(pd.DataFrame({"setting": list(cfg),
                                   "value": [json.dumps(v) if isinstance(v, (list, dict))
                                             else str(v) for v in cfg.values()]}),
                     hide_index=True, width="stretch", height=420)
        st.markdown("Chosen on validation: **k = %s**, neighbour weights **%s**, target "
                    "**%s**." % (mm["k"], mm["weights"], mm["target_kind"]))
    with c2:
        d5 = doc("05_model_design.md")
        hp = md_section(d5, "Hyper-parameters")
        if hp:
            st.markdown("**Why these values** (docs/05_model_design.md)")
            st.markdown(hp)
    if d5:
        with st.expander("Full model design document (components, baselines, evaluation "
                         "protocol)"):
            st.markdown(d5)
    t = mm.get("timings_seconds", {})
    if t:
        tt = pd.DataFrame({"step": list(t), "seconds": list(t.values())})
        fig = px.bar(tt[tt["step"] != "total"], x="seconds", y="step", orientation="h",
                     title="Where the training time goes (total %.0f s)" % t.get("total", 0))
        show(fig, 300)


def tab_results(m, comp_df, test_rows, sample):
    from sklearn.metrics import roc_curve
    mm, bm = m["metrics"], m["baseline_metrics"]
    st.subheader("All methods on the same test split")
    st.markdown(standing_sentence(comp_df))
    c1, c2 = st.columns([3, 2])
    with c1:
        metric = st.radio("Metric", ["ROC-AUC", "accuracy", "F1", "average precision"],
                          horizontal=True, key="r_metric")
        d = comp_df.sort_values(metric)
        fig = px.bar(d, x=metric, y="method", color="group", orientation="h", text=metric,
                     color_discrete_map=GROUP_COLORS, category_orders={"method": d["method"].tolist()[::-1]},
                     title="Test %s by method" % metric)
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        lo = float(d[metric].min())
        fig.update_xaxes(range=[max(0.0, lo - 0.05) if lo > 0.4 else 0.0, 1.0])
        fig.update_layout(legend=dict(orientation="h", y=-0.2, title=None))
        show(fig, 500)
    with c2:
        st.dataframe(comp_df.drop(columns=["group"]).style.format(
            {"ROC-AUC": "{:.4f}", "accuracy": "{:.4f}", "F1": "{:.4f}",
             "average precision": "{:.4f}", "vector size": "{:.0f}", "k": "{:.0f}"},
            na_rep="-"), hide_index=True, width="stretch", height=360)
        st.caption(
            "How to read: all k-NN rows use the same neighbour vote, only the vectors differ, "
            "so differences are due to the representation. 'Supervised' models were trained "
            "with the income labels and are shown as references. The 'vector size' column "
            "matters: a method with more numbers per row has more room to store information.")

    st.subheader("Does the size of the embedding matter?")
    sweep = mm["dimension_sweep"]
    split = st.radio("Split", ["test_roc_auc", "val_roc_auc"], horizontal=True, key="r_split",
                     format_func=lambda s: "Test" if s.startswith("test") else "Validation")
    names = {"gae": "GAE (main)", "pca": "PCA", "spectral": "Spectral", "skipgram": "Skip-gram"}
    rows = [{"method": names.get(k, k), "dimension": int(dim), "ROC-AUC": v}
            for k, dd in sweep[split].items() for dim, v in dd.items()]
    sw = pd.DataFrame(rows).sort_values("dimension")
    fig = px.line(sw, x="dimension", y="ROC-AUC", color="method", markers=True, log_x=True,
                  title="ROC-AUC of the k-NN vote vs embedding dimension")
    if split == "test_roc_auc":
        fig.add_hline(y=bm["roc_auc"], line_dash="dot",
                      annotation_text="raw ordinal k-NN baseline")
    fig.update_xaxes(tickvals=sorted(sw["dimension"].unique()))
    show(fig, 420)
    st.caption("How to read: each line is one label-free embedding method; further right "
               "means more numbers per row. A good universal embedding should already be "
               "strong at small sizes and flatten out after that.")

    st.subheader("Model selection and stability")
    c1, c2 = st.columns(2)
    with c1:
        rows = []
        for name, res in (("GAE main model", mm), ("raw ordinal baseline", bm)):
            for w, dd in res.get("val_roc_auc_grid", {}).items():
                rows += [{"method": "%s, %s weights" % (name, w), "k": int(k),
                          "validation ROC-AUC": v} for k, v in dd.items()]
        fig = px.line(pd.DataFrame(rows).sort_values("k"), x="k", y="validation ROC-AUC",
                      color="method", markers=True,
                      title="Choosing k and the neighbour weighting (validation split)")
        fig.update_layout(legend=dict(orientation="h", y=-0.25, title=None))
        show(fig, 430)
        st.caption("k is the number of neighbours that vote. The best point of each method "
                   "on validation was used once on the test split (main model: k = %s, %s)."
                   % (mm["k"], mm["weights"]))
    with c2:
        rows = []
        for kind, a in mm.get("target_ablation", {}).items():
            for split_name, key in (("validation", "val_roc_auc_by_seed"),
                                    ("test", "test_roc_auc_by_seed")):
                rows += [{"target": "target %s" % kind, "seed": "seed %s" % s,
                          "split": split_name, "ROC-AUC": v} for s, v in a[key].items()]
        fig = px.strip(pd.DataFrame(rows), x="target", y="ROC-AUC", color="split",
                       hover_data=["seed"],
                       title="Reconstruction target ablation, one dot per training seed")
        fig.update_traces(marker=dict(size=11))
        show(fig, 430)
        st.caption("T is the paper's row-stochastic target, S a symmetric variant. Dots close "
                   "together mean the result is stable across random seeds. The target was "
                   "picked using the validation dots only (chosen: %s)." % mm["target_kind"])

    st.subheader("Is the embedding really 'universal'? Three other jobs, same vectors")
    u = mm["universality"]
    c1, c2, c3 = st.columns(3)
    with c1:
        a = u["age_regression"]
        lab = {"gae_knn": "GAE + k-NN", "raw_ordinal_knn": "raw k-NN", "xgboost": "XGBoost",
               "training_mean": "predict the mean"}
        t = pd.DataFrame([{"method": lab[k], "MAE (years)": a[k]["mae"], "R2": a[k]["r2"]}
                          for k in lab if k in a])
        which = st.radio("Age regression metric", ["MAE (years)", "R2"], horizontal=True,
                         key="r_age")
        fig = px.bar(t, x="method", y=which, text=which, title="Predict age (column hidden)")
        fig.update_traces(texttemplate="%{text:.3f}")
        show(fig, 380)
        st.caption("The embedding is rebuilt without the age column, then age is predicted "
                   "from neighbours. MAE: average error in years (lower is better). R2: share "
                   "of age variation explained (higher is better, 0 = no better than the mean).")
    with c2:
        o = u["outlier_detection"]
        lab = {"gae_roc_auc": "GAE", "pca_roc_auc": "PCA", "onehot_roc_auc": "raw one-hot",
               "raw_ordinal_roc_auc": "raw ordinal"}
        t = pd.DataFrame([{"vectors": lab[k], "ROC-AUC": o[k]} for k in lab if k in o])
        fig = px.bar(t, x="vectors", y="ROC-AUC", text="ROC-AUC",
                     title="Spot corrupted rows")
        fig.update_traces(texttemplate="%{text:.3f}")
        fig.add_hline(y=0.5, line_dash="dot", annotation_text="chance")
        fig.update_yaxes(range=[0.4, 1.0])
        show(fig, 380)
        st.caption("Test rows were copied and part of their cells replaced by values from "
                   "other rows (%s of cells actually changed). Score = mean distance to the "
                   "%s nearest training rows. ROC-AUC says how well the score separates real "
                   "from corrupted rows." % (pct(o["share_of_cells_actually_changed"]), o["k"]))
    with c3:
        n = u["neighbour_agreement"]
        t = pd.DataFrame({"": ["GAE", "raw ordinal", "chance"],
                          "share": [n["mean_same_label_share"],
                                    n["raw_ordinal_mean_same_label_share"], n["chance_level"]]})
        fig = px.bar(t, x="", y="share", text="share", title="Similar rows share the label")
        fig.update_traces(texttemplate="%{text:.3f}")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        show(fig, 380)
        st.caption("Share of the %s nearest training rows that have the same income label as "
                   "the test row. For >50K test rows it is %s, for <=50K rows %s: the smaller "
                   "class is harder." % (n["k"], pct(n["share_positive_queries"]),
                                         pct(n["share_negative_queries"])))

    st.subheader("Main model in detail (rebuilt from artifacts/test_rows.csv)")
    y = (test_rows["true_label"] == POS).to_numpy().astype(int)
    p = test_rows["prob_gt_50k"].to_numpy(dtype=float)
    c1, c2 = st.columns(2)
    with c1:
        thr = st.slider("Decision threshold on P(>50K)", 0.05, 0.95, 0.5, 0.01, key="r_thr")
        pred = (p >= thr).astype(int)
        tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
        fn, tn = int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum())
        fig = go.Figure(go.Heatmap(
            z=[[tn, fp], [fn, tp]], x=["predicted " + NEG, "predicted " + POS],
            y=["actual " + NEG, "actual " + POS], text=[[tn, fp], [fn, tp]],
            texttemplate="%{text}", colorscale="Blues", showscale=False))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title="Confusion matrix at threshold %.2f" % thr)
        show(fig, 360)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        k = st.columns(3)
        k[0].metric("Accuracy", pct((tp + tn) / len(y), 2))
        k[1].metric("Precision (>50K)", pct(prec))
        k[2].metric("Recall (>50K)", pct(rec))
        st.caption("Move the threshold: lower catches more high earners (recall up) but "
                   "raises false alarms (precision down). The reported metrics use 0.50.")
    with c2:
        fpr, tpr, th = roc_curve(y, p)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="GAE + k-NN",
                                 line=dict(color=COLORS[POS], width=3)))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="chance",
                                 line=dict(dash="dot", color="grey")))
        fig.add_trace(go.Scatter(x=[fp / max(fp + tn, 1)], y=[tp / max(tp + fn, 1)],
                                 mode="markers", name="current threshold",
                                 marker=dict(size=13, color="black")))
        fig.update_layout(title="ROC curve (area = %s in metrics.json)" % f4(mm["roc_auc"]),
                          xaxis_title="false positive rate", yaxis_title="true positive rate")
        show(fig, 470)
        st.caption("How to read: the curve shows every possible threshold. The closer it "
                   "hugs the top-left corner, the better; the diagonal is random guessing.")

    c1, c2 = st.columns(2)
    with c1:
        tr = test_rows.copy()
        tr["predicted probability"] = tr["prob_gt_50k"].round(2)
        cal = tr.groupby("predicted probability").agg(
            observed=("true_label", lambda s: (s == POS).mean()), rows=("row_id", "size"))
        cal = cal.reset_index()
        fig = px.scatter(cal, x="predicted probability", y="observed", size="rows",
                         title="Calibration: predicted probability vs what really happened")
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                                 line=dict(dash="dot", color="grey")))
        fig.update_yaxes(title="observed share of >50K")
        show(fig, 420)
        st.caption("Each dot groups test rows with the same predicted probability (the vote "
                   "share among k neighbours). Dots on the dotted line mean the probability "
                   "can be taken at face value. Bigger dot = more rows.")
    with c2:
        _, xy = load_projection()
        opts = ["true_label", "predicted_label", "correct", "prob_gt_50k", "outlier_score"] + \
               list(m["metrics"]["config"]["feature_cols"])
        colour = st.selectbox("Colour the map by", opts, key="r_col")
        n_pts = min(len(test_rows), 4000)
        idx = np.random.default_rng(0).choice(len(test_rows), n_pts, replace=False)
        mp = test_rows.iloc[idx].copy()
        mp["pc1"], mp["pc2"] = xy[idx, 0], xy[idx, 1]
        fig = px.scatter(mp, x="pc1", y="pc2", color=colour, opacity=0.6,
                         color_discrete_map=COLORS,
                         hover_data=["age", "education", "occupation", "marital-status",
                                     "true_label", "prob_gt_50k"],
                         title="Map of %d test-row embeddings (2-D PCA view)" % n_pts)
        fig.update_traces(marker=dict(size=5))
        show(fig, 470)
        st.caption("Each dot is one test person placed by their embedding. Colour by "
                   "true_label: the model never saw this label, yet the colours separate. "
                   "Try marital-status or education to see what organises the space.")

    st.subheader("Error analysis on artifacts/sample_predictions.csv")
    sp = sample.copy()
    sp["correct"] = sp["correct"].astype(str).str.lower().eq("true")
    sp["error type"] = np.where(sp["correct"], "correct",
                                np.where(sp["predicted_label"] == POS,
                                         "false positive (said >50K)",
                                         "false negative (missed >50K)"))
    k = st.columns(4)
    k[0].metric("Rows in the sample", "%d" % len(sp))
    k[1].metric("Accuracy in the sample", pct(sp["correct"].mean()))
    k[2].metric("False positives", "%d" % (sp["error type"].str.startswith("false pos")).sum())
    k[3].metric("False negatives", "%d" % (sp["error type"].str.startswith("false neg")).sum())
    st.caption("This is a random sample of the test predictions saved for inspection, so its "
               "accuracy differs slightly from the full-test figure above.")
    c1, c2 = st.columns(2)
    with c1:
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
        sp["confidence band"] = pd.cut(sp["confidence"], bins, right=False,
                                       labels=["50-60%", "60-70%", "70-80%", "80-90%", "90-100%"])
        gb = sp.groupby("confidence band", observed=True).agg(
            accuracy=("correct", "mean"), rows=("correct", "size")).reset_index()
        fig = px.bar(gb, x="confidence band", y="accuracy", text="rows",
                     title="Is the model right when it is confident?")
        fig.update_traces(texttemplate="%{text} rows")
        fig.update_yaxes(range=[0, 1], tickformat=".0%")
        show(fig, 400)
        st.caption("Confidence = share of neighbours that agree with the prediction. Bars "
                   "should rise from left to right: more agreement, fewer mistakes.")
    with c2:
        feat = list(m["metrics"]["config"]["feature_cols"])
        col = st.selectbox("Error rate by", feat, index=feat.index("marital-status")
                           if "marital-status" in feat else 0, key="r_err")
        tmp = sp[[col, "correct"]].copy()
        if pd.api.types.is_numeric_dtype(tmp[col]):
            try:
                tmp[col] = pd.qcut(tmp[col], 5, duplicates="drop").astype(str)
            except ValueError:
                tmp[col] = tmp[col].astype(str)
        gb = tmp.groupby(col).agg(error_rate=("correct", lambda s: 1 - s.mean()),
                                  rows=("correct", "size")).reset_index()
        gb = gb[gb["rows"] >= 5].sort_values("error_rate")
        fig = px.bar(gb, y=col, x="error_rate", orientation="h", text="rows",
                     title="Share of wrong predictions by %s (groups with 5+ rows)" % col)
        fig.update_traces(texttemplate="%{text} rows")
        fig.update_xaxes(tickformat=".0%", title="error rate")
        show(fig, 400)
        st.caption("Longer bar = the model is wrong more often for that group. Groups where "
                   "both income classes are common (for example married people) are harder.")
    kind = st.radio("Show", ["most confident mistakes", "false positives", "false negatives",
                             "least confident predictions", "all rows"],
                    horizontal=True, key="r_kind")
    if kind == "most confident mistakes":
        v = sp[~sp["correct"]].sort_values("confidence", ascending=False)
    elif kind == "false positives":
        v = sp[sp["error type"].str.startswith("false pos")].sort_values("confidence", ascending=False)
    elif kind == "false negatives":
        v = sp[sp["error type"].str.startswith("false neg")].sort_values("confidence", ascending=False)
    elif kind == "least confident predictions":
        v = sp.sort_values("confidence")
    else:
        v = sp
    st.dataframe(v.drop(columns=["confidence band"]).head(100), hide_index=True, width="stretch")
    st.caption("Confident mistakes are the interesting ones: the person looks like their "
               "neighbours in every recorded column but has a different income. The table "
               "simply does not contain what distinguishes them.")

    st.subheader("Static plots saved by train.py")
    titles = {"01": "Data overview", "02": "Auto-encoder loss curves", "03": "Entity map",
              "04": "Row embedding map", "05": "Method comparison", "06": "ROC curves "
              "(includes baseline and XGBoost)", "07": "Dimension sweep",
              "08": "Age regression", "09": "Outlier scores", "10": "Target ablation and seeds",
              "11": "Confusion matrix"}
    pngs = sorted(PLOTS.glob("*.png")) if PLOTS.exists() else []
    if not pngs:
        st.info("No PNG plots found in artifacts/plots. Run `python train.py` to create them.")
    cols = st.columns(2)
    for i, p_ in enumerate(pngs):
        with cols[i % 2]:
            st.image(str(p_), caption=titles.get(p_.name[:2], p_.stem), width="stretch")


def _load_sample_row(row, levels):
    for col, val in row.items():
        key = "in_" + col
        if col in levels:
            st.session_state[key] = val if val in levels[col] else levels[col][0]
        elif col in ("age", "hours-per-week", "capital-gain", "capital-loss"):
            st.session_state[key] = int(round(float(val)))
    st.session_state["loaded_row"] = row


def tab_try(m, test_rows, sample):
    mm = m["metrics"]
    feat = list(mm["config"]["feature_cols"])
    try:
        model = load_model()
    except Exception as exc:
        st.error("The trained model could not be loaded (%s). Run `python train.py` to "
                 "rebuild artifacts/model.joblib." % type(exc).__name__)
        return
    tok = model.tokeniser
    levels = {c: sorted(tok.cat_index[c].keys()) for c in tok.cat_cols}
    num_all = list(tok.num_cols) + list(tok.zero_cols)

    st.subheader("Try the trained model")
    st.markdown(
        "Describe a person, and the model (1) turns the row into its %d-number embedding, "
        "(2) finds the **%d most similar people in the training set** and (3) lets them vote. "
        "The probability is simply the share of those neighbours who earn more than 50K."
        % (model.dim, model.k))
    mode = st.radio("Input", ["One person (form)", "Many rows (CSV upload)"], horizontal=True,
                    key="t_mode")

    if mode == "Many rows (CSV upload)":
        st.markdown("Upload a CSV with these columns: `%s`. Unknown category values and empty "
                    "cells are allowed; they are skipped when the row is embedded." % "`, `".join(feat))
        st.download_button("Download an example CSV (20 test rows)",
                           sample[feat].head(20).to_csv(index=False).encode("utf-8"),
                           "example_rows.csv", "text/csv")
        up = st.file_uploader("CSV file", type=["csv"], key="t_up")
        if up is None:
            return
        try:
            df = pd.read_csv(up, keep_default_na=False, na_values=[""])
        except Exception as exc:
            st.error("Could not read the file as CSV (%s)." % type(exc).__name__)
            return
        missing = [c for c in feat if c not in df.columns]
        if missing:
            st.error("Missing columns: %s" % ", ".join(missing))
            return
        if len(df) == 0:
            st.warning("The file has no rows.")
            return
        if len(df) > 5000:
            st.warning("Only the first 5000 rows are scored.")
            df = df.head(5000)
        X = df[feat].copy()
        for c in levels:
            X[c] = X[c].fillna("Missing").astype(str).str.strip()
        try:
            _, info = tok.transform(X, return_info=True)
            proba = model.predict_proba(X)
            out_s = model.outlier_score(X)
        except Exception as exc:
            st.error("Scoring failed (%s). Check that numeric columns contain numbers."
                     % type(exc).__name__)
            return
        res = df.copy()
        res["prob_gt_50k"] = np.round(proba, 4)
        res["predicted_label"] = np.where(proba >= 0.5, POS, NEG)
        res["confidence"] = np.round(np.maximum(proba, 1 - proba), 4)
        res["outlier_score"] = np.round(out_s, 5)
        k = st.columns(4)
        k[0].metric("Rows scored", "%d" % len(res))
        k[1].metric("Predicted >50K", pct((proba >= 0.5).mean()))
        k[2].metric("Mean confidence", pct(res["confidence"].mean()))
        k[3].metric("Cells not recognised", "%d of %d" % (info["unseen_cells"], info["n_cells"]))
        st.dataframe(res, hide_index=True, width="stretch")
        st.download_button("Download predictions as CSV", res.to_csv(index=False).encode("utf-8"),
                           "predictions.csv", "text/csv")
        fig = px.histogram(res, x="prob_gt_50k", nbins=26, title="Predicted P(>50K) in the upload")
        show(fig, 320)
        return

    # ---------- single row ----------
    with st.expander("Start from a real test person (optional)", expanded=True):
        c1, c2, c3 = st.columns([1, 3, 1])
        which = c1.radio("Pick from", ["any", "model was wrong", "model was right"], key="t_which")
        ok = sample["correct"].astype(str).str.lower().eq("true")
        pool = sample if which == "any" else sample[~ok] if which == "model was wrong" else sample[ok]
        if len(pool) == 0:
            pool = sample
        recs = pool.to_dict("records")
        i = c2.selectbox(
            "Test row", range(len(recs)), key="t_row_" + which,
            format_func=lambda j: "row %s | %s, %s, %s, %s | true %s" % (
                recs[j]["row_id"], int(recs[j]["age"]), recs[j]["sex"], recs[j]["education"],
                recs[j]["occupation"], recs[j]["true_label"]))
        c3.write("")
        c3.button("Load into the form", on_click=_load_sample_row,
                  args=({**{c: recs[i][c] for c in feat}, "row_id": recs[i]["row_id"],
                         "true_label": recs[i]["true_label"]}, levels), key="t_load")

    if "in_age" not in st.session_state:
        first = sample.iloc[0].to_dict()
        _load_sample_row({**{c: first[c] for c in feat}, "row_id": first["row_id"],
                          "true_label": first["true_label"]}, levels)

    lim = {c: (int(test_rows[c].min()), int(test_rows[c].max())) for c in num_all
           if c in test_rows.columns}
    c = st.columns(4)
    c[0].slider("Age", 17, 90, key="in_age")
    c[1].slider("Hours worked per week", 1, 99, key="in_hours-per-week")
    c[2].number_input("Capital gain (USD, 0 for most people)", 0, 99999, step=500,
                      key="in_capital-gain")
    c[3].number_input("Capital loss (USD, 0 for most people)", 0, 5000, step=100,
                      key="in_capital-loss")
    cat_labels = {"workclass": "Work class", "education": "Education",
                  "marital-status": "Marital status", "occupation": "Occupation",
                  "relationship": "Relationship in household", "race": "Race", "sex": "Sex",
                  "native-country": "Native country"}
    c = st.columns(4)
    for j, col in enumerate(tok.cat_cols):
        if st.session_state.get("in_" + col) not in levels[col]:
            st.session_state["in_" + col] = levels[col][0]
        c[j % 4].selectbox(cat_labels.get(col, col), levels[col], key="in_" + col)

    row = {col: (st.session_state["in_" + col] if col in levels
                 else float(st.session_state["in_" + col])) for col in feat}
    X = pd.DataFrame([row] * (1 + len(feat)))[feat]
    for j, col in enumerate(feat):          # row j+1 = the same person with one column hidden
        X.loc[j + 1, col] = "__hidden__" if col in levels else np.nan
    try:
        proba = model.predict_proba(X)
        dist, idx = model.neighbours(X.iloc[[0]])
        out_score = float(model.outlier_score(X.iloc[[0]])[0])
        E_q = model.transform(X.iloc[[0]])
    except Exception as exc:
        st.error("Prediction failed (%s)." % type(exc).__name__)
        return
    p = float(proba[0])
    label = POS if p >= 0.5 else NEG
    conf = max(p, 1 - p)
    votes = model.y_ref[idx[0]]

    st.markdown("---")
    k = st.columns(4)
    k[0].metric("Prediction", "income " + label)
    k[1].metric("P(income > 50K)", pct(p))
    k[2].metric("Confidence", pct(conf),
                help="Share of the neighbours that agree with the prediction.")
    out_pct = float((test_rows["outlier_score"] < out_score).mean())
    k[3].metric("Unusualness", "%s of test rows are more typical" % pct(out_pct, 0),
                help="Outlier score = mean distance to the 10 nearest training rows (%.4f)."
                     % out_score)

    near = test_rows[(test_rows["prob_gt_50k"] - p).abs() < 0.02]
    msg = ("**Why this answer:** %d of the %d most similar training people earn more than "
           "50K, so P(>50K) = %s." % (int(votes.sum()), len(votes), pct(p)))
    if len(near) >= 20:
        obs = (near["true_label"] == POS).mean()
        msg += ("  \n**How far to trust it:** on the %d held-out test rows that got the same "
                "probability, %s really earned more than 50K, so the prediction `%s` was right "
                "for %s of them." % (len(near), pct(obs), label,
                                     pct(obs if label == POS else 1 - obs)))
    if conf < 0.6:
        msg += "  \nThe neighbours are almost evenly split: treat this as a toss-up."
    if out_pct > 0.95:
        msg += ("  \n**Warning:** this combination of values is unusual compared with the "
                "training data, so the neighbours are far away and the vote is less reliable.")
    st.info(msg)
    loaded = st.session_state.get("loaded_row")
    if loaded and all(str(loaded.get(c_)) == str(row[c_]) or
                      (c_ not in levels and float(loaded.get(c_)) == row[c_]) for c_ in feat):
        truth = loaded.get("true_label")
        st.markdown("This is test row **%s**; its true label is **%s** - the model is **%s**."
                    % (loaded.get("row_id"), truth, "right" if truth == label else "wrong"))

    c1, c2 = st.columns(2)
    with c1:
        contrib = pd.DataFrame({"column": feat, "value": [str(row[c_]) for c_ in feat],
                                "effect": p - proba[1:]}).sort_values("effect")
        contrib["direction"] = np.where(contrib["effect"] >= 0, "pushes towards >50K",
                                        "pushes towards <=50K")
        fig = px.bar(contrib, x="effect", y="column", orientation="h", color="direction",
                     hover_data=["value"],
                     color_discrete_map={"pushes towards >50K": COLORS[POS],
                                         "pushes towards <=50K": COLORS[NEG]},
                     title="Which inputs matter for this person")
        fig.update_xaxes(title="change in P(>50K) when the column is included", tickformat="+.0%")
        fig.update_layout(legend=dict(orientation="h", y=-0.25, title=None))
        show(fig, 430)
        st.caption("Each bar: the prediction with all inputs minus the prediction when that "
                   "one column is hidden from the embedding. A simple what-if test for this "
                   "row only, not a general importance ranking; steps are coarse because the "
                   "probability moves in units of one neighbour vote.")
    with c2:
        _proj, xy = load_projection()
        q = _proj.transform(E_q)
        n_pts = min(len(test_rows), 3000)
        sel = np.random.default_rng(1).choice(len(test_rows), n_pts, replace=False)
        bg = pd.DataFrame({"pc1": xy[sel, 0], "pc2": xy[sel, 1],
                           "income": test_rows["true_label"].to_numpy()[sel]})
        fig = px.scatter(bg, x="pc1", y="pc2", color="income", opacity=0.35,
                         color_discrete_map=COLORS, title="Where this person lands on the map")
        fig.update_traces(marker=dict(size=5))
        fig.add_trace(go.Scatter(x=q[:, 0], y=q[:, 1], mode="markers", name="your input",
                                 marker=dict(size=18, color="black", symbol="star",
                                             line=dict(color="white", width=1.5))))
        show(fig, 430)
        st.caption("Dots are test people coloured by true income; the star is your input. A "
                   "star inside an orange area means most similar people are high earners.")

    st.markdown("#### The %d neighbours that voted" % len(votes))
    nb = pd.DataFrame({"rank": np.arange(1, len(votes) + 1),
                       "distance": np.round(dist[0], 4),
                       "income": np.where(votes == 1, POS, NEG)})
    full = load_full_table()
    try:
        ids = load_npz("embeddings.npz")["train_row_id"][idx[0]]
        nb.insert(1, "row_id", ids)
        if full is not None:
            nb = pd.concat([nb, full.loc[ids, feat].reset_index(drop=True)], axis=1)
    except Exception:
        full = None
    st.dataframe(nb, hide_index=True, width="stretch", height=320)
    if full is None:
        st.caption("Only labels and distances are shown because the raw table is not cached "
                   "in data/ (it is downloaded by `python train.py`).")
    else:
        st.caption("Distance 0 means a training person with an identical embedding. Compare "
                   "the columns with your input to see what 'similar' means to the model.")
    with st.expander("The embedding vector of this row (%d numbers)" % model.dim):
        st.bar_chart(pd.DataFrame({"value": E_q[0]},
                                  index=["z%02d" % j for j in range(E_q.shape[1])]))
        st.caption("These numbers are the row's coordinates. They have no individual "
                   "meaning; only distances between rows matter.")
    st.caption("Note: the model learned from 1994 US census data. It reflects that data, "
               "including its social biases, and must not be used to judge real people.")


def tab_research():
    names = [("Literature survey", "01_literature_survey.md"),
             ("Problem definition", "02_problem_definition.md"),
             ("Solution strategy", "03_solution_strategy.md"),
             ("Dataset", "04_dataset.md"), ("Model design", "05_model_design.md"),
             ("Results", "06_results.md")]
    tabs = st.tabs([n for n, _ in names] + ["Paper"])
    for t, (_, fname) in zip(tabs, names):
        with t:
            text = doc(fname)
            if text:
                show_markdown(text, DOCS)
            else:
                st.info("docs/%s was not found." % fname)
    with tabs[-1]:
        papers = find_paper()
        if not papers:
            st.info("The paper has not been written yet (no '*paper*.md' file in the project "
                    "or docs/ folder). It is added in the final project phase.")
        for p_ in papers:
            st.caption("Source: %s" % p_.relative_to(ROOT).as_posix())
            show_markdown(read_doc(str(p_)) or "", p_.parent)
        st.markdown("Source paper for the method: "
                    "[arXiv:2507.05904](http://arxiv.org/abs/2507.05904v1)")


# ------------------------------------------------------------------ page
def main():
    st.title("Universal Embeddings of Tabular Data")
    st.caption("One label-free vector per table row, reused for classification, regression, "
               "outlier detection and similarity search.")
    missing = [f for f in REQUIRED if not (ART / f).exists()]
    data = None
    if not missing:
        try:
            m = load_json("metrics.json")
            data = {"m": m, "test_rows": load_csv("test_rows.csv"),
                    "sample": load_csv("sample_predictions.csv"),
                    "entities": load_csv("entities.csv"), "graph": load_npz("graph.npz"),
                    "comp": comparison_table(m)}
        except Exception as exc:
            st.error("The artifacts could not be read (%s: %s)." % (type(exc).__name__, exc))
    if data is None:
        st.error("Trained artifacts are missing or unreadable%s.\n\nRun **`python train.py`** "
                 "in the project folder (about 3 minutes on a laptop CPU), then reload this "
                 "page." % ((": " + ", ".join(missing)) if missing else ""))
        st.subheader("Research documents")
        tab_research()
        return

    tabs = st.tabs(["Overview", "Data", "Model", "Results", "Try it", "Research"])
    calls = [lambda: tab_overview(data["m"], data["comp"]),
             lambda: tab_data(data["m"], data["test_rows"]),
             lambda: tab_model(data["m"], data["entities"], data["graph"]),
             lambda: tab_results(data["m"], data["comp"], data["test_rows"], data["sample"]),
             lambda: tab_try(data["m"], data["test_rows"], data["sample"]),
             tab_research]
    for t, fn in zip(tabs, calls):
        with t:
            try:
                fn()
            except Exception as exc:  # one broken tab must not take the page down
                st.error("This tab could not be shown (%s: %s). Re-running `python train.py` "
                         "usually fixes stale artifacts." % (type(exc).__name__, exc))


main()
