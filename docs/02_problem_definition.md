# 02 - Problem Definition

## Problem statement

Given one table with mixed categorical and numerical columns, learn - **without using any target
column** - a fixed-length vector for every row, such that simple distance computations in that
vector space are enough to solve several different downstream tasks that were not known when the
embedding was trained.

Concretely: implement the graph auto-encoder framework of "Universal Embeddings of Tabular Data"
(arXiv:2507.05904) from scratch, apply it to the Adult Census Income table, and measure how good
the frozen embeddings are for classification, regression, outlier detection and similarity
search, compared with other task-independent embeddings and with supervised references.

## Motivation

- Company databases are filled long before anyone knows which questions will be asked. Training
  one supervised model per question is costly; one reusable embedding per table is cheap.
- A vector per row makes a table usable by standard vector tools: nearest-neighbour search,
  clustering, anomaly scoring.
- The source paper tested the idea on two datasets with one task each and against one baseline.
  It is not yet clear (a) whether it holds on a larger, noisier table, and (b) whether the graph
  auto-encoder beats much simpler embeddings such as PCA.

## Objectives (measurable)

1. **Implement the pipeline**: table -> weighted entity graph (soft-binned numbers) -> reduced
   entity graph -> one-layer graph auto-encoder -> row embeddings by weighted mean. Unit checks:
   every row's entity weights sum to the number of used columns; the transition matrix rows sum
   to 1.
2. **Primary task - income classification.** Report test ROC-AUC of a distance-weighted k-NN on
   16-dimensional row embeddings. Also report accuracy, F1 and average precision.
3. **Compare at equal dimension** with three other task-independent embeddings: PCA of one-hot
   features, spectral (SVD) factorisation of the same transition matrix, and a random-walk
   skip-gram model (EmbDI-style approximation).
4. **Compare with references**: k-NN on raw features (ordinal-encoded and one-hot), logistic
   regression and XGBoost trained with labels.
5. **Dimension sweep**: test ROC-AUC for d in {2, 4, 8, 16, 32, 64} for all embedding methods.
6. **Universality checks** with the same method and no task-specific tuning:
   - regression: predict `age` from an embedding built without the age column (MAE, R^2);
   - outlier detection: separate real test rows from synthetically corrupted rows using the mean
     k-NN distance (ROC-AUC);
   - neighbour quality: share of the 10 nearest training rows with the same income label.
7. **Stability**: mean and standard deviation of the primary metric over 3 training seeds.
8. **Budget**: the whole run (`python train.py`) finishes in under 15 minutes on a CPU laptop.
9. **Dashboard**: a Streamlit app that shows data, graph, embeddings, results and lets the user
   embed a new row and find its neighbours.

## Scope

In scope: a single table; categorical and numerical columns; missing values; rows unseen during
training; CPU-only training; honest comparison with baselines.

## Non-goals

- Multi-table / relational joins (left as future work in the paper too).
- Text, image or time-series columns.
- Beating supervised gradient boosting. A universal embedding that does not see the label is
  expected to be somewhat worse; the size of that gap is a result, not a failure.
- Reproducing the paper's Rossmann and Titanic numbers (different dataset here).
- Hyper-parameter search beyond a small validation choice of k for k-NN.

## Success criteria

- **Primary metric**: ROC-AUC on the held-out test split for predicting `income > 50K` with a
  distance-weighted k-NN on the 16-dimensional GAE row embeddings.
- **Simple baseline that must be beaten**: distance-weighted k-NN applied directly to the
  ordinal-encoded, standardised raw columns (the "no embedding" shortcut most people try first),
  with k chosen on the same validation split. The majority-class predictor (ROC-AUC 0.5) must
  also be beaten.
- **Stronger comparisons, reported whatever the outcome**:
  - GAE vs PCA / spectral / skip-gram embeddings at d = 16 and across the dimension sweep;
  - gap to supervised XGBoost on raw features.
- **Secondary**: age regression beats the predict-the-mean baseline (R^2 > 0); outlier ROC-AUC is
  above 0.5.
- **Engineering**: full run under 15 minutes; `artifacts/metrics.json` is the single source of
  every number shown in the README and the dashboard.

If a criterion is not met, the README will say so plainly and discuss why.
