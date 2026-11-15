# PLAN - Universal Embeddings of Tabular Data (graph auto-encoder replication on Adult)

Source paper: http://arxiv.org/abs/2507.05904v1. Design documents are in `docs/01` - `docs/05`.

## Dataset

Adult Census Income, OpenML id 1590 (48,842 rows, CC BY 4.0 per UCI).

```python
fetch_openml(data_id=1590, as_frame=True, parser="auto", data_home=str(ROOT / "data"))
```

Drop `fnlwgt`, `education-num`. Target `class` (>50K) is never used for the embedding.
Split 70/15/15 stratified, seed 42.

## Models

- **Main**: entity graph (categories + soft quantile bins) -> reduced co-occurrence graph ->
  one-layer GAE (Z = P W, decoder Z Z^T, L2 loss, Adam lr 0.01) -> row embedding = weighted mean
  of entity vectors -> k-NN (k and neighbour weighting chosen on validation). Default d = 16.
- **Baselines (no labels)**: raw ordinal k-NN (must beat), raw one-hot k-NN, PCA-d, spectral-d,
  skip-gram-d (EmbDI-style approximation).
- **Supervised references**: logistic regression, XGBoost. Trivial: majority class / mean.

## Metric

Primary: test ROC-AUC for income, GAE d = 16, k chosen on validation.
Must beat: raw ordinal k-NN and majority class. Others are reported as they come out.

## Files to create

```
README.md                 overview, results table filled from metrics.json, how to run
requirements.txt          allowed libraries only
train.py                  one command: runs everything, writes artifacts/
app.py                    Streamlit dashboard
src/config.py             paths (from __file__), seeds, hyper-parameters
src/data.py               load, clean, split
src/tokeniser.py          TableTokeniser: entities, soft bins, row-entity matrix B
src/graph.py              reduced graph A, targets T / S, propagation matrix P
src/gae.py                EntityGAE and training loop
src/embed.py              row embeddings, save/load of the fitted pipeline
src/baselines.py          raw k-NN, PCA, spectral, skip-gram, LogReg, XGBoost
src/evaluate.py           k-NN heads, classification / regression / outlier / neighbour metrics
src/plots.py              matplotlib figures
tests/test_pipeline.py    small sanity checks runnable with `python tests/test_pipeline.py`
src/model.py              one import point: UniversalEmbedder (main) and RawKNNBaseline
selfcheck.py              offline smoke test on a synthetic table, prints ok
artifacts/                metrics.json, model.joblib, embeddings.npz, graph.npz, entities.csv,
                          test_rows.csv, sample_predictions.csv, training_history.json, plots/
```

## train.py stages

1. Load, clean, split; dataset statistics.
2. Fit tokeniser, build graph (log node and edge counts).
3. Train GAE d = 16 for both targets (T, S); pick on validation; 3 seeds.
4. Baselines at d = 16 and raw; supervised references.
5. Dimension sweep d in {2, 4, 8, 16, 32, 64}.
6. Universality: age regression (graph without age), outlier detection, neighbour agreement.
7. Save metrics.json, artifacts, figures; print an ASCII summary and total time (< 15 min).

## Plots (artifacts/plots)

1. Class balance and key column distributions.
2. GAE training loss curve (per seed).
3. Entity embedding map (2-D PCA of Z, coloured by column; age bins joined in order).
4. Row embedding map (2-D projection of test rows, coloured by income).
5. Method comparison bar chart (test ROC-AUC, d = 16, plus references).
6. ROC curves: GAE vs simple baseline vs XGBoost.
7. ROC-AUC vs embedding dimension for all embedding methods.
8. Age regression: predicted vs true.
9. Outlier score histograms: original vs corrupted rows.
10. Target ablation and seed stability.
11. Confusion matrix of the main model.

## Dashboard pages (Streamlit, reads artifacts only)

1. **Overview** - idea, pipeline diagram, headline numbers from metrics.json.
2. **Data explorer** - table sample, column distributions, split and class balance.
3. **Entity graph** - entity list, strongest co-occurrence edges, interactive entity embedding
   map, nearest entities of a chosen entity.
4. **Row embedding space** - interactive scatter coloured by any column or the label.
5. **Results** - comparison table, ROC curves, dimension sweep, ablation, seeds.
6. **Universality** - age regression, outlier detection, neighbour agreement.
7. **Try it** - enter a new person, embed without retraining, see k-NN income estimate,
   outlier score and most similar training rows.
8. **About** - limits, ethics note, links to docs and paper.

## Rules to keep

Relative paths from `Path(__file__)`; ASCII console output; only allowed libraries; only
`python ...` commands; no number anywhere that is not in `artifacts/metrics.json`.
