# 03 - Solution Strategy

## Approach in one paragraph

Turn the table into a small weighted graph whose nodes are **entities** (each category value and
each numerical bin), where two entities are connected when they occur in the same row. Train a
one-layer **graph auto-encoder** (GAE) on that graph to get one vector per entity. A row's
embedding is the weighted mean of the vectors of the entities it contains. Labels are never
used. Downstream tasks are solved with distance-weighted k-NN in this space.

## Why this approach

- It is the method of the source paper, so the project is a faithful, checkable replication on a
  new dataset.
- **Cost depends on the number of entities, not rows.** Adult has about 100 category values and
  about 60-80 numerical bins, so the graph has fewer than 250 nodes. Full-batch training on a
  250 x 250 matrix takes seconds on CPU, leaving time for sweeps, seeds and baselines.
- **Rows are embedded inductively** by a sparse matrix product, so validation, test and
  user-entered rows need no retraining.
- **Interpretable.** Entity vectors can be plotted; one can check that neighbouring age bins lie
  next to each other and that "Husband" lies close to "Married-civ-spouse".

## Alternatives considered

| Alternative | Why not chosen as the main model | Role in the project |
|-------------|----------------------------------|---------------------|
| Random-walk + word2vec (EmbDI) | Needs many sampled walks; stochastic; the paper already claims to beat it | Baseline, re-implemented as a small skip-gram model in PyTorch |
| PCA on one-hot + scaled numbers | No learning of entity relations, but very strong and cheap | Baseline at equal dimension |
| Spectral factorisation (truncated SVD of the transition matrix) | Closed form of "low-rank reconstruction of T"; tells us what gradient training of the GAE adds | Baseline / ablation |
| Contrastive row encoders (SCARF) or masked reconstruction (VIME, SubTab) | Row-level only, need augmentation design and longer training | Discussed in the survey, not built |
| TabPFN, TabTransformer | Supervised and/or not installed | Not built |
| Supervised XGBoost / logistic regression | Uses labels, so not universal | Upper reference |

## Pipeline step by step

1. **Load** Adult from OpenML (`fetch_openml(data_id=1590)`), cached under `data/`.
2. **Clean**: drop `fnlwgt` (a sampling weight) and `education-num` (a 1:1 copy of `education`);
   strip strings; map missing categories to an explicit `Missing` entity; binarise the target.
3. **Split** 70 / 15 / 15, stratified on the target, fixed seed. Everything fitted below
   (bins, graph, embeddings, scalers, PCA) uses the **training rows only**.
4. **Tokenise rows into entities** (row-entity incidence matrix B, sparse, shape rows x entities):
   - categorical cell -> weight 1 on node `column=value`;
   - numerical cell -> weights (1 - t, t) on the two neighbouring bin-centre nodes, where t is
     the linear position between the centres; values beyond the outer centres go fully to the
     outer bin. Bins are quantile-based with duplicates merged; zero-inflated columns
     (`capital-gain`, `capital-loss`) get a dedicated zero bin and quantile bins over the
     non-zero values.
5. **Reduce the graph**: A = B^T B with the diagonal set to zero (co-occurrence weight = sum
   over rows of the product of the two cell weights). T = row-normalised A.
6. **Train the GAE**: Z = norm(A + I) W (one GCN layer with identity node features, W is the
   only parameter, n_entities x d). Decoder: Z Z^T. Loss: mean squared error to the scaled target
   matrix (L1 mix alpha = 0 as in the paper). Adam, lr 0.01, gradient-norm clipping, early stop on
   relative loss improvement.
7. **Embed rows**: E = rownorm(B) Z for train, validation and test. Entities never seen in
   training are skipped; a row with no known entity gets the zero vector (counted and reported).
8. **Evaluate** with distance-weighted k-NN (Euclidean); k picked on validation from a small
   grid; final numbers on test only.
9. **Baselines** on the same split, same k-selection rule.
10. **Extra studies**: dimension sweep, seeds, target-normalisation ablation (row-stochastic vs
    symmetric), age regression (graph rebuilt without the age column), outlier detection on
    corrupted rows, neighbour label agreement.
11. **Persist** `artifacts/metrics.json`, embeddings, entity table, model weights and plots.
12. **Dashboard** reads only the artifacts; it retrains nothing but can embed a new row live.

## Risks and how they are handled

| Risk | Handling |
|------|----------|
| OpenML download fails or is slow | Cache in `data/` after the first load; clear ASCII error message telling the user to retry; no manual-download dataset is used |
| Label leakage into a "task-independent" embedding | Target column is excluded from the graph; for age regression the age column is removed and the graph is rebuilt; bins/graph fitted on training rows only |
| Symmetric decoder cannot fit an asymmetric transition matrix | Ablation: row-stochastic target (paper) vs symmetric normalisation D^-1/2 A D^-1/2; the default is chosen on validation AUC and the choice is reported |
| Target values are tiny (rows sum to 1), so gradients vanish | Target is multiplied by the number of entities so the mean entry is 1; documented |
| Loss spikes (seen in the paper) | Gradient-norm clipping and keeping the best-loss weights |
| GAE might not beat PCA | Reported honestly; PCA is a comparison, not a rigged baseline |
| Skip-gram baseline is weaker than real EmbDI because of the small walk budget | Named "EmbDI-style (approximation)", walk budget stated, no claim about the original tool |
| Zero-inflated numerical columns break quantile bins | Dedicated zero bin, duplicate edges merged |
| k-NN on 34K x 100-dim one-hot is slow | Brute-force BLAS k-NN is fine at this size; time is logged per stage and the sweep is trimmed first if over budget |
| Run time over 15 minutes | Entity graph is tiny; epochs capped; seeds limited to 3; timings written to metrics.json |
| Non-ASCII console crash on Windows | All prints are ASCII; category names are sanitised before printing |
| Synthetic outliers are not real anomalies | Stated as a limitation; the corruption recipe is documented and seeded |
