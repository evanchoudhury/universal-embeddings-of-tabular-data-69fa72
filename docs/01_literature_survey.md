# 01 - Literature Survey

Source paper: Franz, Hoppe, Michaelis, Goebel - "Universal Embeddings of Tabular Data",
arXiv:2507.05904v1 (July 2025), accepted at the Tabular Data Analysis (TaDA) workshop at VLDB 2025.
Link: http://arxiv.org/abs/2507.05904v1

How this survey was made: the arXiv abstract page and the arXiv HTML version of the source paper
were read. For each related work the arXiv abstract page (or the NeurIPS proceedings page) was
fetched. Statements below are limited to what those pages say. Where a page did not give a
detail (for example a dataset name or a number), the table says so instead of guessing.

## 1. Related works

| # | Paper | Year | Method | Dataset | Key result | Limitation |
|---|-------|------|--------|---------|------------|------------|
| 1 | [Cappuzzo, Papotti, Thirumuruganathan - Creating Embeddings of Heterogeneous Relational Datasets for Data Integration Tasks (EmbDI)](https://arxiv.org/abs/1909.01120) | 2019 (SIGMOD 2020) | Builds a compact graph from relational tables (cell values, rows, attributes), generates "sentences" by random walks on it, and trains word-embedding models on those sentences to get local embeddings. | Evaluated on schema matching and entity resolution; the abstract page does not name the datasets. | Abstract reports meaningful results for data-integration tasks, supervised and unsupervised; no numbers on the abstract page. | Needs a very large number of sampled walks (the source paper used 1,000,000 sentences for its EmbDI baseline); sampling is stochastic; designed for data integration, not for prediction. |
| 2 | [Kipf, Welling - Variational Graph Auto-Encoders](https://arxiv.org/abs/1611.07308) | 2016 (NIPS BDL workshop) | Unsupervised graph model: a GCN encoder gives node embeddings, an inner-product decoder reconstructs the adjacency matrix. Has a variational (VGAE) and a plain (GAE) form. | Citation networks (dataset names not given on the abstract page). | Competitive link prediction on citation networks; node features improve results. No numbers on the abstract page. | Built for unweighted, undirected link prediction; says nothing about tables, numerical values or row-level embeddings. |
| 3 | [Yoon, Zhang, Jordon, van der Schaar - VIME](https://proceedings.neurips.cc/paper/2020/hash/7d97667a3e056acab9aaf653807b4a03-Abstract.html) | 2020 (NeurIPS) | Self- and semi-supervised learning for tables with two pretext tasks: estimate which cells were masked, and impute the original values of corrupted cells. | "Multiple tabular datasets" from genomics and clinical domains (names not on the abstract page). | Abstract claims it exceeds prior baselines; no numbers on the abstract page. | Learns a row encoder only (no entity embeddings); needs a neural encoder trained per table; corruption design is a tuning choice. |
| 4 | [Bahri, Jiang, Tay, Metzler - SCARF](https://arxiv.org/abs/2106.15147) | 2021 (ICLR 2022 spotlight) | Contrastive pre-training where the second view of a row is made by replacing a random subset of features with values drawn from the column marginals. | 69 classification datasets from OpenML-CC18. | Improves accuracy in fully supervised, label-noise and semi-supervised settings; outperforms auto-encoders (no single headline number on the abstract page). | Evaluated as pre-training followed by supervised fine-tuning, so the representation is not tested as a frozen, task-independent embedding with a simple k-NN. |
| 5 | [Ucar, Hajiramezanali, Edwards - SubTab](https://arxiv.org/abs/2110.04361) | 2021 (NeurIPS) | Splits the features into subsets and trains an auto-encoder to reconstruct all features from each subset (multi-view learning); aggregates subset latents at test time. | MNIST in tabular form plus three other real-world datasets (not named on the abstract page). | 98.31% accuracy on tabular MNIST. | Requires choosing the number and overlap of feature subsets; row-level only. |
| 6 | [Huang, Khetan, Cvitkovic, Karnin - TabTransformer](https://arxiv.org/abs/2012.06678) | 2020 | Transformer layers turn per-category embeddings into contextual embeddings; optional unsupervised pre-training for the semi-supervised case. | Fifteen public datasets (not named on the abstract page). | At least 1.0% mean AUC over other deep tabular models; matches tree ensembles; 2.1% average AUC lift in the semi-supervised setting. | Embeddings are learned for a supervised objective; much heavier than a one-layer graph auto-encoder. |
| 7 | [Hollmann, Mueller, Eggensperger, Hutter - TabPFN](https://arxiv.org/abs/2207.01848) | 2022 | A transformer pre-trained once on synthetic tasks that does classification by in-context learning in one forward pass, no per-dataset training. | 18 small OpenML-CC18 datasets, plus 67 small OpenML datasets for validation. | Outperforms boosted trees on that subset and is on par with AutoML systems, with up to 230x CPU speed-up. | Evaluated only up to 1,000 training points, 100 numerical features, 10 classes. The source paper could not run it on Rossmann for this reason. Supervised, so not a universal embedding. |
| 8 | [Grinsztajn, Oyallon, Varoquaux - Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815) | 2022 | Large benchmark of deep models against XGBoost and Random Forests with equal tuning budgets, plus a study of inductive biases. | 45 tabular datasets. | Tree models stay state of the art on medium-sized data (about 10K samples). | A benchmark of supervised models only; it motivates using a gradient-boosted tree as the strong supervised reference in this project. |
| 9 | [Li, Tsai, Chen, Liao - Graph Neural Networks for Tabular Data Learning: A Survey](https://arxiv.org/abs/2401.02143) | 2024 | Survey organised by (a) how a graph is built from a table and (b) how representations are learned on it. | Survey (no experiments of its own on the abstract page). | Argues that plain deep tabular models under-use correlations among instances and feature values, which graphs can capture. | Survey only; most covered methods are supervised or task-specific. |

All nine pages were reachable. The full text (PDF body) of works 1-9 was not read, so no claim
here depends on anything beyond the abstract-level pages. The exact text of Equation 1 and the
figure values of the source paper could not be extracted from the HTML version (see below).

## 2. Critical review of the source paper

### What the paper does

1. **Table to graph.** Every distinct categorical value becomes an entity node. Every numerical
   column is cut into bins with roughly equal counts (plus two overflow bins) and each bin is an
   entity node. A row is linked to the entities it contains. A numerical value is linked to its
   bin with a weight that depends on its position inside the bin, and also to the neighbouring
   bin when it is close to the border, so the order of numbers survives.
2. **Graph reduction.** Row nodes are removed. Two entities that occur in the same row are linked
   directly; the weight is the product of the two row-entity weights, summed over rows. This
   roughly halves the graph and makes its size independent of the number of rows.
3. **Entity embeddings.** The weighted adjacency is row-normalised to a transition matrix. A
   one-layer graph auto-encoder (GCN encoder, inner-product decoder) is trained to reconstruct
   it, with an L2 loss (the L1 mixing weight alpha is 0 in the experiments).
4. **Row embeddings.** A row is the weighted mean of its entity embeddings. A new row made of
   known entities is embedded the same way, with no retraining.
5. **Use.** Distance-weighted k-NN (Euclidean) in the embedding space does classification and
   regression; distances also support similarity search and outlier detection.

Reported setup: Titanic (817 train / 74 test rows, AUC) and Rossmann store sales (759,917 train /
84,475 test rows, RMSPE). Learning rate 0.01, gradient-norm clipping, stop when the relative loss
improvement over 10,000 epochs is below 0.1%. Supervised references in the paper: XGBoost AUC
0.931 on Titanic and RMSPE 0.2153 on Rossmann; TabPFN AUC 0.946 on Titanic.

### Strengths

- **Task independence is real by construction**: the target column is never used to build the
  graph or to train the auto-encoder.
- **Tiny model.** The reduced graph has one node per entity, so training cost does not grow with
  the number of rows. This is why the method fits a CPU.
- **Numbers are handled in the same graph** as categories, with soft bin membership that keeps
  order. EmbDI-style methods treat numbers as opaque tokens.
- **Deterministic input.** Unlike random-walk methods there is no sentence sampling.
- **Inductive for rows**: new rows need only an average, not a forward pass through a trained
  row encoder.

### Weaknesses

- **Thin evaluation.** Two datasets, one task each, one small test set (74 rows on Titanic). The
  authors say this themselves.
- **Numbers of the proposed method are only in figures.** The HTML text gives exact values for
  XGBoost and TabPFN but not for the proposed embeddings; the claim is "in the same range".
- **One universal baseline.** Only EmbDI is compared. Cheap alternatives such as PCA of a one-hot
  encoding or a direct spectral factorisation of the same transition matrix are not tested, so it
  is unclear how much the graph auto-encoder itself adds.
- **Outlier detection and similarity search are claimed but not measured.**
- **Mean aggregation ignores interactions** between entities of one row; the embedding of a row
  is linear in its entities.
- **Unseen entities** (a category that never occurred in training) have no embedding.
- **Convergence rule is very long** (10,000-epoch windows) and the paper notes loss spikes
  without gradient clipping.
- **Symmetric decoder, asymmetric target.** An inner product Z Z^T is symmetric while a
  row-normalised transition matrix is not. The paper does not discuss this mismatch.

### What can be reproduced on a laptop

- The complete pipeline: graph construction, reduction, one-layer GAE, row embeddings, k-NN.
  The entity graph of a mid-sized table has only a few hundred nodes, so full-batch training
  takes seconds on CPU.
- A dimension sweep like the paper's Figures 7 and 9.
- A random-walk skip-gram baseline in the EmbDI style, but with far fewer walks than 1,000,000
  and written in PyTorch (gensim/word2vec is not available here). It is an approximation of
  EmbDI, not the original tool, and is labelled as such everywhere.

Not reproduced: Rossmann (needs a Kaggle download), TabPFN (not installed), the exact Equation 1
weighting (re-implemented as linear interpolation between neighbouring bin centres, which has
the same properties described in the paper: weights depend on position in the bin and neighbours
share mass near borders).

## 3. The gap this project addresses

1. **Scale and breadth of the test.** The method is evaluated here on Adult Census Income
   (48,842 rows, 8 categorical and 4 numerical input columns), about 60 times larger than
   Titanic, with a 7,000+ row test set instead of 74.
2. **Missing cheap baselines.** The embeddings are compared at equal dimension against PCA of a
   one-hot encoding, a closed-form spectral factorisation of the same matrix, and a random-walk
   skip-gram model, plus k-NN on raw features without any embedding.
3. **"Universal" is tested, not assumed.** The same frozen pipeline is scored on four uses:
   classification (income), regression (age, with the age column held out of the graph),
   outlier detection (synthetically corrupted rows with known labels), and neighbour quality.
4. **Stability.** Results are reported over several random seeds, which the paper does not show.
5. **Usability.** An interactive dashboard lets a visitor inspect the entity graph, the embedding
   space, similar rows, and embed a new hand-entered row without retraining.

No result is promised in advance: all numbers will come from `artifacts/metrics.json`.
