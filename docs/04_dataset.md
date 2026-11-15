# 04 - Dataset

## Name and source

**Adult (Census Income)**, extracted by Barry Becker from the 1994 US Census database and
donated by Barry Becker and Ronny Kohavi in 1996.

- UCI page: https://archive.ics.uci.edu/dataset/2/adult (DOI 10.24432/C5XW20)
- OpenML copy used in code: https://www.openml.org/d/1590 (name `adult`, version 2 - the
  original UCI train and test files merged)
- Reference: R. Kohavi, "Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree
  Hybrid", KDD 1996.

## License

Creative Commons Attribution 4.0 International (CC BY 4.0) according to the UCI page; OpenML
lists the licence as "Public". The data is not redistributed in this repository; it is
downloaded at run time.

## Why it fits the problem

- It is a genuine relational-style table of people with **mixed categorical and numerical
  columns**, exactly the setting of the source paper.
- Category values repeat a lot across rows, so an entity co-occurrence graph is meaningful.
- It has missing values and zero-inflated numerical columns, which stress the binning step.
- It is well known, so readers can compare the supervised references with public results.
- It is about 60 times larger than the paper's Titanic dataset but still small enough for CPU.

## Size and columns

48,842 rows, 14 feature columns plus the target (as stated on the UCI page). The download is a
few megabytes.

| Column | Type | Used as | Notes |
|--------|------|---------|-------|
| age | numerical | input (soft-binned) | also the held-out regression target in the universality check |
| workclass | categorical | input | has missing values |
| fnlwgt | numerical | **dropped** | census sampling weight, not a property of the person |
| education | categorical | input | 16 levels |
| education-num | numerical | **dropped** | 1:1 numeric copy of `education`; keeping both would count it twice |
| marital-status | categorical | input | |
| occupation | categorical | input | has missing values |
| relationship | categorical | input | |
| race | categorical | input | |
| sex | categorical | input | |
| capital-gain | numerical | input (zero bin + quantile bins) | mostly zero |
| capital-loss | numerical | input (zero bin + quantile bins) | mostly zero |
| hours-per-week | numerical | input (soft-binned) | strong peak at 40 |
| native-country | categorical | input | has missing values, many rare levels |
| class | binary target | **never used for the embedding** | `>50K` vs `<=50K`; the positive class is the minority (about a quarter of rows) |

Exact row counts, class balance, missing-value counts and the number of entities are computed by
the code and written to `artifacts/metrics.json`; this document does not state numbers that were
not taken from the dataset pages.

## Exact loading call

```python
from pathlib import Path
from sklearn.datasets import fetch_openml

DATA_DIR = Path(__file__).resolve().parent.parent / "data"   # resolved from the script location
bunch = fetch_openml(data_id=1590, as_frame=True, parser="auto", data_home=str(DATA_DIR))
df = bunch.frame          # 14 feature columns + "class"
```

The first call downloads and caches the file under `data/`; later calls are offline.

## Preprocessing

1. Strip whitespace from all string cells; treat `?` and NaN as missing.
2. Target: `y = 1` if `class` is `>50K`, else 0.
3. Drop `fnlwgt` and `education-num`.
4. Categorical columns: missing -> explicit level `Missing` (its own entity, because "not
   reported" carries information in census data).
5. Numerical columns: no scaling is needed for the graph method (they are binned).
   - `age`, `hours-per-week`: up to 20 quantile bins fitted on the training rows, duplicate
     edges merged; each value is shared between the two nearest bin centres by linear
     interpolation.
   - `capital-gain`, `capital-loss`: one bin for exactly zero, plus up to 10 quantile bins over
     the non-zero training values (soft assignment among the non-zero bins only).
6. For the baselines only: one-hot encoding of categoricals and standardisation of numericals
   (`log1p` first for the two capital columns), or ordinal codes + standardisation for the
   simple baseline. All fitted on the training rows.
7. No rows are removed. No duplicates are removed (identical people are legitimate in a census).

## Split plan

| Split | Share | Use |
|-------|-------|-----|
| Train | 70% | fit bins, build the entity graph, train the GAE, fit baselines, k-NN reference set |
| Validation | 15% | choose k for k-NN, choose the target normalisation, early checks |
| Test | 15% | final numbers only, touched once |

Stratified on the income label, `random_state=42`, done with two calls to
`sklearn.model_selection.train_test_split`. The original UCI train/test file boundary is not
used because the OpenML version merges both files.

## Known issues and ethics note

The data is from 1994 and contains sensitive attributes (race, sex, native country). It is used
here only as a public benchmark of representation quality; the project makes no claim that the
embedding or any predictor built on it is fair or suitable for decisions about people. The
dashboard states this.
