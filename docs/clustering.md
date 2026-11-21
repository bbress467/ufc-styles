# Style clusters

Implemented in [`src/clustering.py`](../src/clustering.py). Analysed in [`notebooks/03_clustering.ipynb`](../notebooks/03_clustering.ipynb).
Clusters are numbered by size (cluster 1 = largest) and are **not named**: the descriptions below are neutral summaries of the features.

## Method

- **Input:** `features.clustering_matrix(profiles)`. These are full-career profiles of the 1,240 eligible fighters (5+ UFC fights since
  2000), as within-division z-scores of the 8 shrunk style features in `CLUSTER_FEATURES` (see [feature_spec.md](feature_spec.md)).
- **PCA:** first 5 principal components, 85.2% of the variance (`notebooks/02_pca.ipynb`). The sign convention is fixed: each component's
  largest loading is positive.
- **Model:** Gaussian mixture with full covariance (scikit-learn `GaussianMixture`), 10 restarts, `random_state=0`.
- **Choosing k:** for k = 3–10, the BIC of the full-data fit, plus bootstrap stability. The mixture is refit on 50 resamples of fighters
  (with replacement), every fighter is assigned with the refit model, and the result is compared with the full-data assignment by
  adjusted Rand index (ARI). The PCA is held fixed during the bootstrap.

## Choosing k

| k | 1* | 2* | **3** | **4** | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| BIC | 18,671 | 18,296 | **18,221** | 18,283 | 18,382 | 18,468 | 18,569 | 18,671 | 18,775 | 18,890 |
| bootstrap ARI, mean | — | 0.81 | **0.70** | 0.49 | 0.34 | 0.34 | 0.32 | 0.35 | 0.26 | 0.29 |
| bootstrap ARI, 10th pct | — | 0.72 | 0.55 | 0.33 | 0.24 | 0.25 | 0.22 | 0.24 | 0.21 | 0.22 |
| smallest cluster | 1,240 | 528 | 294 | 258 | 152 | 107 | 69 | 56 | 52 | 54 |

\* Outside the requested range, shown for context.

**Recommended k = 3; next-best k = 4** (`CHOSEN_K`, `RUNNER_UP_K`).

- Within 3–10, k = 3 is best on **both** BIC and stability, and k = 4 is second on both.
- From k = 5, bootstrap ARI falls to about 0.3, so finer partitions don't reproduce.
- BIC strongly prefers a mixture over a single Gaussian (k = 1 is 450 points worse), so the style space isn't one blob. But even at k = 3
  stability is only moderate (ARI ≥ 0.8 is usually called stable), and 17% of fighters have a highest membership probability below 0.6.
  The clusters are **overlapping regions of a continuum**, not sharply separated types. For the matchup analysis the membership
  probabilities are likely more useful than hard labels.

## k-means baseline

| k | ARI (GMM vs k-means) | fighters in matched clusters |
|---|---|---|
| 3 | 0.34 | 72% |
| 4 | 0.31 | 63% |

The agreement is weak. k-means assumes spherical, equal-size clusters; the full-covariance mixture allows elongated, correlated ones.
They split the same space differently: GMM cluster 1 is spread across all three k-means clusters, while GMM cluster 3 maps mostly onto
one. This is consistent with weak cluster structure. With well-separated clusters, both methods would find about the same groups.

## Cluster profiles (k = 3)

Mean within-division z-scores (what the model saw) of the fighters hard-assigned to each cluster. Raw-unit means, skill measures, the
10 most representative fighters and mixed-membership fighters are in the notebook.

| cluster | fighters | distance | clinch | ground | head | body | leg | TD att/15 | ctrl share | sub att/15 | pace |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 510 | −0.50 | +0.11 | +0.60 | +0.49 | −0.34 | −0.42 | +0.62 | +0.68 | +0.60 | −0.45 |
| 2 | 436 | +0.71 | −0.57 | −0.56 | +0.13 | −0.20 | −0.00 | −0.60 | −0.68 | −0.41 | +0.54 |
| 3 | 294 | −0.19 | +0.66 | −0.22 | −1.04 | +0.88 | +0.73 | −0.19 | −0.17 | −0.42 | −0.01 |

- **Cluster 1:** above average on ground strike share, takedown attempts, control share and submission attempts; below-average pace;
  strikes mostly to the head. High on PC1 (+1.36). Most representative: Jaqueline Amorim, Charles Oliveira, Renato Sobral, TJ Waldburger,
  Frank Mir.
- **Cluster 2:** above average on distance share and pace; below average on clinch, ground, takedowns, control and submissions. Low on
  PC1 (−1.34). Most representative: Sean Strickland, Nasrat Haqparast, Nam Phan, Esteban Ribovics, Eddie Wineland.
- **Cluster 3:** strikes to the body and legs rather than the head (head share −1.04 SD), more clinch work, about average grappling and pace.
  High on PC2 (+1.33). Most representative: Chris Gutierrez, Chidi Njokuani, Caol Uno, Fabio Maldonado, Michel Pereira, Demetrious Johnson.
- Of 30 well-known fighters checked, 3 have a highest probability below 0.8 at k = 3: Daniel Cormier (0.69 / 0.05 / 0.25),
  Francis Ngannou (0.26 / 0.68 / 0.06) and Kamaru Usman (0.60 / 0.00 / 0.40).

**k = 4** keeps cluster 3 almost intact (as k = 4 cluster 4). It splits k = 3 clusters 1 and 2 into a more extreme grappling group
(cluster 2: ground +0.99, takedowns +0.96, control +1.07, submissions +0.97), a more extreme distance-and-pace group (cluster 3: distance
+0.94, pace +0.82), and a new **middle** cluster (cluster 1, 398 fighters) with every mean z-score within ±0.31 of 0. That middle cluster
is the least distinct, which is consistent with the lower stability at k = 4.

## Confounding checks

Cramér's V between hard cluster and each variable (0 = no association):

| | k = 3 | k = 4 |
|---|---|---|
| division | 0.06 | 0.08 |
| era (median year of the fighter's UFC fights: 2000–07, 2008–13, 2014–19, 2020–26) | **0.17** | **0.15** |

- **Division: negligible.** Every division's cluster mix is within a few points of the overall mix, so within-division standardization
  works.
- **Era: weak but real.** At k = 3, cluster 1 is 64% of fighters whose careers centre on 2000–07 but 33% of those in 2020–26. Cluster 2 goes
  from 17% to 44%. Cluster 3 is flat (19–27%).
- The cause is drift in the features. Comparing within-division z-scores of the 2000–07 and 2020–26 eras: distance share −0.84 → +0.30,
  pace −0.62 → +0.24, clinch +0.65 → −0.30, ground +0.69 → −0.19, submissions +0.52 → −0.13. The sport has moved toward higher-volume
  distance striking, and clusters 1 and 2 lie along exactly those features. So **part of the cluster 1 / cluster 2 split is era**.
- Possible remedies (not applied): standardize within division × era (or year) instead of division only; control for era in the matchup
  model; or restrict the analysis window.

## Saved outputs

`python src/clustering.py` (after `src/features.py`) writes, gitignored:

- `data/processed/cluster_probabilities.csv`: per eligible fighter, the k = 3 probabilities `p_cluster_1..3`, the hard `cluster`,
  `max_prob` and PC scores `PC1..PC5`.
- `data/processed/models/style_model_k3.joblib` and `style_model_k4.joblib`: a `clustering.StyleModel` containing the fitted PCA,
  Gaussian mixture, cluster numbering, input columns and metadata (fit time, priors, eligibility rule, seed).

Applying a saved model to profiles built at a date:

```python
import features as F, clustering as C
model = C.StyleModel.load("data/processed/models/style_model_k3.joblib")
probs = model.assign(F.build_profiles(as_of="2018-01-01"))   # eligible fighters; eligible_only=False to score everyone
```

## Leakage concerns for the matchup analysis

The features themselves are leak-free at any `as_of`. **The saved models are not**, because they were fitted on full-career profiles
through the end of the data (2026-08-08).

1. **Cluster definitions use the future.** The PCA axes, the mixture means and covariances, and even the choice of k = 3 were learned
   from profiles that include every fight in the data. Applying them to `as_of` profiles assigns fighters to groups defined partly by
   fights after `as_of`, including the fights the matchup model will try to predict. Every fight a fighter has had contributes to their
   full-career profile, and so to the model.
2. **Standardization mismatch.** `as_of` profiles are standardized with pre-`as_of` division means and SDs, while the model was fitted
   on full-career z-scores. Because of the era drift above, the same raw style maps to different z-scores in 2010 and 2026, so a saved
   model may assign historical fighters differently than a model fitted at that time would.
3. **Tuning choices were made on all data:** the shrinkage priors (60 and 240 min), the 5 PCs, k and the eligibility rule. This is a
   milder, hyperparameter-level leak.
4. **Cluster numbering isn't stable across refits.** Clusters are numbered by size on the fitting data, so a model refitted at another
   cutoff may number them differently. Match clusters across refits (for example by Hungarian matching on component means) before
   comparing them.
5. **Newer fighters.** At an `as_of` date many fighters have fewer than 5 prior fights. With `eligible_only=False` they can be scored,
   but their profiles are dominated by the division prior.

**Recommended for the matchup analysis:** fit the PCA and mixture **only on profiles built at the start of the evaluation period**
(e.g. `fit_style_model(build_profiles(as_of=cutoff), as_of=cutoff)`), then score fights after the cutoff with
`build_profiles(as_of=fight_date)`. Alternatively, refit on an expanding window (for example yearly) with cluster matching. The
full-data model is fine for describing styles. It shouldn't be used to estimate how well styles predict outcomes.
