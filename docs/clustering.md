# Style clusters

Implemented in [`src/clustering.py`](../src/clustering.py). Analysed in [`notebooks/03_clustering.ipynb`](../notebooks/03_clustering.ipynb).

**Clusters are a secondary shorthand.** The main description of a fighter's style is their position on the four continuous style axes
([style_axes.md](style_axes.md)). Clusters are numbered by size (cluster 1 = largest) and are **not named**: the descriptions below are
neutral summaries of the features.

## Method

- **Input:** `features.clustering_matrix(profiles)`. These are full-career profiles of the 1,240 eligible fighters (5+ UFC fights since
  2000), as **era-relative** z-scores of the 8 shrunk style features in `CLUSTER_FEATURES`. Each fighter is standardized against
  same-division fighters whose career midpoints are within ±4 years ([feature_spec.md](feature_spec.md)).
- **PCA:** first 5 principal components, 84.5% of the variance. The sign convention is fixed: each component's largest loading is positive.
- **Model:** Gaussian mixture with full covariance (scikit-learn `GaussianMixture`), 10 restarts, `random_state=0`.
- **Choosing k:** for k = 3–10, the BIC of the full-data fit, plus bootstrap stability. The mixture is refit on 50 resamples of fighters,
  every fighter is assigned with the refit model, and the result is compared with the full-data assignment by adjusted Rand index (ARI).
  The PCA is held fixed during the bootstrap.

## Choosing k

Era-relative features (current):

| k | 1* | 2* | **3** | **4** | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| BIC | 18,533 | 18,233 | **18,168** | 18,227 | 18,315 | 18,411 | 18,521 | 18,620 | 18,758 | 18,825 |
| bootstrap ARI, mean | — | 0.73 | **0.62** | 0.53 | 0.39 | 0.39 | 0.34 | 0.30 | 0.30 | 0.31 |
| bootstrap ARI, 10th pct | — | 0.64 | 0.44 | 0.33 | 0.26 | 0.30 | 0.25 | 0.22 | 0.22 | 0.24 |
| smallest cluster | 1,240 | 482 | 254 | 242 | 163 | 124 | 115 | 55 | 46 | 28 |

\* Outside the requested range, shown for context.

**k = 3 still holds; next-best k = 4** (`CHOSEN_K`, `RUNNER_UP_K`).

- Within 3–10, k = 3 is best on both BIC and stability, and k = 4 is second on both. That's the same ranking as with division-only
  standardization, where the ARIs were 0.70 and 0.49.
- k = 3 is slightly less stable than before (0.62 vs 0.70). Era differences used to help separate fighters into groups, and they no longer do.
- From k = 5, stability falls to about 0.3–0.4.
- Even k = 3 is only moderately stable, and 18% of fighters have a highest membership probability below 0.6. The clusters are
  **overlapping regions of a continuum**, which is the reason the continuous axes are now the main description.

## k-means baseline

| k | ARI (GMM vs k-means) | fighters in matched clusters |
|---|---|---|
| 3 | 0.34 | 71% |
| 4 | 0.39 | 71% |

Agreement is weak, as before. k-means assumes spherical, equal-size clusters, while the full-covariance mixture allows elongated ones.
The weak agreement is consistent with weak cluster structure.

## Cluster profiles (k = 3)

Mean era-relative z-scores of the fighters hard-assigned to each cluster, and their mean scores on the four style axes:

| cluster | fighters | distance | clinch | ground | head | body | leg | TD att/15 | ctrl share | sub att/15 | pace |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 500 | +0.59 | −0.46 | −0.46 | +0.17 | −0.25 | −0.01 | −0.48 | −0.55 | −0.29 | +0.33 |
| 2 | 486 | −0.59 | +0.17 | +0.65 | +0.40 | −0.21 | −0.42 | +0.63 | +0.72 | +0.62 | −0.45 |
| 3 | 254 | −0.11 | +0.60 | −0.27 | −1.12 | +0.90 | +0.86 | −0.27 | −0.26 | −0.56 | +0.08 |

| cluster | axis 1 | axis 2 | axis 3 | axis 4 |
|---|---|---|---|---|
| 1 | −0.54 | −0.42 | −0.19 | −0.01 |
| 2 | +0.67 | −0.03 | +0.48 | −0.38 |
| 3 | −0.22 | +0.88 | −0.54 | +0.74 |

- **Cluster 1:** above average on distance share and pace; below average on clinch, ground, takedowns, control and submissions. Most
  representative: Sean Strickland, Nam Phan, Nasrat Haqparast, Eddie Wineland, Sean Spencer.
- **Cluster 2:** above average on ground strike share, takedown attempts, control share and submission attempts; below-average pace; strikes
  mostly to the head. Most representative: Charles Oliveira, Aleksei Oleinik, Jaqueline Amorim, Jailton Almeida, TJ Waldburger.
- **Cluster 3:** strikes to the body and legs rather than the head (head share −1.12 SD), more clinch work, below-average submissions.
  Most representative: Chris Gutierrez, Brandon Vera, John Lineker, Caol Uno, Rafael Fiziev, Demetrious Johnson.
- **Mixed memberships** among 30 well-known fighters (highest probability < 0.8):

  | Fighter | p(cluster 1) | p(cluster 2) | p(cluster 3) |
  |---|---|---|---|
  | Jon Jones | 0.21 | 0.00 | 0.79 |
  | Georges St-Pierre | 0.00 | 0.75 | 0.25 |
  | Francis Ngannou | 0.63 | 0.34 | 0.03 |
  | Daniel Cormier | 0.31 | 0.61 | 0.08 |
  | Alex Pereira | 0.24 | 0.00 | 0.76 |

- **Numbering changed** from the earlier, division-only version, because clusters are numbered by size. The distance-and-pace group was
  cluster 2 and is now cluster 1, and the grappling group was cluster 1 and is now cluster 2. Their profiles are almost unchanged.

**k = 4** keeps cluster 3 almost intact (as k = 4 cluster 4). It splits the other two into a more extreme distance-and-pace group (cluster 2),
a more extreme grappling group (cluster 3), and a middle group (cluster 1, 409 fighters, all mean z-scores within ±0.35).

## Confounding checks

Cramér's V between hard cluster and each variable (0 = no association; below about 0.1 is usually read as negligible):

| | division-only features (before) | era-relative features (now) |
|---|---|---|
| division, k = 3 | 0.06 | 0.06 |
| era, k = 3 | 0.17 | **0.09** |
| era, k = 4 | 0.15 | **0.07** |

Era is the median year of the fighter's UFC fights since 2000, in bins 2000–07, 2008–13, 2014–19 and 2020–26.

- **The era confounding is largely gone.** The association roughly halved and is now below the usual threshold for negligible. The
  era-relative feature means are flat across eras (within about ±0.1, except in 2000–07, where few fighters overlap and reference sets
  widen into later years). Under division-only standardization they drifted by 0.6–1.1 SD.
- **What remains:** fighters whose careers centre on 2008–13 are somewhat more often in cluster 2 (48%, against 37–40% in other eras).
  That's a modest difference in that era's style mix, not a level shift in the features.
- **Division:** still negligible.

## Saved outputs

`python src/clustering.py` (after `src/features.py`) writes, gitignored:

- `data/processed/cluster_probabilities.csv`: per eligible fighter, the k = 3 probabilities `p_cluster_1..3`, the hard `cluster`,
  `max_prob` and PC scores `PC1..PC5`.
- `data/processed/models/style_model_k3.joblib` and `style_model_k4.joblib`: a `clustering.StyleModel` with the fitted PCA, Gaussian mixture,
  cluster numbering, input columns and metadata.

```python
import features as F, clustering as C
model = C.StyleModel.load("data/processed/models/style_model_k3.joblib")
probs = model.assign(F.build_profiles(as_of="2018-01-01"))   # eligible fighters; eligible_only=False to score everyone
```

## Leakage concerns for the matchup analysis

The features themselves are leak-free at any `as_of`, including the era-relative standardization. **The saved models are not**, because
they were fitted on full-career profiles through the end of the data (2026-08-08).

1. **Cluster definitions use the future.** The PCA axes, the mixture components and the choice of k were learned from profiles that include
   every fight, including fights the matchup model will try to predict.
2. **Standardization at the frontier.** At an `as_of` date, fighters whose careers centre near that date are compared only with fighters
   *before* them, because later fighters don't exist yet. The full-data model was fitted on two-sided windows. Within-era drift is now
   small, so the effect is modest, but it isn't zero.
3. **Tuning choices were made on all data:** the shrinkage priors (60 and 240 min), the era window (±4 years), the 5 PCs, k and the
   eligibility rule. This is a milder, hyperparameter-level leak.
4. **Cluster numbering isn't stable across refits.** Match clusters across refits (for example by Hungarian matching on component
   means) before comparing them.
5. **Newer fighters.** At an `as_of` date many fighters have fewer than 5 prior fights. They can be scored with `eligible_only=False`,
   but their profiles are dominated by the division prior.

**Recommended for the matchup analysis:** fit the PCA and mixture **only on profiles built at the start of the evaluation period**
(`fit_style_model(build_profiles(as_of=cutoff), as_of=cutoff)`), then score fights after the cutoff with `build_profiles(as_of=fight_date)`.
Alternatively, refit on an expanding window with cluster matching. The full-data model is fine for describing styles. It shouldn't be used
to estimate how well styles predict outcomes.
