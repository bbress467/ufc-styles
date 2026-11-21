# Feature spec: fighter style profiles

Implemented in [`src/features.py`](../src/features.py). Explored in [`notebooks/01_feature_eda.ipynb`](../notebooks/01_feature_eda.ipynb)
(distributions, correlations, shrinkage check) and [`notebooks/02_pca.ipynb`](../notebooks/02_pca.ipynb) (PCA of the clustering features).

## Goal

Describe **how each fighter fights, not how good they are.** Style features feed the clustering. Skill measures are
computed alongside them but kept separate and **not used for clustering**. The later matchup analysis uses them to control for
fighter quality.

## Data

- UFC fights from **2000-01-01 onward** (see "Date cutoff" in the README), from `data/processed/ufc.sqlite`.
- Built from the per-round table `fight_round_stats`, summed to one row per fighter per fight. Fight time comes from
  `fights.total_fight_time_s`, the elapsed fight time, the same for both fighters.
- Every fight counts, including draws and no contests: style doesn't depend on the result.

## Style features (per fighter)

All counts are **attempts**, pooled over the fighter's fights (see Decisions). `T` = total fight time in seconds.

| Feature | Column | Definition |
|---|---|---|
| Position mix: distance | `pos_distance` | Σ distance sig. attempts / Σ sig. attempts |
| Position mix: clinch | `pos_clinch` | Σ clinch sig. attempts / Σ sig. attempts |
| Position mix: ground | `pos_ground` | Σ ground sig. attempts / Σ sig. attempts |
| Target mix: head | `tgt_head` | Σ head sig. attempts / Σ sig. attempts |
| Target mix: body | `tgt_body` | Σ body sig. attempts / Σ sig. attempts |
| Target mix: legs | `tgt_leg` | Σ leg sig. attempts / Σ sig. attempts |
| Takedown attempts per 15 min | `td_att_per15` | Σ TD attempts / (T / 900) |
| Control share | `ctrl_share` | Σ control seconds / T |
| Submission attempts per 15 min | `sub_att_per15` | Σ submission attempts / (T / 900) |
| Pace | `pace` | Σ sig. attempts / (T / 60), in attempts per minute |

These are the **raw** definitions, stored as `raw_<feature>`. The main columns are the same ratios **shrunk toward the division
average** (Decision 8). Each mix sums to 1, so each mix has only two free dimensions (Decision 9).

## Skill measures (not used for clustering)

Definitions follow ufcstats.com's career statistics, computed from the same fights.

| Measure | Column | Definition |
|---|---|---|
| Striking accuracy | `str_acc` | Σ sig. landed / Σ sig. attempts |
| Striking defense | `str_def` | 1 − Σ opponent sig. landed / Σ opponent sig. attempts |
| Takedown accuracy | `td_acc` | Σ TD landed / Σ TD attempts, **shrunk** (Decision 8); raw in `raw_td_acc` (NULL if no attempts) |
| Takedown defense | `td_def` | 1 − Σ opponent TD landed / Σ opponent TD attempts, **shrunk**; raw in `raw_td_def` |
| Knockdowns | `kd_per15` | Σ knockdowns / (T / 900) |
| Strike differential | `str_diff_per_min` | (Σ sig. landed − Σ opponent sig. landed) / (T / 60) |

## Decisions

1. **Attempts, not landed**, for style features. Attempts reflect what a fighter is trying to do; landed mixes in how good they are at it.
2. **Minimum fights.** A fighter is `eligible` with at least **5 UFC fights since 2000** (before `as_of`). Profiles are returned for every
   fighter with at least one fight, with `n_fights` and `eligible` columns, so the matchup analysis can still see newer fighters. Clustering
   uses eligible fighters only.
3. **Standardize within weight class**, so clusters reflect style rather than division. Columns `z_<feature>` hold the z-score of each
   style feature within the fighter's division (mean and SD over *eligible* fighters in that division). Ineligible fighters are scored
   against the same eligible-fighter mean and SD.
4. **Division per fighter = most frequent division** among their fights (before `as_of`), **ignoring catch-weight bouts**. Ties go to the
   most recent fight. TUF and Road to UFC tournament bouts are mapped to their division (e.g. "Ultimate Fighter 9 Welterweight Tournament"
   → Welterweight), and the single "Super Heavyweight" bout → Heavyweight. A fighter with only catch-weight fights has no division and NULL z-scores.
5. **Women's Featherweight is standardized together with Women's Bantamweight** (`std_group`). It has only 4 eligible fighters, too few for
   a stable mean and SD. The fighter's own `division` still says Women's Featherweight.
6. **Pooled totals, not averages of per-fight values.** Sum attempts and time over all fights, then divide. Longer fights carry more weight,
   and very short fights can't produce extreme shares or rates.
7. **No leakage.** `build_profiles(as_of=...)` uses only fights **strictly before** `as_of`, including when choosing each fighter's division,
   deciding eligibility, computing the shrinkage priors, and computing the standardization means and SDs. `as_of=None` gives full-career profiles.
8. **Shrinkage toward the division average**, so fighters with little cage time are pulled toward typical values (details below).
   It applies to all 10 style features and to takedown accuracy and defense.
9. **Redundant shares dropped for clustering.** `pos_distance` = 1 − clinch − ground, and `tgt_head` = 1 − body − legs. Keeping them would
   count one dimension of each mix twice. `features.CLUSTER_FEATURES` (8 features) and `features.clustering_matrix(profiles)` leave them out;
   the saved profiles keep them, for interpretation.

## Shrinkage (Decision 8)

### Method: pseudo-counts of division-average fighting

Every feature is a ratio, numerator / denominator: for example clinch attempts / sig. attempts, TD attempts / fight seconds, or
opponent TD landed / opponent TD attempts. Each fighter's totals get **P seconds of a division-average fighter's totals** added:

```
shrunk = (fighter numerator + P × division numerator per second) / (fighter denominator + P × division denominator per second)
```

- "Division per second" values are pooled over **eligible fighters in the fighter's standardization group**, from pre-`as_of` fights only:
  Σ numerator / Σ fight seconds. A group with no eligible fighters yet, or a fighter with no division, falls back to the pooled average of
  all eligible fighters.
- With 0 fight time a fighter gets the division average. With far more than P of fight time the prior barely matters. At fight time T the
  fighter's own data gets weight about T / (T + P).
- It is the same formula for every feature. For rates the denominator is time (P seconds is added to their fight time). For shares the
  denominator is sig. attempts (the division's attempt rate × P is added). For takedown accuracy it is attempts (P minutes' worth of the
  division's TD attempts, at the division's accuracy, is added). Takedown defense works the same way on the opponents' attempts.
- The prior strength is a fight-time amount, so it means the same thing across features. That's why the prior is expressed in minutes
  rather than as an attempt count.
- Other skill measures (striking accuracy and defense, knockdowns, strike differential) are left raw, as specified.

### Choosing P = 60 minutes

Held-out check (`notebooks/01_feature_eda.ipynb`, section 5). For the 618 fighters with 10+ fights since 2000, features from an early
sample of fights, shrunk with prior P, are compared with the **raw** features from their later fights. The error is fight-time-weighted
MSE ÷ the variance of the later values, so 1.0 means "no better than guessing the population mean". Mean over the 12 shrunk features:

| P (min) | 0 | 15 | 30 | 45 | **60** | 90 | 120 | 180 | 240 |
|---|---|---|---|---|---|---|---|---|---|
| first half → second half | 1.36 | 1.02 | 0.90 | 0.84 | **0.81** | 0.78 | 0.78 | 0.80 | 0.81 |
| first 5 fights → fights 6+ | 2.36 | 1.27 | 1.00 | 0.89 | **0.84** | 0.80 | 0.80 | 0.82 | 0.83 |

- **Without shrinkage, early-career values predict later ones worse than the population mean does** (error > 1). Shrinkage is clearly needed.
- The error curve is flat from 60 to 180 minutes. Its minimum is at 90–120, and **60 is the smallest prior within 5% of the minimum on both splits**.
- The best P differs by feature:
  - about 30–60 min: target mix, takedown attempts, pace, control share
  - about 90–180 min: position mix
  - 240 min, the edge of the grid: submission attempts and takedown accuracy, which barely repeat between career halves
- **Why 60 rather than 120:** a smaller prior keeps more of each fighter's own data. The median eligible fighter (100 min of fight time)
  keeps 62% weight on their own data at P = 60 (45% at 120), and the 10th-percentile fighter (51 min) keeps 46% (30% at 120). Larger priors also
  start to rank long-career fighters first simply because their values move least (see below). Part of the held-out error is genuine
  style change over a career, not noise, and a larger prior is rewarded for ignoring that. One number for all features keeps the method
  simple to explain. Per-feature priors would fit better, especially for submissions, and are an option if clustering looks noisy on
  those features.
- **Caveat:** the division averages in this check use all fights. That's fine for choosing one tuning number, but it isn't a leakage-free
  evaluation. `build_profiles` itself only uses pre-`as_of` data.

### Effect on the top-10 lists (full careers, eligible fighters)

Share of the raw (P = 0) top 10 still in the top 10 at P = 60. Full lists at P = 0, 30, 60 and 120 are in notebook 01, section 5.

| Feature | Kept | Raw #1 (fight minutes) | #1 at P = 60 |
|---|---|---|---|
| `td_acc` | 0/10 | 1.00, ten fighters with few attempts (Houston Alexander, 25 min) | Georges St-Pierre 0.70 (343 min) |
| `td_def` | 0/10 | 1.00, ten fighters (Viktor Pesta, 50 min) | Jose Aldo 0.88; Jon Jones 0.87 |
| `pos_clinch` | 2/10 | Ken Shamrock 0.44 (23 min) | Josh Barnett 0.25 (92 min) |
| `pace` | 4/10 | Leslie Smith 18.7 (63 min) | Max Holloway 13.8 (534 min) |
| `td_att_per15` | 5/10 | Quillan Salkilld 17.2 (29 min) | Merab Dvalishvili 14.8 (299 min) |
| `sub_att_per15` | 6/10 | Paul Sass 7.4 (28 min) | Paul Sass 2.8, then TJ Waldburger, Charles Oliveira |
| `ctrl_share` | 7/10 | Bartosz Fabinski 0.79 (51 min) | Chael Sonnen 0.55 (143 min) |

At P = 30, short-career fighters still lead several lists (Leslie Smith in pace, Quillan Salkilld still in the takedown top 10). At P = 120
the lists are mostly fighters with 150+ minutes (Oliveira, Lytle and Lauzon in submissions, Holloway and Strickland in pace). At P = 60 the
leaders are recognizable specialists, and fighters with strong but short records (Paul Sass, Jaqueline Amorim) can still appear.

## Outputs

- `build_profiles(as_of, prior_minutes=60)` returns one row per fighter: ID, name, `division`, `std_group`, `n_fights`, `eligible`,
  `fight_time_min`, first and last fight date, the 10 shrunk style features, the 6 skill measures (td_acc and td_def shrunk),
  `raw_` versions of the 12 shrunk columns, and `z_` versions of the shrunk style features.
- `clustering_matrix(profiles)` gives eligible fighters × `z_` of the 8 `CLUSTER_FEATURES`. It's the input to
  `notebooks/02_pca.ipynb` and to clustering later.
- `python src/features.py` writes the full-career profiles to `data/processed/fighter_profiles.csv` (gitignored).
- `python -m unittest discover tests` runs the tests. They check that a profile as of a fight's date excludes that fight, that profiles
  (shrunk values, priors, z-scores) don't change when later fights are present in the input, that the shrinkage matches the formula, and
  that P = 0 returns the raw values.
