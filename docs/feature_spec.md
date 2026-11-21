# Feature spec: fighter style profiles

Implemented in [`src/features.py`](../src/features.py). Explored in [`notebooks/01_feature_eda.ipynb`](../notebooks/01_feature_eda.ipynb).

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

Each mix sums to 1, so each mix has only two free dimensions. A fighter with zero significant strike attempts gets NULL mixes
(undefined, not 0).

## Skill measures (not used for clustering)

Definitions follow ufcstats.com's career statistics, computed from the same fights.

| Measure | Column | Definition |
|---|---|---|
| Striking accuracy | `str_acc` | Σ sig. landed / Σ sig. attempts |
| Striking defense | `str_def` | 1 − Σ opponent sig. landed / Σ opponent sig. attempts |
| Takedown accuracy | `td_acc` | Σ TD landed / Σ TD attempts (NULL if no attempts) |
| Takedown defense | `td_def` | 1 − Σ opponent TD landed / Σ opponent TD attempts (NULL if never attempted against) |
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
   deciding eligibility, and computing the standardization means and SDs. `as_of=None` gives full-career profiles.

## Outputs

- `build_profiles(as_of)` returns one row per fighter: ID, name, `division`, `std_group`, `n_fights`, `eligible`, `fight_time_min`,
  first and last fight date, the 10 style features, the 6 skill measures, and `z_` versions of the style features.
- `python src/features.py` writes the full-career profiles to `data/processed/fighter_profiles.csv` (gitignored).
- `python -m unittest discover tests` runs the leakage tests.
