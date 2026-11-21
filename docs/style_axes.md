# Style axes

Implemented in [`src/style_axes.py`](../src/style_axes.py). Chosen and described in [`notebooks/05_style_axes.ipynb`](../notebooks/05_style_axes.ipynb),
mapped in [`notebooks/06_style_map.ipynb`](../notebooks/06_style_map.ipynb).

The four continuous style axes are the **main way of describing a fighter's style**. Clusters ([clustering.md](clustering.md)) are a secondary
shorthand. Axes are numbered by the variance they carry.

## Names

| axis | name | code key (`style_axes.AXIS_KEYS`) | high score means | anchor feature |
|---|---|---|---|---|
| 1 | **grappler vs. volume striker** | `grappler_vs_striker` | more takedowns, control and ground striking, lower pace | control share |
| 2 | **clinch and body work** | `clinch_body` | more strikes in the clinch and to the body | clinch share |
| 3 | **submission hunting** | `submission_hunting` | more submission attempts (and lower pace) | submission attempts |
| 4 | **leg kicking** | `leg_kicking` | more strikes to the legs | leg share |

Score columns stay `axis_1..axis_4`. `style_axes.AXIS_NAMES` gives the display names.

**Alignment.** Any 4-axis fit is reordered and signed by its anchor features (`align_to_named_axes`): axis k is the axis its anchor
loads on most, signed so that loading is positive. This uses only the fit's own loadings, so a model fitted on `as_of` profiles
(e.g. the frozen 2019 model in the matchup analysis) gets the same names without seeing later data. If the anchors don't pick out
four distinct axes with |loading| ≥ 0.5, the fit raises an error instead of mislabelling. The full-data fit is unchanged by alignment.

## Method

1. **Input:** the 8 era-relative style z-scores from `features.clustering_matrix(profiles)` (eligible fighters, full careers). Each fighter is
   standardized against same-division fighters with career midpoints within ±4 years ([feature_spec.md](feature_spec.md)).
   The distance and head shares are dropped as redundant.
2. **PCA** on those z-scores. Eigenvalues: 2.89, 1.45, 0.90, 0.84, 0.60, 0.57, 0.48, 0.16.
3. **Varimax rotation** of the first m components' loadings (loadings = eigenvector × √eigenvalue).
4. **Axis scores** are the standardized component scores rotated by the same matrix. They are standardized (mean 0, SD 1 on the fitting
   data) and uncorrelated. Loadings are the correlations between each input feature and each axis.
5. Axes are ordered by variance after rotation, and each axis's sign is set so its largest-magnitude loading is positive.

## Three or four axes: four

Bootstrap: 200 resamples of fighters, refit, match each bootstrap axis to a full-data axis by |Tucker congruence|, and compare.

| | axis | variance | loading congruence, mean (5th pct) | score correlation, mean (5th pct) |
|---|---|---|---|---|
| **4 axes** | 1 | 32.1% | 0.998 (0.996) | 0.998 (0.994) |
| | 2 | 16.5% | 0.996 (0.990) | 0.997 (0.991) |
| | 3 | 14.8% | 0.992 (0.979) | 0.993 (0.979) |
| | 4 | 13.5% | 0.993 (0.982) | 0.995 (0.987) |
| 3 axes | 1 | 36.3% | 0.990 (0.959) | 0.975 (0.894) |
| | 2 | 16.4% | 0.969 (0.892) | 0.958 (0.841) |
| | 3 | 13.6% | **0.820 (0.228)** | **0.845 (0.324)** |

| | 3 axes | 4 axes |
|---|---|---|
| total variance carried | 66% | 77% |
| mean purity (share of a feature's explained variance on its main axis) | 0.93 | 0.88 |
| features loading ≥ 0.4 on more than one axis | 0 | 1 (pace) |
| lowest communality | 0.30 (submissions) | 0.59 (pace) |

**Four axes.** Every axis reproduces almost perfectly under resampling. With three, the third axis is unstable: in some resamples it
barely resembles the full-data axis. This happens because PCs 3 and 4 have nearly equal eigenvalues (0.90 vs 0.84). Keeping only three
cuts through a near-tie, and the third axis becomes a different mix of the two in each sample. Three axes look slightly "purer", but
only because they nearly drop submission attempts (communality 0.30). With four, every feature is well represented, and only pace splits
across axes.

## Loadings (4 axes)

| feature | axis 1 (grappler vs. volume striker) | axis 2 (clinch and body work) | axis 3 (submission hunting) | axis 4 (leg kicking) |
|---|---|---|---|---|
| clinch share | +0.22 | **+0.79** | −0.03 | −0.04 |
| ground share | **+0.78** | +0.03 | +0.23 | −0.04 |
| body share | −0.16 | **+0.79** | +0.05 | +0.19 |
| leg share | −0.08 | +0.07 | −0.06 | **+0.96** |
| TD attempts / 15 | **+0.86** | −0.04 | +0.05 | −0.06 |
| control share | **+0.92** | +0.05 | +0.15 | −0.08 |
| sub attempts / 15 | +0.16 | −0.01 | **+0.92** | −0.09 |
| pace | **−0.50** | −0.17 | **−0.48** | −0.28 |

Implied by the dropped shares: distance share is high where clinch and ground are low, and head share is high where body and legs are low.

- **Axis 1, grappler vs. volume striker:** control share, takedown attempts, ground strike share (positive); pace (negative).
- **Axis 2, clinch and body work:** clinch share and body share (positive).
- **Axis 3, submission hunting:** submission attempts (positive); pace (negative).
- **Axis 4, leg kicking:** leg share (positive).

## Fighters at each end

Eligible fighters, axis scores in SD units. Full top-10 lists with divisions and fight minutes are in notebook 05.

| axis | highest | lowest |
|---|---|---|
| 1 (grappler vs. volume striker) | Khabib Nurmagomedov +3.74, Gregor Gillespie, Tatiana Suarez, Pat Sabatini, Chael Sonnen, Merab Dvalishvili | Chris Lytle −2.60, Nam Phan, Junior Dos Santos, John Lineker, Irene Aldana, Max Holloway |
| 2 (clinch and body work) | Loma Lookboonmee +4.42, Chidi Njokuani, Ben Saunders, Anthony Hamilton, Frankie Saenz, Yana Santos, Demetrious Johnson | Serghei Spivac −2.28, Yoshihiro Akiyama, Renato Moicano, Sean Brady, Khabib Nurmagomedov |
| 3 (submission hunting) | Charles Oliveira +5.44, Aleksei Oleinik, Jaqueline Amorim, TJ Waldburger, Jim Miller, Stefan Struve | Corey Anderson −2.25, Colby Covington, Curtis Blaydes, Kamaru Usman, Cain Velasquez |
| 4 (leg kicking) | Chris Gutierrez +6.14, Caol Uno, Armen Petrosyan, Chris Padilla, Nathaniel Wood, Johnny Walker | Evan Dunham −2.88, Brandon Royval, Darren Elkins, Sean Strickland, Joe Lauzon |

The axes are skewed. The high ends reach +4 to +6 SD (a few specialists), and the low ends stop at about −2 to −3.

## Era and division checks

- Correlation of axis scores with career midpoint: −0.016, −0.015, −0.015 and 0.000. Mean axis scores by career era (2000–07 to 2020–26)
  are all within ±0.10. The same rotation applied to division-only z-scores gives correlations of −0.18, −0.11, −0.19 and +0.04, so the
  era-relative standardization removes the era trend from the axes.
- Mean axis scores by division are within ±0.1, apart from Women's Featherweight, which has only 4 fighters.

## Saved outputs

`python src/style_axes.py` (after `src/features.py`) writes, gitignored:

- `data/processed/style_axes.csv`: `axis_1..axis_4` for each eligible fighter, plus name, division, fights and career midpoint.
- `data/processed/models/style_axes_m4.joblib` (and `_m3`): a `style_axes.StyleAxes` with the feature means, eigenvectors, eigenvalues,
  rotation and metadata.

```python
import features as F, style_axes as A
axes = A.StyleAxes.load("data/processed/models/style_axes_m4.joblib")
scores = axes.transform(F.build_profiles(as_of="2018-01-01"))   # eligible_only=False to score everyone
```

## Leakage concerns for the matchup analysis

As with the clusters, the inputs are leak-free at any `as_of`, but the saved transformation isn't. Its means, eigenvectors and rotation
were fitted on full careers through 2026-08-08, and the number of axes was chosen on all data. Applying it to `as_of` profiles describes
fighters with axes partly defined by later fights.

The axes are very stable under resampling (congruence ≥ 0.98), so refitting them on an earlier subset probably changes them little.
For a leak-free evaluation, though, fit them with `fit_style_axes(build_profiles(as_of=cutoff), as_of=cutoff)` and score later fights
with that fit. The frontier caveat in [clustering.md](clustering.md) also applies: at an `as_of` date, recent fighters' era windows are
one-sided.
