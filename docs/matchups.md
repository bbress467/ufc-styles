# Do style matchups predict outcomes beyond skill?

Code: [`src/elo.py`](../src/elo.py), [`src/matchups.py`](../src/matchups.py). Results: [`notebooks/07_matchups.ipynb`](../notebooks/07_matchups.ipynb).

**Short answer.**

- **Matchups: no.** On about 2,000 UFC fights from 2019 on, adding the four style axes to Elo, as differences or as matchup interactions,
  doesn't improve out-of-sample log loss or Brier score. That holds against both the baseline and an improved Elo. No matchup
  interaction holds up.
- **One pre-specified follow-up passed, modestly.** The difference between fighters in **takedown attempts per 15 minutes** improves
  2019+ predictions over the improved Elo (Δ log loss −0.0027, 95% CI −0.0046 to −0.0008). It's a small main effect of grappling intent,
  not a matchup effect, and it comes with the caveats in the last section.

## Elo (`src/elo.py`)

| Choice | Value | Why |
|---|---|---|
| Fights used | all UFC fights in the database, 1994 (UFC 2) onward | ratings warm up before the 2000+ analysis window; UFC 1 isn't in the data |
| Expected score | 1 / (1 + 10^((R_B − R_A)/400)) | standard logistic Elo |
| Update order | by **event date**: every fight on a date is predicted from pre-date ratings, and updates are applied after the date | the rating for any fight depends only on earlier dates (leak-free); no need to guess fight order within a card; handles 1990s tournament nights, where updates are summed |
| Wins | score 1 / 0 | |
| Draws | score 0.5 each | a draw says the fighters were roughly even, which is informative, especially for an upset draw |
| No contests | no rating change (still listed with pre-fight ratings) | the result was voided; nothing was learned about who is better |
| Tuning window | log loss of the Elo expected score on decisive fights, 2005-01-01 to 2018-12-31 | all settings are chosen before the 2019+ test period |
| Not included | inactivity decay, weight-class changes, non-UFC fights | kept simple and explainable; possible extensions |

### Baseline Elo (`ELO_K`)

Everyone starts at 1500, with one constant **K = 80**. The grid minimum sits in a flat region (64–96).

| K | 16 | 32 | 48 | 64 | **80** | 96 | 128 |
|---|---|---|---|---|---|---|---|
| log loss, 2005–18 | 0.6868 | 0.6827 | 0.6802 | 0.6788 | **0.6782** | 0.6785 | 0.6813 |

### Improved Elo (`ELO_PARAMS`, used by the matchup analysis)

Three refinements, each tuned on its own (with K retuned), then all three together (`elo.tune_variants`). The best setting
per variant is committed in [`elo_tuning.csv`](elo_tuning.csv); `python src/elo.py --tune` reruns the search (about 40 minutes). Grid: K 40–300; debut offset
−100 to +50, or fixed start; finish multiplier 1–2.5; K decay 0–0.6.

- **Debut rating relative to the roster.** A **lower fixed starting rating does nothing** in zero-sum Elo. Every fighter debuts at
  the same value, so all ratings shift together and only differences matter (checked: 1300 and 1500 give identical predictions).
  The meaningful version is that a newcomer starts at the **mean rating of the active roster** (fighters with a UFC fight in the past
  2 years) minus an offset.
- **Larger updates for finishes.** KO/TKO, submission and doctor's-stoppage wins update by K × multiplier. Decisions, DQs and draws update by K.
- **K shrinking with experience.** Each fighter's K is k × (prior UFC fights + 1)^−decay. Updates are no longer zero-sum when the two
  fighters' experience differs.

| variant | log loss, 2005–18 | Δ vs baseline | setting |
|---|---|---|---|
| baseline (constant K, fixed start) | 0.6782 | | K = 80 |
| + debut rating relative to roster | 0.6761 | −0.0022 | K = 80, offset 0 |
| + larger updates for finishes | 0.6781 | −0.0002 | K = 80, finishes × 1.25 |
| + K shrinking with experience | 0.6776 | −0.0007 | K = 100, decay 0.2 |
| **all three combined (chosen)** | **0.6750** | **−0.0032** | **K = 100, offset 0, finishes × 1.25, decay 0.2** |

- **Best offset is 0.** Newcomers should start *at* the roster mean, not below it. Under plain Elo they effectively start below it,
  because the active roster's average drifts above 1500 (fighters who keep winning stay), and that undervalues debutants.
- **Every chosen value is inside its grid**, not at an edge.
- **Out of sample** (2019+ decisive fights, never used for tuning): log loss 0.6827 → **0.6697**, and the favourite wins 55.6% → **58.0%**.
  The improvement is real, not a tuning artefact.

## Evaluation design (`src/matchups.py`)

- **Frozen style model.** The style axes (PCA + varimax, 4 axes) are fitted on profiles built **as of 2019-01-01**: 706 eligible
  fighters, using only earlier fights. They are aligned to the named axes by anchor features, without outside data. Congruence with the
  full-data axes is 0.989–0.997, so the axes mean the same thing.
- **Per-fight features.** For every fight, train or test, both fighters' profiles are built **as of the fight date** with
  `features.build_profiles(as_of=date)`. That means shrinkage priors, era-relative standardization and career midpoints all come from
  earlier fights only. The profiles are transformed with the frozen model. Elo is the pre-fight rating.
- **Split.**
  - Train: decisive fights from 2006-01-01 to 2018-12-31.
  - Test: decisive fights from 2019-01-01 to 2026-08-08.
  - Draws and no contests are excluded from the outcome models.
- **Inclusion.** Both fighters must have **≥ 3** prior UFC fights since 2000: 1,772 training and 2,015 test fights. The feature definitions
  themselves are unchanged (priors and standardization references still use the 5-fight eligibility rule); only fight inclusion differs.
  Results at **≥ 5** prior fights (1,074 / 1,392 fights) are reported too.
- **Order invariance.** Each fight is stored with a random A/B orientation (seeded). All models have **no intercept** and only
  antisymmetric features, so P(A beats B) = 1 − P(B beats A) exactly. Verified: the largest |P(A>B) + P(B>A) − 1| on swapping is 2e-16.
  The models therefore ignore corner assignment, which carries information of its own (red wins 57% of 2019+ decisive fights).
- **Models** (logistic regression):
  1. `elo`: (Elo_A − Elo_B)/100
  2. `elo+style`: + the four axis differences a_k − b_k
  3. `elo+inter`: + the six antisymmetric interactions a_i·b_j − a_j·b_i, for i < j
- **Regularization.** An L2 penalty applies to style terms only; Elo is never penalized.
  - The penalty on axis differences is tuned for model 2.
  - Model 3 keeps that penalty and tunes a separate one on the interactions.
  - Tuning uses time-based validation **inside the training period**: fit on 2006–15, validate on 2016–18. The final model is refit on 2006–18.
- **Uncertainty.** Test metrics get 95% CIs from a paired bootstrap over test fights (2,000 resamples). Coefficients get 95% CIs from a
  bootstrap over fights (500 resamples).
- **What counts as holding up:** improving test-set log loss with a CI excluding zero, **or** a coefficient with the same sign and a
  CI excluding zero when estimated separately in the training and test periods. Nothing is claimed from training-period fits alone.

## Results

Test set (2019+), Δ log loss of each style model against the Elo-only model fitted on the same fights:

| prior fights | Elo | Elo-only test log loss | + axis differences: Δ [95% CI] | + interactions: Δ [95% CI] |
|---|---|---|---|---|
| ≥ 3 | **improved** | 0.6685 | −0.0009 [−0.0039, +0.0021] | −0.0009 [−0.0039, +0.0021] |
| ≥ 3 | baseline | 0.6819 | −0.0004 [−0.0032, +0.0025] | −0.0003 [−0.0032, +0.0025] |
| ≥ 5 | **improved** | 0.6656 | +0.0023 [−0.0024, +0.0067] | +0.0023 [−0.0024, +0.0067] |
| ≥ 5 | baseline | 0.6793 | +0.0041 [−0.0003, +0.0081] | +0.0041 [−0.0003, +0.0082] |

**The conclusions don't change with the improved Elo.** Elo itself got much better (0.6819 → 0.6685), but adding style still doesn't
measurably improve predictions, and with 5+ prior fights it is slightly worse. Brier scores agree.

- **Interactions overfit.** Unregularized, with the improved Elo, the interaction model has the best training log loss (0.6688) and a test
  log loss of 0.6700, worse than Elo alone (0.6685). Validation loss falls monotonically as the interaction penalty grows. The tuned penalty
  (1e5, the top of the grid) shrinks all interactions to about 0, so model 3 equals model 2.
- **Calibration** of the fitted models on the test set is reasonable.

### Coefficients (log-odds per SD of axis difference; unpenalized, improved Elo included, estimated separately per period)

| term | 2006–18 | 2019+ | holds up? |
|---|---|---|---|
| Elo difference (per 100) | +0.45 [0.33, 0.56] | +0.57 [0.45, 0.70] | yes |
| Δ grappler vs. volume striker | +0.08 [−0.00, 0.16] | +0.14 [0.07, 0.21] | **suggestive only** |
| Δ clinch and body work | −0.04 [−0.12, 0.03] | +0.01 [−0.06, 0.08] | no |
| Δ submission hunting | −0.08 [−0.16, 0.01] | +0.01 [−0.06, 0.08] | no |
| Δ leg kicking | −0.01 [−0.07, 0.07] | +0.02 [−0.04, 0.09] | no |
| clinch and body work × leg kicking | −0.04 [−0.11, 0.03] | −0.07 [−0.14, −0.00] | no (2019+ only) |
| submission hunting × leg kicking | −0.03 [−0.10, 0.04] | +0.07 [0.00, 0.14] | no (sign flips) |
| other four interactions | CIs include 0 | CIs include 0 | no |

Two interactions are nominally significant in the 2019+ period alone, and neither is in 2006–18. With ten style terms, one or two
nominal hits in one period is about what chance produces. No interaction improves out-of-sample predictions, so no matchup effect is claimed.

## Interpretation

- **No matchup effects.** There's no evidence that particular style pairings ("grapplers beat leg kickers" and so on) change outcomes beyond
  what Elo already captures, at least not effects large enough to detect in about 2,000 fights. That holds against both Elo versions.
- **Axis 1 (grappler vs. volume striker):** positive in both periods, but not established. It isn't significant in 2006–18, and adding the
  axes doesn't improve 2019+ predictions. It may partly be skill: axis 1 is built mostly from control time and ground striking, which
  fighters accumulate by winning exchanges. That motivated the single follow-up test below.

## Pre-specified follow-up test: takedown attempts only

Specified (here and in `matchups.GRAPPLING_TEST`) **before** running it, as the **only** follow-up test of the style question.

**Why.** The one suggestive result above, the grappler end of axis 1, may partly reflect skill, because axis 1 is built mostly from
control time and ground striking, which fighters earn by winning exchanges. Takedown *attempts* measure what a fighter tries to do,
whether or not it works.

| | |
|---|---|
| Measure | `z_td_att_per15`: shrunk takedown attempts per 15 minutes, era-relative z-score. No control time, no ground strikes |
| Feature | fighter A minus fighter B, from profiles built as of the fight date |
| Model | logistic, no intercept, unpenalized: improved Elo difference + takedown-attempt difference |
| Baseline | improved Elo difference only, fitted on the same fights |
| Train / test | decisive fights 2006–2018 / 2019 on; both fighters ≥ 3 prior UFC fights since 2000 |
| Primary outcome | Δ mean test log loss (model − baseline), 95% paired bootstrap CI (2,000 resamples, seed 0) |
| Pass rule | improves predictions only if the whole CI is below 0 |
| Secondary (descriptive) | Δ Brier; the takedown coefficient with 95% CI fitted on train and, separately, on test |

The result is reported below whatever it is.

### Result: passes (run once)

| | value |
|---|---|
| fights | 1,772 train, 2,015 test |
| test log loss | improved Elo 0.6685 → + takedown attempts 0.6658 |
| **Δ log loss (primary)** | **−0.0027, 95% CI [−0.0046, −0.0008]**: whole CI below 0, **passes** |
| Δ Brier | −0.0013, 95% CI [−0.0022, −0.0004] |
| takedown coefficient, fitted on 2006–18 (used for the test predictions) | +0.068 per SD, 95% CI [−0.011, +0.138] |
| takedown coefficient, fitted on 2019+ (descriptive) | +0.121 per SD, 95% CI [+0.053, +0.190] |

**What it means.** A fighter who attempts more takedowns than their opponent, relative to fighters in their division and era, wins
slightly more often than their Elo predicts. With the 2006–18 coefficient, a 1 SD edge moves an even fight to about 51.7%. The
log-loss gain is about a fifth of what improving Elo bought (0.0027 vs 0.0134). It is a **main effect of grappling intent**, not a matchup
effect: it doesn't say which style beats which.

**Caveats, which limit how much weight it can bear:**

1. **The test set wasn't untouched.** This hypothesis was chosen *after* the earlier analysis had shown a positive 2019+ coefficient for
   axis 1, of which takedown attempts are one input. The test is pre-specified and was run once, but the period it was run on had
   already influenced the choice of hypothesis. This makes it weaker than a test on new data.
2. **The training-period estimate alone isn't significant.** Its CI includes 0, so the evidence rests mostly on the 2019+ fights.
3. **Attempts can still partly reflect skill.** Fighters attempt takedowns more when they expect to land them, even with control time removed.
4. **The clean replication** is the same pre-specified test on fights after 2026-08-08, once a newer dataset version includes them.

## Leakage summary

Elo (all settings), profiles, the style model and all model tuning use only data before each fight or before 2019-01-01. The remaining
full-data choices made earlier in the project are:

- the shrinkage priors
- the era window
- the number of axes
- the feature set

These are hyperparameter-level choices, not outcome information, but they weren't re-derived on pre-2019 data. The follow-up test's
hypothesis was informed by an earlier look at the 2019+ period (caveat 1 above).
