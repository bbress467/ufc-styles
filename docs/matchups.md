# Do style matchups predict outcomes beyond skill?

Code: [`src/elo.py`](../src/elo.py), [`src/matchups.py`](../src/matchups.py). Results: [`notebooks/07_matchups.ipynb`](../notebooks/07_matchups.ipynb).

**Short answer: not detectably.** On ~2,000 UFC fights from 2019 on, adding the four style axes (as differences, or as matchup interactions)
to an Elo rating doesn't improve out-of-sample log loss or Brier score. No matchup interaction holds up. One main effect, being further
toward the grappler end of axis 1 than one's opponent, is suggestive but doesn't meet the out-of-sample bar.

## Elo (`src/elo.py`)

| Choice | Value | Why |
|---|---|---|
| Fights used | all UFC fights in the database, 1994 (UFC 2) onward | ratings warm up before the 2000+ analysis window; UFC 1 isn't in the data |
| Starting rating | 1500 for everyone | standard; newcomers' true level is unknown |
| Expected score | 1 / (1 + 10^((R_B − R_A)/400)) | standard logistic Elo |
| Update order | by **event date**: every fight on a date is predicted from pre-date ratings, and updates are applied after the date | the rating for any fight depends only on earlier dates (leak-free); no need to guess fight order within a card; handles 1990s tournament nights, where updates are summed |
| Wins | score 1 / 0 | |
| Draws | score 0.5 each | a draw says the fighters were roughly even, which is informative, especially for an upset draw |
| No contests | no rating change (still listed with pre-fight ratings) | the result was voided; nothing was learned about who is better |
| K | **80**, chosen by log loss of the Elo expected score on decisive fights from 2005-01-01 to 2018-12-31 | nothing from the 2019+ test period is used; the grid minimum sits in a flat region (64–96) |
| Not included | inactivity decay, weight-class changes, margin of victory, non-UFC fights | kept simple and explainable; possible extensions |

K tuning (log loss on 4,430 decisive fights, 2005–18):

| K | 16 | 32 | 48 | 64 | **80** | 96 | 128 |
|---|---|---|---|---|---|---|---|
| log loss | 0.6868 | 0.6827 | 0.6802 | 0.6788 | **0.6782** | 0.6785 | 0.6813 |

A large K means ratings move quickly, which suits a sport where fighters' form changes fast and careers are short. Elo is a weak
predictor here. The favourite wins about 55% of decisive fights in both 2005–18 and 2019+, against a log loss of 0.693 for a coin flip.

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

Test set, both fighters with ≥ 3 prior fights (2,015 fights):

| model | log loss [95% CI] | Brier [95% CI] | Δ log loss vs Elo [95% CI] |
|---|---|---|---|
| raw Elo expected score | 0.6842 [0.6749, 0.6936] | 0.2456 | |
| `elo` | 0.6819 [0.6749, 0.6889] | 0.2445 [0.2411, 0.2479] | |
| `elo+style` | 0.6815 [0.6739, 0.6890] | 0.2443 [0.2407, 0.2480] | −0.0004 [−0.0032, +0.0025] |
| `elo+inter` | 0.6815 [0.6739, 0.6890] | 0.2443 [0.2407, 0.2480] | −0.0003 [−0.0032, +0.0025] |

With ≥ 5 prior fights (1,392 test fights), adding style made predictions slightly **worse**: Δ log loss +0.0041 [−0.0003, +0.0081].

- **Interactions overfit.** Unregularized, the interaction model has the best training log loss (0.6731) and the worst test log loss (0.6837,
  worse than Elo's 0.6819). Validation loss falls monotonically as the interaction penalty grows. The tuned penalty (1e5, the top of the
  grid) shrinks all interactions to about 0, so model 3 equals model 2.
- **Calibration** of the fitted models on the test set is reasonable. Raw Elo is slightly overconfident at the extremes.

### Coefficients (log-odds per SD of axis difference; unpenalized, Elo included, estimated separately per period)

| term | 2006–18 | 2019+ | holds up? |
|---|---|---|---|
| Elo difference (per 100) | +0.42 [0.29, 0.53] | +0.35 [0.22, 0.47] | yes |
| Δ grappler vs. volume striker | +0.07 [−0.02, 0.15] | +0.16 [0.09, 0.24] | **suggestive only** |
| Δ clinch and body work | −0.03 [−0.11, 0.04] | +0.00 [−0.06, 0.07] | no |
| Δ submission hunting | −0.07 [−0.15, 0.02] | +0.04 [−0.03, 0.12] | no (sign flips) |
| Δ leg kicking | −0.01 [−0.08, 0.07] | +0.03 [−0.04, 0.09] | no |
| all six interactions | CIs include 0 | CIs include 0 | no |

Submission hunting × leg kicking flips sign between periods. Clinch and body work × leg kicking is negative in both, but not significant
in either.

## Interpretation

- **No matchup effects.** There's no evidence that particular style pairings ("grapplers beat leg kickers" and so on) change outcomes beyond
  what Elo already captures, at least not effects large enough to detect in ~2,000 fights.
- **A possible style main effect, not established.** The grappler end of axis 1 has a positive coefficient in both periods. But it isn't
  significant in 2006–18, and using it doesn't measurably improve 2019+ predictions. It may also partly be skill: axis 1 is built mostly
  from control time and ground striking, which fighters accumulate by winning exchanges.
- **Next steps** if this is pursued:
  - rebuild axis 1 from intent-only measures (takedown *attempts*) and retest
  - add an expanding-window refit of the style model
  - use betting odds as a stronger skill baseline than Elo

## Leakage summary

Elo, profiles, the style model and all tuning use only data before each fight or before 2019-01-01. The remaining full-data choices made
earlier in the project are:

- the shrinkage priors
- the era window
- the number of axes
- the feature set

These are hyperparameter-level choices, not outcome information, but they weren't re-derived on pre-2019 data.
