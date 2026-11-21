"""Do style matchups predict fight outcomes beyond Elo? Time-split, leak-free evaluation (see docs/matchups.md).

    import matchups as M
    data = M.build_dataset()                 # one row per decisive UFC fight from TRAIN_START, cached
    res = M.evaluate(data, min_prior=3)      # fits the three models on pre-2019 fights, scores 2019+ fights

Design
------
* Style axes: fitted once on profiles built as of SPLIT_DATE (2019-01-01) and frozen (`style_axes.fit_style_axes`,
  aligned to the named axes). Every fight, train or test, uses both fighters' profiles built as of the fight date
  (only earlier fights), transformed with that frozen model.
* Elo: pre-fight ratings from `elo.run_elo` (K chosen on pre-2019 fights).
* Train: decisive fights from TRAIN_START to SPLIT_DATE. Test: decisive fights from SPLIT_DATE on.
  Draws and no contests are excluded from the outcome models (Elo still uses draws).
* Inclusion: both fighters have at least `min_prior` UFC fights since 2000 before the fight (3 by default; 5 is reported too).
* Fighter order: each fight is stored with a random A/B orientation (seeded). All models have no intercept and only
  antisymmetric features (differences and a_i*b_j - a_j*b_i), so P(A beats B) = 1 - P(B beats A) exactly.
* Models (logistic, no intercept):
    elo        : elo diff
    elo+style  : elo diff + 4 axis differences
    elo+inter  : elo diff + 4 axis differences + 6 antisymmetric interactions a_i*b_j - a_j*b_i (i < j)
  An L2 penalty on the style coefficients (never on Elo) is tuned by time-based validation inside the training
  period: fit on TRAIN_START..VAL_START, validate on VAL_START..SPLIT_DATE. Then refit on the whole training period.
"""
from __future__ import annotations

import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import elo
import features as F
import style_axes as A

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "data" / "processed" / "matchup_dataset.pkl"
MODEL_DIR = ROOT / "data" / "processed" / "models"

SPLIT_DATE = "2019-01-01"
TRAIN_START = "2006-01-01"
VAL_START = "2016-01-01"
MIN_PRIOR = 3
SEED = 0
AXES = ["axis_1", "axis_2", "axis_3", "axis_4"]
PAIRS = list(combinations(range(4), 2))
LAMBDA_GRID = [0.0, 1, 3, 10, 30, 100, 300, 1000, 3000, 1e4, 3e4, 1e5]
ELO_SCALE = 100.0       # elo difference enters as (elo_a - elo_b) / 100


# --------------------------------------------------------------------------- dataset
def frozen_axes(split_date: str = SPLIT_DATE, fighter_fights: pd.DataFrame | None = None) -> A.StyleAxes:
    """Style axes fitted on profiles as of split_date (eligible fighters, only earlier fights), aligned to names."""
    return A.fit_style_axes(F.build_profiles(split_date, fighter_fights), A.CHOSEN_AXES, as_of=split_date)


def build_dataset(refresh: bool = False, start: str = TRAIN_START, split_date: str = SPLIT_DATE) -> pd.DataFrame:
    """One row per decisive UFC fight on/after `start`, with pre-fight Elo and as-of-date axis scores for both fighters."""
    if CACHE_PATH.exists() and not refresh:
        with open(CACHE_PATH, "rb") as fh:
            cached = pickle.load(fh)
        if cached["meta"] == _meta(start, split_date):
            return cached["data"]
    ff = F.load_fighter_fights()
    axes = frozen_axes(split_date, ff)
    ratings = elo.run_elo(elo.load_ufc_fights(), elo.ELO_K)
    fights = ratings[(ratings["event_date"] >= start) & (ratings["result_status"] == "win")].copy()
    rows = []
    for date, day in fights.groupby("event_date"):
        prof = F.build_profiles(date, ff)
        sc = axes.transform(prof, eligible_only=False).set_index("fighter_id")
        for r in day.itertuples(index=False):
            a, b = r.red_fighter_id, r.blue_fighter_id
            if a not in sc.index or b not in sc.index:
                continue                                    # no prior UFC fight since 2000, or no division yet
            rows.append({"fight_id": r.fight_id, "event_date": date, "red": a, "blue": b,
                         "red_won": int(r.winner_id == a), "elo_red": r.elo_red, "elo_blue": r.elo_blue,
                         "n_red": int(sc.at[a, "n_fights"]), "n_blue": int(sc.at[b, "n_fights"]),
                         **{f"red_{x}": sc.at[a, x] for x in AXES}, **{f"blue_{x}": sc.at[b, x] for x in AXES}})
    d = pd.DataFrame(rows)
    # random A/B orientation (seeded), so neither corner is systematically "A"
    flip = np.random.default_rng(SEED).random(len(d)) < 0.5
    out = pd.DataFrame({"fight_id": d["fight_id"], "event_date": d["event_date"]})
    # (A column, B column, red column, blue column)
    pairs = [("a", "b", "red", "blue"), ("elo_a", "elo_b", "elo_red", "elo_blue"), ("n_a", "n_b", "n_red", "n_blue")] + \
            [(f"a_{x}", f"b_{x}", f"red_{x}", f"blue_{x}") for x in AXES]
    for col_a, col_b, col_r, col_bl in pairs:
        out[col_a] = np.where(flip, d[col_bl], d[col_r])
        out[col_b] = np.where(flip, d[col_r], d[col_bl])
    out["y"] = np.where(flip, 1 - d["red_won"], d["red_won"])      # 1 = A won
    out["a_is_red"] = ~flip
    out["set"] = np.where(out["event_date"] < split_date, "train", "test")
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as fh:
        pickle.dump({"meta": _meta(start, split_date), "data": out}, fh)
    return out


def _meta(start, split_date):
    return {"start": start, "split": split_date, "elo_k": elo.ELO_K, "prior_minutes": F.PRIOR_MINUTES,
            "feature_priors": dict(F.FEATURE_PRIORS), "era_window": F.ERA_WINDOW_YEARS, "min_fights": F.MIN_FIGHTS}


# --------------------------------------------------------------------------- design matrices
def design(d: pd.DataFrame, model: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Antisymmetric features for A vs B. Returns (X, column names, term type per column: 'elo' | 'diff' | 'inter')."""
    cols = {"elo_diff": (d["elo_a"] - d["elo_b"]).to_numpy() / ELO_SCALE}
    if model in ("elo+style", "elo+inter"):
        for x in AXES:
            cols[f"d_{x}"] = (d[f"a_{x}"] - d[f"b_{x}"]).to_numpy()
    if model == "elo+inter":
        for i, j in PAIRS:
            ai, aj, bi, bj = (d[f"a_{AXES[i]}"], d[f"a_{AXES[j]}"], d[f"b_{AXES[i]}"], d[f"b_{AXES[j]}"])
            cols[f"x_{AXES[i]}_{AXES[j]}"] = (ai * bj - aj * bi).to_numpy()
    names = list(cols)
    X = np.column_stack([cols[c] for c in names])
    return X, names, np.array(["elo" if c == "elo_diff" else "diff" if c.startswith("d_") else "inter" for c in names])


def swap(d: pd.DataFrame) -> pd.DataFrame:
    """The same fights with A and B exchanged (and the outcome flipped)."""
    s = d.copy()
    for col in ["", "elo_", "n_"] + [f"{x}" for x in AXES]:
        ca, cb = (f"{col}a", f"{col}b") if col in ("", "elo_", "n_") else (f"a_{col}", f"b_{col}")
        s[ca], s[cb] = d[cb].to_numpy(), d[ca].to_numpy()
    s["y"] = 1 - d["y"]
    return s


# --------------------------------------------------------------------------- fitting
def penalty_vector(terms: np.ndarray, lam_diff: float = 0.0, lam_inter: float = 0.0) -> np.ndarray:
    """Per-coefficient L2 weights: Elo is never penalized; axis differences get lam_diff, interactions lam_inter."""
    return np.select([terms == "diff", terms == "inter"], [lam_diff, lam_inter], 0.0).astype(float)


def fit_logistic(X: np.ndarray, y: np.ndarray, lam: np.ndarray | float = 0.0) -> np.ndarray:
    """Logistic regression without intercept; L2 penalty 0.5 * sum(lam_j * beta_j^2) on the summed log-loss."""
    lam = np.broadcast_to(np.asarray(lam, float), (X.shape[1],))

    def f(beta):
        z = X @ beta
        nll = np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(lam * beta ** 2)
        g = X.T @ (1 / (1 + np.exp(-z)) - y) + lam * beta
        return nll, g

    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def predict(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(X @ beta)))


def log_loss(p, y, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def tune_lambda(train: pd.DataFrame, model: str, grid=LAMBDA_GRID, val_start: str = VAL_START,
                lam_diff_fixed: float = 0.0) -> pd.DataFrame:
    """Validation log loss for each lambda: fit on train before val_start, score train from val_start on.

    elo+style: lambda applies to the axis differences. elo+inter: lambda applies to the interactions only,
    with the axis differences held at lam_diff_fixed (the value tuned for elo+style).
    """
    fit_d, val_d = train[train["event_date"] < val_start], train[train["event_date"] >= val_start]
    Xf, _, terms = design(fit_d, model)
    Xv, _, _ = design(val_d, model)
    rows = []
    for lam in grid:
        pv = penalty_vector(terms, lam, 0.0) if model == "elo+style" else penalty_vector(terms, lam_diff_fixed, lam)
        b = fit_logistic(Xf, fit_d["y"].to_numpy(), pv)
        rows.append({"lambda": lam, "val_log_loss": log_loss(predict(Xv, b), val_d["y"].to_numpy()).mean(),
                     "fit_log_loss": log_loss(predict(Xf, b), fit_d["y"].to_numpy()).mean()})
    return pd.DataFrame(rows).set_index("lambda")


MODELS = ["elo", "elo+style", "elo+inter"]


def evaluate(data: pd.DataFrame, min_prior: int = MIN_PRIOR, n_boot: int = 2000, seed: int = SEED) -> dict:
    """Fit the three models on the training period, score the test period, with bootstrap CIs."""
    d = data[(data["n_a"] >= min_prior) & (data["n_b"] >= min_prior)]
    train, test = d[d["set"] == "train"], d[d["set"] == "test"]
    out = {"n_train": len(train), "n_test": len(test), "min_prior": min_prior, "fits": {}, "tuning": {}, "pred": {}}
    yt = test["y"].to_numpy()
    lam_diff = 0.0
    for m in MODELS:
        tun = None if m == "elo" else tune_lambda(train, m, lam_diff_fixed=lam_diff)
        lam = 0.0 if tun is None else float(tun["val_log_loss"].idxmin())
        if m == "elo+style":
            lam_diff = lam
        X, names, terms = design(train, m)
        pv = penalty_vector(terms, lam_diff, lam if m == "elo+inter" else 0.0)
        beta = fit_logistic(X, train["y"].to_numpy(), pv)
        Xt, _, _ = design(test, m)
        p = predict(Xt, beta)
        p_swap = predict(design(swap(test), m)[0], beta)
        out["fits"][m] = {"beta": pd.Series(beta, index=names), "lambda": lam, "penalty": pd.Series(pv, index=names),
                          "max_swap_error": float(np.max(np.abs(p + p_swap - 1)))}
        out["tuning"][m] = tun
        out["pred"][m] = p
    out["pred"]["raw_elo"] = elo.expected(test["elo_a"].to_numpy(), test["elo_b"].to_numpy())
    # bootstrap over test fights (paired across models)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(test), size=(n_boot, len(test)))
    rows, diffs = [], []
    ll = {m: log_loss(p, yt) for m, p in out["pred"].items()}
    br = {m: (p - yt) ** 2 for m, p in out["pred"].items()}
    for m in out["pred"]:
        bl, bb = ll[m][idx].mean(axis=1), br[m][idx].mean(axis=1)
        rows.append({"model": m, "log_loss": ll[m].mean(), "ll_lo": np.quantile(bl, .025), "ll_hi": np.quantile(bl, .975),
                     "brier": br[m].mean(), "brier_lo": np.quantile(bb, .025), "brier_hi": np.quantile(bb, .975),
                     "accuracy": float(np.mean((out["pred"][m] > 0.5) == (yt == 1)))})
        if m not in ("elo", "raw_elo"):
            dl = (ll[m] - ll["elo"])[idx].mean(axis=1)
            db = (br[m] - br["elo"])[idx].mean(axis=1)
            diffs.append({"model": m, "d_log_loss": (ll[m] - ll["elo"]).mean(), "dll_lo": np.quantile(dl, .025),
                          "dll_hi": np.quantile(dl, .975), "p_better": float(np.mean(dl < 0)),
                          "d_brier": (br[m] - br["elo"]).mean(), "dbr_lo": np.quantile(db, .025), "dbr_hi": np.quantile(db, .975)})
    out["metrics"] = pd.DataFrame(rows).set_index("model")
    out["vs_elo"] = pd.DataFrame(diffs).set_index("model")
    out["test"] = test
    out["train"] = train
    return out


def coefficient_cis(d: pd.DataFrame, model: str, lam: np.ndarray | float = 0.0, n_boot: int = 500, seed: int = SEED) -> pd.DataFrame:
    """Point estimates and 95% bootstrap CIs (resampling fights) for a model fitted on `d` with penalty `lam`."""
    X, names, _ = design(d, model)
    y = d["y"].to_numpy()
    est = fit_logistic(X, y, lam)
    rng = np.random.default_rng(seed)
    boots = np.array([fit_logistic(X[i], y[i], lam) for i in (rng.integers(0, len(d), len(d)) for _ in range(n_boot))])
    return pd.DataFrame({"estimate": est, "lo": np.quantile(boots, .025, axis=0), "hi": np.quantile(boots, .975, axis=0)},
                        index=names)


if __name__ == "__main__":
    data = build_dataset(refresh=True)
    print(data["set"].value_counts().to_dict())
    for mp in (3, 5):
        r = evaluate(data, mp, n_boot=1000)
        print(f"\nmin prior fights {mp}: train {r['n_train']}, test {r['n_test']}")
        print(r["metrics"].round(4).to_string())
        print(r["vs_elo"].round(4).to_string())
        print({m: (f["lambda"], round(f["max_swap_error"], 12)) for m, f in r["fits"].items()})
