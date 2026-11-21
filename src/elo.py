"""Leak-free Elo ratings for UFC fighters, updated fight by fight (see docs/matchups.md, "Elo").

    import elo
    fights = elo.load_ufc_fights()
    ratings = elo.run_elo(fights, **elo.ELO_PARAMS)   # one row per fight: pre-fight ratings and expected score

Choices
-------
* All UFC fights in the database (UFC 2, 1994 onward), so ratings warm up before the 2000+ analysis window.
  Only UFC fights count; fighters' records elsewhere are unknown to the model.
* Expected score for A against B: 1 / (1 + 10^((R_B - R_A) / 400)).
* Fights are processed **by event date**. All fights on a date are predicted from the ratings *before* that date,
  and the updates are applied together afterwards. This makes the rating used for any fight depend only on
  earlier dates (leak-free) and avoids guessing the order of fights within a card. It also handles 1990s
  tournament nights, where a fighter fought several times in one evening: their updates are summed.
* Result scoring: win = 1, loss = 0, **draw = 0.5** each (a draw is informative: the fighters were roughly even).
  **No contests don't update** ratings (the result was voided), but the fight is still listed with its pre-fight ratings.
* Two versions:
    - baseline (`ELO_K`): everyone starts at 1500, one constant K = 80.
    - improved (`ELO_PARAMS`, used by the matchup analysis): K = 100 shrinking with experience as
      K * (prior UFC fights + 1) ** -0.2; finishes (KO/TKO, submission, doctor's stoppage) update 1.25x as much as
      decisions; a debuting fighter starts at the mean rating of the active roster (fighters with a UFC fight in the
      past 2 years). A lower *fixed* starting rating is a no-op in zero-sum Elo, so the debut rating is set relative
      to the roster instead; the tuned offset below the roster mean is 0.
* All settings are chosen by log loss of the Elo expected score on decisive fights from TUNE_START to TUNE_END
  (2005-01-01 to 2018-12-31), before the 2019+ test period (`tune_k`, `tune_variants`). Ratings still run from 1994.
* No inactivity decay, weight-class adjustment or non-UFC fights (kept simple; see docs/matchups.md).
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "ufc.sqlite"
OUT_PATH = ROOT / "data" / "processed" / "elo_ratings.csv"

INITIAL_RATING = 1500.0
ROSTER_DAYS = 730          # "active roster" for debut ratings: fighters with a UFC fight in the past 2 years
SCALE = 400.0
TUNE_START, TUNE_END = "2005-01-01", "2019-01-01"   # K is tuned on fights in [TUNE_START, TUNE_END)
K_GRID = [8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128]
ELO_K = 80.0          # baseline Elo: constant K chosen by tune_k on 2005-2018 fights (flat optimum 64-96); see docs/matchups.md
# Improved Elo (used by the matchup analysis): chosen by tune_variants on 2005-2018 fights only (docs/matchups.md).
ELO_PARAMS = {"k": 100.0, "debut_offset": 0.0, "finish_mult": 1.25, "k_decay": 0.2}


def load_ufc_fights(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """All UFC fights in the database with result info, sorted by date."""
    with sqlite3.connect(db_path) as con:
        f = pd.read_sql_query(
            "SELECT fight_id, event_id, event_date, red_fighter_id, blue_fighter_id, winner_id, result_status, method "
            "FROM fights ORDER BY event_date, fight_id", con)
    return f


def expected(r_a, r_b):
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / SCALE))


FINISH_METHODS = ("KO/TKO", "Submission", "TKO - Doctor's Stoppage")


def run_elo(fights: pd.DataFrame, k: float = ELO_K, initial: float = INITIAL_RATING,
            finish_mult: float = 1.0, k_decay: float = 0.0, debut_offset: float | None = None,
            roster_days: int = ROSTER_DAYS) -> pd.DataFrame:
    """Pre-fight ratings for every fight, updating date by date. Returns one row per fight.

    initial     : rating of a debuting fighter when debut_offset is None. With plain zero-sum Elo this is a
                  no-op (every fighter starts at the same value, so all ratings shift together).
    debut_offset: if set, a debuting fighter starts at the mean pre-date rating of the *active roster*
                  (fighters with a UFC fight in the previous `roster_days` days) minus debut_offset
                  (initial - debut_offset if there is no roster yet). This is what makes "newcomers start lower" matter.
    finish_mult : decisive wins by KO/TKO, submission or doctor's stoppage update by finish_mult * K;
                  decisions, DQs and draws by K.
    k_decay     : each fighter's K shrinks with experience: K_i = k * (n_i + 1) ** -k_decay, where n_i is the
                  number of earlier UFC fights. 0 = constant K. Updates are therefore not zero-sum when the two
                  fighters' experience differs.

    Columns: elo_red, elo_blue (pre-fight), exp_red (Elo expected score for red), red_score (1/0/0.5, NaN for NC),
    prior_ufc_red / prior_ufc_blue (number of earlier UFC fights, any result), plus the input columns.
    """
    rating: dict[str, float] = {}
    count: dict[str, int] = defaultdict(int)
    last: dict[str, np.datetime64] = {}
    out = []
    for date, day in fights.groupby("event_date", sort=True):
        if debut_offset is not None:
            cutoff = np.datetime64(date) - np.timedelta64(roster_days, "D")
            active = [rating[f] for f, t in last.items() if t >= cutoff]
            debut = (np.mean(active) if active else initial) - debut_offset
        else:
            debut = initial
        for f in pd.unique(day[["red_fighter_id", "blue_fighter_id"]].to_numpy().ravel()):
            if f not in rating:
                rating[f] = debut
        delta: dict[str, float] = defaultdict(float)
        for r in day.itertuples(index=False):
            a, b = r.red_fighter_id, r.blue_fighter_id
            ra, rb = rating[a], rating[b]
            e = expected(ra, rb)
            if r.result_status == "win":
                s = 1.0 if r.winner_id == a else 0.0
                m = finish_mult if r.method in FINISH_METHODS else 1.0
            elif r.result_status == "draw":
                s, m = 0.5, 1.0
            else:                                   # no contest: no update
                s, m = np.nan, 0.0
            if not np.isnan(s):
                ka = k * (count[a] + 1) ** -k_decay
                kb = k * (count[b] + 1) ** -k_decay
                delta[a] += m * ka * (s - e)
                delta[b] -= m * kb * (s - e)
            out.append((r.fight_id, ra, rb, e, s, count[a], count[b]))
        for a, d in delta.items():
            rating[a] += d
        for r in day.itertuples(index=False):
            count[r.red_fighter_id] += 1
            count[r.blue_fighter_id] += 1
            last[r.red_fighter_id] = last[r.blue_fighter_id] = np.datetime64(date)
    res = pd.DataFrame(out, columns=["fight_id", "elo_red", "elo_blue", "exp_red", "red_score", "prior_ufc_red", "prior_ufc_blue"])
    return fights.merge(res, on="fight_id", validate="one_to_one")


def log_loss(p: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def tune_k(fights: pd.DataFrame, grid=K_GRID, start: str = TUNE_START, end: str = TUNE_END) -> pd.DataFrame:
    """Log loss and Brier score of the Elo expected score on decisive fights in [start, end), for each K."""
    rows = []
    for k in grid:
        r = run_elo(fights, k)
        w = r[(r["event_date"] >= start) & (r["event_date"] < end) & (r["result_status"] == "win")]
        y, p = w["red_score"].to_numpy(), w["exp_red"].to_numpy()
        rows.append({"k": k, "log_loss": log_loss(p, y), "brier": float(np.mean((p - y) ** 2)), "fights": len(w)})
    return pd.DataFrame(rows).set_index("k")


VARIANT_GRID = {
    "k": [40, 60, 80, 100, 120, 150, 200, 250, 300],
    "debut_offset": [None, -100, -75, -50, -25, 0, 25, 50],
    "finish_mult": [1.0, 1.25, 1.5, 1.75, 2.0, 2.5],
    "k_decay": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
}


def score(fights: pd.DataFrame, start: str = TUNE_START, end: str = TUNE_END, **params) -> float:
    """Log loss of the Elo expected score on decisive fights in [start, end) for one setting."""
    r = run_elo(fights, **params)
    w = r[(r["event_date"] >= start) & (r["event_date"] < end) & (r["result_status"] == "win")]
    return log_loss(w["exp_red"].to_numpy(), w["red_score"].to_numpy())


def tune_variants(fights: pd.DataFrame, start: str = TUNE_START, end: str = TUNE_END, grid: dict = VARIANT_GRID) -> pd.DataFrame:
    """Grid search over each refinement on its own (K retuned) and all three combined. One row per variant (best setting)."""
    import itertools
    base = {"debut_offset": [None], "finish_mult": [1.0], "k_decay": [0.0]}
    variants = {
        "baseline (constant K, fixed start)": base,
        "+ debut rating relative to roster": {**base, "debut_offset": grid["debut_offset"]},
        "+ larger updates for finishes": {**base, "finish_mult": grid["finish_mult"]},
        "+ K shrinking with experience": {**base, "k_decay": grid["k_decay"]},
        "all three combined": {k: grid[k] for k in ("debut_offset", "finish_mult", "k_decay")},
    }
    rows, cache = [], {}
    for name, g in variants.items():
        best = None
        for d, m, kd in itertools.product(g["debut_offset"], g["finish_mult"], g["k_decay"]):
            for k in grid["k"]:
                key = (k, d, m, kd)
                if key not in cache:
                    cache[key] = score(fights, start, end, k=k, debut_offset=d, finish_mult=m, k_decay=kd)
                if best is None or cache[key] < best[0]:
                    best = (cache[key], key)
        ll, (k, d, m, kd) = best
        rows.append({"variant": name, "log_loss": ll, "k": k, "debut_offset": d, "finish_mult": m, "k_decay": kd})
    return pd.DataFrame(rows).set_index("variant")


# Grid search results (best setting per variant), committed so the tuning evidence is visible without rerunning it.
TUNING_PATH = ROOT / "docs" / "elo_tuning.csv"


def main(tune: bool = False) -> None:
    """Write pre-fight ratings for every UFC fight using ELO_PARAMS.

    With tune=True (--tune), first rerun the settings search on 2005-2018 fights (about 40 minutes), print it,
    and save it to docs/elo_tuning.csv. The search doesn't change ELO_PARAMS: update the constant by hand if the
    best setting changes.
    """
    fights = load_ufc_fights()
    if tune:
        print(tune_k(fights).round(4).to_string())
        v = tune_variants(fights)
        v.to_csv(TUNING_PATH, float_format="%.6f")
        print(v.to_string())
        best = v.loc["all three combined"]
        print(f"wrote {TUNING_PATH.relative_to(ROOT)}; best combined setting: "
              f"k={best['k']:g}, debut_offset={best['debut_offset']}, finish_mult={best['finish_mult']:g}, k_decay={best['k_decay']:g}")
    print(f"ELO_PARAMS = {ELO_PARAMS}")
    run_elo(fights, **ELO_PARAMS).to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    import sys
    main(tune="--tune" in sys.argv)
