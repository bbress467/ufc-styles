"""Leak-free Elo ratings for UFC fighters, updated fight by fight (see docs/matchups.md, "Elo").

    import elo
    fights = elo.load_ufc_fights()
    ratings = elo.run_elo(fights, k=elo.ELO_K)   # one row per fight: pre-fight ratings and expected score

Choices
-------
* All UFC fights in the database (UFC 2, 1994 onward), so ratings warm up before the 2000+ analysis window.
  Only UFC fights count; fighters' records elsewhere are unknown to the model.
* Every fighter starts at 1500. Expected score for A against B: 1 / (1 + 10^((R_B - R_A) / 400)).
* Fights are processed **by event date**. All fights on a date are predicted from the ratings *before* that date,
  and the updates are applied together afterwards. This makes the rating used for any fight depend only on
  earlier dates (leak-free) and avoids guessing the order of fights within a card. It also handles 1990s
  tournament nights, where a fighter fought several times in one evening: their updates are summed.
* Result scoring: win = 1, loss = 0, **draw = 0.5** each (a draw is informative: the fighters were roughly even).
  **No contests don't update** ratings (the result was voided), but the fight is still listed with its pre-fight ratings.
* One K for every fight. K is chosen by log loss of the Elo expected score on decisive fights from
  TUNE_START to TUNE_END (2005-01-01 to 2018-12-31), all before the 2019+ test period. Ratings still run from 1994.
  Nothing from 2019 on is used to choose K.
* No inactivity decay, weight-class adjustment or margin-of-victory weighting (kept simple; see docs).
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
SCALE = 400.0
TUNE_START, TUNE_END = "2005-01-01", "2019-01-01"   # K is tuned on fights in [TUNE_START, TUNE_END)
K_GRID = [8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128]
ELO_K = 80.0          # chosen by tune_k on 2005-2018 fights (flat optimum 64-96); see docs/matchups.md


def load_ufc_fights(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """All UFC fights in the database with result info, sorted by date."""
    with sqlite3.connect(db_path) as con:
        f = pd.read_sql_query(
            "SELECT fight_id, event_id, event_date, red_fighter_id, blue_fighter_id, winner_id, result_status, method "
            "FROM fights ORDER BY event_date, fight_id", con)
    return f


def expected(r_a, r_b):
    return 1.0 / (1.0 + 10.0 ** ((r_b - r_a) / SCALE))


def run_elo(fights: pd.DataFrame, k: float = ELO_K, initial: float = INITIAL_RATING) -> pd.DataFrame:
    """Pre-fight ratings for every fight, updating date by date. Returns one row per fight.

    Columns: elo_red, elo_blue (pre-fight), exp_red (Elo expected score for red), red_score (1/0/0.5, NaN for NC),
    prior_ufc_red / prior_ufc_blue (number of earlier UFC fights, any result), plus the input columns.
    """
    rating: dict[str, float] = defaultdict(lambda: initial)
    count: dict[str, int] = defaultdict(int)
    out = []
    for date, day in fights.groupby("event_date", sort=True):
        delta: dict[str, float] = defaultdict(float)
        rows = []
        for r in day.itertuples(index=False):
            a, b = r.red_fighter_id, r.blue_fighter_id
            ra, rb = rating[a], rating[b]
            e = expected(ra, rb)
            if r.result_status == "win":
                s = 1.0 if r.winner_id == a else 0.0
            elif r.result_status == "draw":
                s = 0.5
            else:                                   # no contest: no update
                s = np.nan
            if not np.isnan(s):
                delta[a] += k * (s - e)
                delta[b] -= k * (s - e)
            rows.append((r.fight_id, ra, rb, e, s, count[a], count[b]))
        for a, d in delta.items():
            rating[a] += d
        for r in day.itertuples(index=False):
            count[r.red_fighter_id] += 1
            count[r.blue_fighter_id] += 1
        out += rows
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


def main() -> None:
    fights = load_ufc_fights()
    res = tune_k(fights)
    best = float(res["log_loss"].idxmin())
    print(res.round(4).to_string())
    print(f"best K on {TUNE_START}..{TUNE_END}: {best:g} (ELO_K = {ELO_K:g})")
    run_elo(fights, ELO_K).to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
