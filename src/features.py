"""Per-fighter style profiles (see docs/feature_spec.md).

    from features import build_profiles, load_fighter_fights, clustering_matrix
    ff = load_fighter_fights()                   # one row per fighter per fight (load once)
    profiles = build_profiles("2019-03-02", ff)  # only fights strictly before 2019-03-02
    X = clustering_matrix(profiles)              # eligible fighters x CLUSTER_FEATURES z-scores

Run as a script to write full-career profiles to data/processed/fighter_profiles.csv.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "ufc.sqlite"
OUT_PATH = ROOT / "data" / "processed" / "fighter_profiles.csv"

START_DATE = "2000-01-01"
MIN_FIGHTS = 5
# Shrinkage prior strength: minutes of division-average fighting added to every fighter's totals.
# Chosen with a held-out check (docs/feature_spec.md, notebooks/01_feature_eda.ipynb section 5).
PRIOR_MINUTES = 60.0

STYLE_FEATURES = ["pos_distance", "pos_clinch", "pos_ground", "tgt_head", "tgt_body", "tgt_leg",
                  "td_att_per15", "ctrl_share", "sub_att_per15", "pace"]
SKILL_MEASURES = ["str_acc", "str_def", "td_acc", "td_def", "kd_per15", "str_diff_per_min"]
SHRUNK_SKILL = ["td_acc", "td_def"]
# Each mix sums to 1, so one share per mix is redundant: drop distance and head for clustering.
CLUSTER_FEATURES = [f for f in STYLE_FEATURES if f not in ("pos_distance", "tgt_head")]

# feature -> (numerator column, denominator column, scale). value = scale * num / den.
# Shares divide by sig. attempts; rates divide by fight seconds (scaled to per 15 min / per min).
RATIOS = {
    "pos_distance": ("distance_att", "sig_att", 1), "pos_clinch": ("clinch_att", "sig_att", 1),
    "pos_ground": ("ground_att", "sig_att", 1), "tgt_head": ("head_att", "sig_att", 1),
    "tgt_body": ("body_att", "sig_att", 1), "tgt_leg": ("leg_att", "sig_att", 1),
    "td_att_per15": ("td_att", "time_s", 900), "ctrl_share": ("ctrl_s", "ctrl_time_s", 1),
    "sub_att_per15": ("sub_att", "time_s", 900), "pace": ("sig_att", "time_s", 60),
    "td_acc": ("td_landed", "td_att", 1), "td_def": ("opp_td_landed", "opp_td_att", 1),  # td_def = 1 - ratio
}

# Longest names first, so "Light Heavyweight" wins over "Heavyweight" and "Women's Flyweight" over "Flyweight".
DIVISIONS = ["Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Women's Featherweight",
             "Light Heavyweight", "Super Heavyweight", "Heavyweight", "Middleweight", "Welterweight",
             "Lightweight", "Featherweight", "Bantamweight", "Flyweight"]
DIVISION_ALIASES = {"Super Heavyweight": "Heavyweight"}
# Standardization groups: divisions too small to standardize on their own are pooled with a neighbour.
STD_GROUP = {"Women's Featherweight": "Women's Bantamweight"}

_SUM_COLS = ["kd", "sig_landed", "sig_att", "total_landed", "total_att", "td_landed", "td_att", "sub_att",
             "head_att", "body_att", "leg_att", "distance_att", "clinch_att", "ground_att", "ctrl_s"]


def canonical_division(weight_class: str | None) -> str | None:
    """Map a fight's weight class to a division. Catch weight, open weight and unknown -> None."""
    if not isinstance(weight_class, str):
        return None
    for d in DIVISIONS:
        if d.lower() in weight_class.lower():
            return DIVISION_ALIASES.get(d, d)
    return None


def load_fighter_fights(db_path: Path | str = DB_PATH, start_date: str = START_DATE) -> pd.DataFrame:
    """One row per fighter per fight (UFC, on/after start_date), with the opponent's stats as opp_*."""
    sums = ", ".join(f"SUM({c}) AS {c}" for c in _SUM_COLS if c != "ctrl_s")
    sql = f"""
        SELECT r.fight_id, r.fighter_id, r.opponent_id, {sums},
               CASE WHEN COUNT(r.ctrl_s) = COUNT(*) THEN SUM(r.ctrl_s) END AS ctrl_s,
               f.event_date, f.weight_class, f.total_fight_time_s AS time_s
        FROM fight_round_stats r JOIN fights f USING (fight_id)
        WHERE f.event_date >= ?
        GROUP BY r.fight_id, r.fighter_id
    """
    with sqlite3.connect(db_path) as con:
        ff = pd.read_sql_query(sql, con, params=(start_date,))
        names = pd.read_sql_query("SELECT fighter_id, fighter_name FROM fighters", con)
    opp = ff[["fight_id", "fighter_id"] + _SUM_COLS].rename(
        columns={"fighter_id": "opponent_id", **{c: f"opp_{c}" for c in _SUM_COLS}})
    ff = ff.merge(opp, on=["fight_id", "opponent_id"], how="left", validate="one_to_one")
    ff["division"] = ff["weight_class"].map(canonical_division)
    ff = ff.merge(names, on="fighter_id", how="left")
    if ff["time_s"].isna().any():
        raise ValueError("fights without a fight time; check total_fight_time_s")
    return ff


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.where(den > 0)).astype(float)


def _modal_division(ff: pd.DataFrame) -> pd.Series:
    """Most frequent division per fighter, ignoring catch weight; ties go to the most recent fight."""
    d = ff.dropna(subset=["division"])
    stats = d.groupby(["fighter_id", "division"]).agg(n=("fight_id", "size"), last=("event_date", "max")).reset_index()
    stats = stats.sort_values(["fighter_id", "n", "last"], ascending=[True, False, False])
    return stats.drop_duplicates("fighter_id").set_index("fighter_id")["division"]


def aggregate(ff: pd.DataFrame) -> pd.DataFrame:
    """Pooled totals per fighter: own and opponent sums, fight seconds, and seconds with control time recorded."""
    g = ff.groupby("fighter_id")
    agg = g[_SUM_COLS + [f"opp_{c}" for c in _SUM_COLS] + ["time_s"]].sum(min_count=1)
    agg["ctrl_time_s"] = ff.loc[ff["ctrl_s"].notna()].groupby("fighter_id")["time_s"].sum().reindex(agg.index).fillna(0)
    agg["ctrl_s"] = agg["ctrl_s"].fillna(0)
    return agg


def division_priors(agg: pd.DataFrame, std_group: pd.Series, eligible: pd.Series) -> pd.DataFrame:
    """Per-second division-average totals, pooled over eligible fighters in each standardization group.

    Returned per fighter (indexed like agg): for every column c, prior[c] = sum(c) / sum(time_s) over the
    reference fighters in that fighter's group. Groups with no eligible fighters (and fighters without a
    division) fall back to the pooled average of all eligible fighters, or of everyone if none are eligible.
    """
    cols = sorted({c for num, den, _ in RATIOS.values() for c in (num, den)})
    ref = agg.loc[eligible[eligible].index] if eligible.any() else agg
    overall = ref[cols].sum() / ref["time_s"].sum()
    grp = std_group.reindex(ref.index)
    by_group = ref[cols].groupby(grp).sum().div(ref["time_s"].groupby(grp).sum(), axis=0)
    per_fighter = by_group.reindex(std_group.reindex(agg.index)).set_axis(agg.index)
    return per_fighter.fillna(overall)


def ratio_features(agg: pd.DataFrame, priors: pd.DataFrame | None, prior_minutes: float) -> pd.DataFrame:
    """Shrunk ratio features: (num + P * prior_num_rate) / (den + P * prior_den_rate), P = prior seconds.

    With prior_minutes = 0 (or priors None) these are the raw pooled ratios.
    """
    P = prior_minutes * 60
    out = pd.DataFrame(index=agg.index)
    for feat, (num, den, scale) in RATIOS.items():
        n, d = agg[num].astype(float), agg[den].astype(float)
        if priors is not None and P > 0:
            n, d = n + P * priors[num], d + P * priors[den]
        r = scale * _ratio(n, d)
        out[feat] = 1 - r if feat == "td_def" else r
    return out


def build_profiles(as_of: str | pd.Timestamp | None = None, fighter_fights: pd.DataFrame | None = None,
                   min_fights: int = MIN_FIGHTS, prior_minutes: float = PRIOR_MINUTES,
                   db_path: Path | str = DB_PATH) -> pd.DataFrame:
    """One row per fighter, built only from fights strictly before `as_of` (all fights if None).

    Every fighter with at least one such fight is returned; `eligible` marks fighters with >= min_fights.
    Style features and td_acc/td_def are shrunk toward the fighter's division average by `prior_minutes`
    of division-average fighting (raw_<feature> keeps the unshrunk value). z_<feature> standardizes each
    shrunk style feature within the fighter's standardization group, using eligible fighters' mean and SD.
    Division choice, eligibility, priors and standardization all use only pre-as_of fights (no leakage).
    """
    ff = load_fighter_fights(db_path) if fighter_fights is None else fighter_fights
    if as_of is not None:
        as_of = pd.Timestamp(as_of).strftime("%Y-%m-%d")
        ff = ff[ff["event_date"] < as_of]
    if ff.empty:
        return pd.DataFrame(columns=["fighter_id"])

    g = ff.groupby("fighter_id")
    agg = aggregate(ff)
    p = pd.DataFrame(index=agg.index)
    p["fighter_name"] = g["fighter_name"].first()
    p["n_fights"] = g.size()
    p["eligible"] = p["n_fights"] >= min_fights
    p["fight_time_min"] = agg["time_s"] / 60
    p["first_fight"] = g["event_date"].min()
    p["last_fight"] = g["event_date"].max()
    p["division"] = _modal_division(ff)
    p["std_group"] = p["division"].map(lambda d: STD_GROUP.get(d, d) if isinstance(d, str) else None)

    raw = ratio_features(agg, None, 0)
    priors = division_priors(agg, p["std_group"], p["eligible"])
    shrunk = ratio_features(agg, priors, prior_minutes)
    for f in STYLE_FEATURES:
        p[f] = shrunk[f]
    tmin = agg["time_s"] / 60
    p["str_acc"] = _ratio(agg["sig_landed"], agg["sig_att"])
    p["str_def"] = 1 - _ratio(agg["opp_sig_landed"], agg["opp_sig_att"])
    p["td_acc"], p["td_def"] = shrunk["td_acc"], shrunk["td_def"]
    p["kd_per15"] = _ratio(agg["kd"], agg["time_s"] / 900)
    p["str_diff_per_min"] = _ratio(agg["sig_landed"] - agg["opp_sig_landed"], tmin)
    for f in STYLE_FEATURES + SHRUNK_SKILL:
        p[f"raw_{f}"] = raw[f]

    ref = p[p["eligible"] & p["std_group"].notna()]
    mu, sd = ref.groupby("std_group")[STYLE_FEATURES].mean(), ref.groupby("std_group")[STYLE_FEATURES].std(ddof=0)
    m, s = mu.reindex(p["std_group"]).set_axis(p.index), sd.reindex(p["std_group"]).set_axis(p.index)
    for f in STYLE_FEATURES:
        p[f"z_{f}"] = (p[f] - m[f]) / s[f].where(s[f] > 0)

    cols = (["fighter_name", "division", "std_group", "n_fights", "eligible", "fight_time_min", "first_fight", "last_fight"]
            + STYLE_FEATURES + SKILL_MEASURES + [f"raw_{f}" for f in STYLE_FEATURES + SHRUNK_SKILL]
            + [f"z_{f}" for f in STYLE_FEATURES])
    return p[cols].reset_index()


def clustering_matrix(profiles: pd.DataFrame) -> pd.DataFrame:
    """Eligible fighters x z-scored CLUSTER_FEATURES (distance and head shares dropped), indexed by fighter_id."""
    e = profiles[profiles["eligible"] & profiles["std_group"].notna()]
    return e.set_index("fighter_id")[[f"z_{f}" for f in CLUSTER_FEATURES]]


def main() -> None:
    profiles = build_profiles(None)
    profiles.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH.relative_to(ROOT)}: {len(profiles):,} fighters, "
          f"{int(profiles['eligible'].sum()):,} eligible (>= {MIN_FIGHTS} UFC fights since {START_DATE[:4]}), "
          f"shrinkage prior {PRIOR_MINUTES:g} min")


if __name__ == "__main__":
    main()
