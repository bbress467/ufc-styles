"""Clean the pinned neelagiriaditya download into tidy tables in data/processed/ufc.sqlite.

Steps
-----
1. Read every raw CSV as text and convert missing-value markers ('' / '--' / '---') to NaN,
   counting conversions per column.
2. Keep only UFC events: an event is kept iff its ufcstats event_id appears in Greco1899's
   ufc_event_details.csv (the UFC events list). EXTRA_UFC_EVENT_IDS can add events the
   whitelist lacks (see below).
3. Build tidy tables (career-average columns are dropped entirely):
     events             one row per event
     fighters           one row per fighter appearing in a kept fight
     fights             one row per fight (red/blue corner = ufcstats fight-page order)
     fight_round_stats  one row per fighter per round
     fight_total_stats  one row per fighter per fight (the dataset's fight-total columns, with its
                        zero-filled missing values removed: see the totals section of main())
4. Write the SQLite database and data/processed/clean_report.json (row counts, duplicates,
   marker conversions, whitelist results) for src/validate.py.

Missing vs. "no attempts"
-------------------------
Counts are stored as integers. "No attempts" is attempts == 0 (a real value), never NaN.
NaN means the source has no value (e.g. control time not recorded for 1990s fights).
No percentage columns are stored, so landed/attempted with attempts == 0 is left for the
feature code to treat as undefined. A fight with no per-round data has no rows in
fight_round_stats (rather than rows of zeros) and has_round_stats = 0 in fights.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PINS = json.loads((ROOT / "data_versions.json").read_text())
NEEL_DIR = ROOT / "data" / "raw" / "neelagiriaditya" / f"v{PINS['neelagiriaditya']['version']}"
GRECO_DIR = ROOT / "data" / "raw" / "greco1899" / PINS["greco1899"]["commit"][:12]
OUT_DIR = ROOT / "data" / "processed"
DB_PATH = OUT_DIR / "ufc.sqlite"
REPORT_PATH = OUT_DIR / "clean_report.json"

MISSING_MARKERS = ("", "--", "---")

# Events to keep even though the Greco1899 whitelist lacks them. ufcstats.com no longer lists
# UFC 1, so the strict whitelist drops it. Uncomment to keep it (8 fights, all with round stats).
EXTRA_UFC_EVENT_IDS: set[str] = set([
    # "6420efac0578988b",  # UFC 1: The Beginning (1993-11-12)
])

CAREER_AVG_COLS = ["slpm", "str_acc", "sapm", "str_def", "td_avg", "td_acc", "td_def", "sub_avg"]

# round.csv / master.csv column stem -> tidy column name. Each stem is prefixed with r_/b_
# (round.csv) or r_total_/b_total_ (master.csv).
STAT_MAP = {
    "kd": "kd",
    "sig_landed": "sig_landed", "sig_atmp": "sig_att",
    "total_str_landed": "total_landed", "total_str_atmp": "total_att",
    "td_success": "td_landed", "td_atmp": "td_att",
    "sub_att": "sub_att", "rev": "rev",
    **{f"sig_str_{k}_{t}": f"{t}_{'landed' if k == 'landed' else 'att'}"
       for t in ("head", "body", "leg", "distance", "clinch", "ground")
       for k in ("landed", "atmp")},
}
COUNT_COLS = list(STAT_MAP.values())
STAT_COLS = COUNT_COLS + ["ctrl_s"]

SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE events (
    event_id      TEXT PRIMARY KEY,
    event_name    TEXT NOT NULL,
    event_date    TEXT NOT NULL,          -- ISO yyyy-mm-dd
    location      TEXT
);
CREATE TABLE fighters (
    fighter_id    TEXT PRIMARY KEY,
    fighter_name  TEXT NOT NULL,
    nickname      TEXT,
    height_in     REAL,                   -- NULL = not recorded
    weight_lbs    REAL,
    reach_in      REAL,
    stance        TEXT,
    dob           TEXT                    -- ISO yyyy-mm-dd
);
CREATE TABLE fights (
    fight_id            TEXT PRIMARY KEY,
    event_id            TEXT NOT NULL REFERENCES events(event_id),
    event_date          TEXT NOT NULL,
    weight_class        TEXT,
    title_fight         INTEGER NOT NULL,
    red_fighter_id      TEXT NOT NULL REFERENCES fighters(fighter_id),  -- fight-page order, not winner
    blue_fighter_id     TEXT NOT NULL REFERENCES fighters(fighter_id),
    winner_id           TEXT REFERENCES fighters(fighter_id),           -- NULL for draw / no contest
    result_status       TEXT NOT NULL,     -- win | draw | no_contest
    method              TEXT,
    finish_round        INTEGER,
    finish_time_s       INTEGER,           -- elapsed seconds in the final round
    total_fight_time_s  INTEGER,           -- NULL when the format has no fixed round lengths
    time_format         TEXT,
    scheduled_periods   INTEGER,           -- rounds incl. overtime periods; NULL = no time limit
    period_lengths_min  TEXT,              -- e.g. '5-5-5'
    referee             TEXT,
    details             TEXT,
    bonuses             TEXT,              -- '; '-joined bonus types, NULL if none
    has_round_stats     INTEGER NOT NULL
);
CREATE TABLE fight_round_stats (
    fight_id      TEXT NOT NULL REFERENCES fights(fight_id),
    round         INTEGER NOT NULL,
    fighter_id    TEXT NOT NULL REFERENCES fighters(fighter_id),
    opponent_id   TEXT NOT NULL REFERENCES fighters(fighter_id),
    corner        TEXT NOT NULL,           -- red | blue
    {stat_cols},
    PRIMARY KEY (fight_id, round, fighter_id)
);
CREATE TABLE fight_total_stats (
    fight_id      TEXT NOT NULL REFERENCES fights(fight_id),
    fighter_id    TEXT NOT NULL REFERENCES fighters(fighter_id),
    opponent_id   TEXT NOT NULL REFERENCES fighters(fighter_id),
    corner        TEXT NOT NULL,
    {stat_cols},
    PRIMARY KEY (fight_id, fighter_id)
);
CREATE INDEX idx_fights_event ON fights(event_id);
CREATE INDEX idx_rounds_fighter ON fight_round_stats(fighter_id);
CREATE INDEX idx_totals_fighter ON fight_total_stats(fighter_id);
""".replace("{stat_cols}", ",\n    ".join(
    [f"{c} INTEGER NOT NULL" for c in COUNT_COLS] + ["ctrl_s INTEGER /* NULL = not recorded */"]))


# --------------------------------------------------------------------------- helpers
def read_raw(name: str, report: dict) -> pd.DataFrame:
    """Read a raw CSV as text, convert missing markers to NaN, and record what changed."""
    df = pd.read_csv(NEEL_DIR / name, dtype=str, keep_default_na=False)
    conv = {}
    for col in df.columns:
        stripped = df[col].str.strip()
        counts = {m if m else "(empty)": int((stripped == m).sum()) for m in MISSING_MARKERS}
        counts = {k: v for k, v in counts.items() if v}
        if counts:
            conv[col] = counts
        df[col] = stripped.where(~stripped.isin(MISSING_MARKERS), np.nan)
    report["raw"][name] = {"rows": len(df), "exact_duplicate_rows": int(df.duplicated().sum()),
                           "missing_markers_to_nan": conv}
    return df


def mmss_to_s(s: pd.Series) -> pd.Series:
    parts = s.str.extract(r"^(\d+):(\d{2})$")
    bad = s.notna() & parts[0].isna()
    if bad.any():
        raise ValueError(f"unparseable M:SS values: {s[bad].unique()[:10]}")
    return (parts[0].astype(float) * 60 + parts[1].astype(float)).astype("Int64")


def height_to_in(s: pd.Series) -> pd.Series:
    parts = s.str.extract(r"""^(\d+)'\s*(\d+)"?$""")
    bad = s.notna() & parts[0].isna()
    if bad.any():
        raise ValueError(f"unparseable heights: {s[bad].unique()[:10]}")
    return parts[0].astype(float) * 12 + parts[1].astype(float)


def parse_time_format(tf: str | float) -> list[int] | None:
    """'3 Rnd (5-5-5)' -> [5,5,5]; '1 Rnd + 2OT (15-3-3)' -> [15,3,3]; 'No Time Limit' -> None."""
    if not isinstance(tf, str):
        return None
    m = re.search(r"\(([\d-]+)\)\s*$", tf)
    return [int(x) for x in m.group(1).split("-")] if m else None


def dup_count(df: pd.DataFrame, keys: list[str]) -> int:
    return int(df.duplicated(keys).sum())


def to_long(wide: pd.DataFrame, prefix_r: str, prefix_b: str, keys: list[str], ctrl_col: str,
            ctrl_is_mmss: bool) -> pd.DataFrame:
    """Reshape r_/b_ wide stats into one row per fighter."""
    out = []
    for corner, pre, me, opp in (("red", prefix_r, "red_fighter_id", "blue_fighter_id"),
                                 ("blue", prefix_b, "blue_fighter_id", "red_fighter_id")):
        cols = {f"{pre}{stem}": tidy for stem, tidy in STAT_MAP.items()}
        part = wide[keys + [me, opp] + list(cols)].rename(columns=cols)
        part = part.rename(columns={me: "fighter_id", opp: "opponent_id"})
        ctrl = wide[ctrl_col.format(pre=pre)]
        part["ctrl_s"] = mmss_to_s(ctrl) if ctrl_is_mmss else pd.to_numeric(ctrl).astype("Int64")
        part["corner"] = corner
        out.append(part)
    long = pd.concat(out, ignore_index=True)
    for c in COUNT_COLS:
        long[c] = pd.to_numeric(long[c], errors="raise").astype("Int64")
    return long


# --------------------------------------------------------------------------- main
def main() -> None:
    report: dict = {"inputs": {"neelagiriaditya": str(NEEL_DIR.relative_to(ROOT)),
                               "greco1899_whitelist": str((GRECO_DIR / "ufc_event_details.csv").relative_to(ROOT))},
                    "raw": {}, "steps": [], "duplicates": {}, "whitelist": {}, "checks": {}}

    def step(name: str, **counts) -> None:
        report["steps"].append({"step": name, **{k: int(v) for k, v in counts.items()}})

    ev = read_raw("event.csv", report)
    fi = read_raw("fight.csv", report)
    fr = read_raw("fighter.csv", report)
    rd = read_raw("round.csv", report)
    ms = read_raw("master.csv", report)
    bo = read_raw("fighter_bonus.csv", report)
    step("raw", events=len(ev), fights=len(fi), fighters=len(fr), round_rows=len(rd), master_rows=len(ms))

    report["duplicates"]["raw_keys"] = {
        "event.csv event_id": dup_count(ev, ["event_id"]),
        "fight.csv fight_id": dup_count(fi, ["fight_id"]),
        "fighter.csv fighter_id": dup_count(fr, ["fighter_id"]),
        "fighter.csv fighter_name (different ids)": dup_count(fr, ["fighter_name"]),
        "round.csv (fight_id, round_no)": dup_count(rd, ["fight_id", "round_no"]),
        "master.csv fight_id": dup_count(ms, ["fight_id"]),
        "fighter_bonus.csv (fight_id, bonus_type)": dup_count(bo, ["fight_id", "bonus_type"]),
    }
    for name, df, key in (("event.csv", ev, ["event_id"]), ("fight.csv", fi, ["fight_id"]),
                          ("fighter.csv", fr, ["fighter_id"]), ("round.csv", rd, ["fight_id", "round_no"])):
        if dup_count(df, key):
            raise SystemExit(f"duplicate keys in {name}; refusing to continue")

    # ---- UFC whitelist ---------------------------------------------------------------
    g = pd.read_csv(GRECO_DIR / "ufc_event_details.csv", dtype=str)
    g["event_id"] = g["URL"].str.extract(r"event-details/([0-9a-f]{16})")[0]
    if g["event_id"].isna().any() or not g["event_id"].is_unique:
        raise SystemExit("Greco1899 event list has missing or duplicate event IDs")
    whitelist = set(g["event_id"]) | EXTRA_UFC_EVENT_IDS
    keep = ev["event_id"].isin(whitelist)
    dropped = ev.loc[~keep]
    report["whitelist"] = {
        "greco_events": len(g),
        "extra_event_ids": sorted(EXTRA_UFC_EVENT_IDS),
        "kept": int(keep.sum()), "dropped": int((~keep).sum()),
        "dropped_by_first_word": dropped["event_name"].str.extract(r"^(\S+)")[0].value_counts().to_dict(),
        "dropped_names_containing_UFC": sorted(dropped.loc[dropped["event_name"].str.contains("UFC"), "event_name"]),
        "greco_events_not_in_dataset": int((~g["event_id"].isin(ev["event_id"])).sum()),
        "greco_events_not_in_dataset_dates": [g.loc[~g["event_id"].isin(ev["event_id"]), "DATE"].iloc[-1],
                                              g.loc[~g["event_id"].isin(ev["event_id"]), "DATE"].iloc[0]]
        if (~g["event_id"].isin(ev["event_id"])).any() else [],
        "dropped_events_with_location": int(dropped["location"].notna().sum()),
        "kept_events_without_location": int(ev.loc[keep, "location"].isna().sum()),
    }
    ev = ev.loc[keep].copy()
    fi = fi.loc[fi["event_id"].isin(ev["event_id"])].copy()
    step("after UFC whitelist", events=len(ev), fights=len(fi))

    # ---- events --------------------------------------------------------------------
    events = pd.DataFrame({
        "event_id": ev["event_id"], "event_name": ev["event_name"],
        "event_date": pd.to_datetime(ev["date"], format="%Y-%m-%d").dt.strftime("%Y-%m-%d"),
        "location": ev["location"],
    })

    # ---- fights --------------------------------------------------------------------
    fi = fi.merge(events[["event_id", "event_date"]], on="event_id", how="left", validate="many_to_one")
    lengths = fi["time_format"].map(parse_time_format)
    unparsed_tf = sorted(fi.loc[lengths.isna() & fi["time_format"].notna(), "time_format"].unique())
    finish_round = pd.to_numeric(fi["round"]).astype("Int64")
    finish_time_s = mmss_to_s(fi["time"])

    def total_time(i: int):
        L, r, t = lengths.iat[i], finish_round.iat[i], finish_time_s.iat[i]
        if L is None or pd.isna(r) or pd.isna(t) or r > len(L):
            return pd.NA
        return int(sum(L[: r - 1]) * 60 + t)

    bonuses = bo.groupby("fight_id")["bonus_type"].agg(lambda s: "; ".join(sorted(s)))
    fights = pd.DataFrame({
        "fight_id": fi["fight_id"], "event_id": fi["event_id"], "event_date": fi["event_date"],
        "weight_class": fi["weight_class"], "title_fight": pd.to_numeric(fi["title_fight"]).astype(int),
        "red_fighter_id": fi["r_id"], "blue_fighter_id": fi["b_id"], "winner_id": fi["winner_id"],
        "result_status": fi["result_status"], "method": fi["method"],
        "finish_round": finish_round, "finish_time_s": finish_time_s,
        "total_fight_time_s": pd.array([total_time(i) for i in range(len(fi))], dtype="Int64"),
        "time_format": fi["time_format"],
        "scheduled_periods": pd.array([len(L) if L else pd.NA for L in lengths], dtype="Int64"),
        "period_lengths_min": [("-".join(map(str, L)) if L else None) for L in lengths],
        "referee": fi["referee"].str.replace(r"\s+", " ", regex=True).str.strip(),
        "details": fi["details"].str.replace(r"\s+", " ", regex=True).str.strip(),
        "bonuses": fi["fight_id"].map(bonuses),
        "has_round_stats": fi["fight_id"].isin(rd["fight_id"]).astype(int),
    })
    report["checks"]["unparsed_time_formats"] = unparsed_tf
    report["checks"]["result_status_counts"] = fights["result_status"].value_counts().to_dict()
    report["checks"]["winner_id_null_by_status"] = fights.loc[fights["winner_id"].isna(), "result_status"].value_counts().to_dict()
    report["checks"]["winner_not_in_fight"] = int((fights["winner_id"].notna() & (fights["winner_id"] != fights["red_fighter_id"])
                                                   & (fights["winner_id"] != fights["blue_fighter_id"])).sum())
    report["checks"]["same_fighter_both_corners"] = int((fights["red_fighter_id"] == fights["blue_fighter_id"]).sum())
    step("fights table", fights=len(fights), fights_with_round_stats=fights["has_round_stats"].sum(),
         fights_without_round_stats=(1 - fights["has_round_stats"]).sum())

    # ---- fighters (only those in kept fights; career averages dropped) --------------
    used = pd.unique(pd.concat([fights["red_fighter_id"], fights["blue_fighter_id"]]))
    missing_fighters = sorted(set(used) - set(fr["fighter_id"]))
    if missing_fighters:
        raise SystemExit(f"{len(missing_fighters)} fight fighter IDs missing from fighter.csv: {missing_fighters[:5]}")
    frk = fr.loc[fr["fighter_id"].isin(used)].drop(columns=CAREER_AVG_COLS)
    fighters = pd.DataFrame({
        "fighter_id": frk["fighter_id"], "fighter_name": frk["fighter_name"],
        "nickname": frk["fighter_nick_name"], "height_in": height_to_in(frk["height"]),
        "weight_lbs": pd.to_numeric(frk["weight_lbs"]), "reach_in": pd.to_numeric(frk["reach_inches"]),
        "stance": frk["stance"],
        "dob": pd.to_datetime(frk["dob"], format="%Y-%m-%d").dt.strftime("%Y-%m-%d"),
    })
    report["checks"]["career_avg_columns_dropped"] = CAREER_AVG_COLS
    step("fighters table", fighters_in_raw=len(fr), fighters_in_ufc_fights=len(fighters))

    # ---- round stats (long) ----------------------------------------------------------
    rdk = rd.loc[rd["fight_id"].isin(fights["fight_id"])].copy()
    step("round rows after UFC filter", round_rows=len(rdk))
    corner_ids = rdk.merge(fights[["fight_id", "red_fighter_id", "blue_fighter_id"]], on="fight_id")
    report["checks"]["round_corner_ids_disagree_with_fight"] = int(
        ((corner_ids["r_id"] != corner_ids["red_fighter_id"]) | (corner_ids["b_id"] != corner_ids["blue_fighter_id"])).sum())
    rdk = rdk.rename(columns={"r_id": "red_fighter_id", "b_id": "blue_fighter_id", "round_no": "round"})
    rdk["round"] = pd.to_numeric(rdk["round"]).astype(int)
    rounds = to_long(rdk, "r_", "b_", ["fight_id", "round"], "{pre}ctrl", ctrl_is_mmss=True)
    rounds = rounds[["fight_id", "round", "fighter_id", "opponent_id", "corner"] + STAT_COLS]
    report["duplicates"]["fight_round_stats (fight_id, round, fighter_id)"] = dup_count(rounds, ["fight_id", "round", "fighter_id"])
    step("fight_round_stats (long)", rows=len(rounds))

    # ---- fight totals (long) from master.csv -----------------------------------------
    msk = ms.loc[ms["fight_id"].isin(fights["fight_id"])].merge(
        fights[["fight_id", "red_fighter_id", "blue_fighter_id"]], on="fight_id", validate="one_to_one")
    totals = to_long(msk, "r_total_", "b_total_", ["fight_id"], "{pre}ctrl_seconds", ctrl_is_mmss=False)
    totals = totals[["fight_id", "fighter_id", "opponent_id", "corner"] + STAT_COLS]
    has_any_total = totals[COUNT_COLS].notna().any(axis=1)
    report["checks"]["total_rows_all_null"] = int((~has_any_total).sum())
    totals = totals.loc[has_any_total]
    for c in COUNT_COLS:
        if totals[c].isna().any():
            raise SystemExit(f"partially null totals in {c}")
    step("fight_total_stats (long, raw)", rows=len(totals))

    # The dataset's fight totals fill missing data with zeros:
    #  (a) fights with no per-round data have all-zero totals (the site has no stats for them);
    #  (b) when every round's control time is missing ('--' on the site), the total ctrl is 0.
    # Neither zero is real, so (a) rows are dropped and (b) totals are set to NULL.
    no_rounds = ~totals["fight_id"].isin(rounds["fight_id"])
    report["checks"]["totals_dropped_no_round_stats"] = {
        "rows": int(no_rounds.sum()), "fights": int(totals.loc[no_rounds, "fight_id"].nunique()),
        "rows_all_zero": int((totals.loc[no_rounds, STAT_COLS].fillna(0) == 0).all(axis=1).sum())}
    totals = totals.loc[~no_rounds].copy()
    ctrl_all_null = rounds.groupby(["fight_id", "fighter_id"])["ctrl_s"].apply(lambda s: s.isna().all())
    ctrl_some_null = rounds.groupby(["fight_id", "fighter_id"])["ctrl_s"].apply(lambda s: s.isna().any())
    key = pd.MultiIndex.from_frame(totals[["fight_id", "fighter_id"]])
    all_null = ctrl_all_null.reindex(key).to_numpy(dtype=bool)
    report["checks"]["totals_ctrl_zero_set_null"] = {
        "rows": int(all_null.sum()),
        "values_before": {str(k): int(v) for k, v in totals.loc[all_null, "ctrl_s"].value_counts(dropna=False).items()},
        "rows_with_only_some_round_ctrl_null": int((ctrl_some_null.reindex(key).to_numpy(dtype=bool) & ~all_null).sum())}
    totals.loc[all_null, "ctrl_s"] = pd.NA
    report["duplicates"]["fight_total_stats (fight_id, fighter_id)"] = dup_count(totals, ["fight_id", "fighter_id"])
    step("fight_total_stats (long, cleaned)", rows=len(totals),
         dropped_all_zero_rows_for_fights_without_round_stats=int(no_rounds.sum()),
         ctrl_zero_set_to_null=int(all_null.sum()))

    # ---- write -----------------------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if DB_PATH.exists():
        DB_PATH.unlink()
    con = sqlite3.connect(DB_PATH)
    con.executescript(SCHEMA)
    for name, df in (("events", events), ("fighters", fighters), ("fights", fights),
                     ("fight_round_stats", rounds), ("fight_total_stats", totals)):
        df.astype(object).where(df.notna(), None).to_sql(name, con, if_exists="append", index=False)
    fk = con.execute("PRAGMA foreign_key_check").fetchall()
    report["checks"]["foreign_key_violations"] = len(fk)
    con.commit()
    db_counts = {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                 for t in ("events", "fighters", "fights", "fight_round_stats", "fight_total_stats")}
    con.close()
    step("written to SQLite", **db_counts)

    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps(report["steps"], indent=1))
    print(f"wrote {DB_PATH.relative_to(ROOT)} and {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
