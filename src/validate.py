"""Validate data/processed/ufc.sqlite and write docs/data_quality.md.

1. Row counts and duplicates at each cleaning step (from data/processed/clean_report.json).
2. Cross-check against Greco1899 (an independent scrape of the same site): mismatch rate for
   every overlapping round-level stat column, fight-result column and fighter attribute.
3. Internal consistency: round stats sum to fight totals; landed <= attempted; significant <=
   total strikes; target and position splits sum to significant strikes; control time fits in
   the round; round rows match the finish round.
4. Completeness by year: share of fights with complete strike splits and control time.

Run after src/clean.py. Exits non-zero only if the inputs are missing.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PINS = json.loads((ROOT / "data_versions.json").read_text())
GRECO_DIR = ROOT / "data" / "raw" / "greco1899" / PINS["greco1899"]["commit"][:12]
DB_PATH = ROOT / "data" / "processed" / "ufc.sqlite"
REPORT_PATH = ROOT / "data" / "processed" / "clean_report.json"
DOC_PATH = ROOT / "docs" / "data_quality.md"

SPLITS_TARGET = ("head", "body", "leg")
SPLITS_POSITION = ("distance", "clinch", "ground")
PAIRS = ["sig", "total", "td"] + list(SPLITS_TARGET + SPLITS_POSITION)
COUNT_COLS = ["kd", "sig_landed", "sig_att", "total_landed", "total_att", "td_landed", "td_att",
              "sub_att", "rev"] + [f"{t}_{k}" for t in SPLITS_TARGET + SPLITS_POSITION for k in ("landed", "att")]
STAT_COLS = COUNT_COLS + ["ctrl_s"]


# --------------------------------------------------------------------------- formatting
def pct(n: float, d: float, digits: int = 2) -> str:
    return "n/a" if not d else f"{100 * n / d:.{digits}f}%"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- Greco parsing
def of_split(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    p = s.str.extract(r"^\s*(\d+) of (\d+)\s*$")
    return pd.to_numeric(p[0]).astype("Int64"), pd.to_numeric(p[1]).astype("Int64")


def mmss(s: pd.Series) -> pd.Series:
    p = s.astype("string").str.extract(r"^\s*(\d+):(\d{2})\s*$")
    return (pd.to_numeric(p[0]) * 60 + pd.to_numeric(p[1])).astype("Int64")


def height_in(s: pd.Series) -> pd.Series:
    p = s.astype("string").str.extract(r"""^(\d+)'\s*(\d+)"?$""")
    return pd.to_numeric(p[0]) * 12 + pd.to_numeric(p[1])


def compare(a: pd.Series, b: pd.Series) -> dict:
    """Mismatch stats between two aligned series (neel = a, greco = b)."""
    both = a.notna() & b.notna()
    mism = both & (a.astype(object) != b.astype(object))
    return {"rows": len(a), "compared": int(both.sum()), "mismatches": int(mism.sum()),
            "null_only_in_neel": int((a.isna() & b.notna()).sum()),
            "null_only_in_greco": int((a.notna() & b.isna()).sum())}


def main() -> None:
    for p in (DB_PATH, REPORT_PATH, GRECO_DIR / "ufc_fight_stats.csv"):
        if not p.exists():
            raise SystemExit(f"missing {p}; run src/download_data.py and src/clean.py first")
    report = json.loads(REPORT_PATH.read_text())
    con = sqlite3.connect(DB_PATH)
    q = lambda sql: pd.read_sql_query(sql, con)
    events, fighters, fights = q("SELECT * FROM events"), q("SELECT * FROM fighters"), q("SELECT * FROM fights")
    rounds, totals = q("SELECT * FROM fight_round_stats"), q("SELECT * FROM fight_total_stats")
    for df in (rounds, totals):
        for c in STAT_COLS:
            df[c] = df[c].astype("Int64")
    out: list[str] = []
    w = out.append

    # =============================================================== header
    w("# Data quality report\n")
    w(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC by `src/validate.py`. Rerun it after "
      "`src/download_data.py` and `src/clean.py` to regenerate this file.\n")
    w(f"- Data: neelagiriaditya Kaggle **v{PINS['neelagiriaditya']['version']}** "
      f"(`{report['inputs']['neelagiriaditya']}`)")
    w(f"- Reference: Greco1899 commit `{PINS['greco1899']['commit'][:12]}`, used for the UFC event whitelist and the cross-check only")
    w(f"- Output: `data/processed/ufc.sqlite`. Tables: `events`, `fighters`, `fights`, `fight_round_stats`, `fight_total_stats`\n")

    # =============================================================== 1. counts & duplicates
    w("## 1. Row counts and duplicates\n")
    w("### Row counts at each step\n")
    rows = []
    for s in report["steps"]:
        for k, v in s.items():
            if k != "step":
                rows.append({"step": s["step"], "count": k, "rows": f"{v:,}"})
    w(md_table(pd.DataFrame(rows)) + "\n")
    w("`fight_round_stats` and `fight_total_stats` are long format (two rows per fight-round or fight, one per fighter), "
      "so they have twice as many rows as the raw wide tables.\n")

    wl = report["whitelist"]
    w("### UFC whitelist\n")
    w(f"An event is kept only if its ufcstats `event_id` appears in Greco1899's UFC events list ({wl['greco_events']} events). "
      f"**Kept {wl['kept']}, dropped {wl['dropped']}.** Every dropped event has an empty `location` and every kept event has one "
      f"(dropped with a location: {wl['dropped_events_with_location']}; kept without one: {wl['kept_events_without_location']}), "
      "which independently supports the split.\n")
    w(f"- Dropped events whose name contains \"UFC\": {', '.join(wl['dropped_names_containing_UFC'])}. "
      "All are \"Road to UFC\" (a tournament series, not official UFC bouts) except **UFC 1: The Beginning**, which ufcstats.com "
      "no longer lists. To keep UFC 1, uncomment its ID in `EXTRA_UFC_EVENT_IDS` in `src/clean.py`.")
    top = sorted(wl["dropped_by_first_word"].items(), key=lambda kv: -kv[1])[:12]
    w("- Dropped, by first word of the event name: " + ", ".join(f"{k} {v}" for k, v in top) + ", …")
    w(f"- Greco1899 events missing from this dataset version: {wl['greco_events_not_in_dataset']} "
      f"({' to '.join(wl['greco_events_not_in_dataset_dates'])}). The dataset ends at the 2026-08-08 event.\n")

    w("### Duplicates\n")
    d = report["duplicates"]
    drows = [{"check": k, "duplicates": v} for k, v in d["raw_keys"].items()]
    drows += [{"check": k, "duplicates": v} for k, v in d.items() if k != "raw_keys"]
    drows.append({"check": "fighters table: same name, different fighter_id",
                  "duplicates": int(fighters["fighter_name"].duplicated().sum())})
    w(md_table(pd.DataFrame(drows)) + "\n")
    dn = fighters[fighters["fighter_name"].duplicated(keep=False)].sort_values("fighter_name")
    w("Shared names among UFC fighters (harmless here, because every table is keyed by `fighter_id`): "
      + ", ".join(sorted(dn["fighter_name"].unique())) + ".\n")

    w("### Missing-value markers converted to NULL\n")
    w("In this dataset missing values are **empty strings**. No `--` or `---` markers appear in any raw file "
      "(the scraper already converted them). Conversions per column:\n")
    mrows = []
    for f, info in report["raw"].items():
        for col, counts in info["missing_markers_to_nan"].items():
            if f == "master.csv" and not col.startswith(("r_total", "b_total")):
                continue  # master.csv repeats event/fighter columns; only its totals are used
            mrows.append({"file": f, "column": col, **{k: f"{v:,}" for k, v in counts.items()}})
    w(md_table(pd.DataFrame(mrows).fillna("")) + "\n")
    w("**No attempts vs. missing.** Counts are stored as integers: attempts = 0 means no attempts, and that is never NULL. "
      "NULL appears only where the source has no value. Among stats, that is `ctrl_s` in 1990s fights. "
      "Elsewhere it is fighter height, weight, reach, DOB, stance and nickname, and fight referee, details and winner (draw or no contest). "
      "No percentages are stored, so accuracy = landed/attempted must treat attempts = 0 as undefined, not as 0% and not as missing. "
      f"Fights with no per-round data ({int((fights['has_round_stats'] == 0).sum())}) have no rows in `fight_round_stats` "
      "and `has_round_stats = 0`; they are never filled with zeros.\n")
    w("Career-average columns dropped: " + ", ".join(f"`{c}`" for c in report["checks"]["career_avg_columns_dropped"]) + ".\n")

    # =============================================================== 2. Greco cross-check
    w("## 2. Cross-check against Greco1899\n")
    gs = pd.read_csv(GRECO_DIR / "ufc_fight_stats.csv", dtype=str)
    gd = pd.read_csv(GRECO_DIR / "ufc_fight_details.csv", dtype=str)
    gr = pd.read_csv(GRECO_DIR / "ufc_fight_results.csv", dtype=str)
    gt = pd.read_csv(GRECO_DIR / "ufc_fighter_tott.csv", dtype=str)
    for df in (gs, gd, gr):
        df["EVENT"], df["BOUT"] = df["EVENT"].str.strip(), df["BOUT"].str.strip()
    gd["fight_id"] = gd["URL"].str.extract(r"fight-details/([0-9a-f]{16})")[0]
    ambiguous = gd.duplicated(["EVENT", "BOUT"], keep=False)
    gd_u = gd.loc[~ambiguous, ["EVENT", "BOUT", "fight_id"]]

    funnel = [("Greco1899 fighter-round rows", len(gs))]
    gs = gs.dropna(subset=["ROUND"])
    funnel.append(("… with stats (drop all-empty rows)", len(gs)))
    gs = gs.merge(gd_u, on=["EVENT", "BOUT"], how="inner")
    funnel.append(("… mapped to a fight_id via (EVENT, BOUT); ambiguous keys dropped", len(gs)))
    gs = gs[gs["fight_id"].isin(fights["fight_id"])]
    funnel.append(("… fight is in this dataset (removes fights after 2026-08-08)", len(gs)))

    # fighter name -> id within each fight (exact name; if only one side matches, the other side is the other fighter)
    name = fighters.set_index("fighter_id")["fighter_name"]
    fx = fights[["fight_id", "red_fighter_id", "blue_fighter_id"]].copy()
    fx["red_name"], fx["blue_name"] = fx["red_fighter_id"].map(name), fx["blue_fighter_id"].map(name)
    gs = gs.merge(fx, on="fight_id")
    gs["fighter_id"] = np.where(gs["FIGHTER"] == gs["red_name"], gs["red_fighter_id"],
                                np.where(gs["FIGHTER"] == gs["blue_name"], gs["blue_fighter_id"], None))
    names_in_fight = gs.groupby("fight_id")["FIGHTER"].transform(lambda s: s.nunique())
    unresolved = gs["fighter_id"].isna()
    other = gs.loc[~unresolved].groupby("fight_id")["fighter_id"].first()
    fill = gs.loc[unresolved, "fight_id"].map(other)
    flip = np.where(fill == gs.loc[unresolved, "red_fighter_id"], gs.loc[unresolved, "blue_fighter_id"],
                    np.where(fill == gs.loc[unresolved, "blue_fighter_id"], gs.loc[unresolved, "red_fighter_id"], None))
    gs.loc[unresolved & (names_in_fight == 2), "fighter_id"] = pd.Series(flip, index=gs.index[unresolved])[names_in_fight[unresolved] == 2]
    n_by_name = int((~unresolved).sum())
    gs = gs.dropna(subset=["fighter_id"])
    funnel.append((f"… fighter resolved to fighter_id ({n_by_name:,} by exact name, the rest as the other fighter in the bout)", len(gs)))
    gs["round"] = gs["ROUND"].str.extract(r"(\d+)")[0].astype(int)

    g = pd.DataFrame({"fight_id": gs["fight_id"], "round": gs["round"], "fighter_id": gs["fighter_id"]})
    g["kd"] = pd.to_numeric(gs["KD"]).astype("Int64").values
    for col, stem in (("SIG.STR.", "sig"), ("TOTAL STR.", "total"), ("TD", "td"), ("HEAD", "head"), ("BODY", "body"),
                      ("LEG", "leg"), ("DISTANCE", "distance"), ("CLINCH", "clinch"), ("GROUND", "ground")):
        landed, att = of_split(gs[col])
        g[f"{stem}_landed"], g[f"{stem}_att"] = landed.values, att.values
    g["sub_att"] = pd.to_numeric(gs["SUB.ATT"]).astype("Int64").values
    g["rev"] = pd.to_numeric(gs["REV."]).astype("Int64").values
    g["ctrl_s"] = mmss(gs["CTRL"]).values
    g["sig_pct_raw"], g["td_pct_raw"] = gs["SIG.STR. %"].values, gs["TD %"].values
    exact_dupes = int(g.duplicated().sum())
    g = g.drop_duplicates()
    funnel.append((f"… after removing {exact_dupes} exact duplicate rows (Greco1899 stores renamed events twice)", len(g)))
    dupe_g = int(g.duplicated(["fight_id", "round", "fighter_id"], keep=False).sum())
    g = g.drop_duplicates(["fight_id", "round", "fighter_id"], keep=False)
    funnel.append((f"… unique (fight_id, round, fighter_id) ({dupe_g} rows with conflicting duplicate keys dropped)", len(g)))
    m = rounds.merge(g, on=["fight_id", "round", "fighter_id"], how="inner", suffixes=("", "_g"))
    funnel.append(("… matched to a row in `fight_round_stats`", len(m)))
    funnel.append(("`fight_round_stats` rows with no Greco1899 counterpart", len(rounds) - len(m)))

    w("### Matching\n")
    w(md_table(pd.DataFrame(funnel, columns=["stage", "rows"]).assign(rows=lambda x: x["rows"].map("{:,}".format))) + "\n")

    w("### Round-level stats: mismatch rate per column\n")
    w("Mismatch rate = rows where both values are present but differ ÷ rows where both are present.\n")
    crow = []
    for c in STAT_COLS:
        r = compare(m[c], m[f"{c}_g"])
        crow.append({"column": c, "compared": f"{r['compared']:,}", "mismatches": r["mismatches"],
                     "mismatch rate": pct(r["mismatches"], r["compared"], 3),
                     "NULL only here": r["null_only_in_neel"], "NULL only in Greco": r["null_only_in_greco"]})
    w(md_table(pd.DataFrame(crow)) + "\n")
    mm_any = sum(int(x["mismatches"]) for x in crow)
    if mm_any:
        bad = m.loc[np.logical_or.reduce([(m[c].notna() & m[f"{c}_g"].notna() & (m[c] != m[f"{c}_g"])).to_numpy(dtype=bool) for c in STAT_COLS])]
        w("Sample of mismatching rows (fight_id, round, fighter_id):\n")
        w(md_table(bad[["fight_id", "round", "fighter_id"]].head(10)) + "\n")

    # "no attempts" consistency: Greco keeps the site's '---' for 0 attempts
    td_noatt = m["td_pct_raw"].str.strip() == "---"
    sig_noatt = m["sig_pct_raw"].str.strip() == "---"
    w("**No attempts vs. missing, checked against the site markers.** Greco1899 keeps the site's raw `---` percentage marker. "
      f"TD % is `---` on {int(td_noatt.sum()):,} rows: on {pct(int((td_noatt & (m['td_att'] == 0)).sum()), int(td_noatt.sum()))} of them "
      f"this dataset has `td_att = 0`, and none has NULL. Conversely, {int(((m['td_att'] == 0) & ~td_noatt).sum())} rows with `td_att = 0` "
      f"are not `---` in Greco1899. Sig. str. % `---`: {int(sig_noatt.sum())} rows, "
      f"{pct(int((sig_noatt & (m['sig_att'] == 0)).sum()), int(sig_noatt.sum()))} with `sig_att = 0`. "
      f"Control time: Greco1899 `--` on {int((m['ctrl_s_g'].isna()).sum()):,} matched rows; this dataset has NULL on "
      f"{int((m['ctrl_s'].isna()).sum()):,}, both NULL on {int((m['ctrl_s'].isna() & m['ctrl_s_g'].isna()).sum()):,}.\n")

    # ---- fight-level
    w("### Fight results\n")
    gr = gr.merge(gd_u, on=["EVENT", "BOUT"], how="inner")
    gr["fight_id"] = gr["fight_id"].fillna(gr["URL"].str.extract(r"fight-details/([0-9a-f]{16})")[0])
    dup_results = int(gr.duplicated("fight_id").sum())
    gr = gr.drop_duplicates("fight_id")
    fm = fights.merge(gr, on="fight_id", how="inner")
    outcome_map = {"W/L": "red", "L/W": "blue", "D/D": "draw", "NC/NC": "no_contest"}
    neel_outcome = np.select([fights["result_status"] == "draw", fights["result_status"] == "no_contest",
                              fights["winner_id"] == fights["red_fighter_id"], fights["winner_id"] == fights["blue_fighter_id"]],
                             ["draw", "no_contest", "red", "blue"], default=None)
    fm["neel_outcome"] = fm["fight_id"].map(dict(zip(fights["fight_id"], neel_outcome)))
    frows = []
    pairs = [("outcome (winner corner)", fm["neel_outcome"], fm["OUTCOME"].str.strip().map(outcome_map)),
             ("method", fm["method"].str.strip(), fm["METHOD"].str.strip()),
             ("finish_round", fm["finish_round"].astype("Int64"), pd.to_numeric(fm["ROUND"]).astype("Int64")),
             ("finish_time_s", fm["finish_time_s"].astype("Int64"), mmss(fm["TIME"])),
             ("time_format", fm["time_format"].str.strip(), fm["TIME FORMAT"].str.strip()),
             ("referee (exact)", fm["referee"].str.strip(), fm["REFEREE"].str.strip()),
             ("referee (ignoring whitespace)", fm["referee"].str.replace(r"\s+", "", regex=True),
              fm["REFEREE"].str.replace(r"\s+", "", regex=True))]
    for label, a, b in pairs:
        r = compare(a.reset_index(drop=True), b.reset_index(drop=True))
        frows.append({"column": label, "compared": f"{r['compared']:,}", "mismatches": r["mismatches"],
                      "mismatch rate": pct(r["mismatches"], r["compared"], 3),
                      "NULL only here": r["null_only_in_neel"], "NULL only in Greco": r["null_only_in_greco"]})
    w(f"{len(fm):,} fights matched by fight_id ({dup_results} duplicate Greco1899 result rows from renamed events were removed first).\n")
    w(md_table(pd.DataFrame(frows)) + "\n")
    meth = fm[fm["method"].str.strip() != fm["METHOD"].str.strip()]
    if len(meth):
        w("Fights whose method or outcome differs:\n")
        w(md_table(meth[["fight_id", "event_date", "BOUT", "result_status", "method", "OUTCOME", "METHOD"]]
                   .rename(columns={"BOUT": "bout", "OUTCOME": "Greco outcome", "METHOD": "Greco method"})) + "\n")
        w("On the pinned versions this is a single fight, later **overturned to a no contest**. Greco1899's incremental job scrapes "
          "each fight only once (right after the event), so it still shows the original decision. This dataset reflects the overturn.\n")
    refd = fm[fm["referee"].notna() & fm["REFEREE"].notna() & (fm["referee"] != fm["REFEREE"].str.strip())]
    if len(refd):
        w("Referee differences are whitespace only (Greco1899 deletes double spaces instead of collapsing them), e.g. "
          + "; ".join(f"`{a}` ↔ `{b}`" for a, b in refd[["referee", "REFEREE"]].drop_duplicates().head(4).values)
          + f". Greco1899 also has {int((fm['referee'].notna() & fm['REFEREE'].isna()).sum())} empty referees where this dataset has one.\n")

    # ---- fighters
    w("### Fighter attributes\n")
    gt["fighter_id"] = gt["URL"].str.extract(r"fighter-details/([0-9a-f]{16})")[0]
    gt = gt.drop_duplicates("fighter_id")
    for c in ("HEIGHT", "WEIGHT", "REACH", "STANCE", "DOB"):
        gt[c] = gt[c].str.strip().replace({"--": np.nan, "": np.nan})
    fa = fighters.merge(gt, on="fighter_id", how="inner")
    arows = []
    apairs = [("fighter_name", fa["fighter_name"], fa["FIGHTER"].str.strip()),
              ("height_in", fa["height_in"], height_in(fa["HEIGHT"])),
              ("weight_lbs", fa["weight_lbs"], pd.to_numeric(fa["WEIGHT"].str.extract(r"(\d+)")[0])),
              ("reach_in", fa["reach_in"], pd.to_numeric(fa["REACH"].str.extract(r"(\d+(?:\.\d+)?)")[0])),
              ("stance", fa["stance"], fa["STANCE"]),
              ("dob", fa["dob"], pd.to_datetime(fa["DOB"], format="%b %d, %Y", errors="coerce").dt.strftime("%Y-%m-%d"))]
    for label, a, b in apairs:
        r = compare(a.astype(object).where(a.notna(), None), b.astype(object).where(b.notna(), None))
        arows.append({"column": label, "compared": f"{r['compared']:,}", "mismatches": r["mismatches"],
                      "mismatch rate": pct(r["mismatches"], r["compared"], 3),
                      "NULL only here": r["null_only_in_neel"], "NULL only in Greco": r["null_only_in_greco"]})
    w(f"{len(fa):,} of {len(fighters):,} fighters matched by fighter_id.\n")
    w(md_table(pd.DataFrame(arows)) + "\n")
    name_mm = fa[fa["fighter_name"] != fa["FIGHTER"].str.strip()]
    if len(name_mm):
        w(f"All {len(name_mm)} name differences (this dataset ↔ Greco1899): "
          + "; ".join(f"{a} ↔ {b}" for a, b in name_mm[["fighter_name", "FIGHTER"]].values) + ".\n")
    w("Interpretation (observed on the pinned versions): Greco1899 scraped most fighter profiles in a full refresh on 2025-11-22 and only "
      "adds new fighters afterwards, while this dataset re-scraped every profile for v5 (2026-08). Most name differences look like "
      "profile edits between the two scrapes (Cortes-Acosta → Cortes Acosta, Kamaka → Kamaka III, Zachary → Zach Reese). "
      "Weight differences of one division are consistent with ufcstats updating a fighter's listed weight after a division change "
      "(likely, not verified). A few pairs look like **different people under one ID** in one of the sources: "
      "`bc7f4b2d4c2724a2` is \"Regina Malpica\" in Greco1899 but \"Tyrell Fortune\" here, and this dataset has that ID only in "
      "heavyweight fights, so Greco1899 is wrong there. Jose Montanha ↔ Henrique Da Silva Lopes and Jose Souza ↔ Jose Henrique "
      "were not resolved. Joins are by `fighter_id`, so names don't affect them. "
      "Treat height, weight and reach as a snapshot at scrape time, not as values at fight time.\n")
    ex_w = fa[fa["weight_lbs"].notna() & (fa["weight_lbs"] != pd.to_numeric(fa["WEIGHT"].str.extract(r"(\d+)")[0]))]
    if len(ex_w):
        w("Weight differences (sample): " + "; ".join(f"{n} {int(a)} ↔ {b}" for n, a, b in ex_w[["fighter_name", "weight_lbs", "WEIGHT"]].head(6).values) + ".\n")

    # =============================================================== 3. internal consistency
    w("## 3. Internal consistency\n")
    crow = []

    def add(check: str, scope: str, n_bad: int, n: int, note: str = ""):
        crow.append({"check": check, "scope": scope, "rows checked": f"{n:,}", "violations": f"{n_bad:,}",
                     "rate": pct(n_bad, n, 3), "note": note})

    # (a) round sums == totals
    sums = rounds.groupby(["fight_id", "fighter_id"])[COUNT_COLS].sum().reset_index()
    ctrl_sum = rounds.groupby(["fight_id", "fighter_id"])["ctrl_s"].agg(lambda s: s.sum() if s.notna().all() else pd.NA).reset_index()
    st = totals.merge(sums, on=["fight_id", "fighter_id"], suffixes=("", "_sum")).merge(ctrl_sum, on=["fight_id", "fighter_id"], suffixes=("", "_sum"))
    bad_any = np.zeros(len(st), dtype=bool)
    per_col = []
    for c in COUNT_COLS:
        b = (st[c] != st[f"{c}_sum"]).to_numpy(dtype=bool)
        bad_any |= b
        per_col.append((c, int(b.sum())))
    add("round stats sum to fight totals (all 21 count columns)", "fighter-fights with round stats", int(bad_any.sum()), len(st),
        "worst columns: " + ", ".join(f"{c} {n}" for c, n in sorted(per_col, key=lambda x: -x[1])[:3] if n) if bad_any.any() else "")
    cc = st["ctrl_s"].notna() & st["ctrl_s_sum"].notna()
    add("round control time sums to total control time", "fighter-fights with ctrl in every round",
        int((st.loc[cc, "ctrl_s"] != st.loc[cc, "ctrl_s_sum"]).sum()), int(cc.sum()))
    add("total ctrl NULL ⇔ some round ctrl NULL", "fighter-fights with round stats",
        int((st["ctrl_s"].isna() != st["ctrl_s_sum"].isna()).sum()), len(st))

    # (b) landed <= attempted, sig <= total
    for tbl, df in (("round", rounds), ("total", totals)):
        n_bad = 0
        worst = []
        for p in PAIRS:
            b = int((df[f"{p}_landed"] > df[f"{p}_att"]).sum())
            n_bad += b
            if b:
                worst.append(f"{p} {b}")
        add(f"landed ≤ attempted ({len(PAIRS)} pairs)", f"{tbl} rows × pairs", n_bad, len(df) * len(PAIRS), ", ".join(worst))
        add("sig. strikes ≤ total strikes (landed and attempted)", f"{tbl} rows",
            int(((df["sig_landed"] > df["total_landed"]) | (df["sig_att"] > df["total_att"])).sum()), len(df))
        for grp, label in ((SPLITS_TARGET, "head + body + leg"), (SPLITS_POSITION, "distance + clinch + ground")):
            bl = df[[f"{t}_landed" for t in grp]].sum(axis=1) != df["sig_landed"]
            ba = df[[f"{t}_att" for t in grp]].sum(axis=1) != df["sig_att"]
            add(f"{label} = sig. strikes (landed and attempted)", f"{tbl} rows", int((bl | ba).sum()), len(df))

    # (c) control time vs round length
    fl = fights.set_index("fight_id")
    rr = rounds.merge(fights[["fight_id", "period_lengths_min", "finish_round", "finish_time_s"]], on="fight_id")
    def round_len_s(row):
        if not isinstance(row.period_lengths_min, str):
            return np.nan
        L = [int(x) for x in row.period_lengths_min.split("-")]
        return L[row.round - 1] * 60 if row.round <= len(L) else np.nan
    rr["round_len_s"] = [round_len_s(r) for r in rr[["period_lengths_min", "round"]].itertuples()]
    rr["elapsed_s"] = np.where(rr["round"] == rr["finish_round"], rr["finish_time_s"], rr["round_len_s"])
    has = rr["ctrl_s"].notna() & rr["round_len_s"].notna()
    add("ctrl ≤ scheduled round length", "fighter-rounds with ctrl and a fixed format",
        int((rr.loc[has, "ctrl_s"] > rr.loc[has, "round_len_s"]).sum()), int(has.sum()))
    has_e = rr["ctrl_s"].notna() & rr["elapsed_s"].notna()
    bad_e = rr.loc[has_e & (rr["ctrl_s"] > rr["elapsed_s"])]
    add("ctrl ≤ time actually elapsed in the round (final round = finish time)", "fighter-rounds with ctrl and a fixed format",
        len(bad_e), int(has_e.sum()))
    both = rr.groupby(["fight_id", "round"]).agg(ctrl_sum=("ctrl_s", lambda s: s.sum() if s.notna().all() else np.nan),
                                                 elapsed_s=("elapsed_s", "first")).dropna()
    add("red ctrl + blue ctrl ≤ time elapsed in the round", "fight-rounds with ctrl and a fixed format",
        int((both["ctrl_sum"] > both["elapsed_s"]).sum()), len(both))
    ntl = int((fights["time_format"] == "No Time Limit").sum())

    # (d) round rows vs finish round
    rc = rounds.groupby("fight_id")["round"].agg(["max", "nunique", "count"]).join(fl[["finish_round"]])
    add("max(round) in round stats = finish_round", "fights with round stats", int((rc["max"] != rc["finish_round"]).sum()), len(rc))
    add("rounds are contiguous 1..n with two fighters each", "fights with round stats",
        int(((rc["nunique"] != rc["max"]) | (rc["count"] != 2 * rc["max"])).sum()), len(rc))
    add("winner is one of the two fighters", "fights", report["checks"]["winner_not_in_fight"], len(fights))
    add("round-file corner IDs agree with fight file", "round rows (wide)", report["checks"]["round_corner_ids_disagree_with_fight"],
        len(rounds) // 2)

    w(md_table(pd.DataFrame(crow)) + "\n")
    w(f"Control-time checks skip the {ntl} fights with `No Time Limit` (no fixed round length).\n")
    tz, cz = report["checks"]["totals_dropped_no_round_stats"], report["checks"]["totals_ctrl_zero_set_null"]
    w("**Zero-filled totals fixed in `clean.py`.** The raw fight totals (`master.csv`) turn missing data into zeros, and the checks above run "
      f"after these fixes. (1) The {tz['fights']} fights with no per-round data have all-zero totals ({tz['rows_all_zero']} of "
      f"{tz['rows']} fighter rows entirely zero). The site has no stats for them, so those rows are dropped. (2) In {cz['rows']} "
      "fighter-fights every round's control time is missing (the site shows `--`), but the total was "
      f"{', '.join(f'{k} ({v})' for k, v in cz['values_before'].items())}. Those totals are set to NULL. "
      f"There are {cz['rows_with_only_some_round_ctrl_null']} fighter-fights with only some rounds missing control time. "
      "Because the totals match the round sums exactly and invent zeros where rounds are missing, they look derived from the rounds rather than "
      "scraped from the site's Totals table, so they carry no independent information.\n")
    if len(bad_e):
        ex = bad_e.merge(fights[["fight_id", "event_date", "time_format"]], on="fight_id")
        w("Examples of control time exceeding the elapsed time:\n")
        w(md_table(ex[["fight_id", "event_date", "round", "fighter_id", "ctrl_s", "elapsed_s", "time_format"]].head(8)) + "\n")
    bad_sum = st.loc[bad_any]
    if len(bad_sum):
        w("Examples of round stats not summing to fight totals:\n")
        cols = [c for c, n in per_col if n][:3]
        w(md_table(bad_sum[["fight_id", "fighter_id"] + sum(([c, f"{c}_sum"] for c in cols), [])].head(8)) + "\n")

    # =============================================================== 4. completeness by year
    w("## 4. Completeness by year (for choosing a date cutoff)\n")
    rr2 = rounds.copy()
    rr2["split_ok"] = True
    for grp in (SPLITS_TARGET, SPLITS_POSITION):
        rr2["split_ok"] &= (rr2[[f"{t}_landed" for t in grp]].sum(axis=1) == rr2["sig_landed"]) & \
                           (rr2[[f"{t}_att" for t in grp]].sum(axis=1) == rr2["sig_att"])
    per_fight = rr2.groupby("fight_id").agg(split_ok=("split_ok", "all"), ctrl_ok=("ctrl_s", lambda s: s.notna().all()),
                                            max_round=("round", "max"), n=("round", "count"))
    f = fights[["fight_id", "event_date", "finish_round", "has_round_stats"]].merge(per_fight, on="fight_id", how="left")
    f["all_rounds"] = (f["has_round_stats"] == 1) & (f["max_round"] == f["finish_round"]) & (f["n"] == 2 * f["finish_round"])
    f["splits"] = f["all_rounds"] & f["split_ok"].fillna(False).astype(bool)
    f["ctrl"] = f["all_rounds"] & f["ctrl_ok"].fillna(False).astype(bool)
    f["both"] = f["splits"] & f["ctrl"]
    f["year"] = f["event_date"].str[:4].astype(int)
    y = f.groupby("year").agg(fights=("fight_id", "count"), round_stats=("all_rounds", "sum"),
                              splits=("splits", "sum"), ctrl=("ctrl", "sum"), both=("both", "sum")).reset_index()
    table = pd.DataFrame({
        "year": y["year"], "fights": y["fights"],
        "per-round stats for every round": [pct(a, b, 1) for a, b in zip(y["round_stats"], y["fights"])],
        "complete strike splits": [pct(a, b, 1) for a, b in zip(y["splits"], y["fights"])],
        "control time in every round": [pct(a, b, 1) for a, b in zip(y["ctrl"], y["fights"])],
        "splits + control time": [pct(a, b, 1) for a, b in zip(y["both"], y["fights"])],
    })
    w("Definitions, per fight:\n")
    w("- **per-round stats for every round**: `fight_round_stats` has both fighters for every round 1..`finish_round`")
    w("- **complete strike splits**: that, plus in every fighter-round head + body + leg and distance + clinch + ground both equal significant strikes (landed and attempted)")
    w("- **control time in every round**: per-round stats, plus `ctrl_s` present for both fighters in every round\n")
    w(md_table(table) + "\n")
    tot = y[["fights", "round_stats", "splits", "ctrl", "both"]].sum()
    w(f"All years: {tot['fights']:,} fights. Splits + control time complete for {tot['both']:,} ({pct(tot['both'], tot['fights'], 1)}).\n")

    DOC_PATH.write_text("\n".join(out))
    con.close()
    print(f"wrote {DOC_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
