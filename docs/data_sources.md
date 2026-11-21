# Public UFCStats-derived datasets: comparison

Surveyed 2026-11-20. ufcstats.com itself was **not** contacted for this survey.
Paid scraping services (the Apify actors) were skipped.

**How claims were checked**

- ✅ **Verified**: I downloaded the files and checked them with code: row counts, date ranges, and the
  edge-case fights from [site_map.md](site_map.md) §4 looked up by their ufcstats IDs.
- 🟡 **Columns verified**: column names come from Kaggle's machine-readable Croissant metadata
  (`/datasets/{owner}/{slug}/croissant/download`), but the rows were not inspected.
- ⚪ **Description only**: claims come from the README or dataset description.

Public Kaggle datasets downloaded **without a login** (`/api/v1/datasets/download/{owner}/{slug}`), so nothing needs to be downloaded by hand.

---

## Summary

| | Per-round, target + position splits, TD, sub att, ctrl | Coverage (events) | Last updated | Fighter attrs (ht / reach / stance / DOB) | IDs | Non-UFC fights | License | Checked |
|---|---|---|---|---|---|---|---|---|
| **neelagiriaditya** (Kaggle) | ✅ all, as integers | 1993-11-12 → 2026-08-08 | 2026-08-11 | ✅ all | ✅ ufcstats IDs everywhere | ⚠️ **included** (476 events) | CC0 | ✅ |
| **Greco1899/scrape_ufc_stats** (GitHub) | ✅ all, raw `x of y` strings | 1994-03-11 → 2026-11-14 | 2026-11-18 (automated) | ✅ all | ⚠️ **names only** in stats | none | GPL-3.0 | ✅ |
| **scarekrow/ufc-data** (Kaggle) | ✅ all, wide (361 cols) | 1994-03-11 → 2026-11-14 | 2026-11-16 | ✅ all (cm) | fight and fighter URLs | none | "Other", not actually stated | ✅ |
| **mtoto/ufc.stats** (GitHub, R) | ❌ **no control time** | 1993-11-12 → 2026-03-07 | 2026-07-18 | ❌ none | integer fight id, names | none seen | MIT | ✅ |
| jerzyszocik/ufc-fight-forecast… (Kaggle) | claims per-round incl. ctrl | not stated | 2026-06-10 | claims bios | — | — | CC0 | ⚪ |
| alexmagnus24/ufc-fight-statistics… (Kaggle) | 🟡 per-round, head/body/leg/dist/clinch/ground, ctrl (minutes) | 2016-07 → 2024-11 (desc.) | 2025-02-02 | ❌ | names | — | Apache 2.0 | 🟡 |
| Jadevit/fight-canon-db (GitHub) | claims round-by-round + judges' cards | 1963 → 2026-11-20 (all MMA) | 2026-11-20 | claims bios | own IDs + ufcstats aliases | all promotions | MIT | ⚪ (185 MB SQLite, not downloaded) |

**Didn't qualify** (checked from Croissant columns or files 🟡):

| Dataset | Why it doesn't qualify |
|---|---|
| rajeevw/ufcdata (Kaggle, 921 votes, CC0) | `raw_total_fight_data.csv` has **per-fight** totals with all splits, but no per-round data, and it was last updated **2021-03-21** |
| maksbasher/ufc-complete-dataset… (CC0, 2024-03) | Per-fight only; no head/body/leg or position splits |
| leandroiber/ufc-stats-complete-dataset (CC0, 2026-05) | Per-fight sig. strikes, TD, ctrl only; no target or position splits |
| jossilva3110/ufc-dataset-1994-2026 (MIT, 2026-03) | Per-fight totals; head/body/leg etc. appear to be landed only |
| aminealibi/ufc-fights-fighters… (CC0, 2026-04) | Per-fight target and position **percentages** only (raw files have `x of y` per fight, but no rounds) |
| mdabbert/ultimate-ufc-dataset (CC BY 4.0) | Pre-fight career averages plus betting odds |
| calmdownkarm/ufcdataset (CC0, 2018) | Cumulative pre-fight round stats, 2013–2018 |
| fanghuiz/ufc-stats-crawler (GitHub, MIT) | Crawler with a 2019 sample of data |

---

## Candidates in detail

### 1. neelagiriaditya/ufc-datasets-1994-2025 (Kaggle) ✅
Scraper: [github.com/neelagiriaditya/ufc-scraper](https://github.com/neelagiriaditya/ufc-scraper) (MIT). Stated source: ufcstats.com.

- **Files:** `event.csv`, `fight.csv`, `round.csv`, `fighter.csv`, `fighter_bonus.csv`, `master.csv` (a wide join), `scrape_error.csv`.
- **Granularity ✅:** `round.csv` has one row per fight-round (25,131 rows) with `r_`/`b_` integer columns:
  kd, sig landed/atmp, total str, td success/atmp, sub_att, rev, ctrl, and head/body/leg/distance/clinch/ground landed **and** attempted.
  `ctrl` is still an `M:SS` string. No percentage columns (they can be derived).
- **Coverage ✅:** 1,259 events from **UFC 1, 1993-11-12** (which ufcstats no longer lists) to 2026-08-08. 11,441 fights. Kaggle modified 2026-08-11.
  **The 14 UFC events since then are missing.**
- **Fighter attributes ✅:** `height` (string `5' 10"`), `weight_lbs`, `reach_inches` (number), `stance`, `dob` (ISO), plus career averages. 4,581 fighters.
- **Checklist quirks:**
  - *Corner vs winner ✅:* `r_id`/`b_id` follow the fight-page order (UFC 321: Aspinall is r, Gane is b). The result is explicit: `winner_id` (NaN for draw/NC) and `result_status` ∈ {win 11,238, draw 85, no_contest 118}. This is the cleanest design of the candidates.
  - *Missing values ✅:* `--` → NaN (reach NaN for 1,992 fighters, DOB 510, ctrl NaN in 2,658 rounds, e.g. UFC 2). `---` doesn't come up because percentages aren't stored.
    **Fake zeros are kept.** Scott Morris and Tom Aaron have `slpm = 0.0`, `str_acc = 0`.
  - *Non-UFC ⚠️ (verified):* only **783** events match the UFC events list. The other **476** are DWCS (89), PRIDE (66), Strikeforce (63), WEC (50), DREAM, KOTC, EliteXC, …
    There is no promotion column. Filter by event name or against a UFC event list. UFC-only fights: 8,820.
  - *Fights without stats ✅:* 330 fights (21 of them UFC) have no rounds. Each one is logged in `scrape_error.csv`
    (`PartialFightParse: expected 2 round-by-round tables, found 0`).
  - `details` keeps the raw whitespace from the HTML (`"Punches to Head From Mount\n      Submission to Strikes"`).
  - 8 duplicate fighter names, which don't matter because everything is ID-keyed.

### 2. Greco1899/scrape_ufc_stats (GitHub) ✅
178 stars. CSVs are committed to the repo. Stated source: ufcstats.com. README: a GCP Cloud Run job runs daily and pushes refreshed CSVs; the last full refresh was 2025-11-22.

- **Files:** `ufc_event_details`, `ufc_fight_details`, `ufc_fight_results`, `ufc_fight_stats`, `ufc_fighter_details`, `ufc_fighter_tott`.
- **Granularity ✅:** `ufc_fight_stats.csv` has one row per **fighter per round** (42,429 rows), with every column from the site's per-round tables as **raw strings**
  (`41 of 65`, `63%`, `0:03`). There are no overall-total rows, so sum the rounds.
- **Coverage ✅:** 797 events, UFC 2 (1994-03-11) → UFC 334 (2026-11-14). The upcoming event is excluded and UFC 1 is absent, as on the site.
  Recent commits: "2026-11-18 data refresh", 11-15, 11-08, … roughly every 3–7 days.
- **Fighter attributes ✅:** `ufc_fighter_tott.csv` has HEIGHT, WEIGHT, REACH, STANCE, DOB as raw site strings, plus the fighter URL (ID).
- **Checklist quirks:**
  - *Corner vs winner ✅:* rows follow the fight-page order. `OUTCOME` is `W/L` (5,663), `L/W` (3,205), `D/D` (68), `NC/NC` (92). Miller vs McKinney is `L/W`, as on the site.
  - *Missing values ✅:* **left exactly as on the site.** `---` in TD % (19,756 rows) and Sig. str. % (212); CTRL `--` (432 rows, 1990s); attributes `--` (reach 2,140 of 4,630, DOB 779, height 518). Blank stance → NaN (1,023).
    42 stats rows are entirely empty (old fights with no round data, e.g. UFC 17).
  - *Non-UFC ✅:* none. The events list contains only UFC events (TUF Finales and Noche UFC are UFC events).
    But `fighter_tott` lists 1,878 fighters who appear in no UFC stats row (e.g. Strikeforce-only fighters).
  - ⚠️ **No IDs in `fight_stats`.** Its key is (EVENT, BOUT, ROUND, FIGHTER) by name. Mapping a fighter to a fighter_id means joining by name:
    9 names belong to two fighters each (Bruno Silva, Jean Silva, Michael McDonald, …, covering 120 stat rows), and 23 stats names don't appear in `fighter_tott`.
    A fight_id is available via `ufc_fight_details` (EVENT + BOUT → URL), but (EVENT, BOUT) isn't unique: Sakuraba vs Silveira appears twice at UFC Ultimate Japan.
  - ⚠️ **25 duplicate fights** in `fight_results` where an event was renamed between scrapes ("UFC Fight Night: Lopes vs. Silva" vs "Noche UFC: Lopes vs. Silva").
  - Every `EVENT` in `fight_results` (and `METHOD`) has a trailing space. Stats rows don't, so strip before joining.
  - Judges' scores are concatenated with no separator: `Adrian Castro 47 - 48.Sal D'amato 47 - 48.Chris Lee 47 - 48.`, in the site's loser–winner order.

### 3. scarekrow/ufc-data (Kaggle) ✅
Description: "Kept current with an automated scrape after every live UFC card." It credits jerzyszocik and uses the same `UFC_full_data_silver` naming, so it is probably derived from that pipeline (my inference).

- **Granularity ✅:** one wide CSV, 9,003 fights × 361 columns: `f_{1,2}_r{1..5}_{head,body,leg,distance,clinch,ground}_{succ,att}`, `_ctrl`, `_td_1_succ/att`, `_submission_att`, …, plus fight totals such as `f_1_ctrl_time_sec`.
  Reshaping to long format is needed. Some column meanings aren't documented (`td_1` vs `td_2`, `sig_pct_succ`).
- **Coverage ✅:** 1994-03-11 → 2026-11-14 (798 event names). Kaggle modified 2026-11-16. 25 versions since 2026-07-02.
- **Fighter attributes ✅:** height_cm, weight_lbs, reach_cm, stance, dob (ISO) embedded on each fight row for both fighters. These are a snapshot taken at scrape time, not values at fight time.
- **Checklist quirks:**
  - *Corner vs winner ⚠️:* **reordered so the winner is f_1.** McKinney is f_1, although the fight page lists Miller first, so the corner order is lost. `winner` holds the winner's name; NaN for draw/NC.
  - *Missing values ⚠️:* inconsistent. UFC 2 control time is `--` on the site, NaN in the per-round columns, but **0.0** in `f_1_ctrl_time_sec`.
  - *Non-UFC ✅:* none seen (only UFC and TUF Finale events).
- **License ⚠️:** Kaggle says "Other (specified in description)", but the description contains no license statement.

### 4. mtoto/ufc.stats (GitHub, R package) ✅
- **Granularity ❌ for this project:** per fighter per round, with integer landed/attempted for all target and position splits, TD and sub attempts, **but no control-time column.** No fighter attributes. Fights have an integer `id`; fighters are identified by name only.
- **Coverage ✅:** 765 events, 1993-11-12 → 2026-03-07 (UFC 326). Last commit 2026-07-18. 104 rows have no date.
- **Quirks ✅:** `---` → **0** (`takedown_rate = 0` whenever attempts = 0), so "no attempts" can't be told apart from 0%. `winner` per fighter: W/L/D/NC (18 blank).
- License: MIT (2019, Tamas Szilagyi).

### 5. jerzyszocik/ufc-fight-forecast-complete-gold-modeling-dataset (Kaggle) ⚪
The description claims a "silver" file of raw UFCStats facts with per-round striking and grappling, control time and bios, refreshed weekly, plus a "gold" modeling file (rolling stats, odds, rankings). 44.7 MB zip, last modified 2026-06-10 despite the weekly claim, CC0. Croissant has no column list. Not downloaded. Probably the upstream of #3.

### 6. alexmagnus24/ufc-fight-statistics-july-2016-nov-2024 (Kaggle) 🟡
Wide per-round columns (`Head F1R1`, `Ctrl Time (Minutes) F1R1`, …, 194 cols), with Fighter1 = red corner (per description). Landed counts are split by target and position, but the column list suggests attempts aren't. Only 2016-07 → 2024-11, last updated 2025-02. Apache 2.0.

### 7. Jadevit/fight-canon-db (GitHub) ⚪
An open SQLite database (185 MB release, rebuilt daily, MIT) merging UFCStats, Sherdog, PFL, Oktagon, Sofascore, MMA Decisions (judges' cards) and BestFightOdds. It has a `round_stats` table (76,630 rows across sources) with a `source` column and ufcstats ID aliases.
It's interesting for judges' scores mapped to the correct fighter, but it's 2 months old, has 5 stars, and wasn't inspected.

---

## Licensing caveat (applies to all of them)

Every license above is what the uploader states. None of them owns the underlying ufcstats.com data, and I found no
terms of use from ufcstats granting redistribution rights. GPL-3.0 (Greco1899) is a software license applied to a repo that also contains data,
so how it applies to the CSVs is unclear. For a personal or academic clustering project this is a low risk. Check again before publishing a derived dataset.

---

## Cross-check between the two finalists ✅

I joined Greco1899 and neelagiriaditya on (fight_id, round, fighter name). **41,530** fighter-rounds matched; the rest are mostly fights after 2026-08-08 plus name mismatches.
On every matched row, head strikes landed and control time (where both are present) were **identical (100%)**. The two independent scrapes agree, which makes either one trustworthy for the values it has.

---

## Recommendation

**Use neelagiriaditya as the primary dataset, filtered to UFC events. Use Greco1899 to fill in recent events and to validate.**

Reasons:

1. **Fighter IDs matter most for this project.** Clustering fighters by style means grouping fight-rounds by fighter.
   neelagiriaditya keys every round to `r_id`/`b_id` ufcstats IDs and has `winner_id`/`result_status`. Greco1899 would force a
   name join with known collisions (Bruno Silva and others) and duplicate rows from renamed events.
2. **Its round table is already numeric and complete for style features** (target and position splits with attempts, TD, sub attempts, ctrl),
   and fights without stats are logged instead of silently missing.
3. **License:** CC0 is the clearest of the candidates.
4. **Its weaknesses are easy to handle:**
   - Non-UFC events: keep only event_ids in Greco1899's `ufc_event_details.csv`, or filter by name.
   - Staleness (14 events, about 3 months, missing): append them from Greco1899. For those fights the fight_id comes from
     `ufc_fight_details` and each fighter is one of two names in the bout, so the name→ID mapping is unambiguous within a fight.
   - Fake zeros in career averages: ignore them. Compute style features from `round.csv` instead.
5. **mtoto is out** (no control time). **scarekrow** is current, but it reorders corners and has no clear license. It's a good fallback if neelagiriaditya goes unmaintained.

Before building features, decide on a **date cutoff** (for example 2001+, after unified rules, or wherever round stats become consistent). Early-era fights
(no time limit, open weight, `--` control time) would distort style clusters. I didn't measure where the data quality changes; that's a good first task for the cleaning pipeline.
