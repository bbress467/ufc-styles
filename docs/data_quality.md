# Data quality report

Generated 2026-11-20 23:54 UTC by `src/validate.py`. Rerun it after `src/download_data.py` and `src/clean.py` to regenerate this file.

- Data: neelagiriaditya Kaggle **v5** (`data/raw/neelagiriaditya/v5`)
- Reference: Greco1899 commit `e87487e35b0c`, used for the UFC event whitelist and the cross-check only
- Output: `data/processed/ufc.sqlite`. Tables: `events`, `fighters`, `fights`, `fight_round_stats`, `fight_total_stats`

## 1. Row counts and duplicates

### Row counts at each step

| step | count | rows |
|---|---|---|
| raw | events | 1,259 |
| raw | fights | 11,441 |
| raw | fighters | 4,581 |
| raw | round_rows | 25,131 |
| raw | master_rows | 11,441 |
| after UFC whitelist | events | 783 |
| after UFC whitelist | fights | 8,820 |
| fights table | fights | 8,820 |
| fights table | fights_with_round_stats | 8,799 |
| fights table | fights_without_round_stats | 21 |
| fighters table | fighters_in_raw | 4,581 |
| fighters table | fighters_in_ufc_fights | 2,735 |
| round rows after UFC filter | round_rows | 20,720 |
| fight_round_stats (long) | rows | 41,440 |
| fight_total_stats (long, raw) | rows | 17,640 |
| fight_total_stats (long, cleaned) | rows | 17,598 |
| fight_total_stats (long, cleaned) | dropped_all_zero_rows_for_fights_without_round_stats | 42 |
| fight_total_stats (long, cleaned) | ctrl_zero_set_to_null | 362 |
| written to SQLite | events | 783 |
| written to SQLite | fighters | 2,735 |
| written to SQLite | fights | 8,820 |
| written to SQLite | fight_round_stats | 41,440 |
| written to SQLite | fight_total_stats | 17,598 |

`fight_round_stats` and `fight_total_stats` are long format (two rows per fight-round or fight, one per fighter), so they have twice as many rows as the raw wide tables.

### UFC whitelist

An event is kept only if its ufcstats `event_id` appears in Greco1899's UFC events list (797 events). **Kept 783, dropped 476.** Every dropped event has an empty `location` and every kept event has one (dropped with a location: 0; kept without one: 0), which independently supports the split.

- Dropped events whose name contains "UFC": Road To UFC 5.1, Road To UFC 5.2, Road to UFC 1.1, Road to UFC 1.2, Road to UFC 1.3, Road to UFC 1.4, Road to UFC 1.5, Road to UFC 1.6, Road to UFC 2.1 + 2.2, Road to UFC 2.3 + 2.4, Road to UFC 2.5 + 2.6, Road to UFC 2.7, Road to UFC 3.1 + 3.2, Road to UFC 3.3 + 3.4, Road to UFC 3.5 + 3.6, Road to UFC 4.1 + 4.2, Road to UFC 4.3 + 4.4, Road to UFC 4.5 + 4.6, UFC 1: The Beginning. All are "Road to UFC" (a tournament series, not official UFC bouts) except **UFC 1: The Beginning**, which ufcstats.com no longer lists. To keep UFC 1, uncomment its ID in `EXTRA_UFC_EVENT_IDS` in `src/clean.py`.
- Dropped, by first word of the event name: DWCS 89, PRIDE 66, Strikeforce 63, WEC 50, DREAM 21, K-1 20, Road 18, EliteXC 10, KOTC 9, IFL 8, Meca 8, Shooto 7, …
- Greco1899 events missing from this dataset version: 14 (August 15, 2026 to November 14, 2026). The dataset ends at the 2026-08-08 event.

### Duplicates

| check | duplicates |
|---|---|
| event.csv event_id | 0 |
| fight.csv fight_id | 0 |
| fighter.csv fighter_id | 0 |
| fighter.csv fighter_name (different ids) | 8 |
| round.csv (fight_id, round_no) | 0 |
| master.csv fight_id | 0 |
| fighter_bonus.csv (fight_id, bonus_type) | 0 |
| fight_round_stats (fight_id, round, fighter_id) | 0 |
| fight_total_stats (fight_id, fighter_id) | 0 |
| fighters table: same name, different fighter_id | 1 |

Shared names among UFC fighters (harmless here, because every table is keyed by `fighter_id`): Bruno Silva.

### Missing-value markers converted to NULL

In this dataset missing values are **empty strings**. No `--` or `---` markers appear in any raw file (the scraper already converted them). Conversions per column:

| file | column | (empty) |
|---|---|---|
| event.csv | location | 476 |
| fight.csv | winner_id | 203 |
| fight.csv | referee | 260 |
| fight.csv | details | 402 |
| fighter.csv | fighter_nick_name | 2,029 |
| fighter.csv | height | 372 |
| fighter.csv | weight_lbs | 88 |
| fighter.csv | reach_inches | 1,992 |
| fighter.csv | stance | 897 |
| fighter.csv | dob | 510 |
| round.csv | r_ctrl | 2,658 |
| round.csv | b_ctrl | 2,658 |

**No attempts vs. missing.** Counts are stored as integers: attempts = 0 means no attempts, and that is never NULL. NULL appears only where the source has no value. Among stats, that is `ctrl_s` in 1990s fights. Elsewhere it is fighter height, weight, reach, DOB, stance and nickname, and fight referee, details and winner (draw or no contest). No percentages are stored, so accuracy = landed/attempted must treat attempts = 0 as undefined, not as 0% and not as missing. Fights with no per-round data (21) have no rows in `fight_round_stats` and `has_round_stats = 0`; they are never filled with zeros.

Career-average columns dropped: `slpm`, `str_acc`, `sapm`, `str_def`, `td_avg`, `td_acc`, `td_def`, `sub_avg`.

## 2. Cross-check against Greco1899

### Matching

| stage | rows |
|---|---|
| Greco1899 fighter-round rows | 42,428 |
| … with stats (drop all-empty rows) | 42,386 |
| … mapped to a fight_id via (EVENT, BOUT); ambiguous keys dropped | 42,382 |
| … fight is in this dataset (removes fights after 2026-08-08) | 41,552 |
| … fighter resolved to fighter_id (41,522 by exact name, the rest as the other fighter in the bout) | 41,552 |
| … after removing 116 exact duplicate rows (Greco1899 stores renamed events twice) | 41,436 |
| … unique (fight_id, round, fighter_id) (0 rows with conflicting duplicate keys dropped) | 41,436 |
| … matched to a row in `fight_round_stats` | 41,436 |
| `fight_round_stats` rows with no Greco1899 counterpart | 4 |

### Round-level stats: mismatch rate per column

Mismatch rate = rows where both values are present but differ ÷ rows where both are present.

| column | compared | mismatches | mismatch rate | NULL only here | NULL only in Greco |
|---|---|---|---|---|---|
| kd | 41,436 | 0 | 0.000% | 0 | 0 |
| sig_landed | 41,436 | 0 | 0.000% | 0 | 0 |
| sig_att | 41,436 | 0 | 0.000% | 0 | 0 |
| total_landed | 41,436 | 0 | 0.000% | 0 | 0 |
| total_att | 41,436 | 0 | 0.000% | 0 | 0 |
| td_landed | 41,436 | 0 | 0.000% | 0 | 0 |
| td_att | 41,436 | 0 | 0.000% | 0 | 0 |
| sub_att | 41,436 | 0 | 0.000% | 0 | 0 |
| rev | 41,436 | 0 | 0.000% | 0 | 0 |
| head_landed | 41,436 | 0 | 0.000% | 0 | 0 |
| head_att | 41,436 | 0 | 0.000% | 0 | 0 |
| body_landed | 41,436 | 0 | 0.000% | 0 | 0 |
| body_att | 41,436 | 0 | 0.000% | 0 | 0 |
| leg_landed | 41,436 | 0 | 0.000% | 0 | 0 |
| leg_att | 41,436 | 0 | 0.000% | 0 | 0 |
| distance_landed | 41,436 | 0 | 0.000% | 0 | 0 |
| distance_att | 41,436 | 0 | 0.000% | 0 | 0 |
| clinch_landed | 41,436 | 0 | 0.000% | 0 | 0 |
| clinch_att | 41,436 | 0 | 0.000% | 0 | 0 |
| ground_landed | 41,436 | 0 | 0.000% | 0 | 0 |
| ground_att | 41,436 | 0 | 0.000% | 0 | 0 |
| ctrl_s | 41,008 | 0 | 0.000% | 0 | 0 |

**No attempts vs. missing, checked against the site markers.** Greco1899 keeps the site's raw `---` percentage marker. TD % is `---` on 19,290 rows: on 100.00% of them this dataset has `td_att = 0`, and none has NULL. Conversely, 0 rows with `td_att = 0` are not `---` in Greco1899. Sig. str. % `---`: 210 rows, 100.00% with `sig_att = 0`. Control time: Greco1899 `--` on 428 matched rows; this dataset has NULL on 428, both NULL on 428.

### Fight results

8,818 fights matched by fight_id (25 duplicate Greco1899 result rows from renamed events were removed first).

| column | compared | mismatches | mismatch rate | NULL only here | NULL only in Greco |
|---|---|---|---|---|---|
| outcome (winner corner) | 8,818 | 1 | 0.011% | 0 | 0 |
| method | 8,818 | 1 | 0.011% | 0 | 0 |
| finish_round | 8,818 | 0 | 0.000% | 0 | 0 |
| finish_time_s | 8,818 | 0 | 0.000% | 0 | 0 |
| time_format | 8,818 | 0 | 0.000% | 0 | 0 |
| referee (exact) | 8,792 | 13 | 0.148% | 0 | 26 |
| referee (ignoring whitespace) | 8,792 | 0 | 0.000% | 0 | 26 |

Fights whose method or outcome differs:

| fight_id | event_date | bout | result_status | method | Greco outcome | Greco method |
|---|---|---|---|---|---|---|
| 7ffdaa44fc8d111b | 2026-02-21 | Ode Osbourne vs. Alibi Idiris | no_contest | Overturned | L/W | Decision - Unanimous  |

On the pinned versions this is a single fight, later **overturned to a no contest**. Greco1899's incremental job scrapes each fight only once (right after the event), so it still shows the original decision. This dataset reflects the overturn.

Referee differences are whitespace only (Greco1899 deletes double spaces instead of collapsing them), e.g. `Mathew Rocca` ↔ `MathewRocca`; `Bobby Wombacher` ↔ `BobbyWombacher`; `Vyacheslav Kiselev` ↔ `VyacheslavKiselev`. Greco1899 also has 26 empty referees where this dataset has one.

### Fighter attributes

2,735 of 2,735 fighters matched by fighter_id.

| column | compared | mismatches | mismatch rate | NULL only here | NULL only in Greco |
|---|---|---|---|---|---|
| fighter_name | 2,735 | 17 | 0.622% | 0 | 0 |
| height_in | 2,658 | 14 | 0.527% | 0 | 56 |
| weight_lbs | 2,717 | 62 | 2.282% | 0 | 0 |
| reach_in | 2,022 | 9 | 0.445% | 0 | 58 |
| stance | 2,613 | 0 | 0.000% | 0 | 42 |
| dob | 2,575 | 10 | 0.388% | 0 | 57 |

All 17 name differences (this dataset ↔ Greco1899): Ben Johnston ↔ Benjamin Donovan Johnston; Adrian Luna Martinetti ↔ Juan Adrian Luna; Jose Montanha ↔ Henrique Da Silva Lopes; Tyrell Fortune ↔ Regina Malpica; Magomed Bibulatov ↔ Bibulatov Magomed; Waldo Cortes Acosta ↔ Waldo Cortes-Acosta; Kai Kamaka III ↔ Kai Kamaka; YiSak Lee ↔ Yi Sak Lee; Michael Aswell Jr. ↔ Michael Aswell; Patricio Pitbull ↔ Patricio Freire; Zach Reese ↔ Zachary Reese; Shem Rock ↔ Shaqueme Rock; Levi Rodrigues Jr. ↔ Levi Rodrigues; Treston Vines ↔ Tre'ston Vines; Regina Tarin ↔ Regina Malpica Rivera; Muhammad Saidov ↔ Muhammad Said; Jose Souza ↔ Jose Henrique.

Interpretation (observed on the pinned versions): Greco1899 scraped most fighter profiles in a full refresh on 2025-11-22 and only adds new fighters afterwards, while this dataset re-scraped every profile for v5 (2026-08). Most name differences look like profile edits between the two scrapes (Cortes-Acosta → Cortes Acosta, Kamaka → Kamaka III, Zachary → Zach Reese). Weight differences of one division are consistent with ufcstats updating a fighter's listed weight after a division change (likely, not verified). A few pairs look like **different people under one ID** in one of the sources: `bc7f4b2d4c2724a2` is "Regina Malpica" in Greco1899 but "Tyrell Fortune" here, and this dataset has that ID only in heavyweight fights, so Greco1899 is wrong there. Jose Montanha ↔ Henrique Da Silva Lopes and Jose Souza ↔ Jose Henrique were not resolved. Joins are by `fighter_id`, so names don't affect them. Treat height, weight and reach as a snapshot at scrape time, not as values at fight time.

Weight differences (sample): Carli Judice 115 ↔ 125 lbs.; Francis Marshall 155 ↔ 145 lbs.; Jose Montanha 259 ↔ 240 lbs.; Rafael Estevam 135 ↔ 125 lbs.; Ethyn Ewing 135 ↔ 145 lbs.; Tyrell Fortune 259 ↔ 125 lbs..

## 3. Internal consistency

| check | scope | rows checked | violations | rate | note |
|---|---|---|---|---|---|
| round stats sum to fight totals (all 21 count columns) | fighter-fights with round stats | 17,598 | 0 | 0.000% |  |
| round control time sums to total control time | fighter-fights with ctrl in every round | 17,236 | 0 | 0.000% |  |
| total ctrl NULL ⇔ some round ctrl NULL | fighter-fights with round stats | 17,598 | 0 | 0.000% |  |
| landed ≤ attempted (9 pairs) | round rows × pairs | 372,960 | 0 | 0.000% |  |
| sig. strikes ≤ total strikes (landed and attempted) | round rows | 41,440 | 0 | 0.000% |  |
| head + body + leg = sig. strikes (landed and attempted) | round rows | 41,440 | 0 | 0.000% |  |
| distance + clinch + ground = sig. strikes (landed and attempted) | round rows | 41,440 | 0 | 0.000% |  |
| landed ≤ attempted (9 pairs) | total rows × pairs | 158,382 | 0 | 0.000% |  |
| sig. strikes ≤ total strikes (landed and attempted) | total rows | 17,598 | 0 | 0.000% |  |
| head + body + leg = sig. strikes (landed and attempted) | total rows | 17,598 | 0 | 0.000% |  |
| distance + clinch + ground = sig. strikes (landed and attempted) | total rows | 17,598 | 0 | 0.000% |  |
| ctrl ≤ scheduled round length | fighter-rounds with ctrl and a fixed format | 41,008 | 0 | 0.000% |  |
| ctrl ≤ time actually elapsed in the round (final round = finish time) | fighter-rounds with ctrl and a fixed format | 41,008 | 0 | 0.000% |  |
| red ctrl + blue ctrl ≤ time elapsed in the round | fight-rounds with ctrl and a fixed format | 20,504 | 0 | 0.000% |  |
| max(round) in round stats = finish_round | fights with round stats | 8,799 | 0 | 0.000% |  |
| rounds are contiguous 1..n with two fighters each | fights with round stats | 8,799 | 0 | 0.000% |  |
| winner is one of the two fighters | fights | 8,820 | 0 | 0.000% |  |
| round-file corner IDs agree with fight file | round rows (wide) | 20,720 | 0 | 0.000% |  |

Control-time checks skip the 31 fights with `No Time Limit` (no fixed round length).

**Zero-filled totals fixed in `clean.py`.** The raw fight totals (`master.csv`) turn missing data into zeros, and the checks above run after these fixes. (1) The 21 fights with no per-round data have all-zero totals (42 of 42 fighter rows entirely zero). The site has no stats for them, so those rows are dropped. (2) In 362 fighter-fights every round's control time is missing (the site shows `--`), but the total was 0 (362). Those totals are set to NULL. There are 0 fighter-fights with only some rounds missing control time. Because the totals match the round sums exactly and invent zeros where rounds are missing, they look derived from the rounds rather than scraped from the site's Totals table, so they carry no independent information.

## 4. Completeness by year (for choosing a date cutoff)

Definitions, per fight:

- **per-round stats for every round**: `fight_round_stats` has both fighters for every round 1..`finish_round`
- **complete strike splits**: that, plus in every fighter-round head + body + leg and distance + clinch + ground both equal significant strikes (landed and attempted)
- **control time in every round**: per-round stats, plus `ctrl_s` present for both fighters in every round

| year | fights | per-round stats for every round | complete strike splits | control time in every round | splits + control time |
|---|---|---|---|---|---|
| 1994 | 31 | 93.5% | 93.5% | 0.0% | 0.0% |
| 1995 | 40 | 85.0% | 85.0% | 0.0% | 0.0% |
| 1996 | 43 | 83.7% | 83.7% | 0.0% | 0.0% |
| 1997 | 41 | 95.1% | 95.1% | 0.0% | 0.0% |
| 1998 | 25 | 84.0% | 84.0% | 0.0% | 0.0% |
| 1999 | 44 | 100.0% | 100.0% | 50.0% | 50.0% |
| 2000 | 43 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2001 | 40 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2002 | 53 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2003 | 41 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2004 | 39 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2005 | 80 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2006 | 158 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2007 | 171 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2008 | 201 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2009 | 215 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2010 | 253 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2011 | 300 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2012 | 341 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2013 | 386 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2014 | 503 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2015 | 473 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2016 | 493 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2017 | 457 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2018 | 474 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2019 | 516 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2020 | 456 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2021 | 509 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2022 | 511 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2023 | 520 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2024 | 517 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2025 | 520 | 100.0% | 100.0% | 100.0% | 100.0% |
| 2026 | 326 | 100.0% | 100.0% | 100.0% | 100.0% |

All years: 8,820 fights. Splits + control time complete for 8,618 (97.7%).
