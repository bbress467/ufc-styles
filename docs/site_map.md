# ufcstats.com — site map for scraping

Surveyed 2026-11-20. Pages were loaded in a real browser, and the raw server
HTML was then re-requested from inside that browser session (`fetch()`), about
one request per second, roughly 35 requests in all. Every observation below
comes from that raw HTML, which is what a scraper will receive.

---

## 0. Access: bot check in front of every page

A plain HTTP client (`curl`, `requests`) does **not** get the site. Every URL
tried (events list, fighters list) returned the same ~3 KB page:

- `<title>Loading…</title>`, body "Checking your browser…", `<noscript>This site requires JavaScript.</noscript>`
- An inline script runs a SHA-256 proof-of-work (`sha256(nonce + ':' + n)` must
  start with `00`), POSTs `nonce` and `n` to `/__c`, then calls `location.reload()`.
- **The HTTP status is 200**, so checking status codes alone won't detect it.
- `robots.txt` returns 404 (no published crawl rules).

A real browser passes the check automatically and then gets normal pages.
**Decide how the scraper will get access before writing it.** Options: get
permission or access from the site owner, drive a real browser, or start from
an existing public dataset. Any scraper should detect this page (title
`Loading…` or the `/__c` string) and stop instead of saving it as data.

Behind the check, the site is plain server-rendered HTML (see §5).

---

## 1. Page types, links and IDs

All IDs are **16-character lowercase hex strings** and are the last path segment.
Links are absolute `http://ufcstats.com/...` URLs (http, not https).

| Page | URL pattern | Links to |
|---|---|---|
| Events list | `/statistics/events/completed` (`?page=N`, or `?page=all`) | event pages |
| Event | `/event-details/{event_id}` | fight-detail pages, fighter pages |
| Fight detail | `/fight-details/{fight_id}` | fighter pages (no link back to the event) |
| Fighter | `/fighter-details/{fighter_id}` | fight-detail pages (event name and date are text only) |
| Fighters list | `/statistics/fighters?char={a..z}&page=all` | fighter pages |

Crawl order: events list (`?page=all`) → each event → each fight → the
fighters in each fight. The fight-detail page does **not** name its event or
date, so carry `event_id` and the date down from the event page.

### Events list
- `?page=all` returns **all 798 events in one page** (~635 KB). Paginated view: 25 per page, about 32 pages.
- Table: `table.b-statistics__table-events`. Each event row is
  `tr.b-statistics__table-row` → `td:first-child i.b-statistics__table-content`
  containing `a.b-link` (name, href = event URL) and `span.b-statistics__date`
  (`November 14, 2026`). The second `td` is the location (`New York City, New York, USA`).
- The first data row is an empty spacer: `<tr><td class="b-statistics__table-col_type_clear"></td></tr>`.
- **The "completed" list includes the next upcoming event** as its first row
  (`tr.b-statistics__table-row_type_first`, with a `next.png` icon). On the survey
  date, that was 2026-11-21, the day after the survey.
- Newest first. The oldest listed event is **UFC 2 (March 11, 1994)**. UFC 1 is not in the list.

### Event page (`/event-details/{id}`)
- Name: `.b-content__title` (`h2`). Date and location: `li.b-list__box-list-item`
  with labels `Date:` and `Location:` (strip the label).
- Fight table: `tbody tr.b-fight-details__table-row` with
  `data-link="http://ufcstats.com/fight-details/{fight_id}"` (the same URL also appears in `onclick="doNav(...)"`).
- Columns (`thead`): `W/L | Fighter | Kd | Str | Td | Sub | Weight class | Method | Round | Time`.
- Most cells hold **two `<p class="b-fight-details__table-text">`**, one per fighter, in the same order as the Fighter column.
  - W/L: a single `a.b-flag` whose text is `win`, or two flags reading `draw`/`draw` or `nc`/`nc`.
    On this page the **winner is listed first**.
  - Fighter: two `a` links to `/fighter-details/{id}`.
  - `Str` = **significant strikes landed** (not total strikes), and `Td` = takedowns landed.
  - Weight class: text (`Heavyweight`, `Women's Bantamweight`, `Catch Weight`, `Open Weight`)
    followed by icon `<img>`s: `belt.png` = title fight; `fight.png` / `perf.png` (and probably `sub.png` / `ko.png`) = bonus awards.
  - Method: first `<p>` is a short code (`U-DEC`, `S-DEC`, `M-DEC`, `KO/TKO`, `SUB`, `CNC`, …), second `<p>` is the detail (`Rear Naked Choke`, `Punch`) or empty.
  - Round is an integer. Time is `M:SS` (elapsed time in the final round).
- Upcoming events use the same layout, but stat, method, round and time cells are empty, and the Kd column says `View Matchup`.

### Fight-detail page (`/fight-details/{id}`)

**Fighters** — two `div.b-fight-details__person` elements:
- `i.b-fight-details__person-status`: `W`, `L`, `D` or `NC` (class `_style_green` for a win, `_style_gray` otherwise).
- `h3.b-fight-details__person-name a` → name and fighter URL. `p.b-fight-details__person-title` → nickname in quotes, or empty.
- ⚠️ **The order is not winner-first.** For example, Miller (L) vs McKinney (W) is listed L first. It appears to follow the red/blue corner. Always use the status letter.

**Bout info** — `div.b-fight-details__fight`:
- `i.b-fight-details__fight-title`: `UFC Heavyweight Title Bout`, `Lightweight Bout`, etc. (plus the same bonus/belt `<img>`s).
- First `p.b-fight-details__text` contains items `i.b-fight-details__text-item(_first)`, each starting with a label `i.b-fight-details__label`:
  - `Method:` → long form, inside a nested `<i style="font-style: normal">`: `Decision - Unanimous`, `Decision - Split`, `Decision - Majority`, `KO/TKO`, `Submission`, `Could Not Continue`, …
  - `Round:` → integer
  - `Time:` → `M:SS`
  - `Time format:` → `3 Rnd (5-5-5)`, `5 Rnd (5-5-5-5-5)`, `2 Rnd (5-5)`, `No Time Limit` (UFC 2). Other variants likely exist in the 1990s and early 2000s.
  - `Referee:` → inside a `<span>`
- Second `p.b-fight-details__text`, label `Details:`:
  - **Decisions:** one item per judge: `<span>Judge Name</span> 27 - 30.` (trailing period). Usually 3 judges.
  - **Finishes and other results:** free text, e.g. `Punch to Head At Distance`, `Kicks to Leg At Distance Submission to Strikes`, `Eye poke by Gane` (NC).
  - **Old decisions (e.g. UFC 27, 2000) have no scores.** `Details:` is empty.
  - ⚠️ **Score order:** in every decision checked, the lower or losing score came first
    (Borralho, listed first, won 30-27 but shows `27 - 30`; the split decision shows `28 - 29, 29 - 28, 28 - 29`
    with the winner listed second). The scores do **not** follow the order of the fighter blocks. Treat this as unconfirmed and
    check more fights before assigning a score to a specific fighter.

**Stats tables** — four `<table>`s, found by **position**. The two overall tables have no class.

| # | Section | Selector hint | Columns |
|---|---|---|---|
| 0 | Totals (overall) | first `table` after the `Totals` heading; no class | Fighter, KD, Sig. str., Sig. str. %, Total str., Td, Td %, Sub. att, Rev., Ctrl |
| 1 | Totals per round | `table.b-fight-details__table.js-fight-table` (1st) | same 10 columns; header text has a typo, reading `Td %` twice (the 6th column is Td landed of attempted) |
| 2 | Significant Strikes (overall) | third `table`; no class | Fighter, Sig. str, Sig. str. %, Head, Body, Leg, Distance, Clinch, Ground |
| 3 | Sig. strikes per round | `table.b-fight-details__table.js-fight-table` (2nd) | same 9 columns |

- Overall tables: one `tbody tr`. Each `td` has two `<p>`s (fighter A, fighter B, in the fighter-block order).
- Per-round tables have an unusual structure, but every parser handles it:
  `thead(column names) > tbody(empty) > thead.b-fight-details__table-row("Round 1") > tbody > tr > … "Round 2" …`.
  Pair each `Round N` `thead` with the `tbody` after it. The number of rounds = the number of these headers (it matches the final round).
- Head + Body + Leg = Sig. str landed, and Distance + Clinch + Ground = Sig. str landed (checked on the Gane fight).
- A sixth section (`div.b-fight-details__charts`) has bar and stacked charts with `style="height: NN%"`. Those numbers repeat the table data, so skip them.
- The "Per round" sections have collapse toggles (`a.js-fight-collapse-link`), but the data is already in the HTML.

### Fighter page (`/fighter-details/{id}`)
- Name: `span.b-content__title-highlight`. Record: `span.b-content__title-record` → `Record: 15-2-0 (1 NC)` (W-L-D, with an optional ` (N NC)`).
- Nickname: `p.b-content__Nickname` (empty if none).
- Attribute boxes: `div.b-list__info-box` → `li.b-list__box-list-item`, each with a label `i.b-list__box-item-title`:
  - Box 1 (`_style_small-width`): `Height: 6' 4"`, `Weight: 247 lbs.`, `Reach: 81"`, `STANCE: Orthodox`, `DOB: Apr 12, 1990`
  - Box 2 (`_style_middle-width`, career stats): `SLpM: 5.90`, `Str. Acc.: 63%`, `SApM: 2.74`, `Str. Def: 58%`, *(one empty `li` spacer)*,
    `TD Avg.: 0.58`, `TD Acc.: 23%`, `TD Def.: 74%`, `Sub. Avg.: 0.5`
  - Labels are inconsistent: `STANCE:` is uppercase, `Str. Def` has no trailing period. Match labels case-insensitively.
- Fight history: same table layout as the event page, but the 7th column is **Event** (name + date `Nov. 14, 2026`) and the profiled fighter is **always listed first**.
  W/L flag text: `win`, `loss`, `draw`, `nc`. The first `tbody tr` is an empty spacer with no `data-link`.

### Fighters list (`/statistics/fighters?char=a&page=all`)
- Columns: `First | Last | Nickname | Ht. | Wt. | Reach | Stance | W | L | D | Belt`. All three name cells link to the fighter page.
- Useful as a quick fighter index (247 fighters under "a"). **DOB is not in this list.** Get it from the fighter page.

---

## 2. Value formats

| Field | Example | Notes |
|---|---|---|
| Landed of attempted | `237 of 343` | Split on ` of `. Applies to Sig. str., Total str., Td, Head/Body/Leg/Distance/Clinch/Ground |
| Percent | `69%` | `---` when there were 0 attempts (e.g. `0 of 0` takedowns → `---`) |
| Control time | `4:37` | `M:SS`, may exceed 9 minutes (`8:30`). `--` in fights with no control data (1994) |
| Fight time | `5:00`, `12:13` | `M:SS`, elapsed time in the final round. Minutes can exceed 5 in 1990s no-limit fights |
| Height | `6' 4"` | feet' inches" → convert. `--` if missing |
| Weight | `247 lbs.` | `--` if missing |
| Reach | `81"` | inches. `--` if missing |
| Stance | `Orthodox` | **empty string** if missing (not `--`) |
| DOB | `Apr 12, 1990` | abbreviated month. `--` if missing |
| Dates | `November 14, 2026` (events list and event page) · `Nov. 14, 2026` (fighter history) · `Apr 12, 1990` (DOB) | three different formats |
| Record | `Record: 15-2-0 (1 NC)` | NC part is optional |
| Integers (KD, Sub. att, Rev.) | `0` | |

All text cells have heavy whitespace and newlines around the value. Always strip.

### How missing values appear
- `--` : missing fighter attribute (height, weight, reach, DOB); missing Ctrl time; **every stat in non-UFC fights**.
- `---` : percentage with no attempts.
- empty string: missing stance, nickname, method detail, the `Details:` line on old decisions.
- `0.00` / `0%` : **career stats on fighters with no UFC data are shown as zeros, not as missing.** Tom Aaron and Cyborg Abreu both show all zeros. Don't treat these as real values.

---

## 3. Raw HTML vs JavaScript

Once past the bot check, **all data is in the server HTML**. The raw HTML of the
events list, event, fight and fighter pages contained every value listed above.
Page scripts are only jQuery / jQuery UI (a dialog), Google Analytics and
`/js/main.js` (row-click navigation via `doNav`). No API or XHR loads data.
The only JavaScript that matters is the bot check in §0.

---

## 4. Edge cases checked

| Case | Example (ID) | What's different |
|---|---|---|
| 1994 event | UFC 2 `a6a9ab5a824e8f66`; fight `00835554f95fa911` | `Open Weight`, `Time format: No Time Limit`, times like `12:13`/`9:51`, Ctrl `--`, many 1-night tournament fights between the same fighters. Strike tables still present |
| Early-2000s event | UFC 27 (2000-09-22) `31652c9267606d54` | Decisions have **no judges' scores**; 2-round prelims `2 Rnd (5-5)`; full per-round tables present |
| No contest | Aspinall vs Gane, UFC 321 `6390c8e74630473e` | Status `NC`/`NC`; event and fighter flag `nc`; method `Could Not Continue` (event code `CNC`); Details = `Eye poke by Gane`; stats present for round 1 |
| Draw (modern) | Medic vs Holland, UFC 334 `79c49d244eab9005` | Status `D`/`D`; flag text `draw`; `Decision - Majority`; scores `27 - 29, 28 - 28, 28 - 28` |
| Draw (old) | Gumm vs Fernandes, UFC 27 `c078544a32e18ba3` | `D`/`D` with method `Decision - Unanimous` and no scores |
| Round-1 finish | Miller vs McKinney `cb2fd3afd5031f0d` | Per-round tables have only `Round 1`. **Loser listed first** |
| 5-round fight | Gane vs Hokit `30dbf1eceb2e1f8d` | `5 Rnd (5-5-5-5-5)`, Round 1–5 headers, `belt.png` |
| Missing reach / DOB | Scott Morris `be9d259be012e8a4` (reach and DOB `--`); Tom Aaron `93fe7332d16c6ad9` (height and reach `--`, stance empty) | 92 of 247 "A" fighters have no reach |
| Non-UFC fights | Tom Aaron's Strikeforce fights | Appear in fighter history with **every stat `--`**, real fight-detail IDs; event names like `Strikeforce - Henderson vs. Babalu` |
| Fighter with no fights | Cyborg Abreu `f689bd7bbd14b392` | Record `0-0-0`, career stats all zero, only the spacer row |
| Upcoming event | `cbf69fa846c7e3ee` | First row of the "completed" list; fight rows with empty results and `View Matchup` |
| Single-name fighter | `Alatengheili` (UFC 334) | No first/last split |

---

## 5. Things that make scraping tricky

1. **The bot check.** Plain HTTP gets a proof-of-work page with status 200. Detect it explicitly. How to get access is an open decision (see §0).
2. **Fighter order differs by page.** Event page: winner first. Fight page: corner order. Fighter page: profiled fighter first. Join on fighter ID plus the W/L/D/NC status, never on position.
3. **Judges' scores don't follow the fighter order** (apparently loser – winner). They are missing entirely in early events.
4. **Tables have no IDs.** The overall tables have no class, so selection depends on position and section order.
5. **Per-round tables alternate `thead`/`tbody`** for each round, with a misleading duplicate `Td %` header.
6. **Three kinds of missing value** (`--`, `---`, empty), plus **fake zeros** in career stats.
7. **Non-UFC (Strikeforce etc.) fights and fighters with no UFC fights are mixed in.** Filter by event (only events from the UFC events list) before building style features.
8. **Three date formats**, plus fight times and control times in `M:SS` that can exceed 9 minutes.
9. **The "completed" events list includes the next upcoming event**, which has no results.
10. **The fight page has no link to its event.** Keep `event_id` and the date from the crawl.
11. **Rules changed over time** (no time limits, open weight, tournament nights, 2-round fights, no scores before about 2000). Fighters' early-era stats are not comparable to modern ones. Consider a date cutoff for clustering.
12. **Odd labels and whitespace** (`STANCE:`, empty spacer `li`/`tr`, values wrapped in newlines).

Scale: 798 events × about 12 fights ≈ 9–10k fight pages, plus about 4k fighter pages. At 1 request per second, that's roughly 4 hours for a full crawl, so cache raw HTML to disk and make the crawl resumable.
