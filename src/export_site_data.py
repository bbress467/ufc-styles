"""Export the data for the interactive style map in site/ (run after features.py and style_axes.py).

    venv/bin/python src/export_site_data.py

Writes site/data/fighters.json:
  meta      data source, date range, axis and quadrant names, the chart's colours, the 14 labelled fighters
  fighters  one record per eligible fighter (5+ UFC fights since 2000): name, ufcstats ID, most frequent
            division, UFC fights since 2000, first and last year, the four axis scores, and quadrant

Design constants (tints, name colours, labelled fighters) are read from make_figures.py, so the site and the
README chart stay in sync.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

import features as F
import make_figures as MF
import style_axes as A

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "site" / "data" / "fighters.json"
REPO_URL = "https://github.com/bbress467/ufc-styles"
N_SIMILAR = 5     # most similar fighters per card: nearest by Euclidean distance across all four axis scores
KAGGLE_URL = "https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025"
# Labels placed first, so they are always shown even on narrow screens where crowded labels are left out.
LABEL_PRIORITY = ["Khabib Nurmagomedov", "Max Holloway", "Demetrious Johnson"]


def quadrant_of(axis_1: float, axis_2: float) -> str:
    """Quadrant name for a point; the split is at 0 on both axes (the average fighter)."""
    x = "right" if axis_1 >= 0 else "left"
    y = "top" if axis_2 >= 0 else "bottom"
    return MF.QUADRANT_NAMES[(x, y)]


def add_similar(fighters: list[dict], axis_names: list[str], k: int = N_SIMILAR) -> None:
    """Add each fighter's k most similar fighters ('similar': ufcstats IDs, nearest first) by Euclidean distance
    across all four style-axis scores. Ties are broken by name so the output is deterministic."""
    X = np.array([[f[a] for a in axis_names] for f in fighters], float)
    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    for i, f in enumerate(fighters):
        order = sorted(range(len(fighters)), key=lambda j: (D[i, j], fighters[j]["name"]))[:k]
        f["similar"] = [fighters[j]["id"] for j in order]


def build() -> dict:
    profiles = F.build_profiles(None)
    model_path = A.MODEL_DIR / f"style_axes_m{A.CHOSEN_AXES}.joblib"
    axes = A.StyleAxes.load(model_path) if model_path.exists() else A.fit_style_axes(profiles, A.CHOSEN_AXES)
    scores = axes.transform(profiles).set_index("fighter_id")
    p = profiles.set_index("fighter_id").loc[scores.index]

    fighters = []
    for fid, r in scores.iterrows():
        fighters.append({
            "name": r["fighter_name"],
            "id": fid,
            "division": r["division"],
            "fights": int(r["n_fights"]),
            "first_year": int(p.at[fid, "first_fight"][:4]),
            "last_year": int(p.at[fid, "last_fight"][:4]),
            **{k: round(float(r[k]), 3) for k in axes.axis_names},
            "quadrant": quadrant_of(r["axis_1"], r["axis_2"]),
        })
    fighters.sort(key=lambda f: f["name"])
    add_similar(fighters, axes.axis_names)

    with sqlite3.connect(F.DB_PATH) as con:
        first, last = con.execute("SELECT MIN(event_date), MAX(event_date) FROM fights WHERE event_date >= ?",
                                  (F.START_DATE,)).fetchone()
    names = {f["name"] for f in fighters}
    meta = {
        "data_from": first, "data_through": last, "n_fighters": len(fighters), "min_fights": F.MIN_FIGHTS,
        "source": {"name": "neelagiriaditya/ufc-datasets-1994-2025 (Kaggle, CC0)", "url": KAGGLE_URL,
                   "scraped_from": "ufcstats.com"},
        "repo_url": REPO_URL,
        "ufcstats_fighter_url": "http://ufcstats.com/fighter-details/{id}",
        "axes": {k: A.AXIS_NAMES[k] for k in axes.axis_names},
        "axis_ends": {   # plain-language ends for the card bars (low end, high end), matching the map's wording
            "axis_1": ["More striking volume", "More takedowns and control"],
            "axis_2": ["More head strikes at range", "More clinch and body work"],
            "axis_3": ["Fewer submission attempts", "More submission attempts"],
            "axis_4": ["Fewer leg kicks", "More leg kicks"],
        },
        "quadrants": [{"x": x, "y": y, "name": n, "tint": MF.QUADRANT_TINTS[(x, y)],
                       "name_color": MF.QUADRANT_NAME_COLORS[(x, y)]} for (x, y), n in MF.QUADRANT_NAMES.items()],
        "labelled_dot_color": MF.LABELLED_DOT_COLOR,
        "labelled": [n for n in MF.STYLE_MAP_FIGHTERS if n in names],
        "label_priority": [n for n in LABEL_PRIORITY if n in names],
        "note": MF.NOTE.replace("\n", " "),
    }
    return {"meta": meta, "fighters": fighters}


def main() -> None:
    data = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(ROOT)}: {len(data['fighters']):,} fighters, "
          f"{OUT_PATH.stat().st_size / 1024:.0f} KB, data {data['meta']['data_from']} to {data['meta']['data_through']}")


if __name__ == "__main__":
    main()
