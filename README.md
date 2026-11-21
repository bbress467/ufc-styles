# UFC Fighter Styles

Clustering UFC fighters into style archetypes from round-by-round fight
statistics, and testing whether style matchups predict outcomes beyond
fighter skill.

Status: in progress. The data pipeline, style features and style clusters are built; the matchup analysis is next.

## Pipeline

The data comes from a public, CC0 Kaggle dataset scraped from ufcstats.com
([neelagiriaditya/ufc-datasets-1994-2025](https://www.kaggle.com/datasets/neelagiriaditya/ufc-datasets-1994-2025)).
ufcstats.com itself is not scraped. A second public scrape,
[Greco1899/scrape_ufc_stats](https://github.com/Greco1899/scrape_ufc_stats), is used only as a UFC-event whitelist and for cross-checking.
See [docs/data_sources.md](docs/data_sources.md) for why this source was chosen.

| Step | Script | Output (gitignored) |
|---|---|---|
| 1. Download the pinned dataset versions (`data_versions.json`) | `src/download_data.py` | `data/raw/`, plus `data/raw/manifest.json` with checksums |
| 2. Keep UFC events, convert missing values, write tidy tables | `src/clean.py` | `data/processed/ufc.sqlite` |
| 3. Validate and write the data quality report | `src/validate.py` | [docs/data_quality.md](docs/data_quality.md) (committed) |
| 4. Build per-fighter style profiles | `src/features.py` | `data/processed/fighter_profiles.csv` |
| 5. Fit style clusters (PCA + Gaussian mixture) | `src/clustering.py` | `data/processed/cluster_probabilities.csv`, `data/processed/models/` |

Feature definitions and decisions: [docs/feature_spec.md](docs/feature_spec.md). Clustering method, results and leakage caveats:
[docs/clustering.md](docs/clustering.md). Site structure notes: [docs/site_map.md](docs/site_map.md).

## How to run

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python src/download_data.py
venv/bin/python src/clean.py
venv/bin/python src/validate.py
venv/bin/python src/features.py
venv/bin/python src/clustering.py
venv/bin/python -m unittest discover tests
```

`src/download_data.py --latest` moves the pins to the newest dataset versions. No Kaggle login is needed.
Exploratory notebooks: `notebooks/01_feature_eda.ipynb` (feature distributions, correlations, shrinkage priors),
`notebooks/02_pca.ipynb` (PCA of the style features) and `notebooks/03_clustering.ipynb` (choosing k, cluster profiles, confounding checks).

## Date cutoff

The analysis uses **UFC fights from 2000 onward**. Control time isn't recorded before 1999 and is incomplete in 1999
(see the completeness-by-year table in [docs/data_quality.md](docs/data_quality.md)). Starting in 2000 also excludes the early era,
when the rules and weight classes differed from the modern sport.
