"""Download the pinned raw data into data/raw/ (gitignored).

Sources
-------
* neelagiriaditya/ufc-datasets-1994-2025 (Kaggle, CC0) -- the project's only data source.
  Public dataset, so no Kaggle login or API key is needed.
* Greco1899/scrape_ufc_stats (GitHub) -- *reference only*: its event list is the UFC-event
  whitelist, and its stats are used to cross-check neelagiriaditya in src/validate.py.

Pins live in data_versions.json (committed). By default this script downloads exactly those
versions, so reruns reproduce the same files. Use --latest to look up the newest Kaggle version
and Greco1899 commit, download those, and rewrite the pins.

Each download is recorded in data/raw/manifest.json: source URL, version/commit, download time
(UTC) and the sha256 and size of every file.

Usage:
    python src/download_data.py            # pinned versions
    python src/download_data.py --latest   # refresh pins to newest versions
    python src/download_data.py --force    # re-download even if files are present
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import ssl
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PINS_PATH = ROOT / "data_versions.json"
RAW_DIR = ROOT / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.json"

KAGGLE_REF = "neelagiriaditya/ufc-datasets-1994-2025"
KAGGLE_FILES = ["event.csv", "fight.csv", "fighter.csv", "fighter_bonus.csv",
                "master.csv", "round.csv", "scrape_error.csv"]
GRECO_REPO = "Greco1899/scrape_ufc_stats"
GRECO_FILES = ["ufc_event_details.csv", "ufc_fight_details.csv", "ufc_fight_results.csv",
               "ufc_fight_stats.csv", "ufc_fighter_tott.csv"]

USER_AGENT = "ufc-style-clustering/0.1 (data download script)"
REQUEST_DELAY_S = 1.0  # be polite: at most ~1 request per second


def _ssl_context() -> ssl.SSLContext:
    # python.org macOS builds ship without a CA bundle; use certifi's when it is installed.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


SSL_CONTEXT = _ssl_context()


def fetch(url: str) -> bytes:
    time.sleep(REQUEST_DELAY_S)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120, context=SSL_CONTEXT) as resp:
        return resp.read()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def latest_kaggle_version() -> int:
    url = f"https://www.kaggle.com/api/v1/datasets/list?search={KAGGLE_REF.split('/')[0]}"
    for d in json.loads(fetch(url)):
        if d.get("ref") == KAGGLE_REF:
            return int(d["currentVersionNumber"])
    raise RuntimeError(f"{KAGGLE_REF} not found in Kaggle listing")


def latest_greco_commit() -> str:
    url = f"https://api.github.com/repos/{GRECO_REPO}/commits/main"
    return json.loads(fetch(url))["sha"]


def safe_extract(zf: zipfile.ZipFile, dest: Path, wanted: list[str]) -> dict[str, dict]:
    """Extract only the expected top-level CSVs; refuse anything with a path component."""
    names = set(zf.namelist())
    missing = [w for w in wanted if w not in names]
    if missing:
        raise RuntimeError(f"archive is missing expected files: {missing}; has {sorted(names)}")
    extra = sorted(names - set(wanted))
    if extra:
        print(f"  note: archive has unexpected extra files (not extracted): {extra}")
    out = {}
    for name in wanted:
        if "/" in name or "\\" in name or name.startswith("."):
            raise RuntimeError(f"unsafe member name in archive: {name!r}")
        data = zf.read(name)
        (dest / name).write_bytes(data)
        out[name] = {"sha256": sha256(data), "bytes": len(data)}
    return out


def download_kaggle(version: int, force: bool) -> dict:
    dest = RAW_DIR / "neelagiriaditya" / f"v{version}"
    if not force and all((dest / f).exists() for f in KAGGLE_FILES):
        print(f"neelagiriaditya v{version}: already present in {dest.relative_to(ROOT)}")
        return None
    dest.mkdir(parents=True, exist_ok=True)
    url = (f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_REF}"
           f"?datasetVersionNumber={version}")
    print(f"neelagiriaditya v{version}: downloading {url}")
    blob = fetch(url)
    files = safe_extract(zipfile.ZipFile(io.BytesIO(blob)), dest, KAGGLE_FILES)
    return {"source": f"https://www.kaggle.com/datasets/{KAGGLE_REF}", "license": "CC0: Public Domain",
            "version": version, "download_url": url, "archive_sha256": sha256(blob),
            "path": str(dest.relative_to(ROOT)), "files": files}


def download_greco(commit: str, force: bool) -> dict:
    dest = RAW_DIR / "greco1899" / commit[:12]
    if not force and all((dest / f).exists() for f in GRECO_FILES):
        print(f"greco1899 {commit[:12]}: already present in {dest.relative_to(ROOT)}")
        return None
    dest.mkdir(parents=True, exist_ok=True)
    files = {}
    for name in GRECO_FILES:
        url = f"https://raw.githubusercontent.com/{GRECO_REPO}/{commit}/{name}"
        print(f"greco1899 {commit[:12]}: downloading {name}")
        data = fetch(url)
        (dest / name).write_bytes(data)
        files[name] = {"sha256": sha256(data), "bytes": len(data)}
    return {"source": f"https://github.com/{GRECO_REPO}", "license": "GPL-3.0 (repository)",
            "role": "reference only: UFC event whitelist and cross-check",
            "commit": commit, "path": str(dest.relative_to(ROOT)), "files": files}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--latest", action="store_true", help="look up newest versions and update data_versions.json")
    ap.add_argument("--force", action="store_true", help="re-download even if the files are already present")
    args = ap.parse_args()

    pins = json.loads(PINS_PATH.read_text())
    if args.latest:
        pins["neelagiriaditya"]["version"] = latest_kaggle_version()
        pins["greco1899"]["commit"] = latest_greco_commit()
        PINS_PATH.write_text(json.dumps(pins, indent=2) + "\n")
        print(f"updated pins: {pins}")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text()) if MANIFEST_PATH.exists() else {"downloads": []}
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for key, rec in (("neelagiriaditya", download_kaggle(pins["neelagiriaditya"]["version"], args.force)),
                     ("greco1899", download_greco(pins["greco1899"]["commit"], args.force))):
        if rec is not None:
            rec.update({"dataset": key, "downloaded_at_utc": now})
            manifest["downloads"].append(rec)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {MANIFEST_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
