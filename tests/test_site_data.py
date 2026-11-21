"""Tests for src/export_site_data.py and the committed site/data/fighters.json."""
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import export_site_data as E  # noqa: E402

FIELDS = {"name", "id", "division", "fights", "first_year", "last_year", "axis_1", "axis_2", "axis_3", "axis_4", "quadrant"}


class SiteDataFileTests(unittest.TestCase):
    """Checks the committed JSON the site loads (no database needed)."""

    @classmethod
    def setUpClass(cls):
        cls.data = json.loads((ROOT / "site" / "data" / "fighters.json").read_text(encoding="utf-8"))

    def test_records(self):
        fighters = self.data["fighters"]
        self.assertEqual(len(fighters), self.data["meta"]["n_fighters"])
        ids = set()
        for f in fighters:
            self.assertEqual(set(f), FIELDS)
            self.assertRegex(f["id"], r"^[0-9a-f]{16}$")
            self.assertGreaterEqual(f["fights"], self.data["meta"]["min_fights"])
            self.assertLessEqual(2000, f["first_year"])
            self.assertLessEqual(f["first_year"], f["last_year"])
            for k in ("axis_1", "axis_2", "axis_3", "axis_4"):
                self.assertTrue(math.isfinite(f[k]), f"{f['name']} {k}")
            ids.add(f["id"])
        self.assertEqual(len(ids), len(fighters))

    def test_quadrant_matches_axis_signs(self):
        names = {(q["x"], q["y"]): q["name"] for q in self.data["meta"]["quadrants"]}
        for f in self.data["fighters"]:
            x = "right" if f["axis_1"] >= 0 else "left"
            y = "top" if f["axis_2"] >= 0 else "bottom"
            self.assertEqual(f["quadrant"], names[(x, y)], f["name"])

    def test_labelled_and_priority_fighters_present(self):
        names = {f["name"] for f in self.data["fighters"]}
        meta = self.data["meta"]
        self.assertEqual(len(meta["labelled"]), 14)
        self.assertTrue(set(meta["labelled"]) <= names)
        self.assertTrue(set(meta["label_priority"]) <= set(meta["labelled"]))
        self.assertEqual(set(meta["label_priority"]), {"Khabib Nurmagomedov", "Max Holloway", "Demetrious Johnson"})

    def test_colours_match_the_readme_chart(self):
        import make_figures as MF
        for q in self.data["meta"]["quadrants"]:
            self.assertEqual(q["tint"], MF.QUADRANT_TINTS[(q["x"], q["y"])])
            self.assertEqual(q["name_color"], MF.QUADRANT_NAME_COLORS[(q["x"], q["y"])])


@unittest.skipUnless((ROOT / "data" / "processed" / "ufc.sqlite").exists(), "database not built")
class ExportMatchesPipelineTests(unittest.TestCase):
    def test_committed_file_is_up_to_date(self):
        fresh = json.loads(json.dumps(E.build(), ensure_ascii=False))
        committed = json.loads((ROOT / "site" / "data" / "fighters.json").read_text(encoding="utf-8"))
        self.assertEqual(fresh, committed, "site/data/fighters.json is stale: rerun src/export_site_data.py")


if __name__ == "__main__":
    unittest.main()
