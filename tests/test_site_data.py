"""Tests for src/export_site_data.py and the committed site/data/fighters.json."""
import json
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import export_site_data as E  # noqa: E402

FIELDS = {"name", "id", "division", "fights", "first_year", "last_year", "axis_1", "axis_2", "axis_3", "axis_4", "quadrant",
          "similar"}
AXES = ("axis_1", "axis_2", "axis_3", "axis_4")


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

    def test_similar_fighters_are_the_nearest_five(self):
        fighters = self.data["fighters"]
        by_id = {f["id"]: f for f in fighters}
        dist = lambda a, b: math.dist([a[k] for k in AXES], [b[k] for k in AXES])
        for f in fighters:
            sim = f["similar"]
            self.assertEqual(len(sim), E.N_SIMILAR)
            self.assertEqual(len(set(sim)), len(sim))
            self.assertNotIn(f["id"], sim)
            self.assertTrue(all(i in by_id for i in sim))
            d = [dist(f, by_id[i]) for i in sim]
            self.assertEqual(d, sorted(d), f["name"])
            # nobody outside the list is closer than the fifth (scores are rounded in the JSON, so allow a hair)
            fifth = d[-1]
            closer = [g["name"] for g in fighters if g["id"] not in sim and g["id"] != f["id"] and dist(f, g) < fifth - 2e-3]
            self.assertEqual(closer, [], f["name"])

    def test_card_axis_end_labels(self):
        ends = self.data["meta"]["axis_ends"]
        self.assertEqual(set(ends), set(AXES))
        self.assertEqual(ends["axis_1"], ["More striking volume", "More takedowns and control"])   # same as the map
        self.assertEqual(ends["axis_2"], ["More head strikes at range", "More clinch and body work"])

    def test_share_image_is_link_preview_size(self):
        from PIL import Image
        with Image.open(ROOT / "site" / "og-image.png") as im:
            self.assertEqual(im.size, (1200, 630))
        html = (ROOT / "site" / "index.html").read_text(encoding="utf-8")
        for tag in ('property="og:image" content="https://bbress467.github.io/ufc-styles/og-image.png"',
                    'name="twitter:card" content="summary_large_image"',
                    'property="og:url" content="https://bbress467.github.io/ufc-styles/"'):
            self.assertIn(tag, html)

    def test_colours_match_the_readme_chart(self):
        import make_figures as MF
        for q in self.data["meta"]["quadrants"]:
            self.assertEqual(q["tint"], MF.QUADRANT_TINTS[(q["x"], q["y"])])
            self.assertEqual(q["name_color"], MF.QUADRANT_NAME_COLORS[(q["x"], q["y"])])


@unittest.skipUnless((ROOT / "data" / "processed" / "ufc.sqlite").exists(), "database not built")
class ExportMatchesPipelineTests(unittest.TestCase):
    def test_readme_data_range_matches_dataset(self):
        import make_figures as MF
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        expected = f"{MF.DATA_RANGE_START}*{MF.data_range_text()}.*{MF.DATA_RANGE_END}"
        self.assertIn(expected, readme, "README data-range note is stale: rerun src/make_figures.py")
        meta = json.loads((ROOT / "site" / "data" / "fighters.json").read_text(encoding="utf-8"))["meta"]
        y, m, d = (int(v) for v in meta["data_through"].split("-"))
        self.assertIn(f"through {MF.MONTHS[m - 1]} {d}, {y}", MF.data_range_text())   # site and README agree

    def test_committed_file_is_up_to_date(self):
        fresh = json.loads(json.dumps(E.build(), ensure_ascii=False))
        committed = json.loads((ROOT / "site" / "data" / "fighters.json").read_text(encoding="utf-8"))
        self.assertEqual(fresh, committed, "site/data/fighters.json is stale: rerun src/export_site_data.py")


if __name__ == "__main__":
    unittest.main()
