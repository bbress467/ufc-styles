"""Leakage tests for src/features.py. Run: venv/bin/python -m unittest discover tests

Needs data/processed/ufc.sqlite (src/download_data.py + src/clean.py); skipped otherwise.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features  # noqa: E402

HAVE_DB = features.DB_PATH.exists()


@unittest.skipUnless(HAVE_DB, "data/processed/ufc.sqlite not built")
class AsOfExcludesFutureFights(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ff = features.load_fighter_fights()
        # a fighter with plenty of fights; test their 6th fight (so they are eligible before and after)
        counts = cls.ff.groupby("fighter_id").size()
        cls.fighter = counts[counts >= 10].index[0]
        fights = cls.ff[cls.ff["fighter_id"] == cls.fighter].sort_values("event_date")
        cls.fight = fights.iloc[5]
        cls.prior = fights.iloc[:5]

    def profile(self, as_of):
        p = features.build_profiles(as_of, self.ff)
        return p.set_index("fighter_id").loc[self.fighter]

    def test_profile_as_of_fight_date_excludes_that_fight(self):
        p = self.profile(self.fight["event_date"])
        self.assertEqual(p["n_fights"], 5)
        self.assertLess(p["last_fight"], self.fight["event_date"])
        # recompute from the 5 prior fights only and compare
        prior = self.prior
        self.assertAlmostEqual(p["fight_time_min"], prior["time_s"].sum() / 60)
        self.assertAlmostEqual(p["pace"], prior["sig_att"].sum() / (prior["time_s"].sum() / 60))
        self.assertAlmostEqual(p["pos_clinch"], prior["clinch_att"].sum() / prior["sig_att"].sum())
        self.assertAlmostEqual(p["str_acc"], prior["sig_landed"].sum() / prior["sig_att"].sum())

    def test_profile_day_after_includes_the_fight(self):
        after = pd.Timestamp(self.fight["event_date"]) + pd.Timedelta(days=1)
        p = self.profile(after)
        self.assertEqual(p["n_fights"], 6)
        self.assertEqual(p["last_fight"], self.fight["event_date"])

    def test_standardization_ignores_future_fights(self):
        """z-scores as of a date must not change when later fights are added to the input."""
        as_of = self.fight["event_date"]
        full = features.build_profiles(as_of, self.ff).set_index("fighter_id")
        past_only = features.build_profiles(as_of, self.ff[self.ff["event_date"] < as_of]).set_index("fighter_id")
        z = [c for c in full.columns if c.startswith("z_")]
        pd.testing.assert_frame_equal(full[z], past_only.loc[full.index, z])

    def test_no_fights_on_or_after_as_of_anywhere(self):
        as_of = self.fight["event_date"]
        p = features.build_profiles(as_of, self.ff)
        self.assertTrue((p["last_fight"] < as_of).all())


class DivisionMapping(unittest.TestCase):
    def test_canonical_division(self):
        c = features.canonical_division
        self.assertEqual(c("Light Heavyweight"), "Light Heavyweight")
        self.assertEqual(c("Ultimate Fighter 9 Welterweight Tournament"), "Welterweight")
        self.assertEqual(c("Road to 3 Women's Strawweight Tournament"), "Women's Strawweight")
        self.assertEqual(c("Super Heavyweight"), "Heavyweight")
        self.assertIsNone(c("Catch Weight"))
        self.assertIsNone(c(None))


if __name__ == "__main__":
    unittest.main()
