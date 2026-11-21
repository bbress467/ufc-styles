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
        # recompute the unshrunk values from the 5 prior fights only and compare
        prior = self.prior
        self.assertAlmostEqual(p["fight_time_min"], prior["time_s"].sum() / 60)
        self.assertAlmostEqual(p["raw_pace"], prior["sig_att"].sum() / (prior["time_s"].sum() / 60))
        self.assertAlmostEqual(p["raw_pos_clinch"], prior["clinch_att"].sum() / prior["sig_att"].sum())
        self.assertAlmostEqual(p["str_acc"], prior["sig_landed"].sum() / prior["sig_att"].sum())

    def test_profile_day_after_includes_the_fight(self):
        after = pd.Timestamp(self.fight["event_date"]) + pd.Timedelta(days=1)
        p = self.profile(after)
        self.assertEqual(p["n_fights"], 6)
        self.assertEqual(p["last_fight"], self.fight["event_date"])

    def test_shrinkage_and_standardization_ignore_future_fights(self):
        """Shrunk features, z-scores and divisions as of a date must not change when later fights are in the input."""
        as_of = self.fight["event_date"]
        full = features.build_profiles(as_of, self.ff).set_index("fighter_id")
        past_only = features.build_profiles(as_of, self.ff[self.ff["event_date"] < as_of]).set_index("fighter_id")
        pd.testing.assert_frame_equal(full, past_only.loc[full.index])

    def test_shrinkage_matches_pseudo_count_formula(self):
        """pace = (sig_att + P * division sig_att per second) / ((time + P) / 60), with division totals from pre-as_of data."""
        as_of = self.fight["event_date"]
        past = self.ff[self.ff["event_date"] < as_of]
        p = features.build_profiles(as_of, self.ff).set_index("fighter_id")
        grp = p.loc[self.fighter, "std_group"]
        ref = p.index[p["eligible"] & (p["std_group"] == grp)]
        rp = past[past["fighter_id"].isin(ref)]
        mine = past[past["fighter_id"] == self.fighter]
        P = features.PRIOR_MINUTES * 60
        expected = (mine["sig_att"].sum() + P * rp["sig_att"].sum() / rp["time_s"].sum()) / ((mine["time_s"].sum() + P) / 60)
        self.assertAlmostEqual(p.loc[self.fighter, "pace"], expected)

    def test_zero_prior_gives_raw_values(self):
        p = features.build_profiles(self.fight["event_date"], self.ff, prior_minutes=0, feature_priors={})
        for f in features.STYLE_FEATURES + features.SHRUNK_SKILL:
            pd.testing.assert_series_equal(p[f], p[f"raw_{f}"], check_names=False)

    def test_feature_prior_override(self):
        """Submission attempts use their own prior; other features keep the global one."""
        as_of = self.fight["event_date"]
        default = features.build_profiles(as_of, self.ff).set_index("fighter_id")
        uniform = features.build_profiles(as_of, self.ff, feature_priors={}).set_index("fighter_id")
        override = features.build_profiles(as_of, self.ff, prior_minutes=features.PRIOR_MINUTES,
                                           feature_priors={"sub_att_per15": features.PRIOR_MINUTES}).set_index("fighter_id")
        pd.testing.assert_series_equal(uniform["sub_att_per15"], override["sub_att_per15"])
        pd.testing.assert_series_equal(default["pace"], uniform["pace"])
        self.assertFalse(np.allclose(default["sub_att_per15"], uniform["sub_att_per15"]))

    def test_era_relative_z_matches_manual_window(self):
        """z = (x - mean) / sd over eligible same-group fighters with career midpoint within +/- window (pre-as_of)."""
        as_of = "2018-01-01"
        p = features.build_profiles(as_of, self.ff).set_index("fighter_id")
        e = p[p["eligible"] & p["std_group"].notna()]
        fid = e.index[e["era_ref_n"] > features.ERA_MIN_REF][0]   # a fighter whose window did not need widening
        me = p.loc[fid]
        ref = e[(e["std_group"] == me["std_group"]) & ((e["career_mid"] - me["career_mid"]).abs() <= features.ERA_WINDOW_YEARS)]
        expected = (me["pace"] - ref["pace"].mean()) / ref["pace"].std(ddof=0)
        self.assertAlmostEqual(me["z_pace"], expected)
        self.assertEqual(me["era_ref_n"], len(ref))
        # career midpoint uses only pre-as_of fights
        mine = self.ff[(self.ff["fighter_id"] == fid) & (self.ff["event_date"] < as_of)]
        self.assertAlmostEqual(me["career_mid"], features.career_midpoints(mine).iloc[0])

    def test_era_window_widens_to_min_reference(self):
        p = features.build_profiles(None, self.ff)
        e = p[p["eligible"] & p["std_group"].notna()]
        group_sizes = e.groupby("std_group").size()
        expected_min = e["std_group"].map(lambda g: min(features.ERA_MIN_REF, group_sizes[g]))
        self.assertTrue((e["era_ref_n"] >= expected_min).all())
        div_only = features.build_profiles(None, self.ff, era_window_years=None)
        pd.testing.assert_series_equal(div_only["z_pace"], div_only["zdiv_pace"], check_names=False)

    def test_clustering_matrix_drops_redundant_shares(self):
        X = features.clustering_matrix(features.build_profiles(None, self.ff))
        self.assertNotIn("z_pos_distance", X.columns)
        self.assertNotIn("z_tgt_head", X.columns)
        self.assertEqual(len(X.columns), len(features.STYLE_FEATURES) - 2)

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
