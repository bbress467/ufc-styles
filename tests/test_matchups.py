"""Tests for src/elo.py and src/matchups.py. Run: venv/bin/python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import elo  # noqa: E402
import features as F  # noqa: E402
import matchups as M  # noqa: E402


def toy_fights(rows, method="Decision - Unanimous"):
    cols = ["fight_id", "event_date", "red_fighter_id", "blue_fighter_id", "winner_id", "result_status"]
    return pd.DataFrame(rows, columns=cols).assign(event_id="e", method=method)


class EloTests(unittest.TestCase):
    def test_win_draw_and_no_contest(self):
        f = toy_fights([("f1", "2000-01-01", "A", "B", "A", "win"),
                        ("f2", "2000-02-01", "A", "B", None, "draw"),
                        ("f3", "2000-03-01", "A", "B", None, "no_contest"),
                        ("f4", "2000-04-01", "A", "B", "B", "win")])
        r = elo.run_elo(f, k=32).set_index("fight_id")
        self.assertEqual((r.at["f1", "elo_red"], r.at["f1", "elo_blue"]), (1500, 1500))
        self.assertAlmostEqual(r.at["f2", "elo_red"], 1516)          # +K/2 for a win between equals
        a2, b2 = r.at["f2", "elo_red"], r.at["f2", "elo_blue"]
        e = elo.expected(a2, b2)
        self.assertAlmostEqual(r.at["f3", "elo_red"], a2 + 32 * (0.5 - e))   # draw scores 0.5
        self.assertAlmostEqual(r.at["f4", "elo_red"], r.at["f3", "elo_red"])  # no contest: no update
        self.assertTrue(np.isnan(r.at["f3", "red_score"]))

    def test_same_date_uses_pre_date_ratings(self):
        f = toy_fights([("f1", "2000-01-01", "A", "B", "A", "win"),
                        ("f2", "2000-01-01", "A", "C", "A", "win"),
                        ("f3", "2000-02-01", "A", "D", "A", "win")])
        r = elo.run_elo(f, k=32).set_index("fight_id")
        self.assertEqual(r.at["f2", "elo_red"], 1500)                 # second fight that night: still pre-date rating
        self.assertAlmostEqual(r.at["f3", "elo_red"], 1532)           # both updates applied after the date

    def test_finish_multiplier_and_experience_decay(self):
        f = toy_fights([("f1", "2000-01-01", "A", "B", "A", "win"), ("f2", "2000-02-01", "A", "C", "A", "win")], method="KO/TKO")
        r = elo.run_elo(f, k=32, finish_mult=1.5).set_index("fight_id")
        self.assertAlmostEqual(r.at["f2", "elo_red"], 1500 + 1.5 * 16)
        r = elo.run_elo(f.assign(method="Decision - Split"), k=32, finish_mult=1.5).set_index("fight_id")
        self.assertAlmostEqual(r.at["f2", "elo_red"], 1516)                       # decisions: plain K
        g = toy_fights([("f1", "2000-01-01", "A", "B", "A", "win"), ("f2", "2000-02-01", "A", "C", "A", "win"),
                        ("f3", "2000-03-01", "A", "D", "A", "win")])
        r = elo.run_elo(g, k=32, k_decay=0.5).set_index("fight_id")
        e2 = elo.expected(1516, 1500)
        self.assertAlmostEqual(r.at["f3", "elo_red"], 1516 + 32 * 2 ** -0.5 * (1 - e2))   # 2nd fight: K * (1+1)^-0.5

    def test_debut_rating(self):
        f = toy_fights([("f1", "2000-01-01", "A", "B", "A", "win"), ("f2", "2000-02-01", "A", "C", "A", "win")])
        plain = elo.run_elo(f, k=32).set_index("fight_id")
        shifted = elo.run_elo(f, k=32, initial=1300).set_index("fight_id")
        self.assertTrue(np.allclose(plain["exp_red"], shifted["exp_red"]))          # fixed start: no-op
        r = elo.run_elo(f, k=32, debut_offset=10).set_index("fight_id")
        roster_mean = (1506 + 1474) / 2 - 10    # A and B debut at 1500 - 10 (no roster yet); C at their mean - 10
        self.assertAlmostEqual(r.at["f2", "elo_blue"], roster_mean)                  # C debuts relative to the roster

    @unittest.skipUnless(elo.DB_PATH.exists(), "database not built")
    def test_later_results_do_not_change_earlier_ratings(self):
        f = elo.load_ufc_fights()
        f = f[f["event_date"] < "2012-01-01"].copy()
        base = elo.run_elo(f, **elo.ELO_PARAMS)
        late = f["event_date"] >= "2011-06-01"
        g = f.copy()
        g.loc[late & (g["result_status"] == "win"), "winner_id"] = g.loc[late, "blue_fighter_id"]  # rewrite later results
        alt = elo.run_elo(g, **elo.ELO_PARAMS)
        cols = ["elo_red", "elo_blue", "exp_red"]
        early = base["event_date"] <= "2011-06-01"
        pd.testing.assert_frame_equal(base.loc[early, cols].reset_index(drop=True), alt.loc[early, cols].reset_index(drop=True))


class AntisymmetryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        n = 50
        self.d = pd.DataFrame({"a": [f"x{i}" for i in range(n)], "b": [f"y{i}" for i in range(n)],
                               "elo_a": rng.normal(1500, 100, n), "elo_b": rng.normal(1500, 100, n),
                               "n_a": 5, "n_b": 6, "y": rng.integers(0, 2, n),
                               "elo_base_a": rng.normal(1500, 100, n), "elo_base_b": rng.normal(1500, 100, n),
                               "td_a": rng.normal(size=n), "td_b": rng.normal(size=n),
                               **{f"a_{x}": rng.normal(size=n) for x in M.AXES}, **{f"b_{x}": rng.normal(size=n) for x in M.AXES}})

    def test_features_flip_sign_and_predictions_complement(self):
        s = M.swap(self.d)
        for m in M.MODELS + ["elo+td"]:
            X, names, _ = M.design(self.d, m)
            Xs, _, _ = M.design(s, m)
            self.assertTrue(np.allclose(Xs, -X), m)
            beta = np.random.default_rng(2).normal(size=X.shape[1])
            self.assertTrue(np.allclose(M.predict(X, beta) + M.predict(Xs, beta), 1))
        self.assertEqual(len(M.design(self.d, "elo+inter")[1]), 1 + 4 + 6)

    def test_penalty_never_applies_to_elo(self):
        _, _, terms = M.design(self.d, "elo+inter")
        pv = M.penalty_vector(terms, 5.0, 7.0)
        self.assertEqual(pv[0], 0.0)
        self.assertTrue((pv[1:5] == 5.0).all() and (pv[5:] == 7.0).all())
        X, _, _ = M.design(self.d, "elo+inter")
        b = M.fit_logistic(X, self.d["y"].to_numpy(), np.where(terms == "elo", 0, 1e9))
        self.assertTrue(np.allclose(b[1:], 0, atol=1e-6))            # huge penalty: style terms vanish


@unittest.skipUnless(M.CACHE_PATH.exists() and F.DB_PATH.exists(), "matchup dataset not built")
class DatasetLeakageTests(unittest.TestCase):
    def test_test_fight_features_use_profiles_as_of_fight_date_and_frozen_axes(self):
        data = M.build_dataset()
        row = data[data["set"] == "test"].iloc[100]
        ff = F.load_fighter_fights()
        axes = M.frozen_axes(fighter_fights=ff)
        sc = axes.transform(F.build_profiles(row["event_date"], ff), eligible_only=False).set_index("fighter_id")
        for side in ("a", "b"):
            got = [row[f"{side}_{x}"] for x in M.AXES]
            self.assertTrue(np.allclose(got, sc.loc[row[side], M.AXES].to_numpy(float)))
        self.assertEqual(axes.metadata["profiles_as_of"], M.SPLIT_DATE)
        prof = F.build_profiles(row["event_date"], ff).set_index("fighter_id")
        self.assertLess(prof.at[row["a"], "last_fight"], row["event_date"])
        self.assertAlmostEqual(row["td_a"], prof.at[row["a"], "z_td_att_per15"])     # takedown measure, as of fight date
        r = elo.run_elo(elo.load_ufc_fights(), **elo.ELO_PARAMS).set_index("fight_id").loc[row["fight_id"]]
        self.assertAlmostEqual(sorted([row["elo_a"], row["elo_b"]]), sorted([r["elo_red"], r["elo_blue"]]))


if __name__ == "__main__":
    unittest.main()
