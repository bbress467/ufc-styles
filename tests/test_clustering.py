"""Tests for src/clustering.py. Run: venv/bin/python -m unittest discover tests

Needs data/processed/ufc.sqlite; skipped otherwise.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import clustering as C  # noqa: E402
import features as F  # noqa: E402


@unittest.skipUnless(F.DB_PATH.exists(), "data/processed/ufc.sqlite not built")
class StyleModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ff = F.load_fighter_fights()
        cls.profiles = F.build_profiles(None, cls.ff)
        cls.model = C.fit_style_model(cls.profiles, k=3, n_init=3)
        cls.assigned = cls.model.assign(cls.profiles)

    def test_probabilities_sum_to_one_and_clusters_numbered_by_size(self):
        P = self.assigned.filter(like="p_cluster_").values
        self.assertTrue(np.allclose(P.sum(axis=1), 1))
        sizes = self.assigned["cluster"].value_counts().sort_index().values
        self.assertTrue((np.diff(sizes) <= 0).all(), f"cluster sizes not descending: {sizes}")
        self.assertEqual(len(self.assigned), int(self.profiles["eligible"].sum()))

    def test_uses_clustering_features_without_redundant_shares(self):
        self.assertEqual(self.model.feature_columns, [f"z_{f}" for f in F.CLUSTER_FEATURES])
        self.assertEqual(self.model.pca.n_components_, C.N_COMPONENTS)

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = self.model.save(Path(d) / "m.joblib")
            loaded = C.StyleModel.load(path)
        again = loaded.assign(self.profiles)
        self.assertTrue(np.allclose(again.filter(like="p_cluster_").values, self.assigned.filter(like="p_cluster_").values))

    def test_applies_to_as_of_profiles(self):
        as_of = "2015-01-01"
        past = F.build_profiles(as_of, self.ff)
        out = self.model.assign(past)
        self.assertTrue((past.set_index("fighter_id").loc[out["fighter_id"], "last_fight"] < as_of).all())
        self.assertTrue(np.allclose(out.filter(like="p_cluster_").sum(axis=1), 1))
        everyone = self.model.assign(past, eligible_only=False)
        self.assertGreater(len(everyone), len(out))


if __name__ == "__main__":
    unittest.main()
