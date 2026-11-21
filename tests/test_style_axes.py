"""Tests for src/style_axes.py. Run: venv/bin/python -m unittest discover tests

Needs data/processed/ufc.sqlite; skipped otherwise.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features as F  # noqa: E402
import style_axes as A  # noqa: E402


@unittest.skipUnless(F.DB_PATH.exists(), "data/processed/ufc.sqlite not built")
class StyleAxesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ff = F.load_fighter_fights()
        cls.profiles = F.build_profiles(None, cls.ff)
        cls.axes = A.fit_style_axes(cls.profiles, 4)
        cls.scores = cls.axes.transform(cls.profiles)

    def test_scores_standardized_and_uncorrelated(self):
        S = self.scores[self.axes.axis_names].to_numpy()
        self.assertTrue(np.allclose(S.mean(axis=0), 0, atol=1e-8))
        self.assertTrue(np.allclose(np.cov(S, rowvar=False, ddof=0), np.eye(4), atol=1e-8))

    def test_loadings_are_feature_axis_correlations(self):
        X = F.clustering_matrix(self.profiles)
        S = self.axes.scores(X).to_numpy()
        for i, col in enumerate(self.axes.feature_columns):
            x = X[col].to_numpy()
            r = [np.corrcoef(x, S[:, j])[0, 1] * x.std() for j in range(4)]  # loading = cov(x, axis) since axis sd = 1
            self.assertTrue(np.allclose(r, self.axes.loadings.iloc[i].to_numpy(), atol=1e-6))

    def test_axes_ordered_by_variance_and_sign_convention(self):
        ex = self.axes.explained.to_numpy()
        self.assertTrue((np.diff(ex) <= 1e-12).all())
        L = self.axes.loadings.to_numpy()
        self.assertTrue((L[np.abs(L).argmax(axis=0), np.arange(4)] > 0).all())

    def test_varimax_rotation_is_orthogonal(self):
        R = np.abs(self.axes.R)  # signs folded in; |R| columns still unit length
        self.assertTrue(np.allclose(self.axes.R.T @ self.axes.R, np.eye(4), atol=1e-8))
        self.assertTrue(np.allclose((R ** 2).sum(axis=0), 1))

    def test_save_load_and_apply_to_as_of_profiles(self):
        with tempfile.TemporaryDirectory() as d:
            loaded = A.StyleAxes.load(self.axes.save(Path(d) / "axes.joblib"))
        again = loaded.transform(self.profiles)
        self.assertTrue(np.allclose(again[loaded.axis_names].to_numpy(), self.scores[self.axes.axis_names].to_numpy()))
        past = F.build_profiles("2015-01-01", self.ff)
        out = loaded.transform(past)
        self.assertGreater(len(out), 100)
        self.assertTrue(np.isfinite(out[loaded.axis_names].to_numpy()).all())


if __name__ == "__main__":
    unittest.main()
