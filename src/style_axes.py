"""Continuous style axes: PCA on the era-relative style z-scores, varimax-rotated (see docs/style_axes.md).

    import features as F, style_axes as A
    profiles = F.build_profiles(None)
    axes = A.fit_style_axes(profiles, n_axes=A.CHOSEN_AXES)   # fitted on eligible fighters
    scores = axes.transform(profiles)                          # axis_1..axis_m per eligible fighter
    axes.save(); A.StyleAxes.load(path)                        # reuse on build_profiles(as_of=...) output

Axes are numbered by the variance they explain after rotation (axis 1 = most) and are not named.
Axis scores are standardized (mean 0, SD 1 on the fitting data) and uncorrelated; loadings are the
correlations between each input feature and each axis.

Run as a script to fit on full-career profiles and write data/processed/style_axes.csv and
data/processed/models/style_axes_m{m}.joblib.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

import features as F

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "processed" / "models"
SCORES_PATH = ROOT / "data" / "processed" / "style_axes.csv"

CANDIDATE_AXES = (3, 4)
CHOSEN_AXES = 4           # chosen in notebooks/05_style_axes.ipynb (stability + simple structure)
SEED = 0


def varimax(L: np.ndarray, gamma: float = 1.0, max_iter: int = 500, tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Varimax rotation of a loading matrix (features x factors). Returns (rotated loadings, rotation matrix)."""
    p, k = L.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(max_iter):
        Lr = L @ R
        u, s, vt = np.linalg.svd(L.T @ (Lr ** 3 - (gamma / p) * Lr @ np.diag((Lr ** 2).sum(axis=0))))
        R = u @ vt
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
    return L @ R, R


@dataclass
class StyleAxes:
    feature_columns: list[str]
    mean: np.ndarray          # feature means on the fitting data
    V: np.ndarray             # unit eigenvectors, features x m
    eigvals: np.ndarray       # m eigenvalues (variances of the unrotated PCs)
    R: np.ndarray             # m x m rotation, with axis order and signs folded in
    total_variance: float
    metadata: dict = field(default_factory=dict)

    @property
    def n_axes(self) -> int:
        return self.R.shape[1]

    @property
    def axis_names(self) -> list[str]:
        return [f"axis_{i + 1}" for i in range(self.n_axes)]

    @property
    def loadings(self) -> pd.DataFrame:
        """Correlation-scale loadings (features x axes) of the rotated axes."""
        A = (self.V * np.sqrt(self.eigvals)) @ self.R
        return pd.DataFrame(A, index=[c.removeprefix("z_") for c in self.feature_columns], columns=self.axis_names)

    @property
    def explained(self) -> pd.Series:
        """Share of total input variance carried by each rotated axis."""
        return (self.loadings ** 2).sum().div(self.total_variance)

    def scores(self, X: pd.DataFrame) -> pd.DataFrame:
        W = (X[self.feature_columns].to_numpy(float) - self.mean) @ self.V / np.sqrt(self.eigvals)
        return pd.DataFrame(W @ self.R, index=X.index, columns=self.axis_names)

    def transform(self, profiles: pd.DataFrame, eligible_only: bool = True) -> pd.DataFrame:
        """Axis scores for fighters in `profiles` (e.g. from build_profiles(as_of=...))."""
        rows = profiles[profiles["std_group"].notna() & (profiles["eligible"] if eligible_only else True)]
        X = rows.set_index("fighter_id")[self.feature_columns].dropna()
        out = rows.set_index("fighter_id").loc[X.index, ["fighter_name", "division", "n_fights", "career_mid"]]
        return out.join(self.scores(X)).reset_index()

    def save(self, path: Path | None = None) -> Path:
        path = path or MODEL_DIR / f"style_axes_m{self.n_axes}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path | str) -> "StyleAxes":
        return joblib.load(path)


def _fit(X: np.ndarray, n_axes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = np.cov(Xc, rowvar=False, ddof=0)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    Vm, wm = V[:, :n_axes], w[:n_axes]
    _, R = varimax(Vm * np.sqrt(wm))
    A = (Vm * np.sqrt(wm)) @ R
    # order axes by variance after rotation, then make each axis's largest-magnitude loading positive
    idx = np.argsort(-(A ** 2).sum(axis=0))
    R, A = R[:, idx], A[:, idx]
    signs = np.sign(A[np.abs(A).argmax(axis=0), np.arange(n_axes)])
    return mean, Vm, wm, R * signs, float(w.sum())


def fit_style_axes(profiles: pd.DataFrame, n_axes: int = CHOSEN_AXES, as_of: str | None = None) -> StyleAxes:
    """Fit PCA + varimax on the eligible fighters' era-relative style z-scores (features.clustering_matrix)."""
    X = F.clustering_matrix(profiles)
    mean, V, w, R, tot = _fit(X.to_numpy(float), n_axes)
    meta = {"fitted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "profiles_as_of": as_of,
            "n_fighters": len(X), "n_axes": n_axes, "era_window_years": F.ERA_WINDOW_YEARS, "era_min_ref": F.ERA_MIN_REF,
            "prior_minutes": F.PRIOR_MINUTES, "feature_priors": dict(F.FEATURE_PRIORS), "min_fights": F.MIN_FIGHTS,
            "start_date": F.START_DATE}
    return StyleAxes(list(X.columns), mean, V, w, R, tot, meta)


def match_axes(ref: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match `other`'s columns to `ref`'s by |Tucker congruence|. Returns (permutation, signs, congruences)."""
    nr = ref / np.linalg.norm(ref, axis=0)
    no = other / np.linalg.norm(other, axis=0)
    C = nr.T @ no
    r, c = linear_sum_assignment(-np.abs(C))
    signs = np.sign(C[r, c])
    return c, signs, np.abs(C[r, c])


def bootstrap_stability(X: pd.DataFrame, n_axes: int, n_boot: int = 200, seed: int = SEED) -> dict:
    """Refit PCA + varimax on bootstrap resamples of fighters and compare with the full-data axes.

    Returns per-axis loading congruence (Tucker) and per-axis correlation between bootstrap and
    full-data scores for all fighters, each as an (n_boot x n_axes) array.
    """
    Xv = X.to_numpy(float)
    ref = fit_style_axes_from_matrix(Xv, n_axes)
    ref_scores = ref["scores"](Xv)
    rng = np.random.default_rng(seed)
    cong, corr = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(Xv), len(Xv))
        b = fit_style_axes_from_matrix(Xv[idx], n_axes)
        perm, signs, c = match_axes(ref["A"], b["A"])
        s = b["scores"](Xv)[:, perm] * signs
        cong.append(c)
        corr.append([np.corrcoef(ref_scores[:, j], s[:, j])[0, 1] for j in range(n_axes)])
    return {"congruence": np.array(cong), "score_corr": np.array(corr)}


def fit_style_axes_from_matrix(X: np.ndarray, n_axes: int) -> dict:
    mean, V, w, R, tot = _fit(X, n_axes)
    A = (V * np.sqrt(w)) @ R
    return {"A": A, "scores": lambda Y: ((Y - mean) @ V / np.sqrt(w)) @ R, "explained": (A ** 2).sum(axis=0) / tot}


def simple_structure(loadings: pd.DataFrame, salient: float = 0.4) -> dict:
    """Interpretability summaries of a rotated loading matrix."""
    L2 = loadings.values ** 2
    comm = L2.sum(axis=1)
    purity = L2.max(axis=1) / comm                  # share of each feature's explained variance on its main axis
    hoffman = comm ** 2 / (L2 ** 2).sum(axis=1)     # Hofmann complexity: 1 = loads on one axis only
    sal = np.abs(loadings.values) >= salient
    return {"mean_purity": float(purity.mean()), "mean_complexity": float(hoffman.mean()),
            "cross_loading_features": int((sal.sum(axis=1) > 1).sum()),
            "features_without_salient_loading": int((sal.sum(axis=1) == 0).sum()),
            "axes_with_2plus_salient": int((sal.sum(axis=0) >= 2).sum()),
            "min_communality": float(comm.min()), "communality": dict(zip(loadings.index, comm.round(3)))}


def main() -> None:
    profiles = F.build_profiles(None)
    for m in CANDIDATE_AXES:
        axes = fit_style_axes(profiles, m)
        path = axes.save()
        print(f"m={m}: saved {path.relative_to(ROOT)}; explained per axis {axes.explained.round(3).to_dict()}")
        if m == CHOSEN_AXES:
            axes.transform(profiles).to_csv(SCORES_PATH, index=False)
    print(f"wrote {SCORES_PATH.relative_to(ROOT)} (m={CHOSEN_AXES})")


if __name__ == "__main__":
    import style_axes  # pickle as style_axes.StyleAxes, not __main__.StyleAxes
    style_axes.main()
