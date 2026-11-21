"""Style clusters: Gaussian mixture on the first principal components of the style features (see docs/clustering.md).

    import features as F, clustering as C
    profiles = F.build_profiles(None)
    model = C.fit_style_model(profiles, k=C.CHOSEN_K)  # PCA + GMM fitted on eligible fighters
    probs = model.assign(profiles)                      # cluster probabilities for eligible fighters
    model.save(); C.StyleModel.load(path)               # reuse on profiles built with an as_of date

Run as a script to fit the chosen model on full-career profiles and write
data/processed/cluster_probabilities.csv and data/processed/models/style_model_k{K}.joblib.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

import features as F

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "processed" / "models"
PROBS_PATH = ROOT / "data" / "processed" / "cluster_probabilities.csv"

N_COMPONENTS = 5          # first 5 PCs: 85% of the variance (notebooks/02_pca.ipynb)
K_RANGE = range(3, 11)
CHOSEN_K = 3              # recommended in notebooks/03_clustering.ipynb: lowest BIC and most stable for k in 3..10
RUNNER_UP_K = 4           # next-best on both BIC and stability
SEED = 0
N_INIT = 10               # GMM restarts per fit (best log-likelihood kept)


# --------------------------------------------------------------------------- fitting
def fit_pca(X: pd.DataFrame, n_components: int = N_COMPONENTS) -> PCA:
    """PCA with a fixed sign convention: each component's largest-magnitude loading is positive."""
    pca = PCA(n_components=n_components).fit(X.values)
    signs = np.sign(pca.components_[np.arange(n_components), np.abs(pca.components_).argmax(axis=1)])
    pca.components_ *= signs[:, None]
    return pca


def fit_gmm(Z: np.ndarray, k: int, seed: int = SEED, n_init: int = N_INIT) -> GaussianMixture:
    return GaussianMixture(n_components=k, covariance_type="full", n_init=n_init, random_state=seed).fit(Z)


def size_order(gmm: GaussianMixture, Z: np.ndarray) -> np.ndarray:
    """Permutation so cluster 1 is the largest (by hard assignment), cluster 2 the next, and so on."""
    counts = np.bincount(gmm.predict(Z), minlength=gmm.n_components)
    return np.argsort(-counts, kind="stable")


@dataclass
class StyleModel:
    """Fitted PCA + Gaussian mixture, with clusters numbered 1..k by size on the fitting data."""
    pca: PCA
    gmm: GaussianMixture
    order: np.ndarray                       # order[j] = GMM component shown as cluster j + 1
    feature_columns: list[str]
    metadata: dict = field(default_factory=dict)

    @property
    def k(self) -> int:
        return self.gmm.n_components

    def scores(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = self.pca.transform(X[self.feature_columns].values)
        return pd.DataFrame(Z, index=X.index, columns=[f"PC{i + 1}" for i in range(Z.shape[1])])

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        P = self.gmm.predict_proba(self.pca.transform(X[self.feature_columns].values))[:, self.order]
        return pd.DataFrame(P, index=X.index, columns=[f"p_cluster_{j + 1}" for j in range(self.k)])

    def assign(self, profiles: pd.DataFrame, eligible_only: bool = True) -> pd.DataFrame:
        """Cluster probabilities, hard cluster (1..k) and max probability for fighters in `profiles`.

        Profiles can come from build_profiles(as_of=...). With eligible_only=False, every fighter with
        complete z-scores is scored (fighters with < 5 fights get noisier, more shrunk profiles).
        """
        rows = profiles[profiles["std_group"].notna() & (profiles["eligible"] if eligible_only else True)]
        X = rows.set_index("fighter_id")[self.feature_columns].dropna()
        P = self.predict_proba(X)
        out = rows.set_index("fighter_id").loc[X.index, ["fighter_name", "division", "n_fights"]].copy()
        out["cluster"] = P.values.argmax(axis=1) + 1
        out["max_prob"] = P.max(axis=1)
        return out.join(P).join(self.scores(X)).reset_index()

    def save(self, path: Path | None = None) -> Path:
        path = path or MODEL_DIR / f"style_model_k{self.k}.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: Path | str) -> "StyleModel":
        return joblib.load(path)


def fit_style_model(profiles: pd.DataFrame, k: int = CHOSEN_K, n_components: int = N_COMPONENTS,
                    seed: int = SEED, n_init: int = N_INIT, as_of: str | None = None) -> StyleModel:
    """Fit PCA and a full-covariance GMM on the eligible fighters in `profiles`."""
    X = F.clustering_matrix(profiles)
    pca = fit_pca(X, n_components)
    Z = pca.transform(X.values)
    gmm = fit_gmm(Z, k, seed, n_init)
    meta = {"fitted_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "profiles_as_of": as_of, "n_fighters": len(X), "k": k, "n_components": n_components,
            "explained_variance": pca.explained_variance_ratio_.round(4).tolist(),
            "prior_minutes": F.PRIOR_MINUTES, "feature_priors": dict(F.FEATURE_PRIORS),
            "min_fights": F.MIN_FIGHTS, "start_date": F.START_DATE, "seed": seed, "n_init": n_init}
    return StyleModel(pca, gmm, size_order(gmm, Z), list(X.columns), meta)


# --------------------------------------------------------------------------- choosing k
def select_k(Z: np.ndarray, ks=K_RANGE, n_boot: int = 50, seed: int = SEED, n_init: int = N_INIT,
             boot_n_init: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """BIC on the full data and bootstrap stability for each k.

    Stability: refit the GMM on a bootstrap resample of fighters, assign every original fighter with
    the refit model, and compare with the full-data assignments using the adjusted Rand index (ARI).
    The PCA is held fixed: its first 5-PC space barely moves under resampling, and the question is
    whether the mixture's partition is reproducible.
    Returns (summary per k, all bootstrap ARIs).
    """
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, len(Z), len(Z)) for _ in range(n_boot)]
    rows, aris = [], []
    for k in ks:
        ref = fit_gmm(Z, k, seed, n_init)
        ref_labels = ref.predict(Z)
        a = []
        for b, idx in enumerate(boots):
            g = fit_gmm(Z[idx], k, seed + 1 + b, boot_n_init)
            a.append(adjusted_rand_score(ref_labels, g.predict(Z)))
        aris += [{"k": k, "boot": b, "ari": v} for b, v in enumerate(a)]
        a = np.array(a)
        sizes = np.bincount(ref_labels, minlength=k)
        rows.append({"k": k, "bic": ref.bic(Z), "aic": ref.aic(Z), "ari_mean": a.mean(), "ari_sd": a.std(),
                     "ari_p10": np.quantile(a, 0.1), "smallest_cluster": int(sizes.min()),
                     "mean_max_prob": ref.predict_proba(Z).max(axis=1).mean()})
    return pd.DataFrame(rows).set_index("k"), pd.DataFrame(aris)


def kmeans_agreement(Z: np.ndarray, gmm_labels: np.ndarray, k: int, seed: int = SEED) -> dict:
    """Fit k-means at the same k; return ARI and the share of fighters in matched clusters (Hungarian matching)."""
    km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
    ct = pd.crosstab(gmm_labels, km.labels_).values
    r, c = linear_sum_assignment(-ct)
    return {"ari": adjusted_rand_score(gmm_labels, km.labels_), "matched_share": ct[r, c].sum() / len(Z),
            "kmeans_labels": km.labels_, "match": dict(zip(r, c)), "crosstab": ct}


# --------------------------------------------------------------------------- script
def main() -> None:
    profiles = F.build_profiles(None)
    for k in (CHOSEN_K, RUNNER_UP_K):
        model = fit_style_model(profiles, k)
        path = model.save()
        probs = model.assign(profiles)
        if k == CHOSEN_K:
            probs.to_csv(PROBS_PATH, index=False)
        print(f"k={k}: saved {path.relative_to(ROOT)}; cluster sizes "
              f"{probs['cluster'].value_counts().sort_index().to_dict()}")
    print(f"wrote {PROBS_PATH.relative_to(ROOT)} ({len(probs):,} eligible fighters, k={CHOSEN_K})")


if __name__ == "__main__":
    # Run through the imported module so saved models pickle as clustering.StyleModel, not __main__.StyleModel,
    # and can be loaded from notebooks and other scripts.
    import clustering
    clustering.main()
