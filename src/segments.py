"""ML6 - customer map: project customers on 2 components, cluster them, describe each group.

Everything works on matrix row numbers, never Customer IDs (R4).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

SEED = 42


def map_coordinates(customer_factors: np.ndarray, A) -> np.ndarray:
    """First 2 components of the customer factors, with component 1 flipped if needed so that
    a higher value means a bigger buyer (the sign of a component is arbitrary)."""
    Z = np.asarray(customer_factors[:, :2], dtype=float).copy()
    volume = np.asarray(sp.csr_matrix(A).sum(axis=1)).ravel()
    if np.corrcoef(Z[:, 0], volume)[0, 1] < 0:
        Z[:, 0] *= -1
    return Z


def cluster_customers(Z: np.ndarray, ks: Sequence[int] = (3, 4, 5, 6), seed: int = SEED,
                      standardise: bool = True):
    """K-means for each k in ks. Returns (table, {k: labels}).

    Each axis is standardised first so that the high-variance volume axis does not decide everything.
    The table has the silhouette score (higher = better separated) and the size of the smallest group.
    """
    X = StandardScaler().fit_transform(Z) if standardise else Z
    rows, labels = [], {}
    for k in ks:
        lab = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        labels[k] = lab
        rows.append({"n_segments": k,
                     "silhouette": round(float(silhouette_score(X, lab)), 4),
                     "smallest_group_share": round(float(np.bincount(lab).min() / len(lab)), 4)})
    return pd.DataFrame(rows), labels


def relabel_by_volume(labels: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Renumber segments 0, 1, 2, ... in order of increasing median component 1 (buying volume)."""
    order = np.argsort([np.median(Z[labels == s, 0]) for s in np.unique(labels)])
    mapping = {int(old): new for new, old in enumerate(np.unique(labels)[order])}
    return np.array([mapping[int(s)] for s in labels])


def _pct(x: float) -> str:
    """Whole percent for large shares, one decimal for small ones (so 0.4% does not show as 0%)."""
    return f"{x:.0%}" if x >= 0.10 else f"{x:.1%}"


def profile_segments(A, labels: np.ndarray, names: Sequence[str], country: Sequence[str],
                     n_products: int = 5, n_countries: int = 3) -> pd.DataFrame:
    """One row per segment: size, typical number of products bought, top countries, the products
    bought much more often in the segment than overall (share in segment vs overall), and the
    products most commonly bought in the segment. A segment that makes up most of the customers has
    shares close to the overall ones, so it has few distinctive products; use most_common_products."""
    A = sp.csr_matrix(A)
    B = (A > 0).astype(float).tocsr()
    overall = np.asarray(B.mean(axis=0)).ravel()
    n_bought = A.getnnz(axis=1)
    country = np.asarray(country)
    rows = []
    for s in np.unique(labels):
        m = labels == s
        share = np.asarray(B[m].mean(axis=0)).ravel()
        top = np.argsort(-(share - overall))[:n_products]
        common = np.argsort(-share)[:3]
        cs = pd.Series(country[m]).value_counts(normalize=True).head(n_countries)
        rows.append({
            "segment": int(s),
            "customers": int(m.sum()),
            "share_of_customers": round(float(m.mean()), 4),
            "median_products_bought": float(np.median(n_bought[m])),
            "top_countries": "; ".join(f"{c} {v:.0%}" for c, v in cs.items()),
            "top_products": "; ".join(f"{names[j]} ({_pct(share[j])} vs {_pct(overall[j])})" for j in top),
            "most_common_products": "; ".join(f"{names[j]} ({_pct(share[j])})" for j in common),
        })
    return pd.DataFrame(rows)
