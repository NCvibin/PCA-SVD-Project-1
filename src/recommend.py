"""Recommendation functions (ML3).

Item-item cosine similarity in the compressed space:
  * similar_products / all_similar_products: "customers who bought this also bought"
  * recommend_for_customers: score every product by its similarity to what the
    customer already bought
  * recommend_bestsellers: the R3 baseline

Everything works on row/column INDICES, never raw customer IDs (R4).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize


def product_vectors(product_factors: np.ndarray, scale: Optional[np.ndarray] = None) -> np.ndarray:
    """Unit-length product vectors. `scale` (optional, length k) weights components,
    e.g. singular values. Default (None) matches the starter code."""
    X = product_factors if scale is None else product_factors * scale
    return normalize(X)


def _top_n(scores: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n largest scores per row, best first."""
    part = np.argpartition(-scores, n - 1, axis=1)[:, :n]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1)
    return np.take_along_axis(part, order, axis=1)


def similar_products(P_vec: np.ndarray, j: int, n: int = 10):
    """The n products most similar to product j: (indices, cosine similarities)."""
    s = P_vec @ P_vec[j]
    s[j] = -np.inf
    idx = np.argsort(-s)[:n]
    return idx, s[idx]


def all_similar_products(P_vec: np.ndarray, n: int = 10, batch_size: int = 1000) -> np.ndarray:
    """Top-n similar products for EVERY product: (n_products, n) index array.
    This is the table the product-page box reads from; batching keeps memory small."""
    m = P_vec.shape[0]
    out = np.empty((m, n), dtype=np.int64)
    for start in range(0, m, batch_size):
        stop = min(start + batch_size, m)
        S = P_vec[start:stop] @ P_vec.T
        S[np.arange(stop - start), np.arange(start, stop)] = -np.inf
        out[start:stop] = _top_n(S, n)
    return out


def _mask_purchased(scores: np.ndarray, B: sp.csr_matrix) -> None:
    coo = B.tocoo()
    scores[coo.row, coo.col] = -np.inf


def recommend_for_customers(A, P_vec: np.ndarray, rows: Sequence[int], n: int = 10,
                            batch_size: int = 1000, use_values: bool = False) -> np.ndarray:
    """Top-n products per customer row: (len(rows), n) index array.

    score(customer, product) = sum over bought products b of cos(product, b).
    Computed without building the product x product matrix:
        profile = (bought indicator) @ P_vec ;  scores = profile @ P_vec.T
    Summing or averaging gives the same ranking for one customer.
    Products the customer already bought are excluded.
    use_values=True weights bought products by their matrix value instead of 1.
    Pass the raw (un-normalised) product factors as P_vec to score by dot product instead of
    cosine: the vector length, which tracks how widely a product sells, is then kept.
    """
    A = sp.csr_matrix(A)
    rows = np.asarray(rows)
    out = np.empty((len(rows), n), dtype=np.int64)
    for start in range(0, len(rows), batch_size):
        B = A[rows[start:start + batch_size]].copy()
        B.eliminate_zeros()
        W = B if use_values else B.copy()
        if not use_values:
            W.data[:] = 1.0
        scores = np.asarray(W @ P_vec) @ P_vec.T
        _mask_purchased(scores, B)
        out[start:start + len(B.indptr) - 1] = _top_n(scores, n)
    return out


def popularity(A) -> np.ndarray:
    """Number of distinct customers who bought each product (training data)."""
    return np.diff(sp.csc_matrix(A).indptr).astype(float)


def recommend_bestsellers(A, rows: Sequence[int], n: int = 10,
                          batch_size: int = 1000) -> np.ndarray:
    """R3 baseline: the n best-selling products the customer has NOT already bought."""
    A = sp.csr_matrix(A)
    rows = np.asarray(rows)
    pop = popularity(A)
    pop = pop - np.arange(len(pop)) * 1e-9          # deterministic tie-break
    out = np.empty((len(rows), n), dtype=np.int64)
    for start in range(0, len(rows), batch_size):
        B = A[rows[start:start + batch_size]].copy()
        B.eliminate_zeros()
        scores = np.tile(pop, (B.shape[0], 1))
        _mask_purchased(scores, B)
        out[start:start + B.shape[0]] = _top_n(scores, n)
    return out
