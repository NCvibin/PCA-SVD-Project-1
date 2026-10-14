"""Evaluation (ML4): hit rate @ n against held-out purchases, vs the baseline.

R2: the held-out data is used ONLY here, never in a fit.
"""
from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .recommend import (all_similar_products, recommend_bestsellers,
                        recommend_for_customers)


def build_holdout(test_df: pd.DataFrame, cust_index: dict, prod_index: dict, A_train,
                  customer_col: str = "Customer ID", product_col: str = "StockCode"
                  ) -> Tuple[Dict[int, np.ndarray], dict]:
    """Map the test file to {customer_row: array of held-out product columns}.

    Drops (and counts) test lines whose customer or product is not in the training
    matrix, and lines for products the customer already bought in training.
    CHECK dtypes: if keys in cust_index are ints but test_df has floats (13085.0),
    everything will be dropped. The returned stats make that visible.
    """
    df = test_df[[customer_col, product_col]].drop_duplicates()
    rows = df[customer_col].map(cust_index)
    cols = df[product_col].map(prod_index)
    known = rows.notna() & cols.notna()
    stats = {"test_pairs": len(df),
             "dropped_unknown_customer": int(rows.isna().sum()),
             "dropped_unknown_product": int((rows.notna() & cols.isna()).sum())}

    r = rows[known].astype(int).to_numpy()
    c = cols[known].astype(int).to_numpy()
    seen = np.asarray(sp.csr_matrix(A_train)[r, c]).ravel() != 0
    stats["dropped_already_bought"] = int(seen.sum())
    r, c = r[~seen], c[~seen]

    holdout = {int(k): g.to_numpy()
               for k, g in pd.Series(c).groupby(r)}
    stats["test_customers"] = len(holdout)
    stats["test_pairs_used"] = int(len(r))
    return holdout, stats


def hit_rate_at_n(recs: np.ndarray, rows: np.ndarray, holdout: dict) -> float:
    """Share of customers with at least one held-out product in their top-n."""
    hits = [np.isin(recs[i], holdout[int(r)]).any() for i, r in enumerate(rows)]
    return float(np.mean(hits))


def evaluate_recommender(A_train, P_vec: np.ndarray, holdout: dict, n: int = 10,
                         **rec_kw) -> dict:
    """Hit rate @ n for the model and the best-seller baseline on the same customers."""
    rows = np.array(sorted(holdout))
    m = hit_rate_at_n(recommend_for_customers(A_train, P_vec, rows, n, **rec_kw), rows, holdout)
    b = hit_rate_at_n(recommend_bestsellers(A_train, rows, n), rows, holdout)
    N = len(rows)
    se = lambda p: float(np.sqrt(p * (1 - p) / N))
    return {"n_customers": N, f"model_hit_rate@{n}": m, f"baseline_hit_rate@{n}": b,
            "model_se": se(m), "baseline_se": se(b),
            "lift": m / b if b > 0 else float("nan")}


def time_nightly_job(A_train, P_vec: np.ndarray, n: int = 10,
                     P_score: np.ndarray = None) -> dict:
    """Seconds to (1) build the 'also bought' list for every product (the product-page
    box, cosine on P_vec) and (2) score every customer (with P_score; defaults to P_vec).
    Excludes the model fit; add fit_time for the total."""
    t0 = time.perf_counter()
    all_similar_products(P_vec, n)
    t_products = time.perf_counter() - t0

    rows = np.arange(sp.csr_matrix(A_train).shape[0])
    t0 = time.perf_counter()
    recommend_for_customers(A_train, P_vec if P_score is None else P_score, rows, n)
    t_customers = time.perf_counter() - t0
    return {"all_products_similar_s": round(t_products, 3),
            "all_customers_scored_s": round(t_customers, 3)}
