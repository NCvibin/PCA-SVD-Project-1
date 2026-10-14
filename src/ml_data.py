"""ML data access - one place that loads what the Data Engineer produced.

Reads (all written by build_matrix.py; nothing here modifies them):
    data/processed/matrix.npz          sparse CSR, customers x products, log(1 + total quantity)
    data/processed/customer_index.csv  row -> Customer ID
    data/processed/product_index.csv   column -> StockCode, Description
    data/processed/test.parquet        held-out (Customer ID, StockCode) pairs
    data/processed/train.parquet       only used to find each customer's main country

R4: raw Customer IDs are used inside load_all() to line up rows, then dropped. The returned
Data object holds matrix row numbers only, so nothing downstream can print an ID.

Usage from the notebook (run from the repo root):
    from src.ml_data import load_all
    data = load_all()
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .evaluate import build_holdout

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


@dataclass
class Data:
    A: sp.csr_matrix        # training matrix (never densify: ~180 MB)
    names: list             # names[j] = description of product column j
    codes: np.ndarray       # codes[j]  = StockCode of product column j
    country: np.ndarray     # main country of each matrix row
    holdout: dict           # {matrix row: array of held-out product columns}  (scorable customers)
    holdout_stats: dict     # what build_holdout dropped and why
    n_test_customers: int   # ALL test customers, including those the model cannot score


def load_all(proc: Path = PROC) -> Data:
    proc = Path(proc)
    A = sp.load_npz(proc / "matrix.npz").tocsr()
    cust = pd.read_csv(proc / "customer_index.csv")
    prod = pd.read_csv(proc / "product_index.csv", dtype={"StockCode": str})
    assert A.shape == (len(cust), len(prod)), "matrix and index maps disagree"

    test = pd.read_parquet(proc / "test.parquet")
    cust_index = dict(zip(cust["Customer ID"], cust["row"]))
    prod_index = dict(zip(prod["StockCode"], prod["col"]))
    holdout, stats = build_holdout(test, cust_index, prod_index, A)

    try:  # main country per customer, for describing segments (ML6)
        tr = pd.read_parquet(proc / "train.parquet", columns=["Customer ID", "Country"])
        main = tr.groupby("Customer ID")["Country"].agg(lambda s: s.mode().iloc[0])
        country = main.reindex(cust["Customer ID"]).fillna("Unknown").to_numpy()
    except FileNotFoundError:
        country = np.full(len(cust), "Unknown", dtype=object)

    names = prod["Description"].fillna(prod["StockCode"]).tolist()
    return Data(A=A, names=names, codes=prod["StockCode"].to_numpy(), country=country,
                holdout=holdout, holdout_stats=stats,
                n_test_customers=int(test["Customer ID"].nunique()))


def hit_rate_all_customers(hit_rate_scorable: float, data: Data) -> float:
    """Headline hit rate. Test customers the model cannot score (not in the matrix, or with no
    reachable held-out product) count as misses, as the Data Engineer's hand-off note says."""
    return hit_rate_scorable * len(data.holdout) / data.n_test_customers


def change_cell_value(A: sp.csr_matrix, mode: str) -> sp.csr_matrix:
    """ML7: other purchase values, rebuilt from the saved matrix (no pipeline re-run needed).

    'log'    as delivered, log(1 + total quantity)
    'count'  total quantity (recovered exactly with expm1)
    'binary' 1 if the customer ever bought the product
    """
    B = A.copy()
    if mode == "count":
        B.data = np.rint(np.expm1(B.data))
    elif mode == "binary":
        B.data = np.ones_like(B.data)
    elif mode != "log":
        raise ValueError("mode must be 'log', 'count' or 'binary'")
    return B
