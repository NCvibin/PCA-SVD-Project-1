"""Model fitting helpers (ML2, ML5).

Fits Truncated SVD or PCA on the sparse customer x product training matrix and
records everything the comparison table needs: fit time, peak memory, size of
stored factors, and what is lost by compressing.

R2: only ever pass the TRAINING matrix to these functions.
R6: every fit uses a fixed seed (SEED).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
import sklearn
from sklearn.decomposition import PCA, TruncatedSVD

SEED = 42
MB = 1024 ** 2


# --------------------------------------------------------------------------
# Peak memory measurement
# --------------------------------------------------------------------------
class PeakMemory:
    """Context manager: peak extra RAM (MB) used by this process inside the block.

    Samples the process resident memory (RSS) every few milliseconds in a
    background thread and subtracts the starting level. It is an estimate
    (very short spikes can be missed), so run each fit in a fresh kernel state
    if you want clean numbers, and say "approximate" in the report.
    """

    def __init__(self, interval: float = 0.005):
        self.interval = interval
        self.peak_mb = 0.0

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, self._proc.memory_info().rss)
            time.sleep(self.interval)

    def __enter__(self):
        self._proc = psutil.Process()
        self.baseline = self.peak = self._proc.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, self._proc.memory_info().rss)
        self.peak_mb = (self.peak - self.baseline) / MB


# --------------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------------
@dataclass
class FitResult:
    method: str                     # "svd" or "pca"
    k: int
    customer_factors: np.ndarray    # (n_customers, k)  scores
    product_factors: np.ndarray     # (n_products, k)   loadings (V)
    fit_time: float                 # seconds (fit + computing customer factors)
    peak_mem_mb: float
    rel_error: float                # ||A - A_k||_F / ||A||_F, measured against the raw A
    energy_kept: float              # 1 - rel_error**2 (share of A's squared norm kept)
    evr: float                      # PCA explained variance ratio (sum); NaN for SVD
    model: object

    @property
    def factors_mb(self) -> float:
        return (self.customer_factors.nbytes + self.product_factors.nbytes) / MB

    def as_row(self) -> dict:
        return {
            "method": self.method,
            "k": self.k,
            "fit_time_s": round(self.fit_time, 3),
            "peak_mem_mb": round(self.peak_mem_mb, 1),
            "factors_mb": round(self.factors_mb, 2),
            "rel_error": round(self.rel_error, 4),
            "energy_kept": round(self.energy_kept, 4),
            "pca_evr": round(self.evr, 4) if not np.isnan(self.evr) else np.nan,
        }


def _fro_sq(A: sp.spmatrix) -> float:
    return float(np.square(A.data).sum())


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
def fit_svd(A, k: int, seed: int = SEED, n_iter: int = 7) -> FitResult:
    """Truncated SVD on the raw (uncentred) sparse matrix."""
    A = sp.csr_matrix(A)
    with PeakMemory() as mem:
        t0 = time.perf_counter()
        model = TruncatedSVD(n_components=k, n_iter=n_iter, random_state=seed).fit(A)
        V = model.components_.T                      # (n_products, k), orthonormal columns
        cust = np.asarray(A @ V)                     # = U S, customer factors
        fit_time = time.perf_counter() - t0

    # Because V has orthonormal columns: ||A - A V V^T||^2 = ||A||^2 - ||A V||^2
    total = _fro_sq(A)
    err2 = max(total - float(np.square(cust).sum()), 0.0)
    rel_err = float(np.sqrt(err2 / total))
    return FitResult("svd", k, cust, V, fit_time, mem.peak_mb, rel_err,
                     1 - rel_err ** 2, float("nan"), model)


def fit_pca(A, k: int, solver: str = "arpack", seed: int = SEED) -> FitResult:
    """PCA on the sparse matrix with implicit centring (needs scikit-learn >= 1.4).

    solver: "arpack" (as in the starter code) or "covariance_eigh"
    (builds a dense products x products covariance matrix, so check memory).
    """
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    if (major, minor) < (1, 4):
        raise ImportError("PCA on sparse input needs scikit-learn >= 1.4")

    A = sp.csr_matrix(A)
    n = A.shape[0]
    with PeakMemory() as mem:
        t0 = time.perf_counter()
        model = PCA(n_components=k, svd_solver=solver, random_state=seed).fit(A)
        V = model.components_.T
        mu = model.mean_
        cust = np.asarray(A @ V) - mu @ V            # centred scores, (n_customers, k)
        fit_time = time.perf_counter() - t0

    # Error measured against the RAW matrix so it is comparable with SVD:
    # A - A_k = (A - mu)(I - V V^T)  =>  err^2 = ||A - mu||^2 - ||(A - mu) V||^2
    total = _fro_sq(A)
    centred_total = total - n * float(mu @ mu)
    err2 = max(centred_total - float(np.square(cust).sum()), 0.0)
    rel_err = float(np.sqrt(err2 / total))
    return FitResult("pca", k, cust, V, fit_time, mem.peak_mb, rel_err,
                     1 - rel_err ** 2, float(model.explained_variance_ratio_.sum()), model)


def fit(A, k: int, method: str, **kw) -> FitResult:
    return {"svd": fit_svd, "pca": fit_pca}[method](A, k, **kw)


# --------------------------------------------------------------------------
# ML2: k sweep
# --------------------------------------------------------------------------
def sweep_k(A, ks: Sequence[int] = (10, 20, 50, 100, 200), method: str = "svd",
            **kw) -> pd.DataFrame:
    """Fit once per k and return one table row per k."""
    rows = []
    for k in ks:
        res = fit(A, k, method, **kw)
        rows.append(res.as_row())
        print(f"{method} k={k:<4d} err={res.rel_error:.3f} "
              f"time={res.fit_time:.1f}s size={res.factors_mb:.1f}MB")
    return pd.DataFrame(rows)


def plot_k_sweep(sweeps: dict, chosen_k: Optional[int] = None, ax=None):
    """sweeps: {"svd": df, "pca": df}. Plots relative reconstruction error vs k."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    for name, df in sweeps.items():
        ax.plot(df["k"], df["rel_error"], marker="o", label=name.upper())
    if chosen_k is not None:
        ax.axvline(chosen_k, ls="--", color="grey", label=f"chosen k = {chosen_k}")
    ax.set_xlabel("number of components k")
    ax.set_ylabel("relative reconstruction error")
    ax.set_title("What is lost by compressing")
    ax.legend()
    return ax


# --------------------------------------------------------------------------
# Interpreting component 1 (ML5 last row)
# --------------------------------------------------------------------------
def volume_correlation(res: FitResult, A) -> float:
    """Correlation of customer component-1 scores with total amount bought.
    Close to +/-1 means component 1 is basically 'how much a customer buys'."""
    volume = np.asarray(sp.csr_matrix(A).sum(axis=1)).ravel()
    return float(np.corrcoef(res.customer_factors[:, 0], volume)[0, 1])


def top_products_for_component(res: FitResult, names: Sequence[str],
                               component: int = 0, n: int = 10) -> pd.DataFrame:
    """Products with the largest loadings (sign flipped so the mean loading is positive).
    `names[j]` must be the description of product column j."""
    load = res.product_factors[:, component]
    if load.mean() < 0:
        load = -load
    order = np.argsort(-load)[:n]
    return pd.DataFrame({"product": [names[j] for j in order], "loading": load[order]})