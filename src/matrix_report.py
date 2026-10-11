"""DE5 - Matrix report: shape, non-zeros, sparsity, memory dense vs sparse.
Also writes a short hand-off note for the ML Engineer.

Usage:  python src/matrix_report.py   (run after make_matrix.py)
Writes: outputs/de5_matrix_report.md
"""
from pathlib import Path
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs"


def mb(n_bytes):
    return f"{n_bytes / 1e6:,.1f} MB"


def main():
    A = sp.load_npz(PROC / "matrix.npz")
    n_rows, n_cols = A.shape
    cells = n_rows * n_cols
    dense = cells * 8                                       # float64 = 8 bytes per cell
    sparse = A.data.nbytes + A.indices.nbytes + A.indptr.nbytes

    # how many held-out test pairs the matrix can reach (reporting only, nothing is fitted)
    test = pd.read_parquet(PROC / "test.parquet")
    customers = set(pd.read_csv(PROC / "customer_index.csv")["Customer ID"])
    products = set(pd.read_csv(PROC / "product_index.csv", dtype={"StockCode": str})["StockCode"])
    reachable = test["Customer ID"].isin(customers) & test["StockCode"].isin(products)

    md = [
        "# DE5 - Matrix report", "",
        "| Metric | Value |", "|---|---:|",
        f"| Shape (customers x products) | {n_rows:,} x {n_cols:,} |",
        f"| Non-zero cells | {A.nnz:,} |",
        f"| Sparsity | {1 - A.nnz / cells:.2%} |",
        f"| Memory, dense (float64) | {mb(dense)} |",
        f"| Memory, sparse (CSR) | {mb(sparse)} |",
        f"| Sparse is smaller by | {dense / sparse:,.0f}x |",
        "",
        "## Note for the ML Engineer", "",
        "- Load: `A = scipy.sparse.load_npz('data/processed/matrix.npz')`; row i = `customer_index.csv`, "
        "column j = `product_index.csv` (read with `dtype={'StockCode': str}`).",
        "- Cell = log(1 + total quantity) over the training period; products bought by >= 5 customers. "
        "Training rows only, no test data inside.",
        f"- Test file: {len(test):,} held-out pairs; {reachable.mean():.1%} have both customer and product "
        "in the matrix. The rest can never be recommended and count as misses for the model and the baseline.",
        "- Keep A sparse (never `.toarray()` the full matrix). PCA on sparse input needs "
        "scikit-learn >= 1.4 with `svd_solver='arpack'`.",
    ]
    OUT.mkdir(exist_ok=True)
    (OUT / "de5_matrix_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
