"""DE4 - Sparse customer x product matrix from TRAINING rows only.

Cell value:     log(1 + total quantity) per customer-product pair
Product filter: keep products bought by at least MIN_CUSTOMERS different customers

Usage:  python src/make_matrix.py
Writes: data/processed/matrix.npz          sparse CSR matrix (customers x products)
        data/processed/customer_index.csv  row    -> Customer ID
        data/processed/product_index.csv   column -> StockCode, Description
        outputs/de4_decisions.md           the two choices and why
"""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
TRAIN_PATH = PROC / "train.parquet"
PRODUCTS_PATH = ROOT / "data" / "interim" / "products.parquet"
OUT = ROOT / "outputs"
MIN_CUSTOMERS = 5


def main():
    train = pd.read_parquet(TRAIN_PATH)

    # product filter (counted on training data only)
    buyers = train.groupby("StockCode")["Customer ID"].nunique()
    keep = buyers[buyers >= MIN_CUSTOMERS].index
    kept = train[train["StockCode"].isin(keep)]

    # sum quantity per customer-product FIRST, then log (log of the total, not a sum of logs)
    pairs = kept.groupby(["Customer ID", "StockCode"], as_index=False)["Quantity"].sum()
    customers = np.sort(pairs["Customer ID"].unique())
    products = np.sort(pairs["StockCode"].unique())
    rows = pd.Index(customers).get_indexer(pairs["Customer ID"])
    cols = pd.Index(products).get_indexer(pairs["StockCode"])
    A = sp.csr_matrix((np.log1p(pairs["Quantity"].to_numpy(dtype=float)), (rows, cols)),
                      shape=(len(customers), len(products)))

    # save matrix + index maps
    sp.save_npz(PROC / "matrix.npz", A)
    pd.DataFrame({"row": range(len(customers)), "Customer ID": customers}) \
      .to_csv(PROC / "customer_index.csv", index=False)
    desc = pd.read_parquet(PRODUCTS_PATH).set_index("StockCode")["Description"]
    pd.DataFrame({"col": range(len(products)), "StockCode": products,
                  "Description": desc.reindex(products).to_numpy()}) \
      .to_csv(PROC / "product_index.csv", index=False)

    # decisions (numbers computed from the data)
    dropped = len(buyers) - len(keep)
    lines_dropped = 1 - len(kept) / len(train)
    md = [
        "# DE4 - Matrix decisions", "",
        "| Choice | Decision | Why |", "|---|---|---|",
        "| Cell value | log(1 + total quantity) | Most buyers are wholesalers, so raw quantity mostly "
        "measures shop size; the log keeps \"bought more\" as a signal without letting big buyers dominate. |",
        f"| Product filter | Bought by >= {MIN_CUSTOMERS} customers | Products with fewer buyers are too rare "
        f"to learn from: {dropped:,} of {len(buyers):,} products ({dropped / len(buyers):.0%}) but only "
        f"{lines_dropped:.1%} of purchase lines. |",
        "", f"Result: {A.shape[0]:,} customers x {A.shape[1]:,} products, built from training rows only.",
    ]
    OUT.mkdir(exist_ok=True)
    (OUT / "de4_decisions.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
