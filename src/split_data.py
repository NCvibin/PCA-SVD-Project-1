"""DE3 - Time split. The last 2 months are the test period.

Train: every cleaned purchase before the cutoff.
Test:  for each test customer, the products they bought in the test period
       but never bought before the cutoff (one row per customer-product pair).
Test-period data never enters train (R2).

Usage:  python src/split_data.py
Writes: data/processed/train.parquet   transactions before the cutoff
        data/processed/test.parquet    held-out (Customer ID, StockCode) pairs
        outputs/de3_split_log.md
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "data" / "interim" / "clean.parquet"
TRAIN_PATH = ROOT / "data" / "processed" / "train.parquet"
TEST_PATH = ROOT / "data" / "processed" / "test.parquet"
OUT = ROOT / "outputs"
TEST_MONTHS = 2


def main():
    df = pd.read_parquet(CLEAN_PATH)
    cutoff = df["InvoiceDate"].max().normalize() - pd.DateOffset(months=TEST_MONTHS)

    train = df[df["InvoiceDate"] < cutoff]
    test_period = df[df["InvoiceDate"] >= cutoff]

    # unique customer-product pairs bought in the test period
    pairs = test_period[["Customer ID", "StockCode"]].drop_duplicates()
    known_customer = pairs["Customer ID"].isin(train["Customer ID"].unique())
    bought_before = pairs.set_index(["Customer ID", "StockCode"]).index.isin(
        train.set_index(["Customer ID", "StockCode"]).index)

    test = pairs[known_customer & ~bought_before].sort_values(["Customer ID", "StockCode"])

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(TRAIN_PATH, index=False)
    test.to_parquet(TEST_PATH, index=False)

    log = pd.DataFrame([
        ("Cutoff date (test = on or after)", cutoff.date()),
        ("Train period", f"{train['InvoiceDate'].min().date()} to {train['InvoiceDate'].max().date()}"),
        ("Test period", f"{test_period['InvoiceDate'].min().date()} to {test_period['InvoiceDate'].max().date()}"),
        ("Train rows", f"{len(train):,}"),
        ("Train customers", f"{train['Customer ID'].nunique():,}"),
        ("Test-period customer-product pairs", f"{len(pairs):,}"),
        ("  dropped: customer not in train", f"{(~known_customer).sum():,}"),
        ("  dropped: product already bought in train", f"{(known_customer & bought_before).sum():,}"),
        ("Held-out test pairs", f"{len(test):,}"),
        ("Test customers", f"{test['Customer ID'].nunique():,}"),
    ], columns=["Item", "Value"])

    OUT.mkdir(exist_ok=True)
    md = ["# DE3 - Time split", "", "| Item | Value |", "|---|---:|"]
    md += [f"| {a} | {b} |" for a, b in log.itertuples(index=False)]
    (OUT / "de3_split_log.md").write_text("\n".join(md), encoding="utf-8")
    print(log.to_string(index=False))


if __name__ == "__main__":
    main()
