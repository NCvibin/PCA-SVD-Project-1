"""DE2 - Clean the data and log how many rows each rule removes.

Usage:  python src/clean_data.py
Writes: outputs/de2_cleaning_log.md           rows removed per rule
        data/interim/clean.parquet           cleaned transactions
        data/interim/products.parquet        StockCode -> description (for naming segments later)
"""
from pathlib import Path
import pandas as pd

from load_data import load_raw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
CLEAN_PATH = ROOT / "data" / "interim" / "clean.parquet"
PRODUCTS_PATH = ROOT / "data" / "interim" / "products.parquet"

# Real products: 5 digits, optionally followed by letters (85123A, 15056BL).
# Everything else (POST, DOT, M, D, C2, BANK CHARGES, gift vouchers, ...) is not a product.
PRODUCT_CODE = r"^\d{5}[A-Z]*$"


def overlap_mask(df):
    """Rows of the 2010-2011 sheet whose invoice is already in the 2009-2010 sheet (Dec 2010)."""
    first, second = list(dict.fromkeys(df["sheet"]))
    seen = set(df.loc[df["sheet"] == first, "Invoice"])
    return (df["sheet"] == second) & df["Invoice"].isin(seen)


def main():
    df = load_raw()
    df["Invoice"] = df["Invoice"].str.strip().str.upper()
    df["StockCode"] = df["StockCode"].str.strip().str.upper()

    rules = [
        ("Duplicate rows (sheets overlap in Dec 2010)", overlap_mask),
        ("Cancellations (Invoice starts with 'C')",     lambda d: d["Invoice"].str.startswith("C")),
        ("Quantity <= 0",                                lambda d: d["Quantity"] <= 0),
        ("Price <= 0",                                   lambda d: d["Price"] <= 0),
        ("Non-product StockCode",                        lambda d: ~d["StockCode"].str.match(PRODUCT_CODE)),
        ("Missing Customer ID",                          lambda d: d["Customer ID"].isna()),
    ]

    log = [("Raw data", 0, len(df))]
    for name, rule in rules:
        drop = rule(df)
        df = df.loc[~drop]
        log.append((name, int(drop.sum()), len(df)))

    log = pd.DataFrame(log, columns=["Rule", "Rows removed", "Rows remaining"])
    OUT.mkdir(exist_ok=True)
    md = ["# DE2 - Cleaning log", "",
          "| Rule | Rows removed | Rows remaining |", "|---|---:|---:|"]
    md += [f"| {r} | {a:,} | {b:,} |" for r, a, b in log.itertuples(index=False)]
    (OUT / "de2_cleaning_log.md").write_text("\n".join(md), encoding="utf-8")

    df = df.drop(columns="sheet").astype({"Customer ID": "int64"})
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("InvoiceDate").to_parquet(CLEAN_PATH, index=False)
    (df.dropna(subset=["Description"])
       .groupby("StockCode")["Description"].agg(lambda s: s.mode().iloc[0])
       .reset_index().to_parquet(PRODUCTS_PATH, index=False))

    print(log.to_string(index=False, formatters={c: "{:,}".format for c in log.columns[1:]}))


if __name__ == "__main__":
    main()
