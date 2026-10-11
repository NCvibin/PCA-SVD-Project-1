"""Load the raw Online Retail II workbook (both sheets) and cache it as parquet.

Reading the 45 MB xlsx takes a few minutes, so every later step reads the cache.
Usage:  python src/load_data.py            (builds cache if missing)
        python src/load_data.py --refresh  (forces a re-read of the xlsx)
"""
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "raw" / "online_retail_II.xlsx"
RAW_CACHE = ROOT / "data" / "interim" / "raw_combined.parquet"


def read_raw_sheets(path: Path = RAW_XLSX) -> dict:
    """Return {sheet_name: DataFrame} exactly as pandas reads them (no dtype forcing)."""
    if not path.exists():
        sys.exit(f"Raw file not found: {path}\nDownload it from the UCI repository into data/raw/.")
    return pd.read_excel(path, sheet_name=None, engine="openpyxl")


def combine(sheets: dict) -> pd.DataFrame:
    """Stack the sheets, keep the source sheet, and give mixed columns stable types."""
    frames = []
    for name, df in sheets.items():
        df = df.copy()
        df["sheet"] = name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    # Invoice and StockCode mix ints with strings ('C489449', '85123A') -> store as str
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    df["Description"] = df["Description"].astype("string").str.strip()
    # Customer ID is read as float because of NaNs -> nullable integer
    df["Customer ID"] = df["Customer ID"].astype("Int64")
    df["Country"] = df["Country"].astype("string")
    return df


def load_raw(refresh: bool = False) -> pd.DataFrame:
    """Combined raw data, from cache when available."""
    if RAW_CACHE.exists() and not refresh:
        return pd.read_parquet(RAW_CACHE)
    df = combine(read_raw_sheets())
    RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RAW_CACHE, index=False)
    return df


if __name__ == "__main__":
    d = load_raw(refresh="--refresh" in sys.argv)
    print(f"{len(d):,} rows cached at {RAW_CACHE}")
