"""DE1 - Data profile table of the raw Online Retail II data (both sheets).

One row per column: type, example value, missing values per sheet and in total,
and distinct values. Row counts per sheet are in the column headers.

Usage:  python src/profile_data.py
Writes: outputs/de1_data_profile.csv
        data/interim/raw_combined.parquet (cache used by clean_data.py)
"""
from pathlib import Path
import pandas as pd

from load_data import RAW_CACHE, combine, read_raw_sheets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

# How each column is stored after loading (see load_data.combine)
STORED_AS = {"Invoice": "text", "StockCode": "text", "Description": "text",
             "Quantity": "integer", "InvoiceDate": "date-time", "Price": "decimal",
             "Customer ID": "integer", "Country": "text"}


def type_as_read(s: pd.Series) -> str:
    """Plain-English type pandas gave the column, flagging mixed columns."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date-time"
    if pd.api.types.is_integer_dtype(s):
        return "integer"
    if pd.api.types.is_float_dtype(s):
        return "decimal"
    kinds = {type(v).__name__ for v in s.dropna()}
    return "mixed (numbers + text)" if kinds - {"str"} else "text"


def example(s: pd.Series) -> str:
    """First non-missing value, printed the way a person would write it."""
    v = s.dropna().iloc[0]
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)[:30]


def main():
    OUT.mkdir(exist_ok=True)
    sheets = read_raw_sheets()
    raw = pd.concat(sheets.values(), ignore_index=True)
    total = len(raw)

    rows = []
    for c in raw.columns:
        miss = {n: int(d[c].isna().sum()) for n, d in sheets.items()}
        rows.append({
            "Column": c,
            "Type as read": type_as_read(raw[c]),
            "Stored as": STORED_AS.get(c, str(raw[c].dtype)),
            "Example": example(raw[c]),
            **{f"Missing in {n.replace('Year ', '')} ({len(sheets[n]):,} rows)": v for n, v in miss.items()},
            f"Missing total ({total:,} rows)": sum(miss.values()),
            "Missing %": f"{sum(miss.values()) / total:.1%}",
            "Distinct values": raw[c].nunique(),
        })
    profile = pd.DataFrame(rows)
    profile.to_csv(OUT / "de1_data_profile.csv", index=False)

    # ---- Cache for the next steps ------------------------------------------
    RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
    combine(sheets).to_parquet(RAW_CACHE, index=False)

    with pd.option_context("display.width", 250, "display.max_columns", None):
        print(profile.to_string(index=False))
    print("\nSaved outputs/de1_data_profile.csv")


if __name__ == "__main__":
    main()
