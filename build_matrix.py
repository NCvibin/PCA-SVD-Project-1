"""DE6 - Rebuild everything from the raw file with one command (R6).

Usage:  python build_matrix.py
Needs:  data/raw/online_retail_II.xlsx

Runs, in order:
  DE1  src/profile_data.py   raw xlsx -> outputs/de1_data_profile.csv (+ raw cache)
  DE2  src/clean_data.py     -> outputs/de2_cleaning_log.md, data/interim/clean.parquet
  DE3  src/split_data.py     -> outputs/de3_split_log.md, data/processed/train/test.parquet
  DE4  src/make_matrix.py    -> data/processed/matrix.npz + index maps, outputs/de4_decisions.md
  DE5  src/matrix_report.py  -> outputs/de5_matrix_report.md
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import profile_data, clean_data, split_data, make_matrix, matrix_report  # noqa: E402

STEPS = [
    ("DE1 Data profile", profile_data.main),
    ("DE2 Cleaning", clean_data.main),
    ("DE3 Time split", split_data.main),
    ("DE4 Sparse matrix", make_matrix.main),
    ("DE5 Matrix report", matrix_report.main),
]


def main():
    start = time.time()
    for name, step in STEPS:
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
        t = time.time()
        step()
        print(f"[{name} done in {time.time() - t:.1f}s]")
    print(f"\nAll steps finished in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
