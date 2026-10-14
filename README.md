# Customers Like You: PCA or SVD

GiftLane mini-project: product recommendations and customer segments from the UCI Online Retail II data.

## Team
| Role | Member | Tasks |
|---|---|---|
| Data Engineer | Trung Pham | DE1–DE6 |
| ML Engineer | Nikhil Chaganti | ML1–ML7 |
| Both | Trung Pham, Nikhil Chaganti | J1–J3 |

## Success targets (J1)
| Target | Measure | Result |
|---|---|---|
| Compressed data much smaller than the full matrix | Stored factors vs dense matrix | 7.3 MB vs 178.7 MB |
| Nightly job is fast | Fit + "also bought" table + scoring every customer | ~1.1 s |
| Recommendations beat best sellers on unseen purchases | Hit rate at 10 vs best-seller baseline | 38.9% vs 26.8% |
| Customer map with groups marketing can name | Named segments, no customer IDs | 6 segments (3 product groups × UK / outside UK) |

## Data
UCI Online Retail II: https://archive.ics.uci.edu/dataset/502/online+retail+ii (CC BY 4.0)

| Sheet | Rows |
|---|---:|
| Year 2009-2010 | 525,461 |
| Year 2010-2011 | 541,910 |
| **Total** | **1,067,371** |

## How to run
Put the file at `data/raw/online_retail_II.xlsx`. The `data/` folder is not committed.
```bash
pip install -r requirements.txt
python build_matrix.py
jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb
```
1. `build_matrix.py` rebuilds the cleaned data, the train/test split, and the sparse matrix from the raw file (DE1–DE5).
2. The notebook runs ML2–ML7. You can also open it and choose Run All.

All random seeds are fixed at 42.

## Where things are
| Path | Contents |
|---|---|
| `src/` | Data pipeline and ML helpers |
| `notebooks/analysis.ipynb` | ML analysis with results |
| `outputs/de*` | Data profile, cleaning log, split log, matrix decisions and report |
| `outputs/ml_decisions.md` | ML method choice and decision log |
