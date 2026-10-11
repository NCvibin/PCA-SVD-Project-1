# DE2 - Cleaning log

| Rule | Rows removed | Rows remaining |
|---|---:|---:|
| Raw data | 0 | 1,067,371 |
| Duplicate rows (sheets overlap in Dec 2010) | 22,523 | 1,044,848 |
| Cancellations (Invoice starts with 'C') | 19,165 | 1,025,683 |
| Quantity <= 0 | 3,393 | 1,022,290 |
| Price <= 0 | 2,636 | 1,019,654 |
| Non-product StockCode | 4,709 | 1,014,945 |
| Missing Customer ID | 226,758 | 788,187 |