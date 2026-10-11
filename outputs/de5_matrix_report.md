# DE5 - Matrix report

| Metric | Value |
|---|---:|
| Shape (customers x products) | 5,471 x 4,082 |
| Non-zero cells | 416,440 |
| Sparsity | 98.14% |
| Memory, dense (float64) | 178.7 MB |
| Memory, sparse (CSR) | 5.0 MB |
| Sparse is smaller by | 36x |

## Note for the ML Engineer

- Load: `A = scipy.sparse.load_npz('data/processed/matrix.npz')`; row i = `customer_index.csv`, column j = `product_index.csv` (read with `dtype={'StockCode': str}`).
- Cell = log(1 + total quantity) over the training period; products bought by >= 5 customers. Training rows only, no test data inside.
- Test file: 49,647 held-out pairs; 92.2% have both customer and product in the matrix. The rest can never be recommended and count as misses for the model and the baseline.
- Keep A sparse (never `.toarray()` the full matrix). PCA on sparse input needs scikit-learn >= 1.4 with `svd_solver='arpack'`.