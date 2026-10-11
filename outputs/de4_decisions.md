# DE4 - Matrix decisions

| Choice | Decision | Why |
|---|---|---|
| Cell value | log(1 + total quantity) | Most buyers are wholesalers, so raw quantity mostly measures shop size; the log keeps "bought more" as a signal without letting big buyers dominate. |
| Product filter | Bought by >= 5 customers | Products with fewer buyers are too rare to learn from: 447 of 4,529 products (10%) but only 0.2% of purchase lines. |

Result: 5,471 customers x 4,082 products, built from training rows only.