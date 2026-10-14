# ML1 - Method decision and ML decision log

**Author:** Nikhil Chaganti, ML Engineer  
**Written** before any model was fitted on the real matrix

## 1. Decision

**Main method: Truncated SVD.** PCA is run second, on the same matrix and the same k, as the comparison (ML5).

## 2. Why

| Reason | Explanation |
|---|---|
| The matrix is sparse | 98.14% of cells are zero (5,471 x 4,082, 416,440 non-zeros). SVD works on it as it is. |
| A zero means "not bought" | SVD keeps that meaning. PCA subtracts the average customer first, so a zero becomes "bought less than average", which is not what we want to learn from. |
| It gives what the recommender needs | One fit returns customer factors and product factors in the same space. Product-to-product similarity comes straight from the product factors. |
| It is easy to explain | "k components rebuild the purchase data with X% error" is a clear sentence for Sarah. |
| It is the standard choice | Truncated SVD is the usual starting point for latent-factor recommenders on sparse data. |

This is a reasoned starting choice, not a result. ML5 will test it with our own numbers.

## 3. What I expect (to be checked in ML5)

1. Component 1 will mostly measure overall buying volume. I will check this by correlating component 1 with each customer's total purchases.
2. SVD and PCA will give similar hit rates, because the data is the same. If PCA is clearly better, I will say so and explain why.
3. PCA's memory cost may be smaller than the textbook suggests, because scikit-learn centres sparse data implicitly. I will report what I measure, not what I expected.

## 4. Evaluation decisions

| Item | Decision |
|---|---|
| Data | Fit on the training matrix only. The test file is used only to score recommendations (R2). |
| Product similarity | Cosine similarity between product factor vectors, each scaled to length 1. |
| Customer score | For each product, add up its similarity to every product the customer bought. Products already bought are excluded. |
| Baseline (R3) | The 10 products with the most distinct buyers in training, skipping those the customer already bought. |
| Hit rate at 10 | Share of test customers with at least one held-out product in their top 10. |
| Headline denominator | All 1,931 test customers. Customers the model cannot score count as misses, as the Data Engineer's hand-off note says. |
| Second figure | Hit rate on the 1,916 scorable customers only, labelled clearly. |
| Test difficulty | Held-out products are ones the customer never bought before, so this tests discovery, not repeat buying. Both the model and the baseline face the same test, and hit rates will be modest. |
| Seeds | `random_state = 42` everywhere (R6). |

## 5. How k is chosen (set before seeing results)

- k is chosen from the ML2 curve (error against k), balancing what is kept against size and speed.
- Hit rate on the test set is **not** used to choose k, because that would tune the model on the data it is judged on.
- Hit rates for the other k values may be reported for information only.

## 6. Running log (R1)

| Decision | Reason |
|---|---|
| ML1: Truncated SVD as the main method | See section 2 |
| ML2: fitted SVD for k = 10, 20, 50, 100, 200 | The k values from the brief; recorded error, fit time, memory and factor size for each |
| ML2: chose k = 50 | Error falls steadily with no elbow and cost is not a constraint, so a mid-curve point: under 1 s to fit, 3.6 MB of factors |
| ML2: test hit rate not used to choose k | Choosing k on the test set would tune the model on the data it is graded on (R2) |
| ML3: score = sum of cosine similarities to bought products, with product vectors scaled to length 1 | Simple item-based method that works from the compressed factors without a product-by-product matrix |
| ML3: products already bought are never recommended; bought products count equally (1/0) | The same rule is applied to the baseline, so the comparison is fair |
| ML4: reported hit rate at every k for information only; k stays at 50 | Keeps k tied to the ML2 curve, not to the test result |
| ML4: reported the loss to the baseline as it is (20.9% vs 26.8%, lift 0.78) | The gap is about 4 standard errors and holds at every k; no model changes after seeing the test result |
| ML5: ran PCA on the same matrix, with the same k (50) and the same recommender as SVD | Like-for-like comparison: the only difference is centring |
| ML5: PCA uses the arpack solver with implicit centring, so the matrix stays sparse | The matrix is 98% zeros and a dense centred copy would be about 180 MB |
| ML5: PCA error measured against the raw matrix, and PCA’s stored size includes the column means | Makes the error comparable with SVD’s and counts everything PCA has to keep |
| ML5: peak memory read from process memory, so treated as approximate | It is a sampled measure, so small differences are not meaningful |
| ML5: component 1 described from its correlation with total purchases and its top products | Checks the ML1 prediction that it measures buying volume, with evidence |
| ML5: SVD and PCA treated as equal on accuracy and cost | Hit-rate gap of 0.7 points is within the standard error (about 0.9); error, size, time and memory are the same |
| ML5: kept Truncated SVD for the nightly job | Nothing favours PCA, and SVD is simpler and gives both factor sets in one fit; it did not measure better |
| ML5: component 1 described as overall buying volume for both methods | Correlation with total purchases is 0.97 for each |
| ML6: map built from the first two SVD components, with component 1 flipped so higher means a bigger buyer | The sign of a component is arbitrary; this makes the axis readable |
| ML6: each axis standardised before k-means, with the seed fixed at 42 | Stops the volume axis dominating the clusters (R6) |
| ML6: number of segments (3 to 6) chosen by silhouette score | Rule set before looking at the groups |
| ML6: segments described by the products bought much more often than overall, plus top countries | Gives marketing recognisable themes, not just rare products |
| ML6: chose 3 segments | Highest silhouette (0.64); 4 to 6 segments add groups of about 10 to 12 customers, too small to target |
| ML6: named the segments “Light buyers”, “Heavy buyers: home décor and hearts” and “Heavy buyers: party and lunch bags” | Named from segment size, products bought and top countries |
| ML6: added a most-common-products column and one-decimal percentages to the profile table | Segment 'Light Buyers'' is 83% of customers, so it has no distinctive products and showed 0% vs 0%; display change only, clustering unchanged |
| ML6: noted that the map is partly a volume story, and that the light-buyer group is too broad for one campaign | Component 1 is buying volume; the two heavy groups differ by theme instead |
| ML7: compared log, raw count and 1/0 purchase values, each fitted with SVD at the same k and seed | Isolates the effect of the purchase value |
| ML7: same held-out purchases, recommender and baseline for every version | Makes the hit rates directly comparable |
| ML7: main matrix left as delivered (log) | The matrix is frozen on Day 4; ML7 is a sensitivity check and not used to pick the main model |
| ML7: 1/0 gave the best hit rate (24.8%, against 20.9% for log and 12.6% for raw count) | Quantity mostly reflects shop size, so who bought what is a better signal than how much; the 1/0 vs log gap is nearly 3 standard errors |
| ML7: raw counts performed worst (lift 0.47) | A few very large orders dominate the matrix; the low relative error (0.19) reflects that, not recommendation quality |
| ML7: no purchase value clearly beat the best-seller baseline (best: 1/0, 24.8% vs 26.8%) | The gap is about 1.4 standard errors, so a loss cannot be called clearly, but it is not a win |
| ML7: kept log as the main purchase value; 1/0 reported as a finding and a suggested next step | Switching because of the test result would tune on the test set (R2); the matrix is frozen; a validation split is the fair way to test it |
| | |