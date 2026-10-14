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
| Customer score | For each product, add up its similarity to every product the customer bought. Products already bought are excluded. The similarity (cosine or dot product of the factor vectors) is chosen on validation (section 5). |
| Baseline (R3) | The 10 products with the most distinct buyers in training, skipping those the customer already bought. |
| Hit rate at 10 | Share of test customers with at least one held-out product in their top 10. |
| Headline denominator | All 1,931 test customers. Customers the model cannot score count as misses, as the Data Engineer's hand-off note says. |
| Second figure | Hit rate on the 1,916 scorable customers only, labelled clearly. |
| Test difficulty | Held-out products are ones the customer never bought before, so this tests discovery, not repeat buying. Both the model and the baseline face the same test, and hit rates will be modest. |
| Seeds | `random_state = 42` everywhere (R6). |

## 5. How k is chosen

- **Revised (see the running log):** k and the customer-scoring rule are chosen on the hit rate of a **validation split**
  carved out of the training data: the last 2 months of `train.parquet` are held out with the Data Engineer's rules
  (DE3, DE4) and the model is refit on the months before. The test file is never read for this.
- The ML2 error curve is still reported; it shows what is lost, but it has no elbow, so it cannot pick k on its own.
- Hit rate on the test set is **not** used to choose k, because that would tune the model on the data it is judged on.
- Hit rates for the other k values may be reported for information only.

## 6. Running log (R1)

| Decision | Reason |
|---|---|
| ML1: Truncated SVD as the main method | See section 2 |
| ML2: fitted SVD for k = 10, 20, 50, 100, 200 | The k values from the brief; recorded error, fit time, memory and factor size for each |
| ML2: chose k = 100 (first version: k = 50 from the curve) | The error curve has no elbow, so k = 50 was a judgement call. Revised to the validation hit rate, which peaks at k = 100 (40.7%) and falls at k = 200 (37.9%); 7.3 MB of factors, under 1 s to fit |
| ML2: test hit rate not used to choose k or the scoring rule | Both are chosen on the validation split only; choosing on the test set would tune the model on the data it is graded on (R2) |
| ML2/ML3: customer scoring by dot product of the factor vectors (first version: cosine) | Cosine scores 13% to 21% on validation at every k, below best sellers (25.9%), because length-1 vectors lose how widely each product sells; the dot product keeps it (40.7% at k = 100). This tested the reason given in the first ML4 write-up |
| ML3: product-page box = 10 most similar products by cosine (vectors scaled to length 1); customer score = sum of dot-product similarities to bought products | Cosine as the brief asks for product-to-product similarity; dot product for customers, chosen on validation (above) |
| ML3: products already bought are never recommended; bought products count equally (1/0) | The same rule is applied to the baseline, so the comparison is fair |
| ML4: reported hit rate at every k for information only; k stays at 100 | Keeps k tied to the validation choice, not to the test result (the test peak at k = 200 is within one standard error) |
| ML4: model beats the baseline (38.9% vs 26.8%, lift 1.45); cosine scoring reported alongside (22.0%) | The first version (k = 50, cosine) lost, 20.9% vs 26.8%; the validation-chosen version wins by 12 points, and test (38.9%) is close to validation (40.7%) |
| ML5: ran PCA on the same matrix, with the same k (100) and the same recommender as SVD | Like-for-like comparison: the only difference is centring |
| ML5: PCA uses the arpack solver with implicit centring, so the matrix stays sparse | The matrix is 98% zeros and a dense centred copy would be about 180 MB |
| ML5: PCA error measured against the raw matrix, and PCA’s stored size includes the column means | Makes the error comparable with SVD’s and counts everything PCA has to keep |
| ML5: peak memory read from process memory, so treated as approximate | It is a sampled measure, so small differences are not meaningful |
| ML5: component 1 described from its correlation with total purchases and its top products | Checks the ML1 prediction that it measures buying volume, with evidence |
| ML5: SVD and PCA treated as equal on accuracy and cost | Hit-rate gap of 0.4 points (39.3% vs 38.9%) is within the standard error (about 1.1); error, size, time and memory are the same |
| ML5: kept Truncated SVD for the nightly job | Nothing favours PCA, and SVD is simpler and gives both factor sets in one fit; it did not measure better |
| ML5: component 1 described as overall buying volume for both methods | Correlation with total purchases is 0.97 for each |
| ML6: map built from the customer factors scaled to length 1, projected on their 2 main directions (first version: raw components 1 and 2) | Component 1 is buying volume, so the raw map sorted shops by size and put 83% in one group with no theme; scaling to length 1 removes size and keeps the product mix |
| ML6: each axis standardised before k-means, with the seed fixed at 42 | Stops the volume axis dominating the clusters (R6) |
| ML6: number of segments (3 to 6) chosen by silhouette score | Rule set before looking at the groups |
| ML6: segments described by the products bought much more often than overall, plus top countries | Gives marketing recognisable themes, not just rare products |
| ML6: chose 3 product groups, each split into UK and outside UK = 6 segments | Highest silhouette at 3 product groups (0.39; 0.33 to 0.35 for 4 to 6), same rule as before. The country split is a rule on each customer's main country, not a cluster, because marketing plans campaigns by country too |
| ML6: named the 6 segments (e.g. “Bags & baking - UK”, “Bags & baking - outside UK”, “Light buyers: kids' accessories - outside UK”) | Named from each segment's own over-represented products and main countries; the outside-UK halves differ (cake cases and charlotte bags; snack boxes and plasters in tins) and are mainly Germany and France |
| ML6: added a most-common-products column and one-decimal percentages to the profile table | Display change only, clustering unchanged |
| ML6: noted the limits: “Hearts & home decor - outside UK” has only 44 customers, the UK light buyers (44%) have the weakest theme, and the 2 map axes keep only 8% of the spread | Honest limits of a 2-D picture and of a small segment |
| ML6: figure keeps the original chart format (one colour per segment, same size, title and legend), drawn on the product-mix axes the groups were built from | On the original volume axes the 6 groups overlapped, because they are cut by product mix, not size |
| ML7: compared log, raw count and 1/0 purchase values, each fitted with SVD at the same k, seed and scoring rule | Isolates the effect of the purchase value |
| ML7: same held-out purchases, recommender and baseline for every version | Makes the hit rates directly comparable |
| ML7: main matrix left as delivered (log) | The matrix is frozen on Day 4; ML7 is a sensitivity check and not used to pick the main model |
| ML7: 1/0 and log are close (39.9% vs 38.9%, about one standard error); raw count is worst (27.9%) | Quantity mostly reflects shop size, so who bought what carries most of the signal |
| ML7: raw counts only match the baseline (lift 1.04) | A few very large orders dominate the matrix; the low relative error (0.12) reflects that, not recommendation quality |
| ML7: log and 1/0 both beat the best-seller baseline (lift 1.45 and 1.49) | Both well above 26.8% |
| ML7: kept log as the main purchase value; 1/0 reported as a finding | The gap is within noise, the matrix is frozen, and switching on the test result would tune on the test set (R2); compare the two on the validation split next |
| | |