# EDA findings and workbook answers

> Support note written by Claude Code (non-scoring) on 2026-10-12. Descriptive statistics only (pandas). No model was trained or evaluated and no submission file was written.
> The official **G-EDA** step (`explore-ml-data`, which writes `data/eda.py`, `eda.md` and the HTML pages) still has to be done **in Bob**. Use these numbers to check Bob's output and to decide what to ask it.

## The six things to remember

1. **The target only ever goes up within a patient.** 100% of consecutive visits have `target` ≥ the previous visit, and it is **almost linear in disease duration** (`age - age_at_diagnosis`): the median per-patient correlation is **0.98** and the median slope is **≈2.4 points/year**. This is the "temporal progression" problem from CONTEXT.md and the strongest signal in the data.
2. **Measured `off` is the best feature** (r = 0.87), but it **underestimates** the true target: mean bias **−5.4 points**, std 8.7. Predicting `target ≈ off` would give an RMSE of about 10.2 on the visits that have an `off` value.
3. **Drug timing explains the bias in `on`.** Right after intake (< 0.5 h), `on - target` is about −11; around 2–4 h (peak effect) it is about −24. For `off`, early measurements (6–12 h after intake) are more biased (−8 to −14) than late ones. This is the "drug timing" problem.
4. **Missing values carry information.** `ledd` is missing for 100% of visits in patients diagnosed less than 6 months earlier, against 12% after 10 years. A visit **with no `on`** has a mean target of 21, against 44 with `on`. A plausible clinical reading: not treated yet, so no ON exam.
5. **There is a rater effect (`rater_id`).** The mean `off - target` gap ranges from **−10.5 to +2.2** depending on the rater (std 2.6 across raters). All 50 raters in test also appear in train.
6. **The split is by patient** (no patient appears in both train and test), but **test contains every visit of each test patient** (4 to 12). Using the other visits of the same test patient is therefore legitimate.

## Day 1 workbook, section 2 (EDA)

**Q1. Visits and patients**
| | visits | patients | visits/patient (median / mean / min / max) |
| --- | --- | --- | --- |
| train | 44,590 | 5,576 | 7 / 8.0 / 4 / 12 |
| test | 11,013 | 1,395 | 7 / 7.9 / 4 / 12 |

No patient or `Index` is shared between train and test. Visits are about 1.05 years apart on average (range 0.2 to 5.7 years). Within a patient, `Index` order matches age order.

**Q2. Missing values (train; test is very close)**
- The 3 columns with the most gaps: `time_since_intake_off` **78.8%**, `time_since_intake_on` **46.4%**, `off` **42.4%**. Then `ledd` 36.6%, `gene` 32.4%, `on` 29.6%, `age_at_diagnosis` 5.2%.
- No `off`: **42.4%** of visits. No `on`: **29.6%**.
- `on` **and** `off` both missing: **no visit** (as CONTEXT.md says: "at least one of the two"). Breakdown: `on` only 42.4%, `off` only 29.6%, both 27.9%.
- Worth noting: whenever `on` is missing, `ledd` is **always** missing too (missing `on` ⊂ missing `ledd`).

**Q3. Distribution of `target`**
- Range 0 to 109.5 (theoretical: 0 to 132). Mean 37.5, median 37.3, **std 16.5**. 1% of visits are below 1.8 and 99% are below 73.4.
- **No skew** (skew = 0.05): the distribution is close to symmetric, so no log transform is needed a priori.

**Q4. Columns most associated with `target`** (Pearson correlation, pairwise complete)
| column | r |
| --- | --- |
| `off` | **0.87** |
| `on` | 0.67 |
| disease duration (`age - age_at_diagnosis`) | **0.54** |
| `age` | 0.31 |
| `ledd` | 0.30 |
| `age_at_diagnosis` | 0.13 |
| `sexM`, `time_since_intake_*` | ≈ 0 (linear) |

- `off` is **close but not a copy**: bias −5.4, std 8.7, and only 23% of visits are within 2 points of the target.
- `on` sits far below (−21.8 on average): it is the score **under treatment**. CONTEXT.md quotes a 50–100% improvement from OFF to ON.
- The intake delays are **not linearly correlated** with the target, but they **change the gap** between `on`/`off` and the target (see point 3 above). That is an interaction: a linear model (Ridge) cannot capture it, a tree model (HGBR) can.

**Q5. Is `ledd` missing at random?** No.
- By cohort: A 35.9%, B **42.6%**. Cohort B is also missing `gene` more often (44% vs 31%).
- The deciding factor is **disease duration**: `ledd` is missing for 100% of visits before 0.5 years, 86% between 0.5 and 2 years, 40% between 2 and 5, 18% between 5 and 10, and 12% beyond 10 years.
- Clinical reading: early in the disease (the "honeymoon" phase, CONTEXT.md), many patients **are not on levodopa yet**. No dose means no LEDD and no ON exam. A missing value therefore signals an early stage, hence a lower target (25.6 when `ledd` is missing, 44.3 otherwise). Cohort differences may also come from different data-collection protocols.

**Q6. Columns not to feed the model**
- `Index`: a row identifier. It encodes visit order, which is redundant with `age`, and it is a leakage risk.
- `patient_id`: an identifier. The model would memorise patients, while the test patients are new. Use it to **group** the CV, not as a feature.
- `target`: obviously, and never any column derived from it.
- `rater_id`: **open question**. It is not a leak (every test rater is in train) and it carries a real bias, so it can be used as a categorical feature, to be validated with grouped CV.
- `cohort`, `gene`, `sexM`: legitimate features. `sexM` and `gene` have little direct effect (mean target between 34 and 39 across `gene` values).

## Day 1, section 3 (dummy): answers to check on the Hub

1. The dummy RMSE should be **about 16.5**, the std of `target`. Predicting the mean for everyone gives a mean squared error equal to the variance (up to the 80/20 split).
2. Predicted-vs-actual plot: a **horizontal line** (the same prediction for everyone) against a wide spread along the x-axis. A perfect model would put every point on the **y = x diagonal**.
3. An improvement is real when it is larger than the **between-fold** spread of RMSE in grouped CV (Day 2). On a single random split, a few tenths of a point are noise.

## Day 2 conceptual answers (no code run)

- **Unscaled coefficients (Ridge, Q3)**: a coefficient is in "points per unit of the column". `ledd` is in hundreds of mg and `age` in tens of years, so raw sizes cannot be compared. Standardise first (StandardScaler) or use permutation importance.
- **Why `alpha` barely matters**: with about 8 features and about 44,000 rows, regularisation weighs very little against the amount of data. The coefficients hardly move.
- **Expected signs**: `off` and `on` positive (more measured symptoms, more true severity). The `off` coefficient should be clearly larger, with the intercept absorbing part of the −5 bias.
- **Random vs grouped CV**: a random split puts visits from the same patient on both sides. Since the target is almost a straight line per patient, the model "sees" the neighbouring visits and the score is too optimistic. Flexible models (HGBR) should lose the most when moving to grouped CV.

## Ideas to hand to Bob (ideas, not code)

Ranked by expected impact based on the findings above. Each one must be implemented and evaluated **in Bob + skore**, with `GroupKFold` by patient.

1. **Per-patient aggregate features** (temporal progression): test gives every visit of each patient, and the target is almost linear in disease duration. Aggregates over the same patient's visits (mean or median of `off`, of `on`, slope of `off` against duration, number of visits) should help a lot. Watch for CV leakage: compute these aggregates **without the target** and per patient. With GroupKFold a patient is entirely on one side of the split, so this is safe.
2. **Disease duration** `age - age_at_diagnosis` as an explicit feature (r = 0.54). Trees rebuild it poorly from two separate columns.
3. **Missing-value indicators** (`off_missing`, `on_missing`, `ledd_missing`), or HGBR, which handles NaN natively (Day 2 §4).
4. **Interactions with the delays** (`on` × `time_since_intake_on`, `off` × `time_since_intake_off`): HGBR learns them on its own, Ridge does not.
5. **`rater_id` as a categorical** to correct each rater's bias.
6. **Temporal smoothing**: the target is monotonic per patient. Smoothing predictions per patient (isotonic or linear regression on duration) could lower the RMSE. This is a post-processing step, also to be done in Bob.

## Doc inconsistencies found

- DAY1.md says "test may already have `time_since_diagnosis`": **not true for these CSVs**. Train and test have exactly the same 13 columns and no `time_since_diagnosis`. Duration has to be computed on both sides.
- `age_at_diagnosis` is missing for about 5% of patients (missing for the whole patient), so disease duration is missing for them.
- README.md: the "Submission rules" paragraph is duplicated, with two almost identical versions one after the other (harmless).
