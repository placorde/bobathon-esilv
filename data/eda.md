# EDA: Parkinson's Disease UPDRS Score Prediction

Interactive table report: [data/eda_train.html](eda_train.html)

---

## Dataset at a glance

| Property | Value |
|---|---|
| Train rows × columns | 44 590 × 14 (X_train merged with y_train) |
| Test rows | 11 013 |
| Task | **Regression** — predict continuous `target` (UPDRS-like motor score) |
| Target range | 0 – 109.5 |
| Target mean / std | 37.5 ± 16.5 |
| Target median | 37.3 |

---

## Per-column findings

| Column | dtype | Null % | n_unique | Notes |
|---|---|---|---|---|
| `Index` | Int64 | 0 % | 44 590 | Row ID — unique per row, drop before modelling |
| `patient_id` | string | 0 % | 5 576 | 5 576 unique patients across 44 590 rows → ~8 rows/patient |
| `cohort` | string | 0 % | 2 | Two study cohorts (A / B) |
| `sexM` | Int64 | 0 % | 2 | Binary sex flag |
| `gene` | string | **32 %** | 4 | Gene mutation label — substantial missingness |
| `age_at_diagnosis` | Float64 | **5 %** | 563 | Age when PD was diagnosed |
| `age` | Float64 | 0 % | 1 082 | Current age at visit |
| `ledd` | Float64 | **37 %** | 1 320 | Levodopa equivalent daily dose — heavy missingness |
| `time_since_intake_on` | Float64 | **46 %** | 64 | Hours since last levodopa (ON state) — heavy missingness |
| `time_since_intake_off` | Float64 | **79 %** | 178 | Hours since last levodopa (OFF state) — very heavy missingness |
| `rater_id` | string | 0 % | 50 | 50 distinct clinical raters |
| `on` | Float64 | **30 %** | 83 | UPDRS score in ON state — partial observation |
| `off` | Float64 | **42 %** | 101 | UPDRS score in OFF state — partial observation |
| `target` | Float64 | 0 % | 867 | **Target** — no nulls |

**Key missingness pattern:** `off` (42 %), `on` (30 %), `time_since_intake_off` (79 %), `time_since_intake_on` (46 %), `ledd` (37 %), `gene` (32 %). These are not random — a patient may lack an OFF-state measurement altogether if the visit was only ON-state. The pipeline must handle these via imputation or model-native missing-value support rather than row-dropping.

---

## Target

`target` is a continuous UPDRS-like motor score:

- Range: 0 – 109.5
- Mean: 37.5 | Median: 37.3 | IQR: 25.6 – 49.3
- Std: 16.5
- Distribution: roughly bell-shaped, centred near 37, with a right tail extending to ~110. Histogram bin counts confirm the bulk of observations lie between 20–55.

**Implication:** pure regression task. MAE and RMSE are natural metrics. No stratification needed for cross-validation. A right-skewed tail suggests checking whether log-transforming the target helps a linear baseline.

---

## Structure signals

**Datetime columns:** none detected. No explicit timestamp feature is present.

**High-cardinality / ID-like columns (top unique ratios):**

| Column | Unique ratio |
|---|---|
| `Index` | 1.00 (row ID — must drop) |
| `patient_id` | 0.125 (~8 visits per patient) |
| `ledd` | 0.030 |
| `age` | 0.024 |

**Key finding:** `patient_id` repeats across rows — the same patient contributes multiple measurements. A standard `KFold` will leak the patient across train and validation folds, leading to optimistic estimates. **`GroupKFold` or `GroupShuffleSplit` with `groups=patient_id` is required.**

`rater_id` has 50 unique values — rater effects may introduce systematic offsets; a rater feature or leave-one-group-out analysis could be informative later.

---

## Associations

### Feature → target

| Feature | Pearson r | Cramér V |
|---|---|---|
| `off` | **0.87** | 0.41 |
| `on` | **0.67** | 0.37 |
| `ledd` | 0.30 | 0.24 |
| `age` | 0.31 | 0.11 |
| `time_since_intake_on` | ~0.00 | 0.18 |
| `time_since_intake_off` | 0.01 | 0.09 |
| `cohort` | n/a | 0.09 |
| `gene` | n/a | 0.05 |
| `age_at_diagnosis` | 0.13 | 0.05 |
| `rater_id` | n/a | 0.02 |
| `sexM` | ~0.00 | 0.02 |

**⚠️ Leakage flag:** `off` (Pearson r = 0.87) and `on` (r = 0.67) are extremely strong predictors of `target`. These are UPDRS sub-scores measured in the same clinical visit that produced `target`. If `target` is a composite that includes `on`/`off`, using them as features constitutes **target leakage**. This must be confirmed with the problem statement before including them in the model. A safe first baseline should be built *without* `on` and `off`.

### Strongest inter-feature associations

| Pair | Pearson r | Note |
|---|---|---|
| `age_at_diagnosis` ↔ `age` | 0.94 | Near-redundant — high collinearity; keep `age` or derive `years_since_diagnosis` |
| `on` ↔ `off` | 0.87 | Both sub-scores aligned |
| `ledd` ↔ `on` | 0.20 | Mild |

---

## Modelling implications

1. **Splitter → `GroupKFold` (groups = `patient_id`).** Patients repeat across rows; standard CV leaks identity. This is the single most important structural finding.
2. **Metric → MAE or RMSE.** Continuous target, roughly symmetric. Start with MAE (more interpretable in UPDRS units).
3. **Missingness → imputation required.** Up to 79 % missing on `time_since_intake_off`; several columns above 30 %. Use a simple imputer (median for numerics, mode for categoricals) or a tree model that handles NaN natively (e.g. `HistGradientBoostingRegressor`).
4. **`Index` → must drop.** Unique-per-row identifier with no predictive value.
5. **`age_at_diagnosis` + `age` collinearity.** Consider replacing with `age - age_at_diagnosis` (years since diagnosis) to reduce redundancy.
6. **`on` / `off` leakage risk.** Defer their inclusion to a second experiment, after the problem statement is clarified. First baseline: exclude them.
7. **`rater_id` as a categorical feature.** 50 categories; encode with `OrdinalEncoder` or `TargetEncoder` for tree models.
8. **`gene` (32 % missing, 4 categories).** Use `SimpleImputer(strategy="most_frequent")` or let a tree handle NaN.

---

## Open questions

- **Is `target` a composite of `on` and / or `off`?** If so, including them in features is leakage. Confirm with the competition data description.
- **What is `cohort` A vs B?** Could represent different study arms with different measurement protocols, which would affect generalization.
- **Are the test rows from the same patients as train?** If yes, `GroupKFold` covers it; if no, the splitter is moot for generalization but still needed for unbiased CV.
- **Is there a temporal ordering?** No datetime column found, but `Index` increases monotonically — could proxy visit order within a patient.
