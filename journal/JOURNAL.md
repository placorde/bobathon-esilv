# JOURNAL

## Status

| Item | Value |
|---|---|
| Workspace | Parkinson's Disease UPDRS Score Prediction |
| Tabular library | pandas |
| Last updated | 2025-10-12 |

## Workspace decisions

| Decision | Value | Date |
|---|---|---|
| Tabular library | pandas | 2025-10-12 |
| student prior | comfortable - recorded: 2025-10-12 |
| skore mode | hub - recorded: 2025-10-12 |
| skore hub workspace | databoys - recorded: 2025-10-12 |

---

## Data understanding (EDA)

- **Status:** done - 2025-10-12
- **Summary:** 44 590 train rows × 13 features + 1 target. Regression task predicting a continuous UPDRS-like motor score (mean 37.5, range 0–109.5). Key findings: `patient_id` repeats (~8 visits/patient) → `GroupKFold` required; `on`/`off` are suspiciously strong predictors (r ≈ 0.87 / 0.67) and may be target-leaking sub-scores; up to 79 % missingness in several columns → imputation or native-NaN model needed.
- **Report:** [data/eda.md](../data/eda.md)

---

## History

| # | Stem | Intent | Status | Headline | Note |
|---|---|---|---|---|---|
| 01 | `01_dummy` | Floor performance — DummyRegressor(mean) on 8 features, 20 % row holdout | done | MAE 13.48 \| RMSE 16.48 \| R² ≈ 0.00 | [journal/01_dummy.md](01_dummy.md) |
| 02 | `02_ridge` | First linear model — Ridge(alpha=0.1) on same 8 features, same 20 % row holdout | done | MAE 8.35 \| RMSE 10.40 \| R² 0.60 | [journal/02_ridge.md](02_ridge.md) |
| 03 | `03_grouped_cv_comparison` | Re-evaluate dummy + ridge under GroupKFold(patient_id, 5 folds) | done | Dummy MAE 13.49 ± 0.32 \| Ridge MAE 8.39 ± 0.14 \| Ridge R² 0.60 | [journal/03_grouped_cv_comparison.md](03_grouped_cv_comparison.md) |
| 04 | `04_hgbr_grouped` | HistGradientBoostingRegressor, no imputation, GroupKFold(5) | done | MAE 6.10 ± 0.11 \| RMSE 7.75 ± 0.15 \| R² 0.779 | [journal/04_hgbr_grouped.md](04_hgbr_grouped.md) |
| 05 | `05_fe_outliers_grouped` | FE: years_since_diagnosis, rater_id/cohort as categoricals, clip off>108; HGBR, GroupKFold(5) | done | MAE 5.92 ± 0.11 \| RMSE 7.54 ± 0.15 \| R² 0.791 | [journal/05_fe_outliers_grouped.md](05_fe_outliers_grouped.md) |
| 06 | `06_skrub_tabular_grouped` | skrub.tabular_pipeline("regressor"): TableVectorizer auto-encodes all columns; same GroupKFold(5) | done | MAE 5.93 ± 0.11 \| RMSE 7.53 ± 0.15 \| R² 0.792 | [journal/06_skrub_tabular_grouped.md](06_skrub_tabular_grouped.md) |
| 07 | `07_dataops_grouped` | DataOps graph: 05 FE + GroupKFold(5) baked via mark_as_X(cv=..., split_kwargs={"groups": patient_id}) | done | MAE 5.92 ± 0.11 \| RMSE 7.54 ± 0.15 \| R² 0.791 | [journal/07_dataops_grouped.md](07_dataops_grouped.md) |
| 08 | `08_temporal_grouped` | Temporal features: expanding mean+trend of past off/on + visit count per patient, before 05 FE + HGBR, GroupKFold(5) | done | MAE 13.58 ± 0.33 \| RMSE 16.62 ± 0.37 \| R² –0.016 | [journal/08_temporal_grouped.md](08_temporal_grouped.md) |

---

## Backlog

| # | Item | Source |
|---|---|---|
| ~~B1~~ | ~~Ridge regression on same 8 features, same 20 % holdout~~ | resolved in 02_ridge |
| B2 | Replace row holdout with GroupKFold(patient_id) to get honest CV estimate | EDA finding |
