# 03_grouped_cv_comparison

## Question / hypothesis

How much does the row-holdout MAE overstate Ridge's performance compared to
a proper patient-grouped cross-validation? And does the dummy baseline change
meaningfully under grouped CV?

## Motivation

- **Sourcing strategy:** user + EDA finding (Backlog B2)
- **Source(s):**
  - EDA finding: `patient_id` repeats ~8 times per patient; a random holdout
    leaks identity across train/test, producing an optimistic estimate.
  - User request: "Re-evaluate the dummy and ridge models using patient-grouped
    cross-validation (GroupKFold on patient_id) instead of the default random
    split, and push the comparison report to the hub with a new key."
- **Why this matters:** Row-holdout MAE (dummy 13.48, ridge 8.35) may be
  optimistic because the same patient's visits appear on both sides. GroupKFold
  gives the honest generalisation estimate to unseen patients.

## Method

- **Files touched:** `src/parkinsons/data.py` (add `load_groups()`),
  `src/parkinsons/evaluate.py` (add `grouped_splitter`),
  `experiments/03_grouped_cv_comparison.py`,
  `tests/smoke/test_03_grouped_cv_comparison.py`
- **Change versus 02_ridge:** Same two pipelines (dummy + ridge α=0.1).
  Splitter switched from `0.2` (random row holdout) to
  `GroupKFold(n_splits=5)` with `patient_id` baked in via a thin wrapper so
  skore's `split(X, y)` call works without a `groups` kwarg.
  Both models evaluated together → `ComparisonReport` pushed under key
  `03_grouped_cv_comparison`.
- **Cross-validation:** `GroupKFold(n_splits=5)` on `patient_id`. Zero patient
  overlap confirmed across all 5 folds.
- **Out of scope:** new feature engineering, hyperparameter search.

## Risks / things that could invalidate the result

- GroupKFold assigns patients to folds by sort order (no shuffle), so fold
  sizes depend on patient ordering in the CSV. This is fine for an honest
  estimate but worth noting.
- `on`/`off` are still included per the user's feature list. Ridge's grouped-CV
  MAE may still be inflated if those columns encode partial target information.

## Status

- **State:** done
- **Approved by user on:** 2025-10-12
- **Headline result:** Dummy grouped-CV: MAE 13.49 ± 0.32 | RMSE 16.50 ± 0.34 | R² ≈ 0.00. Ridge grouped-CV: MAE 8.39 ± 0.14 | RMSE 10.45 ± 0.22 | R² 0.598 ± 0.014. Hub: 03_dummy_grouped → https://skore.probabl.ai/databoys/databoys/cross-validations/65479 | 03_ridge_grouped → https://skore.probabl.ai/databoys/databoys/cross-validations/65497
- **Implication for next iteration:** Grouped CV confirms the row-holdout results were already honest — Ridge's MAE barely changes (8.35 → 8.39). The gap vs the dummy holds at ~38 %. Low fold-to-fold std (±0.14 MAE) means results are stable. Next: try a non-linear model (e.g. HistGradientBoosting) under the same grouped CV.
