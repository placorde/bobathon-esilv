# 04_hgbr_grouped

## Question / hypothesis

Does ``HistGradientBoostingRegressor`` — which handles missing values
natively — outperform Ridge (MAE 8.39 ± 0.14) under the same
``GroupKFold(patient_id, n_splits=5)`` evaluation?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request: "evaluate a HistGradientBoostingRegressor with the same
    grouped CV, keeping missing values as-is with no imputation."
  - 03_grouped_cv_comparison implication: low fold std for Ridge (±0.14 MAE)
    → stable baseline to beat; a non-linear tree model is the natural next step.
- **Why this matters:** Ridge is linear and uses mean-imputation which may
  destroy the missingness signal. HGBR is gradient-boosted, handles NaN via
  its native missing-value split, and can model non-linear feature interactions
  (e.g. `age` × `ledd`, `on`/`off` interaction with `time_since_intake_*`).

## Method

- **Files touched:** `src/parkinsons/pipeline.py` (add `build_hgbr_learner`),
  `experiments/04_hgbr_grouped.py`,
  `tests/smoke/test_04_hgbr_grouped.py`
- **Change versus 03_grouped_cv_comparison:** Single estimator
  ``HistGradientBoostingRegressor(random_state=0)``, no ``SimpleImputer``
  wrapping. Same 8 feature columns from `load_dataset()`. Same
  ``GroupKFold(n_splits=5)`` splitter via ``make_grouped_splitter()``.
  Pushed to Hub under key ``04_hgbr_grouped``.
- **Cross-validation:** ``GroupKFold(n_splits=5)`` on ``patient_id`` —
  same as experiment 03.
- **Out of scope:** hyperparameter tuning, additional features, Kaggle
  submission (no new test predictions needed unless requested).

## Risks / things that could invalidate the result

- ``on`` / ``off`` have r ≈ 0.87 / 0.67 with ``target``; HGBR may exploit
  this non-linearly and appear much better than Ridge while still being
  partially driven by that near-leakage signal.
- Default HGBR hyperparameters may underfit (``max_iter=100``) or overfit
  (``max_leaf_nodes=31``) on this 44 k-row dataset — reported result is
  for defaults only.

## Status

- **State:** done
- **Approved by user on:** 2025-10-12
- **Headline result:** MAE 6.10 ± 0.11 | RMSE 7.75 ± 0.15 | R² 0.779 ± 0.005 (GroupKFold 5-fold, patient_id). Hub: https://skore.probabl.ai/databoys/databoys/cross-validations/65546
- **Implication for next iteration:** HGBR cuts MAE by 27 % vs Ridge (6.10 vs 8.39) and R² jumps from 0.60 to 0.78. Native NaN handling clearly outperforms mean-imputation + linear model. Very low fold std (±0.11) — stable result. Next: tune HGBR hyperparameters (max_iter, learning_rate, max_leaf_nodes) or try adding remaining features (cohort, gene, rater_id).
