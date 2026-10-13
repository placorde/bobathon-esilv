# 01_dummy

## Question / hypothesis

What is the floor performance of a mean-predicting dummy regressor on the
Parkinson's UPDRS data, using features `sexM`, `age_at_diagnosis`, `age`,
`ledd`, `time_since_intake_on`, `time_since_intake_off`, `on`, and `off`
with mean-imputation for missing values?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request: "Run a dummy regressor from sklearn to act as my baseline
    model, using feature columns sexM, age_at_diagnosis, age, ledd,
    time_since_intake_on, time_since_intake_off, on, off."
- **Why this matters:** Every meaningful model must beat the dummy. This
  establishes the floor MAE/RMSE for the UPDRS regression task and records
  a reproducible report on Skore Hub so later experiments have a reference
  point.

## Method

- **Files touched:** `src/parkinsons/data.py`, `src/parkinsons/pipeline.py`,
  `src/parkinsons/evaluate.py`, `src/parkinsons/hub.py`,
  `experiments/01_dummy.py`
- **Change versus baseline:** First experiment — no prior. A `Pipeline` of
  `SimpleImputer(strategy="mean")` + `DummyRegressor(strategy="mean")` is fit
  on the eight specified numeric/float features selected from `X_train`.
  Target is the `target` column from `y_train`. `X_test` predictions are
  written to `submissions/01_dummy.csv`.
- **Cross-validation:** `splitter=0.2` (random row-level 20 % holdout per
  lab guide for `01_dummy`). No `GroupKFold` yet — the holdout intentionally
  leaks `patient_id` across train/test to mirror the guide's "row holdout"
  baseline; the grouped-CV experiment comes later.
- **Out of scope for this experiment:** grouped CV, any learnable feature
  transformation, hyperparameter tuning.

## Risks / things that could invalidate the result

- The 20 % row holdout lets the same `patient_id` appear on both sides,
  making the dummy's RMSE slightly optimistic vs. a true patient holdout.
  This is intentional for the guide's sequence; note it in the outcome.
- `on` and `off` are included per the user's explicit request; they may
  encode partial target information (EDA: r ≈ 0.87/0.67). The dummy ignores
  features entirely, so there is no leakage risk here — but the same feature
  set in a learned model would require careful scrutiny.

## Status

- **State:** done
- **Approved by user on:** 2025-10-12
- **Headline result:** MAE 13.48 | RMSE 16.48 | R² ≈ 0.00 (20 % row holdout, 8 918 test rows). Report on Hub: https://skore.probabl.ai/databoys/databoys/estimators/63811
- **Implication for next iteration:** Any real model must beat MAE < 13.48. The dummy ignores all features and predicts the training mean (~37.5). The row holdout leaks patient_id across train/test — grouped CV will give a more honest estimate. Next step: Ridge regression on the same features with the same holdout (02_ridge).
