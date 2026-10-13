# 02_ridge

## Question / hypothesis

Can a Ridge regression on the same 8 features beat the dummy baseline
(MAE 13.48 | RMSE 16.48), using the same 20 % row holdout?

## Motivation

- **Sourcing strategy:** my-pick:01_dummy (Backlog B1)
- **Source(s):**
  - `01_dummy` established MAE 13.48 / RMSE 16.48 as the floor.
    Ridge is the simplest learned linear model; its L2 regularisation
    handles the moderate missingness and collinearity (`age` ↔
    `age_at_diagnosis` r = 0.94) better than OLS.
- **Why this matters:** Establishes whether a linear relationship exists
  between these 8 features and the UPDRS target before trying
  non-linear or grouped-CV approaches.

## Method

- **Files touched:** `src/parkinsons/pipeline.py` (new `build_ridge_learner`
  function), `experiments/02_ridge.py`, `tests/smoke/test_02_ridge.py`
- **Change versus 01_dummy:** Swap `DummyRegressor` for
  `Ridge(alpha=1.0)`. Keep the same `SimpleImputer(strategy="mean")`
  prepro step and the same 8 feature columns. Splitter stays `0.2`
  (20 % row holdout) per lab guide for `02_ridge`.
- **Cross-validation:** `splitter=0.2` — same row-level holdout as
  `01_dummy`. Patient-id grouped CV is deferred to a later experiment
  (Backlog B2).
- **Out of scope for this experiment:** hyperparameter search on `alpha`,
  grouped CV, non-linear features.

## Risks / things that could invalidate the result

- The same `patient_id` appears on both sides of the 20 % holdout,
  so the reported MAE is optimistic vs. a true patient holdout. This
  is intentional; it mirrors `01_dummy` exactly so the two results
  are directly comparable.
- `on` / `off` have r ≈ 0.87 / 0.67 with `target` and may encode
  partial target information. Including them inflates Ridge's apparent
  improvement over the dummy. Noted as a risk; confirmed inclusion
  per user's explicit feature list.

## Status

- **State:** done
- **Approved by user on:** 2025-10-12
- **Headline result:** MAE 8.35 | RMSE 10.40 | R² 0.60 (20 % row holdout, 8 918 test rows). Alpha tuned: 0.1 = 1.0 = 10.0 (identical metrics — regularisation has no effect at these scales). Report on Hub: https://skore.probabl.ai/databoys/databoys/estimators/65415
- **Implication for next iteration:** Ridge beats the dummy by 38 % on MAE (8.35 vs 13.48) and 37 % on RMSE (10.40 vs 16.48). R² = 0.60. Alpha choice is irrelevant here — switch to GroupKFold(patient_id) for an honest CV estimate (Backlog B2).
