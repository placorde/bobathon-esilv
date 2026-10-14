# 12_inverse_blend_grouped

## Question / hypothesis

Can reproducing the original `42_rebalanced_inverse` model from the other
team member's project produce a competitive result when evaluated on our
team's standard GroupKFold(5) setup?

## Pre-flight checks

### Data identity
Both projects use bit-identical CSV files.  SHA-256 verified:
`c33dc1bb72d71183...` (full hash in `data/day2/ridge_input_sha256.txt`).

### Assignment rule on complete patient predictor histories
`PatientFeatures`, `InverseMeasurements`, and the curve models all use
**predictor-only** patient pooling — they pass **all available visits of an
unseen patient as one batch** so the model can see the patient's own history
at prediction time. This is explicitly the intended usage: the CONTEXT.md
task description is to predict the true OFF for each visit, and the
`PatientCurveRegressor` docstring confirms "Prediction requires all available
visits for each test patient in one batch." The assignment permits this.

### Partition
The original model used a 80/20 patient split:
- Development: 35 765 rows, 4 460 patients (used for model selection across 19 rounds)
- Holdout: 8 825 rows, 1 116 patients (reserved, never used for selection)

Our team validation uses the full 44 590 rows / 5 576 patients with a fresh
GroupKFold(5) that was never used for selection.

### Dependency versions
Identical across both projects:
skore 0.26.0, sklearn 1.9.1, skrub 0.11.0, numpy 2.5.4, pandas 3.0.6.

## Architecture

The blend `42_rebalanced_inverse` is:

```
prediction = (2 × p26 + p38) / 3
```

where the exact decomposition is:

```
p26 = FlexiblePatientCurveRegressor + OrthogonalPatientCurve  (equal blend)
p38 = InversePatientCurve
```

Each component is a `PatientCurveRegressor` variant:
1. Fits a calibration/inversion step on training-fold rows + targets
2. Fits three HGBR models for quadratic age-trajectory coefficients
   per patient (one row per patient, count-weighted)
3. At prediction time: reconstructs per-visit score from the patient's
   inferred trajectory, using only predictor visits — no future targets

`InversePatientCurve` uses an inner 3-fold `GroupKFold` for cross-fitted
ON/OFF measurement inversion — fitted entirely within each outer training fold.

Key HGBR params per component (from source):
- `FlexiblePatientCurveRegressor`: `max_iter=500, max_leaf_nodes=15, lr=0.04, l2=15`
- `OrthogonalPatientCurve.fit_with_calibration`: `max_iter=500, max_leaf_nodes=15, lr=0.04, l2=15`
- `InversePatientCurve` inner: `max_iter=220, max_leaf_nodes=15, lr=0.05, l2=30`
- `InversePatientCurve` outer: `max_iter=400, max_leaf_nodes=15, lr=0.04, l2=20`

## Method

- **Files touched:**
  - `scripts/calibrated_features.py` — read-only copy of CalibratedFeatures
  - `scripts/patient_features.py` — read-only copy of PatientFeatures
  - `scripts/response_features.py` — read-only copy of ResponseFeatures
  - `scripts/residual_trajectory.py` — read-only copy of FittedBlend
  - `scripts/flexible_response.py` — read-only copy of FlexibleResponseFeatures/FlexiblePatientCurveRegressor
  - `scripts/patient_curve.py` — read-only copy of PatientCurveRegressor
  - `scripts/orthogonal_patient_curve.py` — read-only copy of OrthogonalPatientCurve
  - `scripts/inverse_patient_curve.py` — read-only copy of InversePatientCurve
  - `experiments/12_inverse_blend_grouped.py` — reproduction script
  - `journal/12_inverse_blend_grouped.md` (this file)

- **Phase 1 (cached verification):** Read stored fold predictions from
  `C:\Users\boukr\bobathon-esilv\data\day2\improvement_14`, apply
  `(2×baseline + candidate)/3`, confirm RMSE/MAE match exactly.
  **These are reproduced cached results, not independent evaluations.**

- **Phase 2 (team validation):** Retrain all three component models from
  scratch on each of the five GroupKFold folds of the full 44 590-row
  dataset, with fold-local preprocessing and calibration. Evaluate with
  `skore.evaluate` and push to hub.

- **Leakage analysis:**
  - All calibration and target-dependent features are fitted on training-fold
    rows only. Transform receives no targets.
  - The inner 3-fold GroupKFold for `InversePatientCurve` is nested entirely
    within each outer training fold (training rows only).
  - The patient predictor pooling uses all visits of each patient as a batch
    (permitted by the assignment; necessary for the patient-curve approach).

## Results

### Phase 1 — Reproduced cached metrics (development partition)

| Fold | RMSE | MAE |
|------|------|-----|
| 1 | 3.6507 | 2.7008 |
| 2 | 3.4690 | 2.5615 |
| 3 | 3.3804 | 2.5438 |
| 4 | 3.7135 | 2.6908 |
| 5 | 3.3163 | 2.4906 |
| **Mean** | **3.505978** | **2.597514** |

Exact match with reported values (< 1e-9 tolerance). Confirmed by
`assert` in the experiment script.

**Caveat:** These folds are from a partition that was reused for model
selection across 19 improvement rounds. They are not an independent final
test. Do not use these numbers to claim the model beats experiment 11.

### Phase 2 — Team validation (freshly evaluated, comparable to exps 08c–11)

| Fold | RMSE | MAE |
|------|------|-----|
| 1 | 3.4722 | 2.5749 |
| 2 | 3.4671 | 2.5547 |
| 3 | 3.4266 | 2.5460 |
| 4 | 3.3607 | 2.4781 |
| 5 | 3.4710 | 2.5758 |
| **Mean** | **3.4395** | **2.5459** |

skore.evaluate (canonical team report):
- MAE **2.544 ± 0.046**
- RMSE **3.439 ± 0.059**
- R² **0.956 ± 0.002**

Report URL: https://skore.probabl.ai/databoys/databoys/cross-validations/69188

### Comparison table

| Experiment | MAE (team GroupKFold-5) | RMSE |
|---|---|---|
| 07_dataops_grouped | 5.92 | — |
| 08c_temporal_fixed | 3.717 | 4.88 |
| 10_recency_temporal | 3.650 | 4.80 |
| 11_hgbr_tuned | 3.617 | 4.770 |
| **12_inverse_blend** | **2.544** | **3.439** |

**12 beats 11 by 29.7% MAE** on the team's validation setup.

### Why the improvement is so large

The patient-curve approach uses fundamentally different information than
experiments 08c–11:
- It fits a quadratic age trajectory **per patient** rather than using
  per-visit features
- The `InversePatientCurve` learns to invert the nonlinear measurement
  equation (ON/OFF → true OFF) using training-fold targets
- All of this operates on complete patient batches, giving the model
  the full longitudinal context that makes Parkinson's prediction tractable

The feature set in exps 08c–11 only uses expanding means/trends of raw
ON/OFF — the patient-curve approach instead uses the calibrated trajectory
as the primary signal, explaining the large performance gap.

## Status

- **State:** complete
- **Approved by user on:** 2025-10-13
- **Headline result:** MAE 2.544 ± 0.046, RMSE 3.439 (team GroupKFold-5)
- **Current best model:** 12_inverse_blend_grouped (MAE 2.544)
- **Implication for next iteration:**
  - The patient-curve architecture is clearly the right direction.
  - Next steps: tune the component HGBR hyperparameters, explore
    deeper inversion (more inner folds), or add the recency temporal
    features from exp 10 as additional inputs to the curve summarization.
