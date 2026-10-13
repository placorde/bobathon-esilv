# 05_fe_outliers_grouped

## Question / hypothesis

Does replacing collinear `age`/`age_at_diagnosis` with
`years_since_diagnosis`, adding `rater_id` as a native categorical
feature, and clipping the one plausible data-entry error (`off > 108`)
improve on HGBR's grouped-CV MAE 6.10 ± 0.11?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request: feature engineering + outlier handling on top of
    `04_hgbr_grouped`.
  - EDA: `age` ↔ `age_at_diagnosis` Pearson r = 0.94 — near-redundant.
  - EDA: `rater_id` has 50 categories; `off`-target bias range
    [-8.7, +2.9] across raters (13 raters with |bias| > 5).
  - Outlier investigation (scratch/2025-10-12_outlier_investigation.py):
    `off` max = 111, exceeds clinical UPDRS scale max (108) by 3 points
    → 1 plausible data-entry error.
- **Why this matters:** Collinear features add noise without information;
  rater bias is a systematic offset HGBR can exploit as a categorical;
  the `off > 108` clip removes one physically impossible value.

## Outlier decision (per column, from investigation)

| Column | IQR outliers above fence | p99 | Max | Decision |
|---|---|---|---|---|
| `ledd` | 524 | 1280 | 1796 | **No clip** — medically plausible (advanced PD) |
| `on` | 506 | 54 | 99 | **No clip** — within UPDRS scale (max 108) |
| `off` | 138 | 71 | 111 | **Clip at 108** — 1 row exceeds clinical max |
| `time_since_intake_on` | 701 | 5.4 | 6.3 | **No clip** — long tail, not data error |
| `time_since_intake_off` | 295 | 23.1 | 24.7 | **No clip** — long tail, not data error |

All "long tails" are left untouched — tree models are robust; clipping
them would discard real clinical variation.

## Method

- **Files touched:** `src/parkinsons/features.py` (new
  `engineer_features` transformer), `src/parkinsons/data.py` (update
  `FEATURE_COLS` and `load_dataset`/`load_test_dataset` to include
  `rater_id` + `cohort`), `src/parkinsons/pipeline.py` (new
  `build_fe_hgbr_learner`), `experiments/05_fe_outliers_grouped.py`,
  `tests/smoke/test_05_fe_outliers_grouped.py`
- **Change versus 04_hgbr_grouped:**
  1. Feature set expands: drop `age` and `age_at_diagnosis`, add
     `years_since_diagnosis = age - age_at_diagnosis` + `rater_id`
     (categorical) + `cohort` (categorical).
  2. Clip `off > 108` → 108 (stateless, one-line, applied before fit
     via a `FunctionTransformer`).
  3. `rater_id` and `cohort` passed as object/category dtype so HGBR
     treats them as categorical splits.
  4. No imputation — HGBR handles NaN natively.
  5. Hub key: `05_fe_outliers_grouped`.
- **Cross-validation:** `GroupKFold(n_splits=5)` on `patient_id` —
  same as experiments 03 and 04.
- **Out of scope:** hyperparameter tuning, target transform.

## Risks / things that could invalidate the result

- `rater_id` encodes assessor identity, not a biological signal — if
  rater assignment is confounded with patient severity, HGBR may learn
  the rater→severity mapping instead of a causal feature. Flag if MAE
  drops implausibly (< 4).
- `years_since_diagnosis` has 2 319 nulls (where `age_at_diagnosis` is
  missing). HGBR handles NaN natively — these rows still contribute to
  training, unlike mean-imputation which would impute a noisy value.
- Clipping `off` at 108 removes 1 row from training when the value was
  previously 111 — effectively a correction, not a removal.

## Status

- **State:** approved
- **Approved by user on:** 2025-10-12
- **Headline result:** n/a
- **Implication for next iteration:** n/a
