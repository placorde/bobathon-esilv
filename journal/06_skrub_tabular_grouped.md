# 06_skrub_tabular_grouped

## Question / hypothesis

Does skrub's `tabular_pipeline("regressor")` — which automatically
encodes all column types including high-cardinality strings — match or
beat the hand-crafted FE pipeline (05, MAE 5.92 ± 0.11) under the
same GroupKFold(5) on `patient_id`?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request: "Try skrub's tabular_pipeline('regressor') so
    categorical columns like gene and cohort are included."
  - Experiment 05 used a hand-crafted `EngineerFeatures` step to get
    `rater_id` and `cohort` into HGBR as native categoricals. skrub's
    `TableVectorizer` does this automatically for any object/string
    column, and also handles `age_at_diagnosis` / `age` without manual
    derivation — a useful reference point for how much the FE step is
    actually worth.
- **Why this matters:** `tabular_pipeline` bundles `TableVectorizer`
  (auto-encoding) + HGBR in one call. If it matches 05's MAE with zero
  manual feature engineering, it signals that the FE step was "free"
  but not essential; if it falls behind, the hand-crafted
  `years_since_diagnosis` + clipping was genuinely useful.

## Method

- **Files touched:** `src/parkinsons/pipeline.py` (add
  `build_skrub_tabular_learner`), `experiments/06_skrub_tabular_grouped.py`,
  `tests/smoke/test_06_skrub_tabular_grouped.py`. No changes to
  `data.py` or `features.py` — `load_dataset_fe()` already exposes all
  10 features including `rater_id`/`cohort` as object dtype.
- **Change versus 05_fe_outliers_grouped:**
  1. Replace `EngineerFeatures → HGBR` pipeline with
     `skrub.tabular_pipeline("regressor")`.
  2. Feed the same 10-column `X` from `load_dataset_fe()` directly —
     no manual derivation or clipping; `TableVectorizer` encodes
     `rater_id`/`cohort` as categoricals, keeps `age` and
     `age_at_diagnosis` as separate numeric features, and does **not**
     clip `off`.
  3. Hub key: `06_skrub_tabular_grouped`.
- **Cross-validation:** `GroupKFold(n_splits=5)` on `patient_id` —
  same as experiments 03, 04, 05.
- **Out of scope:** custom hyperparameters for HGBR, any manual FE on
  top of the tabular pipeline.

## Risks / things that could invalidate the result

- `tabular_pipeline("regressor")` uses `StringEncoder` for
  high-cardinality strings (like `rater_id` with ~50 categories). This
  is a learned encoding — it may not generalise as cleanly as
  `ToCategorical` + HGBR's native categorical splits when the category
  distribution shifts between training and test patients.
- Not applying the `off > 108` clip means the one plausible data-entry
  error row is retained. This is a one-row effect, unlikely to matter
  materially.
- `TableVectorizer` keeps `age` and `age_at_diagnosis` as separate
  numeric features; the collinearity (r = 0.94) may cause slight
  variance inflation for the HGBR splits, but tree models are generally
  robust to it.
- Comparison is with experiment 05, which already includes
  `rater_id`/`cohort`. Experiment 04 (no categoricals) is the more
  meaningful lower bound; skrub should at least match 05.

## Status

- **State:** done
- **Approved by user on:** 2025-10-13
- **Headline result:** MAE 5.93 ± 0.11 | RMSE 7.53 ± 0.15 | R² 0.792 (GroupKFold 5)
- **Implication for next iteration:** Matches experiment 05 (MAE 5.92 ± 0.11) within noise — the auto-encoding `TableVectorizer` is as good as hand-crafted FE. The manual derivation of `years_since_diagnosis` and the `off` clip were not worth the effort. Next step: hyperparameter search on top of this pipeline (e.g. HGBR `max_iter`, `learning_rate`, `max_depth`), or try target transform to handle the skewed OFF distribution.
