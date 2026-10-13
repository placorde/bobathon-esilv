# 08_temporal_grouped

## Question / hypothesis

Do patient-level temporal features — expanding-window mean and trend
of past `off`/`on` scores plus visit count — improve on the DataOps FE
baseline (07, MAE 5.92 ± 0.11) by letting HGBR see a patient's disease
trajectory, not just their snapshot values?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request: per-patient expanding mean/trend of past `off`/`on`
    values (no leakage from future visits or target) + visit count;
    evaluate with same GroupKFold.
  - EDA: visits are chronologically ordered within each patient by `age`
    (confirmed by scratch probe: rows sorted by `patient_id` then `age`,
    `Index` increments monotonically within each patient). Each patient
    has 4–12 visits (mean 8). There is a clear disease progression
    signal in `off` and `on` within patients that snapshot features
    miss.
- **Why this matters:** HGBR with snapshot features (experiment 05–07,
  MAE ≈ 5.92) sees each visit in isolation. Adding "how this patient's
  `off`/`on` has been trending" captures intra-patient trajectory
  information that snapshot `off`/`on` values alone do not. The
  hypothesis is that trajectory features → tighter residuals for
  patients whose disease is progressing faster or slower than average.

## Method

- **Files touched:**
  - `src/parkinsons/features.py`: add `TemporalPatientFeatures` — a
    stateful sklearn `BaseEstimator + TransformerMixin` that:
    - `fit_transform(X)`: sorts rows by `patient_id` then `age`;
      for each row computes expanding stats over the *preceding* rows
      of the same patient (strictly past, excluding the current row);
      stores per-patient last-observed stats for predict-time use.
    - `transform(X)`: for test rows, initialises each patient's running
      state from the stored training history, then advances row-by-row
      in age order.
    - Features added: `past_off_mean`, `past_off_trend`,
      `past_on_mean`, `past_on_trend`, `visit_count` (all NaN for the
      first visit of a patient, which HGBR handles natively).
    - After computing, drops `patient_id` (grouping key, not a feature)
      and preserves all other columns.
  - `src/parkinsons/pipeline.py`: add `build_temporal_fe_learner` —
    same DataOps graph as 07 but with `TemporalPatientFeatures` chained
    as an additional `.skb.apply` step in Layer 3, before HGBR.
  - `experiments/08_temporal_grouped.py`
  - `tests/smoke/test_08_temporal_grouped.py`

- **Change versus 07_dataops_grouped:**
  1. `TemporalPatientFeatures` is inserted between `EngineerFeatures`
     and HGBR in Layer 3.
  2. The raw `patient_id` column is included in X (needed by
     `TemporalPatientFeatures`), then dropped inside the transformer
     before HGBR sees it.
  3. All other choices (hub key `08_temporal_grouped`, GroupKFold 5,
     `EngineerFeatures` FE, no hyperparameter tuning) are unchanged.

- **Cross-validation:** `GroupKFold(n_splits=5)` on `patient_id` — same
  as experiments 03–07. GroupKFold ensures the same patient never
  appears in both train and test within a fold.

- **Out of scope:** lags beyond visit count (no absolute time stamps
  available), target-based features, hyperparameter tuning.

## Risks / things that could invalidate the result

- **GroupKFold leakage check**: the transformer must compute `transform`
  stats from training history only. At test-fold time, a patient in the
  test set should not have their training visits visible to the
  transformer's `transform` call — but since GroupKFold puts an entire
  patient in either train or test (never both), the test fold contains
  only patients the transformer has **never seen in training**. This
  means `transform` will produce all-NaN temporal features for those
  patients (cold start). HGBR handles NaN natively, so this is
  structurally correct — but it means the temporal features are
  unavailable for the test fold by design. The experiment will still
  reveal whether fit-time trajectory signal generalises.
- **Exactly because of GroupKFold cold start**: the CV MAE may not
  improve (or may worsen) even if the features are informative on
  seen patients. The benefit would show more clearly with random-fold
  CV (where the same patient appears in both train and test in different
  visits) — but we do NOT use random CV because it leaks patient
  identity. This is the honest evaluation.
- `past_off_trend` / `past_on_trend` require at least 2 prior visits;
  fewer than 2 priors → NaN trend (HGBR-safe).
- `off` and `on` have substantial missingness (up to 79 %); expanding
  stats skip NaN values (using `nanmean` / `nanstd`) but may still be
  NaN if all prior values are NaN for that patient.

## Status

- **State:** done
- **Approved by user on:** 2025-10-13
- **Headline result:** MAE 13.58 ± 0.33 | RMSE 16.62 ± 0.37 | R² –0.016 (GroupKFold 5)
- **Implication for next iteration:** The temporal features hurt badly under GroupKFold — entire patients are held out in each test fold, so test visits have all-NaN temporal features (cold start). HGBR was trained with these features as strong signals and falls back to near-dummy performance on the test fold. The approach only makes sense with a within-patient splitter (e.g. random-fold CV) or if temporal features are derived from the test rows' own visit sequence without leaking the target. Abandoning this direction for now; the DataOps baseline (07, MAE 5.92) remains the best model.
