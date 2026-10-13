# 08b_temporal_selfcontained_grouped

## Question / hypothesis

Does redesigning the temporal feature transformer to compute expanding
stats purely within the rows of the current frame — never referencing
stored training history — allow the GroupKFold CV to recover from the
cold-start failure in 08, and does it improve on the DataOps baseline
(07, MAE 5.92 ± 0.11)?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User feedback: 08 failed because ``TemporalPatientFeatures`` stored
    training-fold history and produced all-NaN for held-out patients at
    CV test time. This exactly matches the Kaggle test set structure
    (test patients are never in train). The fix is a stateless
    transformer that computes expanding stats from whatever rows it
    receives.
  - Design constraint: keep ``GroupKFold`` — the honest evaluation.

- **Why this matters:** Under ``GroupKFold``, each test fold contains
  entire patients never seen in training. If the temporal transformer
  operates only on the rows it receives, those test patients get
  expanding stats from their own visits (sorted by age), which is
  exactly what the Kaggle test-time pipeline will do. No cold start, no
  all-NaN features — the CV is now a faithful proxy for the submission.

## Method

- **Files touched:**
  - `src/parkinsons/features.py`: add ``SelfContainedTemporalFeatures``
    — stateless; ``fit`` is a no-op; ``transform`` calls
    ``_compute_expanding`` on whatever frame is passed, with no
    reference to ``patient_history_``.
  - `src/parkinsons/pipeline.py`: add
    ``build_temporal_selfcontained_learner`` — identical graph to 08
    but wires ``SelfContainedTemporalFeatures`` instead of
    ``TemporalPatientFeatures``.
  - `experiments/08b_temporal_selfcontained_grouped.py`
  - `tests/smoke/test_08b_temporal_selfcontained_grouped.py`

- **Change versus 08_temporal_grouped:**
  - ``TemporalPatientFeatures`` → ``SelfContainedTemporalFeatures``
    (one-line swap in the DataOps graph).
  - Everything else unchanged: same GroupKFold(5), same FE, same HGBR.

- **Cross-validation:** ``GroupKFold(n_splits=5)`` on ``patient_id``.

- **Out of scope:** hyperparameter tuning, changes to the FE step.

## Risks / things that could invalidate the result

- The transformer is now stateless — it sees only the rows of its
  current fold. For a test patient with 8 visits, visit 1 gets NaN
  (no prior), visits 2–8 get stats from their own earlier visits.
  The first-visit NaN is unavoidable; HGBR handles it.
- The expanding stats on the test fold are computed from test-fold
  rows only. Since GroupKFold never leaks the target into X, and the
  temporal features use only ``off`` and ``on`` (not ``target``), there
  is no leakage.
- If ``off``/``on`` are themselves strong signals because they
  correlate with the target (EDA: r ≈ 0.87 / 0.67), then their
  expanding means will also be strong — possibly more so, because the
  mean smooths noise. Whether that helps or hurts generalisation is
  what this experiment answers.

## Status

- **State:** planned
- **Approved by user on:** n/a
- **Headline result:** n/a
- **Implication for next iteration:** n/a
