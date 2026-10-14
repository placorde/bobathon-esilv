# 08c_temporal_fixed_grouped

## Question / hypothesis

Does fixing the row-order misalignment in ``_compute_expanding`` and
``_apply_from_history`` allow ``SelfContainedTemporalFeatures`` to
produce correct per-patient expanding stats and improve over the
DataOps baseline (07, MAE 5.92 ± 0.11)?

## Motivation

- **Sourcing strategy:** audit
- **Source(s):**
  - Audit of 08b output: MAE 13.58, R² -0.016 — identical to 08.
  - Code inspection of ``features.py``: both ``_compute_expanding`` and
    ``_apply_from_history`` sorted by ``[patient_id, age]`` internally,
    assigned feature arrays positionally into the **sorted** copy, then
    returned that sorted frame without restoring the original row order.
    Because skrub/sklearn aligns transformer output with ``y`` by integer
    position (not by index label), the new columns were paired with the
    wrong target values in every fold. This is the sole cause of the
    identical broken metrics in 08 and 08b.
  - The ``SelfContainedTemporalFeatures`` design (no stored history) was
    already correct in principle; only the index alignment was wrong.

- **Why this matters:** Without this fix, all temporal-feature experiments
  are silent no-ops — the model sees correctly-shaped random-looking
  temporal features that are unrelated to ``y``, so it learns nothing
  from them and falls back to the same non-temporal signal as 07.

## Method

- **Files touched:**
  - `src/parkinsons/features.py`:
    - ``_compute_expanding``: capture ``original_index = X.index`` before
      sort; work on ``X_sorted`` copy; call ``result.loc[original_index]``
      before returning.
    - ``_apply_from_history``: same one-line fix (also restores order
      after sort). Fixes ``TemporalPatientFeatures`` for completeness.
  - `experiments/08c_temporal_fixed_grouped.py`
  - `tests/smoke/test_08c_temporal_fixed_grouped.py`
  - `journal/08c_temporal_fixed_grouped.md` (this file)

- **Change versus 08b_temporal_selfcontained_grouped:**
  - Only the index-restoration lines in the two helper methods changed.
  - Pipeline, transformer class, GroupKFold, FE, HGBR: all unchanged.

- **Cross-validation:** ``GroupKFold(n_splits=5)`` on ``patient_id``.

- **Out of scope:** hyperparameter tuning, changes to the FE step.

## Risks / things that could invalidate the result

- First visit of every patient still produces NaN temporal features
  (no prior row in the fold); HGBR handles this natively.
- Temporal features use ``off``/``on`` from the test fold's own rows.
  These are feature columns (not the target), so no target leakage.
- If the per-patient expanding signal is weak relative to the static
  features already in 07, the gain may be small or zero.

## Status

- **State:** complete
- **Approved by user on:** 2025-10-13
- **Headline result:**
  - MAE **3.72 ± 0.03** (GroupKFold-5), R² **0.912 ± 0.004**, RMSE 4.88 ± 0.06
  - Beats 07_dataops_grouped (MAE 5.92 ± 0.11) by **37 %** — a clear win.
  - Beats 08b (MAE 13.58) by 73 % — confirms the index-order fix was the sole blocker.
  - Report: https://skore.probabl.ai/databoys/databoys/cross-validations/67089

- **Implication for next iteration:**
  - The expanding per-patient mean/trend of ``off``/``on`` is the strongest
    signal found so far; it captures individual disease trajectory that
    the static features cannot.
  - Next steps: tune HGBR (``max_iter``, ``max_depth``, ``learning_rate``),
    add more temporal windows (e.g. last-2 visits), or explore interaction
    features between ``years_since_diagnosis`` and the temporal slope.
