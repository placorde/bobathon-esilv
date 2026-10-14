# 09_rater_encoding_grouped

## Question / hypothesis

Does adding a per-rater mean bias feature — computed from training-fold
rows only, leakage-free — improve over 08c (MAE 3.72)?

## Motivation

- **Sourcing strategy:** user + EDA
- **Source(s):**
  - User-specified: EDA found the ``off → target`` gap varies by rater
    (range -10.5 to +2.2 UPDRS points, -0.20 correlation with
    ``years_since_dx``). HGBR sees ``rater_id`` as a categorical in 08c,
    but a tree split on a categorical with 40+ levels is noisy; an
    explicit numeric summary of the rater-specific bias may be a cleaner
    signal.
  - Design: classic out-of-fold target encoding — fit the per-rater mean
    of ``(target - off)`` using only the training fold, then use that
    scalar as a feature for both train and test rows in that fold.

- **Why this matters:** If rater bias is systematic and the HGBR in 08c
  is not fully capturing it through the raw categorical split, a
  pre-computed numeric encoding could improve predictions. If the tree
  was already handling it well, we expect no significant change.

## Method

- **Files touched:**
  - `src/parkinsons/features.py`: add ``RaterBiasEncoder`` — supervised
    transformer; ``fit(X, y)`` computes per-rater mean of ``(y - off)``
    from training rows only; ``transform`` maps rater labels to the
    stored means, with a global-mean fallback for unseen raters. No
    index-sort, so no row-alignment bug.
  - `src/parkinsons/pipeline.py`: add ``build_rater_encoding_learner``
    — identical to 08c pipeline plus ``RaterBiasEncoder`` wired with
    ``y=y`` after ``EngineerFeatures``.
  - `experiments/09_rater_encoding_grouped.py`
  - `tests/smoke/test_09_rater_encoding_grouped.py`
  - `journal/09_rater_encoding_grouped.md` (this file)

- **Change versus 08c_temporal_fixed_grouped:**
  - One additional step: ``X_fe.skb.apply(RaterBiasEncoder(), y=y)``.
  - Everything else unchanged: same GroupKFold(5), same FE, same HGBR.

- **Cross-validation:** ``GroupKFold(n_splits=5)`` on ``patient_id``.

- **Leakage analysis:**
  - ``RaterBiasEncoder.fit`` is called by skrub with training-fold rows
    and their ``y`` values only. The test fold's ``y`` is never passed
    to ``fit``.
  - ``transform`` uses only the fitted means (no ``y`` argument).
  - Therefore no target leakage: the feature value for a test row is
    determined entirely by its ``rater_id`` and training-fold statistics.

- **Out of scope:** hyperparameter tuning, changes to temporal FE.

## Risks / things that could invalidate the result

- If HGBR in 08c already captures the rater bias well via the
  ``rater_id`` category column and the temporal features, adding a
  redundant numeric encoding of the same signal may not help (and
  could slightly hurt if the encoding introduces small fold-to-fold
  variance).
- The rater-bias signal uses ``off`` as the baseline; if ``off`` is NaN,
  the bias is undefined. These rows are excluded from the per-rater
  mean computation; the global mean is used as fallback in their
  ``rater_bias`` column value at transform time.

## Status

- **State:** complete
- **Approved by user on:** 2025-10-13
- **Headline result:**
  - MAE **3.714 ± 0.034** (GroupKFold-5), R² **0.912 ± 0.004**, RMSE 4.88 ± 0.06
  - 08c baseline: MAE 3.717 ± 0.032 — **difference is +0.003 MAE (effectively zero)**.
  - The rater bias encoding adds no measurable signal over 08c.
  - Verdict: **do not keep** — 08c (MAE 3.717) remains the best model.
  - Report: https://skore.probabl.ai/databoys/databoys/cross-validations/67341

- **Implication for next iteration:**
  - The null result confirms HGBR in 08c already extracts the full rater-bias
    signal from the raw ``rater_id`` categorical column. Adding an explicit
    numeric encoding of the same information is redundant.
  - 08c (temporal features only) stays as the current best at MAE 3.72.
  - Next directions: HGBR hyperparameter tuning on 08c's feature set, or
    additional temporal windows (last-N visits vs full expanding mean).
