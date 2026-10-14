# 10_recency_temporal_grouped

## Question / hypothesis

Do recency-window features (last-1 and last-3 visit stats for ``off``/``on``)
improve over the full expanding mean/trend of 08c (MAE 3.72)?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User hypothesis: the most recent 1–3 visits may carry more predictive
    signal for the current UPDRS score than the full expanding history,
    because Parkinson's is a progressive disease whose short-term trajectory
    (recent acceleration or stabilisation) may differ from the long-run mean.
  - Complementary to the existing 5 features: HGBR can then choose the most
    informative temporal scale rather than having to extract it implicitly
    from the full history.

- **Why this matters:** If recency matters more than the full history, the
  new features should drive a measurable MAE improvement. If they add noise
  (e.g. because ``off``/``on`` are highly variable visit-to-visit), the
  improvement will be zero or negative.

## Method

- **Files touched:**
  - `src/parkinsons/features.py`: add ``RecencyTemporalFeatures`` — stateless
    transformer computing 9 per-patient features from strictly-prior rows in
    the current frame: the 5 existing expanding stats (``past_off_mean``,
    ``past_off_trend``, ``past_on_mean``, ``past_on_trend``, ``visit_count``)
    plus 4 recency features (``last_off``, ``last_on``, ``last3_off_mean``,
    ``last3_on_mean``). Same index-restore pattern as 08c.
  - `src/parkinsons/pipeline.py`: add ``build_recency_temporal_learner`` —
    identical graph to 08c but wires ``RecencyTemporalFeatures`` in place of
    ``SelfContainedTemporalFeatures``.
  - `experiments/10_recency_temporal_grouped.py`
  - `tests/smoke/test_10_recency_temporal_grouped.py`
  - `journal/10_recency_temporal_grouped.md` (this file)

- **Change versus 08c_temporal_fixed_grouped:**
  - One class swap: ``SelfContainedTemporalFeatures`` → ``RecencyTemporalFeatures``.
  - 4 extra features: ``last_off``, ``last_on``, ``last3_off_mean``, ``last3_on_mean``.
  - GroupKFold(5), EngineerFeatures, HGBR defaults: all unchanged.

- **Cross-validation:** ``GroupKFold(n_splits=5)`` on ``patient_id``.

- **Feature definitions (strictly prior visits only):**
  - ``last_off``: raw ``off`` value at the chronologically most recent
    prior visit in the current frame. NaN if no prior visit or if that
    visit's ``off`` was NaN.
  - ``last_on``: same for ``on``.
  - ``last3_off_mean``: ``nanmean`` of ``off`` over the last min(3, n_prior)
    visits. NaN if all last-3 ``off`` values are NaN.
  - ``last3_on_mean``: same for ``on``.

## Risks / things that could invalidate the result

- ``off``/``on`` have high missingness (up to 79 %). ``last_off`` will be NaN
  for the first visit of each patient and also for subsequent visits if the
  immediately prior visit had a missing ``off``. HGBR handles this natively.
- The last-3 mean is correlated with the full expanding mean, especially for
  patients with few visits. The redundancy may dilute rather than add signal.
- Under GroupKFold, test patients start cold (all first-visit NaN). This is
  the same situation as 08c; the recency features add no cold-start advantage.

## Status

- **State:** complete
- **Approved by user on:** 2025-10-13
- **Headline result:**
  - MAE **3.650 ± 0.036** (GroupKFold-5), R² **0.915 ± 0.003**, RMSE 4.80 ± 0.06
  - 08c baseline: MAE 3.717 ± 0.032 — **improvement of 0.067 MAE (1.8 %)**.
  - Beats 08c: **yes** — 10 is the new best model.
  - Report: https://skore.probabl.ai/databoys/databoys/cross-validations/67426

- **Implication for next iteration:**
  - Recency signal is genuinely useful on top of the full expanding history.
    The last-1 and last-3 features let HGBR weight recent visits more
    heavily without discarding the long-run trend.
  - The gain is modest (0.07 MAE). Next directions:
    - HGBR hyperparameter tuning (``max_iter``, ``learning_rate``,
      ``max_depth``) on this feature set — default params are rarely optimal.
    - Wider recency windows (last-5, last-10) to find the sweet spot.
    - Trend over the last-3 visits (short-term slope) as an additional feature.
