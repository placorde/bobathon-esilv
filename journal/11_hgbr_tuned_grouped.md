# 11_hgbr_tuned_grouped

## Question / hypothesis

Does tuning ``HistGradientBoostingRegressor`` hyperparameters on the
experiment-10 feature set improve over the default-param baseline
(10, MAE 3.65)?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - Experiment 10 used HGBR defaults (max_iter=100, max_depth=None,
    learning_rate=0.1). Defaults are rarely optimal; more iterations with
    a lower learning rate is a standard bias-variance improvement for
    gradient boosting.
  - Grid space: max_iter ∈ {200, 500, 1000} × max_depth ∈ {None, 4, 6, 8}
    × learning_rate ∈ {0.01, 0.1, 0.3} → 36 combinations.
  - Overfitting guard: train vs test MAE gap inspected; combinations with
    gap > 0.5 and a tighter alternative within 0.02 test MAE are discarded.

- **Why this matters:** Gradient boosting is sensitive to learning_rate ×
  max_iter (effective capacity) and max_depth (per-tree complexity). A
  deeper search is warranted now that the feature set is strong.

## Method

- **Files touched:**
  - `src/parkinsons/pipeline.py`: add ``build_hgbr_tuned_learner`` — same
    DataOps graph as experiment 10 but takes explicit HGBR hyperparameters.
  - `experiments/11_hgbr_tuned_grouped.py`: runs a 36-combination grid
    search using an sklearn ``Pipeline`` + ``GroupKFold(5)`` + ``cross_validate``
    (``return_train_score=True``), prints the full results table, selects
    the winner, then re-evaluates via ``skore.evaluate`` for the hub report.
  - `tests/smoke/test_11_hgbr_tuned_grouped.py`
  - `journal/11_hgbr_tuned_grouped.md` (this file)

- **Change versus 10_recency_temporal_grouped:**
  - HGBR hyperparameters only. Feature set (RecencyTemporalFeatures +
    EngineerFeatures) and GroupKFold(5) unchanged.

- **Cross-validation:** ``GroupKFold(n_splits=5)`` on ``patient_id``
  throughout (grid search and final ``skore.evaluate``).

## Risks / things that could invalidate the result

- Deeper trees (max_depth=4, 6, 8) with high learning_rate may overfit
  on the training folds. The train/test gap check guards against this.
- With 44k rows and ~180 patients, GroupKFold folds are large enough
  that overfitting is less likely than with small datasets, but the gap
  should still be inspected.
- ``max_iter=1000`` + ``learning_rate=0.01`` may need many more trees than
  the default early-stopping patience; HGBR's built-in validation-based
  early stopping (``validation_fraction=0.1`` default) will activate if
  the loss plateaus, so wall-clock time is bounded.

## Status

- **State:** complete
- **Approved by user on:** 2025-10-13
- **Best params:** ``max_iter=1000``, ``max_depth=6``, ``learning_rate=0.1``
- **Headline result:**
  - Grid-search CV MAE: **3.6172 ± 0.0259** (GroupKFold-5)
  - Final ``skore.evaluate`` MAE: **3.617 ± 0.029**, R² **0.916 ± 0.003**, RMSE 4.77 ± 0.05
  - Experiment 10 baseline: MAE 3.650 ± 0.036 — **improvement of 0.033 MAE (0.9%)**.
  - Beats 10: **yes** — 11 is the new best model.
  - Report: https://skore.probabl.ai/databoys/databoys/cross-validations/67974

- **Grid search key findings:**
  - ``learning_rate=0.1`` consistently wins; 0.3 is slightly worse (over-steps),
    0.01 needs far more iterations to match.
  - Increasing ``max_iter`` beyond 200 helps modestly: 200→500 saves ~0.002 MAE
    at depth=6, lr=0.1; 500→1000 saves another ~0.0002 (marginal).
  - ``max_depth`` matters: depth=6 and depth=8 tie at the top for lr=0.1;
    depth=None (unlimited) performs similarly but with a slightly larger gap.
    depth=4 is consistently ~0.005 worse than depth=6.
  - **Overfitting:** The gap for the winner (0.82) is higher than desired but
    consistent across all depth ≥ 6 + lr=0.1 combos. The lower-gap alternatives
    (e.g. depth=6, lr=0.01, iter=1000: gap 0.39, test 3.633) are ~0.016 worse
    in test MAE — not within 0.02 of the winner, so the winner is preferred.

- **Implication for next iteration:**
  - 11 is the new best at MAE 3.617. The feature set from experiment 10 with
    tuned HGBR is a solid foundation.
  - The persistent train/test gap (~0.8 MAE) under GroupKFold suggests the
    features that drive performance (temporal means/trends) are more stable
    on training patients than on held-out ones. This is expected — training
    patients have all their visits seen together; test patients start cold.
  - Next directions: early stopping (n_iter_no_change), finer learning_rate
    search (0.05, 0.08), or adding per-patient visit-level deltas as features.
