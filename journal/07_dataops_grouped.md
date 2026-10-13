# 07_dataops_grouped

## Question / hypothesis

Does rebuilding the experiment-05 FE pipeline as a skrub DataOps
graph — with `GroupKFold(n_splits=5)` and `groups=patient_id` baked
into the graph via `mark_as_X(cv=..., split_kwargs=...)` — reproduce
the same MAE (≈ 5.92) while making the CV strategy a self-contained
property of the learner object rather than an external argument?

## Motivation

- **Sourcing strategy:** user
- **Source(s):**
  - User request: "Rebuild the best model (HGBR with the feature
    engineering from step 05) using skrub DataOps, with groups baked
    into the graph via `mark_as_X(cv=GroupKFold(n_splits=5),
    split_kwargs={'groups': groups})`."
  - Experiments 04–06 all pass `grouped_splitter` externally; the
    DataOps form makes the split strategy declarative and attached to
    the learner, so future callers (audit, submission, hyperparameter
    search) can omit `splitter=` and still get a grouped CV.

- **Why this matters:** The DataOps graph is the idiomatic skrub form.
  Baking the CV into the graph means `skore.evaluate(learner, data=...)`
  (no `splitter=` kwarg) automatically produces a `CrossValidationReport`
  with the correct patient-grouped folds. It also wires `patient_id` as
  a DataOp node, confirming the column-selection and groups-derivation
  pattern compiles and runs without leakage.

## Method

- **Files touched:**
  - `src/parkinsons/data.py`: add `load_raw_fe()` — returns the merged
    DataFrame including `patient_id` and the 10 feature columns plus
    `target`; no column selection (the DataOps graph handles that).
  - `src/parkinsons/pipeline.py`: add `build_dataops_fe_learner()` —
    a DataOps graph rooted on `skrub.var("data_dir")`, loading via
    `load_raw_fe`, deriving `groups` from `patient_id`, marking X with
    `cv=GroupKFold(n_splits=5)` and `split_kwargs={"groups": groups}`,
    applying `EngineerFeatures` in Layer 3, then `HGBR`.
  - `experiments/07_dataops_grouped.py`
  - `tests/smoke/test_07_dataops_grouped.py`

- **Change versus 05_fe_outliers_grouped:**
  1. Learner is a `SkrubLearner` (from `.skb.make_learner()`), not a
     bare sklearn `Pipeline`.
  2. `GroupKFold(n_splits=5)` with `groups=patient_id` is baked into
     `mark_as_X(cv=..., split_kwargs=...)` rather than passed externally.
  3. `skore.evaluate(learner, data={"data_dir": ...})` — env-dict form,
     no explicit `splitter=`.
  4. Feature engineering (`EngineerFeatures`) is a stateful Layer-3
     `.skb.apply` step after the X marker.
  5. Column selection, target extraction, and group derivation all happen
     inside the DataOps graph.

- **Cross-validation:** `GroupKFold(n_splits=5)` on `patient_id`,
  baked into the graph — same split family as 03/04/05/06.

- **Out of scope:** hyperparameter tuning, any change to the FE logic,
  any change to the `off`-clipping decision.

## Risks / things that could invalidate the result

- `GroupKFold` is imported in `pipeline.py` — this is a deliberate
  departure from the "splitter lives only in `evaluate.py`" convention,
  justified by the DataOps design goal. The user explicitly requested it.
- The env-dict call form (`data={"data_dir": ...}`) means the experiment
  script and smoke test pass a path, not a pre-loaded DataFrame. The
  `load_raw_fe` loader must be a pure function of the path; if it
  references any global state the CV may leak.
- `EngineerFeatures` in Layer 3 is stateful (it now sets
  `n_features_in_` in `fit`). Wrapping it via `.skb.apply` is correct —
  the DataOps engine will fit it on each training fold separately.

## Status

- **State:** done
- **Approved by user on:** 2025-10-13
- **Headline result:** MAE 5.92 ± 0.11 | RMSE 7.54 ± 0.15 | R² 0.791 (GroupKFold 5 — baked)
- **Implication for next iteration:** Exactly reproduces experiment 05 (same FE, same HGBR, same splits) in DataOps form. The baked CV confirms the patient-grouped split strategy is self-contained in the learner. Next natural step: add hyperparameter tuning via ``mark_as_X`` + ``skb.make_randomized_search``, or a target transform.
