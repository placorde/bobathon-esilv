"""Learner declarations for the Parkinson's UPDRS experiments.

Each public function returns an unfitted estimator ready for
``skore.evaluate(estimator, X, y, splitter=...)``.

- ``build_learner``: ``DummyRegressor(mean)`` — floor baseline (01_dummy).
- ``build_ridge_learner``: ``Ridge(alpha=0.1)`` — first linear model (02_ridge).
- ``build_hgbr_learner``: ``HistGradientBoostingRegressor`` — native NaN
  support, no imputation needed (04_hgbr_grouped).
- ``build_skrub_tabular_learner``: ``skrub.tabular_pipeline("regressor")`` —
  auto-encodes all column types including high-cardinality strings
  (06_skrub_tabular_grouped).
- ``build_dataops_fe_learner``: DataOps graph replicating experiment-05
  FE with ``GroupKFold(5)`` and ``groups=patient_id`` baked in via
  ``mark_as_X(cv=..., split_kwargs=...)`` (07_dataops_grouped).
- ``build_temporal_fe_learner``: DataOps graph adding per-patient
  expanding-window temporal features (past off/on mean + trend + visit
  count) before the experiment-05 FE and HGBR (08_temporal_grouped).
- ``build_temporal_selfcontained_learner``: same but uses
  ``SelfContainedTemporalFeatures`` — stateless, computes expanding
  stats purely from the current fold's rows (08b).
"""

from __future__ import annotations

import skrub
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


def build_learner(data_dir_preview=None) -> Pipeline:  # noqa: ARG001
    """Return the unfit dummy pipeline (01_dummy).

    Parameters
    ----------
    data_dir_preview :
        Ignored — present only to match the workspace convention used
        by skrub-DataOps-based learners.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted ``SimpleImputer(mean) → DummyRegressor(mean)`` pipeline.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("dummy", DummyRegressor(strategy="mean")),
        ]
    )


def build_ridge_learner(data_dir_preview=None) -> Pipeline:  # noqa: ARG001
    """Return the unfit Ridge pipeline (02_ridge).

    Parameters
    ----------
    data_dir_preview :
        Ignored — present only to match the workspace convention used
        by skrub-DataOps-based learners.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted ``SimpleImputer(mean) → Ridge(alpha=1.0)`` pipeline.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("ridge", Ridge(alpha=0.1)),
        ]
    )


def build_hgbr_learner(data_dir_preview=None) -> HistGradientBoostingRegressor:  # noqa: ARG001
    """Return the unfit HistGradientBoostingRegressor (04_hgbr_grouped).

    ``HistGradientBoostingRegressor`` handles ``NaN`` natively — no
    ``SimpleImputer`` step is needed. This makes it a fairer test of
    what the missingness pattern (up to 79 % in some columns) actually
    costs, since imputation itself can mask or distort that signal.

    Parameters
    ----------
    data_dir_preview :
        Ignored — present only to match the workspace convention.

    Returns
    -------
    sklearn.ensemble.HistGradientBoostingRegressor
        Unfitted estimator with default hyperparameters and
        ``random_state=0`` for reproducibility.
    """
    return HistGradientBoostingRegressor(random_state=0)


def build_fe_hgbr_learner(data_dir_preview=None) -> Pipeline:  # noqa: ARG001
    """Return the unfit feature-engineered HGBR pipeline (05_fe_outliers_grouped).

    Applies ``EngineerFeatures`` before ``HistGradientBoostingRegressor``:

    1. Derives ``years_since_diagnosis = age - age_at_diagnosis`` and
       drops the two collinear originals.
    2. Clips ``off`` at 108 (clinical UPDRS scale maximum).
    3. Passes ``rater_id`` and ``cohort`` as object columns — HGBR
       treats them as native categoricals automatically.
    4. No ``SimpleImputer`` — HGBR handles NaN natively.

    Parameters
    ----------
    data_dir_preview :
        Ignored — present only to match the workspace convention.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted ``EngineerFeatures → HistGradientBoostingRegressor``
        pipeline.
    """
    from parkinsons.features import EngineerFeatures  # noqa: PLC0415

    return Pipeline(
        steps=[
            ("fe", EngineerFeatures(clip_off=True)),
            ("hgbr", HistGradientBoostingRegressor(random_state=0)),
        ]
    )


def build_skrub_tabular_learner(data_dir_preview=None) -> Pipeline:  # noqa: ARG001
    """Return the unfit skrub tabular pipeline (06_skrub_tabular_grouped).

    Uses ``skrub.tabular_pipeline("regressor")`` which chains a
    ``TableVectorizer`` (auto-encodes numeric, low-cardinality, and
    high-cardinality string columns) with a
    ``HistGradientBoostingRegressor(categorical_features='from_dtype')``.
    No manual feature engineering or imputation step is required.

    Parameters
    ----------
    data_dir_preview :
        Ignored — present only to match the workspace convention.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Unfitted ``TableVectorizer → HistGradientBoostingRegressor`` pipeline
        as returned by ``skrub.tabular_pipeline("regressor")``.
    """
    return skrub.tabular_pipeline("regressor")


def build_dataops_fe_learner(data_dir_preview=None):
    """Return the unfit DataOps learner for experiment 07.

    Builds a skrub DataOps graph that replicates the experiment-05 FE
    pipeline (``EngineerFeatures → HistGradientBoostingRegressor``) with
    the patient-grouped CV strategy baked into the graph:

    - Layer 1: ``skrub.var("data_dir")`` → ``load_raw_fe`` (pure loader).
    - Layer 2: ``patient_id`` column extracted as the ``groups`` DataOp;
      ``X`` marked with ``cv=GroupKFold(n_splits=5)`` and
      ``split_kwargs={"groups": groups}``; ``y`` marked separately.
    - Layer 3: ``EngineerFeatures`` applied per fold via ``.skb.apply``;
      ``HistGradientBoostingRegressor(random_state=0)`` as the predictor.

    The returned ``SkrubLearner`` is evaluated with the env-dict form::

        skore.evaluate(learner, data={"data_dir": str(DATA_DIR)})

    No ``splitter=`` kwarg is needed; ``skore.evaluate`` picks up the
    baked-in ``GroupKFold`` and produces a ``CrossValidationReport``.

    Parameters
    ----------
    data_dir_preview : str or Path or None, default=None
        Optional path used for ``learner.skb.preview()``. Does not affect
        the CV or final fit.

    Returns
    -------
    skrub SkrubLearner
        Unfitted learner whose ``.fit({"data_dir": str(path)})`` triggers
        grouped cross-validation internally.
    """
    from parkinsons.data import TARGET_COL, load_raw_fe  # noqa: PLC0415
    from parkinsons.features import EngineerFeatures  # noqa: PLC0415

    data_dir = (
        skrub.var("data_dir", value=str(data_dir_preview))
        if data_dir_preview is not None
        else skrub.var("data_dir")
    )

    # Layer 1: load the raw frame (patient_id + features + target).
    data = data_dir.skb.apply_func(load_raw_fe)

    # Layer 2: derive groups DataOp, mark X and y.
    groups = data["patient_id"]
    X = data.drop(columns=["patient_id", TARGET_COL]).skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": groups},
    )
    y = data[TARGET_COL].skb.mark_as_y()

    # Layer 3: feature engineering per fold, then HGBR.
    X = X.skb.apply(EngineerFeatures(clip_off=True))
    predictions = X.skb.apply(
        HistGradientBoostingRegressor(random_state=0), y=y
    )
    return predictions.skb.make_learner()


def build_temporal_fe_learner(data_dir_preview=None):
    """Return the unfit DataOps learner for experiment 08.

    Extends ``build_dataops_fe_learner`` with per-patient expanding-window
    temporal features inserted before ``EngineerFeatures`` in Layer 3:

    - Layer 1: ``skrub.var("data_dir")`` → ``load_raw_fe`` (raw frame
      including ``patient_id``).
    - Layer 2: ``groups = data["patient_id"]``; ``X`` marked with
      ``cv=GroupKFold(n_splits=5)`` and
      ``split_kwargs={"groups": groups}``; ``y`` marked separately.
    - Layer 3a: ``TemporalPatientFeatures`` — computes expanding mean
      and trend of past ``off``/``on`` and visit count per patient.
      Drops ``patient_id`` from the output.
    - Layer 3b: ``EngineerFeatures`` — derives ``years_since_diagnosis``,
      clips ``off``, drops ``age``/``age_at_diagnosis``.
    - Predictor: ``HistGradientBoostingRegressor(random_state=0)``.

    Parameters
    ----------
    data_dir_preview : str or Path or None, default=None
        Optional preview path for ``learner.skb.preview()``.

    Returns
    -------
    skrub SkrubLearner
        Unfitted learner evaluated via
        ``skore.evaluate(learner, data={"data_dir": str(path)})``.
    """
    from parkinsons.data import TARGET_COL, load_raw_fe  # noqa: PLC0415
    from parkinsons.features import EngineerFeatures, TemporalPatientFeatures  # noqa: PLC0415

    data_dir = (
        skrub.var("data_dir", value=str(data_dir_preview))
        if data_dir_preview is not None
        else skrub.var("data_dir")
    )

    # Layer 1: load the raw frame (patient_id + features + target).
    data = data_dir.skb.apply_func(load_raw_fe)

    # Layer 2: keep patient_id in X so TemporalPatientFeatures can group by it.
    # TemporalPatientFeatures drops patient_id from its output, so HGBR never
    # sees the raw identifier.
    groups = data["patient_id"]
    X_with_pid = data.drop(columns=[TARGET_COL]).skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": groups},
    )
    y = data[TARGET_COL].skb.mark_as_y()

    # Layer 3a: temporal features (patient_id required for grouping; dropped inside)
    X_temporal = X_with_pid.skb.apply(TemporalPatientFeatures())
    # Layer 3b: experiment-05 FE (needs age + age_at_diagnosis; drops them)
    X_fe = X_temporal.skb.apply(EngineerFeatures(clip_off=True))
    # Predictor
    predictions = X_fe.skb.apply(
        HistGradientBoostingRegressor(random_state=0), y=y
    )
    return predictions.skb.make_learner()


def build_temporal_selfcontained_learner(data_dir_preview=None):
    """Return the unfit DataOps learner for experiment 08b.

    Identical to ``build_temporal_fe_learner`` except the temporal step
    uses ``SelfContainedTemporalFeatures`` instead of
    ``TemporalPatientFeatures``. The key difference:

    - ``SelfContainedTemporalFeatures`` computes expanding stats purely
      from the rows present in the frame passed to ``transform`` — it
      never references a stored training history.
    - At CV test time under ``GroupKFold``, the test fold contains each
      held-out patient's own visits. The transformer sorts them by age
      and computes expanding stats within that patient's test visits,
      exactly mirroring what the Kaggle test set looks like.

    Layer structure is identical to experiment 08:

    - Layer 1: ``skrub.var("data_dir")`` → ``load_raw_fe``.
    - Layer 2: ``patient_id`` as ``groups``; ``X`` (including
      ``patient_id``) marked with ``cv=GroupKFold(n_splits=5)`` and
      ``split_kwargs={"groups": groups}``; ``y`` marked.
    - Layer 3a: ``SelfContainedTemporalFeatures`` (drops ``patient_id``).
    - Layer 3b: ``EngineerFeatures``.
    - Predictor: ``HistGradientBoostingRegressor(random_state=0)``.

    Parameters
    ----------
    data_dir_preview : str or Path or None, default=None
        Optional preview path for ``learner.skb.preview()``.

    Returns
    -------
    skrub SkrubLearner
    """
    from parkinsons.data import TARGET_COL, load_raw_fe  # noqa: PLC0415
    from parkinsons.features import (  # noqa: PLC0415
        EngineerFeatures,
        SelfContainedTemporalFeatures,
    )

    data_dir = (
        skrub.var("data_dir", value=str(data_dir_preview))
        if data_dir_preview is not None
        else skrub.var("data_dir")
    )

    data = data_dir.skb.apply_func(load_raw_fe)

    groups = data["patient_id"]
    X_with_pid = data.drop(columns=[TARGET_COL]).skb.mark_as_X(
        cv=GroupKFold(n_splits=5),
        split_kwargs={"groups": groups},
    )
    y = data[TARGET_COL].skb.mark_as_y()

    X_temporal = X_with_pid.skb.apply(SelfContainedTemporalFeatures())
    X_fe = X_temporal.skb.apply(EngineerFeatures(clip_off=True))
    predictions = X_fe.skb.apply(
        HistGradientBoostingRegressor(random_state=0), y=y
    )
    return predictions.skb.make_learner()
