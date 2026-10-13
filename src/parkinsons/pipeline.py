"""Learner declarations for the Parkinson's UPDRS experiments.

Each public function returns an unfitted estimator ready for
``skore.evaluate(estimator, X, y, splitter=...)``.

- ``build_learner``: ``DummyRegressor(mean)`` — floor baseline (01_dummy).
- ``build_ridge_learner``: ``Ridge(alpha=0.1)`` — first linear model (02_ridge).
- ``build_hgbr_learner``: ``HistGradientBoostingRegressor`` — native NaN
  support, no imputation needed (04_hgbr_grouped).
"""

from __future__ import annotations

from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
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
