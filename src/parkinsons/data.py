"""Data loading for the Parkinson's UPDRS prediction workspace.

Owns: loading raw CSVs from ``data/``, selecting the eight feature
columns requested for the baseline experiments, and providing
``X_test`` for the Kaggle submission cell. Data paths are resolved
via ``PROJECT_ROOT`` from the package root.

Also exports ``load_raw_fe(data_dir)`` — a pure path-to-DataFrame
function used by the DataOps graph in experiment 07.
"""

from __future__ import annotations

import pandas as pd

from parkinsons import PROJECT_ROOT

# Eight numeric features selected for baseline experiments.
FEATURE_COLS = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
]

# Extended feature set for experiment 05: includes rater_id and cohort as
# categoricals; age / age_at_diagnosis are present so the feature-engineering
# step can derive years_since_diagnosis and then drop them.
FEATURE_COLS_FE = [
    "sexM",
    "age_at_diagnosis",
    "age",
    "ledd",
    "time_since_intake_on",
    "time_since_intake_off",
    "on",
    "off",
    "rater_id",
    "cohort",
]

TARGET_COL = "target"
_DATA_DIR = PROJECT_ROOT / "data"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` from the training CSVs.

    Reads ``data/X_train.csv`` and ``data/y_train.csv``, merges on
    ``Index``, and returns the eight selected feature columns and the
    ``target`` series.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``X`` — shape ``(n_train, 8)`` with the baseline feature columns.
        ``y`` — shape ``(n_train,)`` with the continuous UPDRS target.
    """
    X_train = pd.read_csv(_DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(_DATA_DIR / "y_train.csv")
    merged = X_train.merge(y_train, on="Index")
    X = merged[FEATURE_COLS]
    y = merged[TARGET_COL]
    return X, y


def load_dataset_fe() -> "tuple[pd.DataFrame, pd.Series]":
    """Return ``(X, y)`` with the extended feature set for experiment 05.

    Includes ``rater_id`` and ``cohort`` as object dtype columns for HGBR's
    native categorical support, plus ``age`` and ``age_at_diagnosis`` so the
    feature-engineering step can derive ``years_since_diagnosis``.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``X`` with ``FEATURE_COLS_FE`` columns, ``y`` the target.
    """
    X_train = pd.read_csv(_DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(_DATA_DIR / "y_train.csv")
    merged = X_train.merge(y_train, on="Index")
    X = merged[FEATURE_COLS_FE].copy()
    # Ensure string columns stay as object so HGBR treats them as categorical
    X["rater_id"] = X["rater_id"].astype(object)
    X["cohort"] = X["cohort"].astype(object)
    y = merged[TARGET_COL]
    return X, y


def load_test_dataset_fe() -> "pd.DataFrame":
    """Return ``X_test`` with the extended feature set for experiment 05.

    Returns
    -------
    pd.DataFrame
        ``Index`` plus ``FEATURE_COLS_FE`` columns, object dtype for
        ``rater_id`` and ``cohort``.
    """
    X_test = pd.read_csv(_DATA_DIR / "X_test.csv")
    result = X_test[["Index"] + FEATURE_COLS_FE].copy()
    result["rater_id"] = result["rater_id"].astype(object)
    result["cohort"] = result["cohort"].astype(object)
    return result


def load_groups() -> "pd.Series":
    """Return the ``patient_id`` series aligned with ``load_dataset()``'s X.

    Used as the ``groups`` argument for ``GroupKFold`` so no patient appears
    on both sides of a fold boundary.

    Returns
    -------
    pd.Series
        Shape ``(n_train,)`` — one ``patient_id`` per training row, in the
        same order as the ``X`` returned by ``load_dataset()``.
    """
    X_train = pd.read_csv(_DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(_DATA_DIR / "y_train.csv")
    merged = X_train.merge(y_train, on="Index")
    return merged["patient_id"]


def load_raw_fe(data_dir: "str") -> pd.DataFrame:
    """Return the merged training frame for the DataOps FE pipeline.

    Pure function of ``data_dir``; suitable for use with
    ``skrub.var("data_dir").skb.apply_func(load_raw_fe)``.

    Returns a DataFrame with ``patient_id``, the ten feature columns
    from ``FEATURE_COLS_FE``, and the ``target`` column — no column
    selection is performed here; the DataOps graph handles that.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing ``X_train.csv`` and ``y_train.csv``.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_train, 12)`` — ``patient_id`` + ``FEATURE_COLS_FE``
        (10 columns) + ``target``.
    """
    import pathlib  # noqa: PLC0415

    data_dir = pathlib.Path(data_dir)
    X_train = pd.read_csv(data_dir / "X_train.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv")
    merged = X_train.merge(y_train, on="Index")
    keep = ["patient_id"] + FEATURE_COLS_FE + [TARGET_COL]
    result = merged[keep].copy()
    result["rater_id"] = result["rater_id"].astype(object)
    result["cohort"] = result["cohort"].astype(object)
    return result


def load_test_dataset() -> pd.DataFrame:
    """Return ``X_test`` for the Kaggle submission cell.

    Reads ``data/X_test.csv`` and returns the ``Index`` identifier
    column alongside the eight selected feature columns so the
    submission cell can write ``Index,target`` without a re-join.

    Returns
    -------
    pd.DataFrame
        Shape ``(n_test, 9)`` — ``Index`` plus the eight feature columns.
    """
    X_test = pd.read_csv(_DATA_DIR / "X_test.csv")
    return X_test[["Index"] + FEATURE_COLS]
