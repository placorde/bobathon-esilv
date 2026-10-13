"""Smoke test for ``experiments/03_grouped_cv_comparison.py``.

Verifies that both pipelines (dummy and ridge) produce one prediction
per held-out row under a GroupKFold-style split with no patient overlap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_groups
from parkinsons.evaluate import GroupedSplitter
from parkinsons.pipeline import build_learner, build_ridge_learner

_SOFT_MAE_THRESHOLD_DUMMY = 60.0
_SOFT_MAE_THRESHOLD_RIDGE = 14.0


@pytest.fixture
def small_grouped_split():
    """Return a tiny (train, test) split that respects patient groups.

    Uses only the first 500 rows to keep the smoke test fast.

    Returns
    -------
    tuple
        ``(X_train, y_train, X_test, y_test)``.
    """
    from sklearn.metrics import mean_absolute_error

    X, y = load_dataset()
    groups = load_groups()

    X_small = X.iloc[:500]
    y_small = y.iloc[:500]
    g_small = groups.iloc[:500]

    gkf = GroupKFold(n_splits=3)
    train_idx, test_idx = next(gkf.split(X_small, y_small, groups=g_small))

    return (
        X_small.iloc[train_idx],
        y_small.iloc[train_idx],
        X_small.iloc[test_idx],
        y_small.iloc[test_idx],
    )


def test_03_grouped_cv_comparison_dummy(small_grouped_split):
    """Dummy predict count must equal held-out row count."""
    X_train, y_train, X_test, y_test = small_grouped_split
    learner = build_learner()
    learner.fit(X_train, y_train)
    preds = learner.predict(X_test)
    assert len(preds) == len(X_test)


def test_03_grouped_cv_comparison_ridge(small_grouped_split):
    """Ridge predict count must equal held-out row count."""
    X_train, y_train, X_test, y_test = small_grouped_split
    learner = build_ridge_learner()
    learner.fit(X_train, y_train)
    preds = learner.predict(X_test)
    assert len(preds) == len(X_test)


def test_grouped_splitter_no_patient_overlap():
    """GroupedSplitter must produce zero patient overlap in every fold."""
    X, y = load_dataset()
    groups = load_groups().values
    splitter = GroupedSplitter(groups=groups, n_splits=5)

    for train_idx, test_idx in splitter.split(X, y):
        overlap = set(groups[train_idx]) & set(groups[test_idx])
        assert len(overlap) == 0, f"Patient overlap detected: {len(overlap)} patients"
