"""Smoke test for ``experiments/01_dummy.py``.

Fits the dummy pipeline on a 90 % training slice and predicts on
the remaining 10 %. Asserts that the prediction count matches the
held-out row count exactly (structural check) and that the smoke
MAE is within reason (NaN-poisoning check).

The dummy ignores features and always predicts the training mean,
so the soft-assertion bound is set generously to 60 (well above
any realistic UPDRS MAE — this catches only outright NaN failures).
"""

from __future__ import annotations

import pytest
from sklearn.metrics import mean_absolute_error

from parkinsons.data import load_dataset
from parkinsons.pipeline import build_learner

# Soft-assertion threshold: smoke MAE must be below this value.
# Updated to the CV headline after the first run. For now, set
# conservatively (max UPDRS score is ~110, so 60 is a loose bound
# that only fires on NaN/garbage predictions).
_SOFT_MAE_THRESHOLD = 60.0


@pytest.fixture
def train_predict_split():
    """Return (X_train, y_train, X_test, y_test) from an in-memory 90/10 split.

    Uses the real ``data/`` CSVs — no synthetic fixtures. The split
    is deterministic (first 90 % of rows as train, last 10 % as test)
    to keep the smoke test reproducible without introducing a random
    seed dependency.

    Returns
    -------
    tuple
        ``(X_train, y_train, X_test, y_test)`` as pandas DataFrames /
        Series.
    """
    X, y = load_dataset()
    n = len(X)
    cutoff = int(n * 0.9)
    return X.iloc[:cutoff], y.iloc[:cutoff], X.iloc[cutoff:], y.iloc[cutoff:]


def test_01_dummy(train_predict_split):
    """Predict count must equal the held-out row count."""
    X_train, y_train, X_test, y_test = train_predict_split

    learner = build_learner()
    learner.fit(X_train, y_train)
    predictions = learner.predict(X_test)

    # HARD: structural correctness — one prediction per row.
    assert len(predictions) == len(X_test), (
        f"got {len(predictions)} predictions for {len(X_test)} rows"
    )

    # SOFT: predictions are not NaN-poisoned.
    smoke_mae = mean_absolute_error(y_test, predictions)
    assert smoke_mae < _SOFT_MAE_THRESHOLD, (
        f"smoke MAE {smoke_mae:.2f} >= threshold {_SOFT_MAE_THRESHOLD}; "
        "predictions may be NaN-poisoned or constant-wrong."
    )
