"""Smoke test for ``experiments/02_ridge.py``.

Fits the Ridge pipeline on a 90 % training slice and predicts on
the remaining 10 %. Asserts that the prediction count matches the
held-out row count exactly and that the smoke MAE is reasonable.
"""

from __future__ import annotations

import pytest
from sklearn.metrics import mean_absolute_error

from parkinsons.data import load_dataset
from parkinsons.pipeline import build_ridge_learner

# Soft-assertion threshold: updated to CV headline after first run.
# Set to 14.0 (slightly above dummy MAE 13.48) as a conservative
# starting bound — Ridge must at least be in the same order of magnitude.
_SOFT_MAE_THRESHOLD = 14.0


@pytest.fixture
def train_predict_split():
    """Return (X_train, y_train, X_test, y_test) from a 90/10 row split.

    Returns
    -------
    tuple
        ``(X_train, y_train, X_test, y_test)``.
    """
    X, y = load_dataset()
    n = len(X)
    cutoff = int(n * 0.9)
    return X.iloc[:cutoff], y.iloc[:cutoff], X.iloc[cutoff:], y.iloc[cutoff:]


def test_02_ridge(train_predict_split):
    """Predict count must equal the held-out row count."""
    X_train, y_train, X_test, y_test = train_predict_split

    learner = build_ridge_learner()
    learner.fit(X_train, y_train)
    predictions = learner.predict(X_test)

    # HARD: structural correctness.
    assert len(predictions) == len(X_test), (
        f"got {len(predictions)} predictions for {len(X_test)} rows"
    )

    # SOFT: predictions are not NaN-poisoned.
    smoke_mae = mean_absolute_error(y_test, predictions)
    assert smoke_mae < _SOFT_MAE_THRESHOLD, (
        f"smoke MAE {smoke_mae:.2f} >= threshold {_SOFT_MAE_THRESHOLD}; "
        "predictions may be NaN-poisoned."
    )
