"""Smoke test for ``experiments/04_hgbr_grouped.py``.

Fits HGBR on a 90 % slice and predicts on the remaining 10 %.
No imputer — NaN values are passed through directly.
"""

from __future__ import annotations

import pytest
from sklearn.metrics import mean_absolute_error

from parkinsons.data import load_dataset
from parkinsons.pipeline import build_hgbr_learner

# Ridge grouped-CV MAE was 8.39; set threshold loosely above that.
_SOFT_MAE_THRESHOLD = 14.0


@pytest.fixture
def train_predict_split():
    """Return a deterministic 90/10 row split of the training data."""
    X, y = load_dataset()
    n = len(X)
    cutoff = int(n * 0.9)
    return X.iloc[:cutoff], y.iloc[:cutoff], X.iloc[cutoff:], y.iloc[cutoff:]


def test_04_hgbr_grouped(train_predict_split):
    """Predict count must equal held-out row count; MAE must be sane."""
    X_train, y_train, X_test, y_test = train_predict_split

    learner = build_hgbr_learner()
    learner.fit(X_train, y_train)
    predictions = learner.predict(X_test)

    # HARD: structural correctness.
    assert len(predictions) == len(X_test), (
        f"got {len(predictions)} predictions for {len(X_test)} rows"
    )

    # SOFT: predictions are not NaN-poisoned.
    smoke_mae = mean_absolute_error(y_test, predictions)
    assert smoke_mae < _SOFT_MAE_THRESHOLD, (
        f"smoke MAE {smoke_mae:.2f} >= threshold {_SOFT_MAE_THRESHOLD}"
    )
