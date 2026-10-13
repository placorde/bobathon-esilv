"""Smoke test for ``experiments/06_skrub_tabular_grouped.py``."""

from __future__ import annotations

import pytest
from sklearn.metrics import mean_absolute_error

from parkinsons.data import load_dataset_fe
from parkinsons.pipeline import build_skrub_tabular_learner


@pytest.fixture
def train_predict_split():
    """Return a deterministic 90/10 row split of the FE training data."""
    X, y = load_dataset_fe()
    n = len(X)
    cutoff = int(n * 0.9)
    return X.iloc[:cutoff], y.iloc[:cutoff], X.iloc[cutoff:], y.iloc[cutoff:]


def test_06_skrub_tabular_grouped(train_predict_split):
    """Predict count must equal held-out row count; MAE must be sane."""
    X_train, y_train, X_test, y_test = train_predict_split

    learner = build_skrub_tabular_learner()
    learner.fit(X_train, y_train)
    predictions = learner.predict(X_test)

    # HARD: structural correctness — TableVectorizer must not drop rows.
    assert len(predictions) == len(X_test), (
        f"got {len(predictions)} predictions for {len(X_test)} rows"
    )

    # SOFT: predictions are not NaN-poisoned (threshold above HGBR grouped MAE 6.10).
    smoke_mae = mean_absolute_error(y_test, predictions)
    assert smoke_mae < 14.0, (
        f"smoke MAE {smoke_mae:.2f} >= 14.0 — check TableVectorizer encoding"
    )
