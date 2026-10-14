"""Smoke test for ``experiments/10_recency_temporal_grouped.py``."""

from __future__ import annotations

from sklearn.metrics import mean_absolute_error

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_raw_fe
from parkinsons.pipeline import build_recency_temporal_learner

DATA_DIR = PROJECT_ROOT / "data"


def test_10_recency_temporal_grouped():
    """Row count must match; train-time MAE must beat the 08b ceiling (13.0)."""
    raw = load_raw_fe(DATA_DIR)
    n_rows = len(raw)

    learner = build_recency_temporal_learner(data_dir_preview=DATA_DIR)
    fitted = learner.fit({"data_dir": str(DATA_DIR)})
    predictions = fitted.predict({"data_dir": str(DATA_DIR)})

    assert len(predictions) == n_rows, (
        f"got {len(predictions)} predictions for {n_rows} rows"
    )

    smoke_mae = mean_absolute_error(raw["target"], predictions)
    assert smoke_mae < 13.0, (
        f"smoke MAE {smoke_mae:.2f} >= 13.0 — RecencyTemporalFeatures may be broken"
    )
