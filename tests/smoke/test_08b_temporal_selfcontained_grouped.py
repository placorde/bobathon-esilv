"""Smoke test for ``experiments/08b_temporal_selfcontained_grouped.py``."""

from __future__ import annotations

from sklearn.metrics import mean_absolute_error

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_raw_fe
from parkinsons.pipeline import build_temporal_selfcontained_learner

DATA_DIR = PROJECT_ROOT / "data"


def test_08b_temporal_selfcontained_grouped():
    """Row count must match; MAE must stay below dummy ceiling."""
    raw = load_raw_fe(DATA_DIR)
    n_rows = len(raw)

    learner = build_temporal_selfcontained_learner(data_dir_preview=DATA_DIR)
    fitted = learner.fit({"data_dir": str(DATA_DIR)})
    predictions = fitted.predict({"data_dir": str(DATA_DIR)})

    assert len(predictions) == n_rows, (
        f"got {len(predictions)} predictions for {n_rows} rows"
    )

    smoke_mae = mean_absolute_error(raw["target"], predictions)
    assert smoke_mae < 14.0, (
        f"smoke MAE {smoke_mae:.2f} >= 14.0 — check SelfContainedTemporalFeatures"
    )
