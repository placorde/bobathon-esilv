"""Smoke test for ``experiments/08c_temporal_fixed_grouped.py``."""

from __future__ import annotations

from sklearn.metrics import mean_absolute_error

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_raw_fe
from parkinsons.pipeline import build_temporal_selfcontained_learner

DATA_DIR = PROJECT_ROOT / "data"


def test_08c_temporal_fixed_grouped():
    """Row count must match; MAE must beat 08b (13.58) and approach 07 (5.92)."""
    raw = load_raw_fe(DATA_DIR)
    n_rows = len(raw)

    learner = build_temporal_selfcontained_learner(data_dir_preview=DATA_DIR)
    fitted = learner.fit({"data_dir": str(DATA_DIR)})
    predictions = fitted.predict({"data_dir": str(DATA_DIR)})

    assert len(predictions) == n_rows, (
        f"got {len(predictions)} predictions for {n_rows} rows"
    )

    smoke_mae = mean_absolute_error(raw["target"], predictions)
    assert smoke_mae < 13.0, (
        f"smoke MAE {smoke_mae:.2f} >= 13.0 — index-order fix may not have applied"
    )
