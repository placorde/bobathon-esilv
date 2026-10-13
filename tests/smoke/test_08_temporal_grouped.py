"""Smoke test for ``experiments/08_temporal_grouped.py``."""

from __future__ import annotations

from sklearn.metrics import mean_absolute_error

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_raw_fe
from parkinsons.pipeline import build_temporal_fe_learner

DATA_DIR = PROJECT_ROOT / "data"


def test_08_temporal_grouped():
    """Fit on training path, predict on same path; check structural row count."""
    raw = load_raw_fe(DATA_DIR)
    n_rows = len(raw)

    learner = build_temporal_fe_learner(data_dir_preview=DATA_DIR)
    fitted = learner.fit({"data_dir": str(DATA_DIR)})
    predictions = fitted.predict({"data_dir": str(DATA_DIR)})

    # HARD: structural correctness — one prediction per training row.
    assert len(predictions) == n_rows, (
        f"got {len(predictions)} predictions for {n_rows} training rows"
    )

    # SOFT: predictions are not NaN-poisoned (threshold well above HGBR baseline).
    y = raw["target"]
    smoke_mae = mean_absolute_error(y, predictions)
    assert smoke_mae < 14.0, (
        f"smoke MAE {smoke_mae:.2f} >= 14.0 — check TemporalPatientFeatures"
    )
