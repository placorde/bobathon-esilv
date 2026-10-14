"""Smoke test for ``experiments/11_hgbr_tuned_grouped.py``.

Checks that ``build_hgbr_tuned_learner`` with a specific set of params
runs end-to-end (row count correct, MAE sensible).  Uses a fast param
combo (max_iter=200, max_depth=6, learning_rate=0.1) — not the grid winner,
just a sanity check that the builder wires correctly.
"""

from __future__ import annotations

from sklearn.metrics import mean_absolute_error

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_raw_fe
from parkinsons.pipeline import build_hgbr_tuned_learner

DATA_DIR = PROJECT_ROOT / "data"


def test_11_hgbr_tuned_grouped():
    """Row count must match; train-time MAE must be under 13.0."""
    raw = load_raw_fe(DATA_DIR)
    n_rows = len(raw)

    learner = build_hgbr_tuned_learner(
        max_iter=200,
        max_depth=6,
        learning_rate=0.1,
        data_dir_preview=DATA_DIR,
    )
    fitted = learner.fit({"data_dir": str(DATA_DIR)})
    predictions = fitted.predict({"data_dir": str(DATA_DIR)})

    assert len(predictions) == n_rows, (
        f"got {len(predictions)} predictions for {n_rows} rows"
    )

    smoke_mae = mean_absolute_error(raw["target"], predictions)
    assert smoke_mae < 13.0, (
        f"smoke MAE {smoke_mae:.2f} >= 13.0 — build_hgbr_tuned_learner may be broken"
    )
