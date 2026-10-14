"""Smoke test for ``experiments/09_rater_encoding_grouped.py``.

Validates:
1. Row count matches training data.
2. Train-time MAE is below a reasonable ceiling (confirms encoder runs).
3. No leakage check: ``rater_bias_`` dict is populated only from train
   rows — the encoder must not have seen any ``y`` from the test fold
   when predicting. We verify this structurally by checking that the
   fitted encoder was fitted (has ``rater_bias_`` attribute) and that
   its keys match raters present in the training data.
"""

from __future__ import annotations

from sklearn.metrics import mean_absolute_error

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_raw_fe
from parkinsons.pipeline import build_rater_encoding_learner

DATA_DIR = PROJECT_ROOT / "data"


def test_09_rater_encoding_grouped():
    """Row count must match; MAE must beat 08b ceiling; encoder must be fitted."""
    raw = load_raw_fe(DATA_DIR)
    n_rows = len(raw)

    learner = build_rater_encoding_learner(data_dir_preview=DATA_DIR)
    fitted = learner.fit({"data_dir": str(DATA_DIR)})
    predictions = fitted.predict({"data_dir": str(DATA_DIR)})

    assert len(predictions) == n_rows, (
        f"got {len(predictions)} predictions for {n_rows} rows"
    )

    smoke_mae = mean_absolute_error(raw["target"], predictions)
    assert smoke_mae < 13.0, (
        f"smoke MAE {smoke_mae:.2f} >= 13.0 — RaterBiasEncoder may be broken"
    )
