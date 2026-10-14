# %% [markdown]
# # Experiment 08c: Temporal features — index-order fix — GroupKFold
#
# **Date:** 2025-10-13
# **Fix vs 08b_temporal_selfcontained_grouped:**
#   - Root cause: ``_compute_expanding`` and ``_apply_from_history`` both
#     sorted by ``[patient_id, age]`` but returned the DataFrame in sorted
#     order, not the original row order. This misaligned the new temporal
#     feature columns with ``y`` for every fold, explaining why 08b produced
#     the same broken metrics (MAE 13.58, R² -0.016) as 08.
#   - Fix: both helpers now capture ``original_index = X.index`` before
#     sorting, compute features on the sorted copy, then call
#     ``result.loc[original_index]`` to restore the original row order
#     before returning. One-line change per helper; no logic change.
#   - ``SelfContainedTemporalFeatures`` inherits the fix via the shared
#     helper; ``TemporalPatientFeatures._apply_from_history`` also fixed.

# %%
import skore
from skore import login

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset_fe, load_test_dataset_fe
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import (
    build_fe_hgbr_learner,
    build_temporal_selfcontained_learner,
)

DATA_DIR = PROJECT_ROOT / "data"

# %% [markdown]
# ## Hub credentials

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="databoys",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Evaluate

# %%
learner = build_temporal_selfcontained_learner(data_dir_preview=DATA_DIR)
report = skore.evaluate(learner, data={"data_dir": str(DATA_DIR)})
report

# %% [markdown]
# ## Persist to Skore Hub

# %%
project.put("08c_temporal_fixed_grouped", report)

# %% [markdown]
# ## Metrics summary

# %%
print(report.metrics.summarize().frame().to_string())

# %% [markdown]
# ## Kaggle submission
#
# Refit the equivalent sklearn pipeline on all training rows.  At
# submission time, X_test patients are unseen; SelfContainedTemporalFeatures
# will compute expanding stats from those test rows' own visit order —
# but since we cannot easily call the DataOps graph on the test CSV
# (no target column), we fall back to the plain EngineerFeatures → HGBR
# pipeline refitted on training data.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X, y = load_dataset_fe()
X_test_full = load_test_dataset_fe()
X_test_features = X_test_full.drop(columns=["Index"])

final = build_fe_hgbr_learner()
final.fit(X, y)
predictions = final.predict(X_test_features)

submission = X_test_full[["Index"]].copy()
submission["target"] = predictions
submission.to_csv(submissions_dir / "08c_temporal_fixed_grouped.csv", index=False)
print(
    f"Submission written: submissions/08c_temporal_fixed_grouped.csv "
    f"({len(submission)} rows)"
)
