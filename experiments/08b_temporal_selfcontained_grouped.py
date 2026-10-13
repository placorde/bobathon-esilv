# %% [markdown]
# # Experiment 08b: Self-contained temporal features — GroupKFold
#
# **Date:** 2025-10-13
# **Fix vs 08_temporal_grouped:**
#   - ``SelfContainedTemporalFeatures`` replaces ``TemporalPatientFeatures``.
#   - Expanding stats are computed purely from the rows in the current
#     frame (train fold or test fold); no stored training history used.
#   - Test patients under GroupKFold get stats from their own visits —
#     matching the Kaggle submission scenario exactly.

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
project.put("08b_temporal_selfcontained_grouped", report)

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
submission.to_csv(submissions_dir / "08b_temporal_selfcontained_grouped.csv", index=False)
print(
    f"Submission written: submissions/08b_temporal_selfcontained_grouped.csv "
    f"({len(submission)} rows)"
)
