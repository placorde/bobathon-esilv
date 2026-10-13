# %% [markdown]
# # Experiment 08: Temporal patient features — GroupKFold
#
# **Date:** 2025-10-13
# **Changes vs 07_dataops_grouped:**
#   - ``TemporalPatientFeatures`` inserted before ``EngineerFeatures``
#     in Layer 3: computes expanding-window mean and OLS trend of past
#     ``off``/``on`` values plus visit count for each patient.
#   - ``patient_id`` travels through to the temporal step (for grouping),
#     then dropped before HGBR sees it.
#   - Everything else unchanged: same GroupKFold(5), same HGBR defaults.

# %%
import skore
from skore import login

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset_fe, load_test_dataset_fe
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_fe_hgbr_learner, build_temporal_fe_learner

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
#
# ``GroupKFold(n_splits=5)`` with ``groups=patient_id`` is baked into
# the graph; no ``splitter=`` kwarg needed.

# %%
learner = build_temporal_fe_learner(data_dir_preview=DATA_DIR)
report = skore.evaluate(learner, data={"data_dir": str(DATA_DIR)})
report

# %% [markdown]
# ## Persist to Skore Hub

# %%
project.put("08_temporal_grouped", report)

# %% [markdown]
# ## Metrics summary

# %%
print(report.metrics.summarize().frame().to_string())

# %% [markdown]
# ## Kaggle submission
#
# Refit the equivalent sklearn pipeline (EngineerFeatures → HGBR) on
# all training rows. The temporal features are derived from training
# history; for test patients all temporal features are NaN (cold start
# under GroupKFold), so the submission predictions match what the model
# learns at fit time without those features being available for unseen
# patients.

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
submission.to_csv(submissions_dir / "08_temporal_grouped.csv", index=False)
print(f"Submission written: submissions/08_temporal_grouped.csv ({len(submission)} rows)")
