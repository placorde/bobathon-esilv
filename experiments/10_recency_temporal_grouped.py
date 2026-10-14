# %% [markdown]
# # Experiment 10: Recency temporal features — GroupKFold
#
# **Date:** 2025-10-13
# **Builds on:** 08c_temporal_fixed_grouped (MAE 3.72, current best)
#
# **Change vs 08c:**
#   - Replaces ``SelfContainedTemporalFeatures`` with
#     ``RecencyTemporalFeatures``, which computes 9 features per row
#     (the existing 5 from 08c + 4 new recency-window features):
#       - ``last_off`` / ``last_on``: value from the single most recent
#         prior visit of this patient in the current frame.
#       - ``last3_off_mean`` / ``last3_on_mean``: mean of up to 3 most
#         recent prior visits (finite values only).
#   - Same cold-start-safe design: stateless, no stored training history,
#     index-order restored before return.
#   - Everything else unchanged: GroupKFold(5), EngineerFeatures, HGBR defaults.
#
# **Hypothesis:** The full expanding mean/trend captures long-run trajectory,
# but the most recent 1–3 visits may carry a stronger recency signal for
# UPDRS prediction. Adding explicit recency features lets HGBR pick the
# most informative temporal scale without having to learn it implicitly.

# %%
import skore
from skore import login

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset_fe, load_test_dataset_fe
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import (
    build_fe_hgbr_learner,
    build_recency_temporal_learner,
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
learner = build_recency_temporal_learner(data_dir_preview=DATA_DIR)
report = skore.evaluate(learner, data={"data_dir": str(DATA_DIR)})
report

# %% [markdown]
# ## Persist to Skore Hub

# %%
project.put("10_recency_temporal_grouped", report)

# %% [markdown]
# ## Metrics summary

# %%
print(report.metrics.summarize().frame().to_string())

# %% [markdown]
# ## Kaggle submission

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
submission.to_csv(submissions_dir / "10_recency_temporal_grouped.csv", index=False)
print(
    f"Submission written: submissions/10_recency_temporal_grouped.csv "
    f"({len(submission)} rows)"
)
