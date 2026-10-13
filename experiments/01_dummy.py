# %% [markdown]
# # Experiment 01: Dummy Regressor Baseline
#
# **Date:** 2025-10-12
# **Goal:** Establish the floor RMSE/MAE for the Parkinson's UPDRS regression
# task using a ``DummyRegressor(strategy="mean")`` on eight selected features.
# All real models in later experiments must beat this result.

# %%
import skore
from skore import login

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.evaluate import splitter
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_learner

# %% [markdown]
# ## Hub credentials
#
# Load the ``.skore`` credentials so ``skore.login`` authenticates without
# opening a browser, then open the shared workspace project.

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="databoys",
    mode="hub",
    workspace=cfg["workspace"],
)

# %% [markdown]
# ## Data
#
# Load the training features and target from ``data/``.

# %%
X, y = load_dataset()

# %% [markdown]
# ## Evaluate
#
# ``splitter=0.2`` is the 20 % random row-level holdout specified by the lab
# guide for ``01_dummy`` and ``02_ridge``. The dummy ignores features entirely
# and predicts the training mean for every row; skore computes MAE, RMSE, R²
# and writes a full ``EstimatorReport``.

# %%
learner = build_learner()
report = skore.evaluate(learner, X, y, splitter=splitter)
report

# %% [markdown]
# ## Persist to Skore Hub
#
# Key ``01_dummy`` identifies this experiment in the ``databoys`` project.
# Re-running this cell overwrites the previous report under the same key.

# %%
project.put("01_dummy", report)

# %% [markdown]
# ## Kaggle submission
#
# Refit on **all** training rows (no CV split), predict the test visits,
# and write ``submissions/01_dummy.csv`` (header ``Index,target``).

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test_full = load_test_dataset()
X_test_features = X_test_full.drop(columns=["Index"])

final = build_learner()
final.fit(X, y)
predictions = final.predict(X_test_features)

submission = X_test_full[["Index"]].copy()
submission["target"] = predictions
submission.to_csv(submissions_dir / "01_dummy.csv", index=False)
print(f"Submission written to submissions/01_dummy.csv  ({len(submission)} rows)")
