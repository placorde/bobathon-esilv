# %% [markdown]
# # Experiment 02: Ridge Regression Baseline
#
# **Date:** 2025-10-12
# **Goal:** Beat the dummy floor (MAE 13.48 | RMSE 16.48) with a Ridge
# regression on the same 8 features and same 20 % row holdout.

# %%
import skore
from skore import login

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.evaluate import splitter
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_ridge_learner

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
# ## Data

# %%
X, y = load_dataset()

# %% [markdown]
# ## Evaluate
#
# ``splitter=0.2`` — same 20 % row-level holdout as ``01_dummy`` so
# results are directly comparable.

# %%
learner = build_ridge_learner()
report = skore.evaluate(learner, X, y, splitter=splitter)
report

# %% [markdown]
# ## Persist to Skore Hub

# %%
project.put("02_ridge", report)

# %% [markdown]
# ## Kaggle submission
#
# Refit on all training rows, predict test visits, write
# ``submissions/02_ridge.csv``.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test_full = load_test_dataset()
X_test_features = X_test_full.drop(columns=["Index"])

final = build_ridge_learner()
final.fit(X, y)
predictions = final.predict(X_test_features)

submission = X_test_full[["Index"]].copy()
submission["target"] = predictions
submission.to_csv(submissions_dir / "02_ridge.csv", index=False)
print(f"Submission written to submissions/02_ridge.csv  ({len(submission)} rows)")
