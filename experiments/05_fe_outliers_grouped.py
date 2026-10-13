# %% [markdown]
# # Experiment 05: Feature Engineering + Outlier Handling — grouped CV
#
# **Date:** 2025-10-12
# **Changes vs 04_hgbr_grouped:**
#   - `years_since_diagnosis` replaces collinear `age` + `age_at_diagnosis`
#   - `rater_id`, `cohort` added as native HGBR categoricals
#   - `off` clipped at 108 (clinical UPDRS max; 1 row had value 111)
#   - No imputation — HGBR handles NaN natively

# %%
import skore
from skore import login

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset_fe, load_test_dataset_fe
from parkinsons.evaluate import make_grouped_splitter
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_fe_hgbr_learner

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
# ## Data + grouped splitter

# %%
X, y = load_dataset_fe()
grouped_splitter = make_grouped_splitter(n_splits=5)

# %% [markdown]
# ## Evaluate

# %%
learner = build_fe_hgbr_learner()
report = skore.evaluate(learner, X, y, splitter=grouped_splitter)
report

# %% [markdown]
# ## Persist to Skore Hub

# %%
project.put("05_fe_outliers_grouped", report)

# %% [markdown]
# ## Metrics summary

# %%
print(report.metrics.summarize().frame().to_string())

# %% [markdown]
# ## Kaggle submission
#
# Refit on all training rows, predict test visits.

# %%
submissions_dir = PROJECT_ROOT / "submissions"
submissions_dir.mkdir(exist_ok=True)

X_test_full = load_test_dataset_fe()
X_test_features = X_test_full.drop(columns=["Index"])

final = build_fe_hgbr_learner()
final.fit(X, y)
predictions = final.predict(X_test_features)

submission = X_test_full[["Index"]].copy()
submission["target"] = predictions
submission.to_csv(submissions_dir / "05_fe_outliers_grouped.csv", index=False)
print(f"Submission written: submissions/05_fe_outliers_grouped.csv ({len(submission)} rows)")
