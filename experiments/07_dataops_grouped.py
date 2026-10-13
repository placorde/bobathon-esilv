# %% [markdown]
# # Experiment 07: DataOps FE pipeline — GroupKFold baked into graph
#
# **Date:** 2025-10-13
# **Changes vs 05_fe_outliers_grouped:**
#   - Learner is a skrub DataOps ``SkrubLearner`` rooted on
#     ``skrub.var("data_dir")``; ``GroupKFold(n_splits=5)`` and
#     ``groups=patient_id`` are baked into the graph via
#     ``mark_as_X(cv=..., split_kwargs=...)``.
#   - ``skore.evaluate`` is called with ``data={"data_dir": ...}``
#     (env-dict form); no ``splitter=`` kwarg needed.
#   - ``EngineerFeatures`` is a Layer-3 ``.skb.apply`` step, fitted
#     independently per fold.

# %%
import skore
from skore import login

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset_fe, load_test_dataset_fe
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_dataops_fe_learner, build_fe_hgbr_learner

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
# The baked ``GroupKFold(n_splits=5)`` is picked up automatically;
# no ``splitter=`` kwarg is passed to ``skore.evaluate``.

# %%
learner = build_dataops_fe_learner(data_dir_preview=DATA_DIR)
report = skore.evaluate(learner, data={"data_dir": str(DATA_DIR)})
report

# %% [markdown]
# ## Persist to Skore Hub

# %%
project.put("07_dataops_grouped", report)

# %% [markdown]
# ## Metrics summary

# %%
print(report.metrics.summarize().frame().to_string())

# %% [markdown]
# ## Kaggle submission
#
# Refit the equivalent sklearn pipeline on all training rows and predict
# test visits.  The DataOps learner is designed for CV evaluation; the
# plain sklearn pipeline (``build_fe_hgbr_learner``) is used for the
# final refit because the test CSV has no ``target`` column and the
# DataOps ``predict`` path calls the same loader as ``fit``.

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
submission.to_csv(submissions_dir / "07_dataops_grouped.csv", index=False)
print(f"Submission written: submissions/07_dataops_grouped.csv ({len(submission)} rows)")
