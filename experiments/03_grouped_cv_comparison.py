# %% [markdown]
# # Experiment 03: Grouped CV Comparison (Dummy vs Ridge)
#
# **Date:** 2025-10-12
# **Goal:** Re-evaluate both models under ``GroupKFold(n_splits=5)`` on
# ``patient_id`` so no patient leaks across folds. Each model's
# ``CrossValidationReport`` is pushed to Hub under its own key.

# %%
import skore
from skore import login

from parkinsons.data import load_dataset
from parkinsons.evaluate import make_grouped_splitter
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_learner, build_ridge_learner

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
X, y = load_dataset()
grouped_splitter = make_grouped_splitter(n_splits=5)

# %% [markdown]
# ## Evaluate dummy under grouped CV

# %%
dummy_report = skore.evaluate(build_learner(), X, y, splitter=grouped_splitter)
dummy_report

# %% [markdown]
# ## Evaluate ridge under grouped CV

# %%
ridge_report = skore.evaluate(build_ridge_learner(), X, y, splitter=grouped_splitter)
ridge_report

# %% [markdown]
# ## Push both reports to Hub

# %%
project.put("03_dummy_grouped", dummy_report)
project.put("03_ridge_grouped", ridge_report)

# %% [markdown]
# ## Print per-model metric summaries

# %%
print("=== DUMMY (grouped CV) ===")
print(dummy_report.metrics.summarize().frame().to_string())
print()
print("=== RIDGE (grouped CV) ===")
print(ridge_report.metrics.summarize().frame().to_string())
