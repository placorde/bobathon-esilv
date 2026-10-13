# %% [markdown]
# # Experiment 04: HistGradientBoostingRegressor — grouped CV
#
# **Date:** 2025-10-12
# **Goal:** Beat Ridge (MAE 8.39 ± 0.14) with a native-NaN tree model
# under the same ``GroupKFold(n_splits=5)`` on ``patient_id``.

# %%
import skore
from skore import login

from parkinsons.data import load_dataset
from parkinsons.evaluate import make_grouped_splitter
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_hgbr_learner

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
# ## Evaluate
#
# ``HistGradientBoostingRegressor`` handles ``NaN`` natively — no imputer.

# %%
learner = build_hgbr_learner()
report = skore.evaluate(learner, X, y, splitter=grouped_splitter)
report

# %% [markdown]
# ## Persist to Skore Hub

# %%
project.put("04_hgbr_grouped", report)

# %% [markdown]
# ## Metrics summary

# %%
print(report.metrics.summarize().frame().to_string())
