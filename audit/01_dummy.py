# %% [markdown]
# # Audit — 01_dummy: DummyRegressor baseline
#
# Read-only review of the stored report. Checks and metrics only —
# no `skore.evaluate()` or `project.put()` in this file.

# %%
import skore
from skore import login

from parkinsons.hub import load_skore_credentials

# %% [markdown]
# ## Open the project
#
# Same hub project the experiment wrote to. Credentials come from
# the nearest ``.skore`` file via ``load_skore_credentials()``.

# %%
cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(
    name="databoys",
    mode="hub",
    workspace=cfg["workspace"],
)
project

# %% [markdown]
# ## List the available reports

# %%
summary = project.summarize()
summary

# %% [markdown]
# ## Load the report
#
# URL printed by ``project.put()`` was:
#   https://skore.probabl.ai/databoys/databoys/estimators/63811
# → type-plural ``estimators`` → id ``skore:report:estimator:63811``

# %%
REPORT_ID = "skore:report:estimator:63811"

report = project.get(REPORT_ID)
report

# %% [markdown]
# ## Checks summary
#
# Each row carries a ``severity`` (passed / issue / tip) and a
# ``documentation_url`` pointing at the mitigation the docs recommend.

# %%
report.checks.summarize().frame()

# %% [markdown]
# ## Metrics summary
#
# Regression defaults: RMSE, MAE, R² plus fit/predict timings.

# %%
report.metrics.summarize().frame()

# %% [markdown]
# ## End of audit
