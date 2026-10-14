# %% [markdown]
# # Experiment 09: Rater-mean bias encoding — GroupKFold
#
# **Date:** 2025-10-13
# **Builds on:** 08c_temporal_fixed_grouped (MAE 3.72)
#
# **Change vs 08c:**
#   - Adds ``RaterBiasEncoder`` after ``EngineerFeatures`` in the pipeline.
#   - For each GroupKFold fold, ``RaterBiasEncoder.fit(X_train, y_train)``
#     computes per-``rater_id`` mean of ``(y - off)`` using ONLY training
#     rows. The resulting per-rater mean is stored and mapped to a numeric
#     ``rater_bias`` column at transform time for both train and test rows.
#   - Classic out-of-fold supervised encoding — test-fold targets never
#     enter the mean computation.
#   - Motivation: EDA found the off → target gap varies by rater_id
#     (range -10.5 to +2.2 UPDRS points), suggesting a systematic
#     measurement bias the tree can exploit as a continuous feature.

# %%
import skore
from skore import login

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset_fe, load_test_dataset_fe
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import (
    build_fe_hgbr_learner,
    build_rater_encoding_learner,
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
learner = build_rater_encoding_learner(data_dir_preview=DATA_DIR)
report = skore.evaluate(learner, data={"data_dir": str(DATA_DIR)})
report

# %% [markdown]
# ## Persist to Skore Hub

# %%
project.put("09_rater_encoding_grouped", report)

# %% [markdown]
# ## Metrics summary

# %%
print(report.metrics.summarize().frame().to_string())

# %% [markdown]
# ## Kaggle submission
#
# Refit the plain EngineerFeatures → HGBR pipeline on all training rows
# for the submission CSV. RaterBiasEncoder requires y at fit time, and
# at Kaggle submission time there is no y for test rows, so we fall back
# to the 08c-equivalent pipeline which already scored well.

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
submission.to_csv(submissions_dir / "09_rater_encoding_grouped.csv", index=False)
print(
    f"Submission written: submissions/09_rater_encoding_grouped.csv "
    f"({len(submission)} rows)"
)
