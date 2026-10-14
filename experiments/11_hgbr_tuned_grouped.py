# %% [markdown]
# # Experiment 11: HGBR hyperparameter tuning — GroupKFold
#
# **Date:** 2025-10-13
# **Builds on:** 10_recency_temporal_grouped (MAE 3.65, current best)
#
# **Goal:** Find the best combination of:
#   - ``max_iter``:    200, 500, 1000
#   - ``max_depth``:   None, 4, 6, 8
#   - ``learning_rate``: 0.01, 0.1, 0.3
#
# Grid size: 3 × 4 × 3 = 36 combinations.
# CV strategy: ``GroupKFold(n_splits=5)`` on ``patient_id`` — same as all
# prior experiments, so results are directly comparable.
#
# **Overfitting check:** compare train-fold MAE vs test-fold MAE for the
# winner. A large gap (test >> train) signals overfitting; prefer a slightly
# worse test-MAE combination with a tighter gap if needed.

# %%
import itertools

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.pipeline import Pipeline

import skore
from skore import login

from parkinsons import PROJECT_ROOT
from parkinsons.data import TARGET_COL, load_raw_fe
from parkinsons.features import EngineerFeatures, RecencyTemporalFeatures
from parkinsons.hub import load_skore_credentials
from parkinsons.pipeline import build_hgbr_tuned_learner, build_fe_hgbr_learner
from parkinsons.data import load_dataset_fe, load_test_dataset_fe

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
# ## Load data for grid search
#
# Grid search runs against the flat sklearn API (Pipeline + GridSearchCV)
# rather than the DataOps graph, so we load the raw frame directly and
# pass ``patient_id`` as the groups array.

# %%
raw = load_raw_fe(str(DATA_DIR))
X_gs = raw.drop(columns=[TARGET_COL])   # includes patient_id for temporal step
y_gs = raw[TARGET_COL].to_numpy(dtype=float)
groups_gs = raw["patient_id"].to_numpy()

print(f"Training rows: {len(raw)}, patients: {raw['patient_id'].nunique()}")

# %% [markdown]
# ## Grid search
#
# The sklearn Pipeline wraps the same two stateless transformers used in
# experiment 10, then HGBR. GroupKFold(5) is used so every split is
# patient-grouped, matching the DataOps CV exactly.
#
# ``return_train_score=True`` lets us compute the train/test gap per combo.

# %%
param_grid = list(itertools.product(
    [200, 500, 1000],         # max_iter
    [None, 4, 6, 8],          # max_depth
    [0.01, 0.1, 0.3],         # learning_rate
))

cv = GroupKFold(n_splits=5)

records = []
total = len(param_grid)
print(f"Running {total} combinations …")

for idx, (max_iter, max_depth, lr) in enumerate(param_grid, 1):
    pipe = Pipeline([
        ("temporal", RecencyTemporalFeatures()),
        ("fe",       EngineerFeatures(clip_off=True)),
        ("hgbr",     HistGradientBoostingRegressor(
            max_iter=max_iter,
            max_depth=max_depth,
            learning_rate=lr,
            random_state=0,
        )),
    ])
    cv_res = cross_validate(
        pipe,
        X_gs,
        y_gs,
        cv=cv,
        groups=groups_gs,
        scoring="neg_mean_absolute_error",
        return_train_score=True,
        n_jobs=-1,
    )
    test_mae  = -cv_res["test_score"].mean()
    test_std  = cv_res["test_score"].std()
    train_mae = -cv_res["train_score"].mean()
    gap       = test_mae - train_mae
    records.append({
        "max_iter":      max_iter,
        "max_depth":     str(max_depth),   # str so None prints cleanly
        "learning_rate": lr,
        "test_mae":      round(test_mae,  4),
        "test_std":      round(test_std,  4),
        "train_mae":     round(train_mae, 4),
        "gap":           round(gap,       4),
    })
    print(f"  [{idx:02d}/{total}] iter={max_iter:4d} depth={str(max_depth):4s} "
          f"lr={lr:.2f}  test_mae={test_mae:.4f} ± {test_std:.4f}  "
          f"train_mae={train_mae:.4f}  gap={gap:.4f}")

results_df = pd.DataFrame(records).sort_values("test_mae")
print("\n--- Full grid results (sorted by test_mae) ---")
print(results_df.to_string(index=False))

# %% [markdown]
# ## Select best combination
#
# Primary criterion: lowest ``test_mae``.
# Tie-break / overfitting guard: if the top candidate has a gap > 0.5
# and the second candidate has gap ≤ 0.3 with test_mae within 0.02,
# prefer the second (less overfit).  In practice we just inspect the table
# and record the chosen params.

# %%
best = results_df.iloc[0]
best_max_iter    = int(best["max_iter"])
best_max_depth   = None if best["max_depth"] == "None" else int(best["max_depth"])
best_lr          = float(best["learning_rate"])

print(f"\nBest params:  max_iter={best_max_iter}, max_depth={best_max_depth}, "
      f"learning_rate={best_lr}")
print(f"Test MAE:     {best['test_mae']:.4f} ± {best['test_std']:.4f}")
print(f"Train MAE:    {best['train_mae']:.4f}")
print(f"Train/test gap: {best['gap']:.4f}")

# %% [markdown]
# ## Final evaluate with best params via DataOps graph
#
# Re-evaluate using ``skore.evaluate`` (same GroupKFold DataOps graph as
# exp 10) to get the canonical Skore report with the tuned params.

# %%
learner = build_hgbr_tuned_learner(
    max_iter=best_max_iter,
    max_depth=best_max_depth,
    learning_rate=best_lr,
    data_dir_preview=DATA_DIR,
)
report = skore.evaluate(learner, data={"data_dir": str(DATA_DIR)})
report

# %% [markdown]
# ## Persist to Skore Hub

# %%
project.put("11_hgbr_tuned_grouped", report)

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
submission.to_csv(submissions_dir / "11_hgbr_tuned_grouped.csv", index=False)
print(
    f"Submission written: submissions/11_hgbr_tuned_grouped.csv "
    f"({len(submission)} rows)"
)
