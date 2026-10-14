# %% [markdown]
# # Experiment 12: Reproduce 42_rebalanced_inverse — team GroupKFold validation
#
# **Date:** 2025-10-13
# **Source model:** ``42_rebalanced_inverse`` from C:\Users\boukr\bobathon-esilv
#   (improvement_16), reported MAE 2.5975, RMSE 3.5060 over five development folds.
#
# ## Architecture of the original model
#
# The blend formula is: prediction = (2 × p26 + p38) / 3, where:
#   - p26 = ``35_orthogonal_blend``: blend(flexible_patient_curve, orthogonal_patient_curve)
#   - p38 = ``38_inverse_curve``: InversePatientCurve (cross-fitted ON/OFF inversion)
#
# Each component is a PatientCurveRegressor variant that:
#   1. Fits calibrated/inverted measurement features on training-fold rows only
#   2. Predicts a quadratic age trajectory per patient
#   3. Uses only predictor visits (no targets) for test patients
#
# The inverse model uses an inner 3-fold GroupKFold for cross-fitted ON/OFF inversion,
# nested entirely within each outer training fold.
#
# ## Reproduction plan
#
# **Phase 1 — Cached verification (read-only):**
#   Read the stored fold predictions from the original project, recompute the
#   blend formula, and confirm RMSE/MAE match the reported numbers exactly.
#   These are development-partition folds (35 765 rows, 4 460 patients).
#
# **Phase 2 — Team validation (fresh training):**
#   Retrain all component models on our team's GroupKFold(5) of the full
#   44 590-row training set, skore.evaluate, and push to hub.
#   The cached predictions cannot be reused: they cover only the development
#   partition, not the full dataset. Preprocessing and calibration are
#   fitted inside each training fold (no leakage).
#
# ## Comparability caveats
#
# - Original RMSE 3.506 / MAE 2.598 used **repeated development CV** on a
#   partition of 4 460 patients. That partition was reused for model selection
#   across 19 improvement rounds, so the score is optimistic relative to a
#   truly independent test.
# - Team validation uses all 5 576 patients, one fresh GroupKFold(5). The
#   holdout patients (~20 % of patients) were never used for selection in our
#   team's pipeline. These numbers are directly comparable to experiments 08c–11.
# - Data files are bit-identical between projects (SHA-256 verified).
# - Dependency versions are identical (skore 0.26.0, sklearn 1.9.1, skrub 0.11.0).

# %%
import sys, os, hashlib, json
from pathlib import Path

# Add our scripts/ dir to sys.path so the copied modules are importable
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

os.environ.setdefault("OMP_NUM_THREADS", "4")

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GroupKFold, cross_validate

import skore
from skore import login

from parkinsons import PROJECT_ROOT
from parkinsons.data import TARGET_COL
from parkinsons.hub import load_skore_credentials

# Module imports from copied scripts
from residual_trajectory import FittedBlend
from inverse_patient_curve import InversePatientCurve
from orthogonal_patient_curve import OrthogonalPatientCurve
from flexible_response import FlexiblePatientCurveRegressor

DATA_DIR = PROJECT_ROOT / "data"
THEIR_ROOT = Path("C:/Users/boukr/bobathon-esilv")
THEIR_BASE = THEIR_ROOT / "data" / "day2"

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
# ## Data — identical between projects (SHA-256 verified)

# %%
fingerprint = hashlib.sha256(b"day2-ridge-v1")
for name in ["X_train", "y_train", "X_test", "sample_submission"]:
    fingerprint.update((DATA_DIR / f"{name}.csv").read_bytes())
their_hash = (THEIR_BASE / "ridge_input_sha256.txt").read_text().strip()
our_hash = fingerprint.hexdigest()
assert our_hash == their_hash, (
    f"Data hash mismatch!\n  Ours: {our_hash}\n  Theirs: {their_hash}")
print(f"Data SHA-256 verified: {our_hash[:16]}...  OK")

train = pd.read_csv(DATA_DIR / "X_train.csv")
labels = pd.read_csv(DATA_DIR / "y_train.csv").set_index("Index")
y_full = labels.loc[train.Index, "target"].reset_index(drop=True)
print(f"Training set: {len(train)} rows, {train.patient_id.nunique()} patients")

# %% [markdown]
# ## Phase 1 — Cached verification (reproduced, not newly evaluated)
#
# Read the stored fold predictions from improvement_14, apply the rebalanced
# blend formula (2×baseline + candidate)/3, and confirm the reported metrics.
# These are development-partition rows only (35 765 rows, 4 460 patients).
# The result is a **reproduced cached result**, not an independent evaluation.

# %%
partition = np.load(THEIR_BASE / "patient_partition.npz")
dev = partition["development"]
X_dev = train.iloc[dev].reset_index(drop=True)
y_dev = y_full.iloc[dev].reset_index(drop=True)
print(f"\nPhase 1 — Development partition: {len(X_dev)} rows, {X_dev.patient_id.nunique()} patients")
assert not set(X_dev.patient_id) & set(train.iloc[partition["holdout"]].patient_id), (
    "Holdout leak detected — partition is corrupt.")

folds_dev = list(GroupKFold(5).split(X_dev, y_dev, X_dev.patient_id))

phase1_rows = []
all_true, all_pred = [], []
for i, (fit, val) in enumerate(folds_dev, 1):
    cached = np.load(THEIR_BASE / "improvement_14" / f"fold_{i}_predictions.npz")
    np.testing.assert_array_equal(cached["Index"], X_dev.iloc[val].Index,
        err_msg=f"Index mismatch on fold {i}")
    baseline  = cached["baseline"]   # = (35_orthogonal_blend) already a blend
    candidate = cached["candidate"]  # = 38_inverse_curve
    prediction = (2.0 * baseline + candidate) / 3.0   # 42_rebalanced_inverse
    true_vals  = y_dev.iloc[val].to_numpy()

    rmse = float(root_mean_squared_error(true_vals, prediction))
    mae  = float(mean_absolute_error(true_vals, prediction))
    phase1_rows.append({"fold": i, "rmse": rmse, "mae": mae, "n_rows": len(val)})
    all_true.append(true_vals)
    all_pred.append(prediction)
    print(f"  Fold {i}: RMSE={rmse:.6f}, MAE={mae:.6f}")

phase1_df = pd.DataFrame(phase1_rows)
phase1_mean_rmse = phase1_df.rmse.mean()
phase1_mean_mae  = phase1_df.mae.mean()
print(f"\n  Mean RMSE: {phase1_mean_rmse:.6f}  (reported 3.505978)")
print(f"  Mean MAE:  {phase1_mean_mae:.6f}  (reported 2.597514)")
assert abs(phase1_mean_rmse - 3.5059782190421416) < 1e-9, "RMSE mismatch — cached predictions may have changed"
assert abs(phase1_mean_mae  - 2.5975138988525694) < 1e-9, "MAE mismatch -- cached predictions may have changed"
print("  Exact match with reported metrics: OK")

# %% [markdown]
# ## Phase 2 — Team validation (fresh retraining, full dataset)
#
# Retrain all component models from scratch on the team's GroupKFold(5) of
# the full 44 590-row dataset.  Preprocessing and target-dependent calibration
# are fitted inside each training fold only.
#
# This retrains:
#   - FlexiblePatientCurveRegressor (model 25) — per fold
#   - OrthogonalPatientCurve with calibration (model 34) — per fold
#   - InversePatientCurve (model 38) — per fold (itself uses inner 3-fold GroupKFold)
# The blend is assembled as FittedBlend wrappers and evaluated per fold.
#
# NOTE: The inner cross-fitting of InversePatientCurve (3-fold GroupKFold inside
# each outer training fold) is entirely self-contained within each outer fold's
# training rows, so there is no leakage.

# %%
OUT_CACHE = DATA_DIR / "day2_12_cache"
OUT_CACHE.mkdir(exist_ok=True)

X_full = train
y_arr  = y_full.to_numpy(dtype=float)
groups = X_full.patient_id.to_numpy()
folds_full = list(GroupKFold(5).split(X_full, y_arr, groups))

print(f"\nPhase 2 — Full dataset: {len(X_full)} rows, {X_full.patient_id.nunique()} patients")
for i, (fit, val) in enumerate(folds_full, 1):
    assert not set(X_full.iloc[fit].patient_id) & set(X_full.iloc[val].patient_id), (
        f"Fold {i} patient leak")
print("  GroupKFold(5) patient-leakage check: OK")

phase2_rows = []
for i, (fit, val) in enumerate(folds_full, 1):
    print(f"\n  === Fold {i}/5 ===", flush=True)
    X_tr = X_full.iloc[fit].reset_index(drop=True)
    y_tr = y_arr[fit]
    X_va = X_full.iloc[val].reset_index(drop=True)
    y_va = y_arr[val]

    # ---- Component A: FlexiblePatientCurveRegressor (= model 25/26 equivalent) ----
    model_a_path = OUT_CACHE / f"fold_{i}_flexible_patient_curve.joblib"
    if model_a_path.exists():
        model_a = joblib.load(model_a_path)
        print(f"    Model A (flexible curve): loaded from cache", flush=True)
    else:
        print(f"    Model A (flexible curve): fitting…", flush=True)
        model_a = FlexiblePatientCurveRegressor(residual=True)
        model_a.fit(X_tr, y_tr)
        joblib.dump(model_a, model_a_path)
        print(f"    Model A: done", flush=True)

    # ---- Component B: OrthogonalPatientCurve with A's calibration (= model 34/35) ----
    model_b_path = OUT_CACHE / f"fold_{i}_orthogonal_curve.joblib"
    if model_b_path.exists():
        model_b = joblib.load(model_b_path)
        print(f"    Model B (orthogonal curve): loaded from cache", flush=True)
    else:
        print(f"    Model B (orthogonal curve): fitting…", flush=True)
        model_b = OrthogonalPatientCurve(residual=True)
        model_b.fit_with_calibration(X_tr, y_tr, model_a.features_)
        joblib.dump(model_b, model_b_path)
        print(f"    Model B: done", flush=True)

    # Blend A+B with equal weight: 35_orthogonal_blend equivalent
    blend_ab = FittedBlend(model_a, model_b, second_weight=0.5)

    # ---- Component C: InversePatientCurve (= model 38) ----
    # Uses inner 3-fold GroupKFold entirely within X_tr.
    model_c_path = OUT_CACHE / f"fold_{i}_inverse_curve.joblib"
    if model_c_path.exists():
        model_c = joblib.load(model_c_path)
        print(f"    Model C (inverse curve): loaded from cache", flush=True)
    else:
        print(f"    Model C (inverse curve): fitting with inner cross-fitting…", flush=True)
        model_c = InversePatientCurve(residual=True)
        model_c.fit(X_tr, y_tr)
        joblib.dump(model_c, model_c_path)
        print(f"    Model C: done", flush=True)

    # ---- Final blend: (2 × blend_AB + C) / 3  = 42_rebalanced_inverse ----
    final_blend = FittedBlend(blend_ab, model_c, second_weight=1.0 / 3.0)

    pred_val = final_blend.predict(X_va)
    assert np.isfinite(pred_val).all(), f"Non-finite predictions on fold {i}"

    rmse = float(root_mean_squared_error(y_va, pred_val))
    mae  = float(mean_absolute_error(y_va, pred_val))
    phase2_rows.append({"fold": i, "rmse": rmse, "mae": mae, "n_rows": len(val)})
    print(f"    Fold {i}: RMSE={rmse:.4f}, MAE={mae:.4f}", flush=True)

phase2_df = pd.DataFrame(phase2_rows)
phase2_mean_rmse = phase2_df.rmse.mean()
phase2_mean_mae  = phase2_df.mae.mean()
print(f"\nPhase 2 team-validation mean RMSE: {phase2_mean_rmse:.4f}")
print(f"Phase 2 team-validation mean MAE:  {phase2_mean_mae:.4f}")

# %% [markdown]
# ## Build final model and skore.evaluate for the hub report
#
# We build a single sklearn estimator that wraps the three-component blend
# and use skore.evaluate with the same GroupKFold(5) splitter so the report
# on the hub is comparable to experiments 08c–11.

# %%
from sklearn.base import BaseEstimator, RegressorMixin

class InverseBlendRegressor(RegressorMixin, BaseEstimator):
    """Three-component blend: (2 * FlexibleCurve/OrthogonalCurve blend + InverseCurve) / 3.

    Replicates 42_rebalanced_inverse from the original project.
    All three components are fitted inside fit(); the InversePatientCurve
    uses an inner 3-fold GroupKFold for cross-fitted ON/OFF inversion.
    """

    def fit(self, X, y):
        print("  Fitting FlexiblePatientCurveRegressor…", flush=True)
        self.model_a_ = FlexiblePatientCurveRegressor(residual=True).fit(X, y)
        print("  Fitting OrthogonalPatientCurve with calibration…", flush=True)
        self.model_b_ = OrthogonalPatientCurve(residual=True)
        self.model_b_.fit_with_calibration(X, y, self.model_a_.features_)
        print("  Fitting InversePatientCurve (inner 3-fold)…", flush=True)
        self.model_c_ = InversePatientCurve(residual=True).fit(X, y)
        self.blend_ab_ = FittedBlend(self.model_a_, self.model_b_, second_weight=0.5)
        self.blend_final_ = FittedBlend(self.blend_ab_, self.model_c_, second_weight=1.0 / 3.0)
        return self

    def predict(self, X):
        return self.blend_final_.predict(X)


# skore.evaluate on the full dataset with GroupKFold(5)
# This retrains 5 × 3 component models (15 total fits).
# We check whether we already have a cached skore report.
report_path = DATA_DIR / "day2_12_cache" / "skore_report.joblib"
if report_path.exists():
    print("Loading cached skore report…")
    report = joblib.load(report_path)
else:
    print("Running skore.evaluate (5 × 3 component model fits)…", flush=True)
    cv_splitter = list(GroupKFold(5).split(X_full, y_arr, groups))
    report = skore.evaluate(
        InverseBlendRegressor(),
        X_full,
        y_full,
        splitter=cv_splitter,
    )
    joblib.dump(report, report_path)

# %% [markdown]
# ## Persist to Skore Hub

# %%
project.put("12_inverse_blend_grouped", report)

# %% [markdown]
# ## Metrics summary

# %%
print("\n=== EXPERIMENT 12 RESULTS ===\n")
print("Phase 1: CACHED reproduction of original development-fold metrics")
print("  (35 765 dev rows, 4 460 patients; original selection partition)")
print("  These are REPRODUCED CACHED RESULTS, not newly evaluated.")
print(f"  Fold scores (RMSE): " + "  ".join(f"{r:.4f}" for r in phase1_df.rmse))
print(f"  Mean RMSE: {phase1_mean_rmse:.6f}  (original reported: 3.505978)")
print(f"  Mean MAE:  {phase1_mean_mae:.6f}  (original reported: 2.597514)")

print()
print("Phase 2: TEAM VALIDATION — fresh retraining, full dataset")
print("  (44 590 rows, 5 576 patients; same GroupKFold(5) as exps 08c–11)")
print("  These are NEWLY EVALUATED results.")
print(f"  Fold scores (RMSE): " + "  ".join(f"{r:.4f}" for r in phase2_df.rmse))
print(f"  Fold scores (MAE):  " + "  ".join(f"{m:.4f}" for m in phase2_df.mae))
print(f"  Mean RMSE: {phase2_mean_rmse:.6f}")
print(f"  Mean MAE:  {phase2_mean_mae:.6f}")

print()
print("Comparison vs experiment 11 (team validation, same GroupKFold setup):")
print("  11_hgbr_tuned_grouped: MAE=3.617, RMSE=4.770")
print(f"  12_inverse_blend:      MAE={phase2_mean_mae:.3f}, RMSE={phase2_mean_rmse:.3f}")
mae_summary = report.metrics.summarize().frame()
print()
print("skore.evaluate report metrics:")
print(mae_summary.to_string())
