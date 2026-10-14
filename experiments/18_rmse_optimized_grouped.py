"""Experiment 18: RMSE-optimized OLS blend of the three patient-curve components.

Builds on experiment 14 (best model so far: clipped equal-weight blend).
Replaces the fixed (1/3, 1/3, 1/3) combination rule with per-fold OLS weights
fit on the training fold's component predictions to minimise squared error
directly (the loss RMSE rewards).

Architecture (unchanged from exp 14):
  A = FlexiblePatientCurveRegressor  (residual=True)
  B = OrthogonalPatientCurve         (residual=True, reuses A calibration)
  C = InversePatientCurve            (residual=True, inner 3-fold cross-fitting)
  Final = clip(w_A*A + w_B*B + w_C*C, 0)

OLS weight fitting:
  At fit time, generate OOF predictions for A/B/C on the training fold via an
  inner 4-fold GroupKFold (patient-disjoint), then solve the unconstrained OLS
  system.  No intercept: the equal-weight sum already centres near y, and the
  per-fold OLS weights across the 5 outer folds are very stable
  (A≈0.16, B≈0.46, C≈0.37) with near-zero intercept from the OOF analysis.

Authored by user/Bob. No Kaggle upload.
"""
from pathlib import Path
import os
import sys
import argparse
import hashlib
import json

os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import skore
from skore import login
from parkinsons.hub import load_skore_credentials

from flexible_response import FlexiblePatientCurveRegressor
from orthogonal_patient_curve import OrthogonalPatientCurve
from inverse_patient_curve import InversePatientCurve
from mae_patient_blend import atomic_dump


CACHE12 = ROOT / "data" / "day2_12_cache"
OUT     = ROOT / "data" / "day2_18_rmse_ols"


def training_signature(X, y):
    ordered = pd.DataFrame(
        {"Index": X.Index.to_numpy(), "target": np.asarray(y, dtype=float)}
    ).sort_values("Index")
    return hashlib.sha256(
        pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes()
    ).hexdigest()


def load_exp12_fold_models(prefix: str):
    """Load the three cached components for a given fold prefix."""
    a = joblib.load(CACHE12 / f"{prefix}_flexible_patient_curve.joblib")
    b = joblib.load(CACHE12 / f"{prefix}_orthogonal_curve.joblib")
    c = joblib.load(CACHE12 / f"{prefix}_inverse_curve.joblib")
    return a, b, c


class OLSBlendRegressor(RegressorMixin, BaseEstimator):
    """Three-component OLS blend with non-negative clipping.

    Fit: generate OOF component predictions on the training fold via an inner
    GroupKFold, solve OLS, store weights.  Reuse cached exp-12 component models
    when the training rows match an exact exp-12 fold.

    Predict: apply stored OLS weights, clip to zero.
    """

    def __init__(self, inner_splits: int = 4):
        self.inner_splits = inner_splits

    def fit(self, X, y):
        if "target" in X.columns:
            raise ValueError("Targets must not enter the predictor table.")

        full_X   = pd.read_csv(ROOT / "data" / "X_train.csv")
        full_y   = pd.read_csv(ROOT / "data" / "y_train.csv").set_index("Index")["target"]
        full_y   = full_y.loc[full_X.Index].to_numpy()
        sig      = training_signature(X, y)
        full_sig = training_signature(full_X, full_y)

        if sig == full_sig:
            prefix = "full_train"
        else:
            prefix = None
            for fold, (fit_idx, _) in enumerate(
                GroupKFold(5).split(full_X, full_y, full_X.patient_id), 1
            ):
                if sig == training_signature(full_X.iloc[fit_idx], full_y[fit_idx]):
                    prefix = f"fold_{fold}"
                    break
            if prefix is None:
                raise ValueError("Training rows do not match any exp-12 fold.")

        print(f"  [{prefix}] loading A, B, C from exp-12 cache", flush=True)
        model_a, model_b, model_c = load_exp12_fold_models(prefix)

        # ---- OLS weight estimation via inner GroupKFold ----
        y_arr  = np.asarray(y, dtype=float)
        inner  = list(
            GroupKFold(self.inner_splits).split(X, y_arr, X.patient_id)
        )
        oof_a = np.full(len(X), np.nan)
        oof_b = np.full(len(X), np.nan)
        oof_c = np.full(len(X), np.nan)

        for k, (ifit, ival) in enumerate(inner, 1):
            print(f"    inner fold {k}/{self.inner_splits}: fitting A, B, C for OLS weight estimation", flush=True)
            X_itr = X.iloc[ifit].reset_index(drop=True)
            y_itr = y_arr[ifit]
            X_iva = X.iloc[ival].reset_index(drop=True)

            # Fit fresh inner component models (no cache: these are sub-folds)
            a_inner = FlexiblePatientCurveRegressor(residual=True).fit(X_itr, y_itr)
            b_inner = OrthogonalPatientCurve(residual=True)
            b_inner.fit_with_calibration(X_itr, y_itr, a_inner.features_)
            c_inner = InversePatientCurve(residual=True).fit(X_itr, y_itr)

            oof_a[ival] = a_inner.predict(X_iva)
            oof_b[ival] = b_inner.predict(X_iva)
            oof_c[ival] = c_inner.predict(X_iva)

        assert not np.any(np.isnan(oof_a)), "Missing inner OOF predictions"

        # Solve unconstrained OLS: minimizes sum of squared errors directly
        stack = np.column_stack([oof_a, oof_b, oof_c])
        w, _, _, _ = np.linalg.lstsq(stack, y_arr, rcond=None)
        self.weights_ = w
        print(
            f"  OLS weights: A={w[0]:.4f}  B={w[1]:.4f}  C={w[2]:.4f}  sum={w.sum():.4f}",
            flush=True,
        )

        # Store outer (full-fold) component models for predict
        self.model_a_ = model_a
        self.model_b_ = model_b
        self.model_c_ = model_c
        return self

    def predict(self, X):
        if "target" in X.columns:
            raise ValueError("Targets cannot enter prediction.")
        pa = self.model_a_.predict(X)
        pb = self.model_b_.predict(X)
        pc = self.model_c_.predict(X)
        pred = pa * self.weights_[0] + pb * self.weights_[1] + pc * self.weights_[2]
        return np.maximum(pred, 0.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-csv", action="store_true")
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)

    # Verify data integrity
    data_hash = hashlib.sha256(b"day2-ridge-v1")
    for name in ["X_train", "y_train", "X_test", "sample_submission"]:
        data_hash.update((ROOT / "data" / f"{name}.csv").read_bytes())
    original = Path(r"C:\Users\boukr\bobathon-esilv\data\day2\ridge_input_sha256.txt")
    if data_hash.hexdigest() != original.read_text().strip():
        raise ValueError("Raw CSVs changed; cached exp-12 models cannot be reused.")

    X = pd.read_csv(ROOT / "data" / "X_train.csv")
    labels = pd.read_csv(ROOT / "data" / "y_train.csv").set_index("Index")["target"]
    y = labels.loc[X.Index].to_numpy()
    assert X.Index.is_unique and np.isfinite(y).all()
    if y.min() < 0:
        raise ValueError("Target contains negatives; review clipping.")

    splits = list(GroupKFold(5).split(X, y, X.patient_id))

    # Hub login
    cfg = load_skore_credentials(ROOT / ".skore")
    login(mode="hub")
    project = skore.Project(
        name="databoys", mode="hub", workspace=cfg["workspace"]
    )

    # ---- Per-fold evaluation (cache each fold report) ----
    fold_states, rows = [], None
    for fold, (fit, val) in enumerate(splits, 1):
        assert not set(X.iloc[fit].patient_id) & set(X.iloc[val].patient_id)
        path = OUT / f"18_rmse_optimized_grouped_fold_{fold}_report.joblib"
        print(f"\n18_rmse_optimized_grouped: fold {fold}/5", flush=True)
        if path.exists():
            single = joblib.load(path)
            print("  Cached report reused", flush=True)
        else:
            single = skore.evaluate(
                OLSBlendRegressor(inner_splits=4),
                X, y,
                splitter=[(fit, val)],
                n_jobs=1,
            )
            atomic_dump(single, path)
        fold_rpt = single.reports_[0]
        np.testing.assert_array_equal(fold_rpt.X_test.Index, X.iloc[val].Index)
        np.testing.assert_allclose(fold_rpt.y_test, y[val])
        mae  = float(fold_rpt.metrics.mae())
        rmse = float(fold_rpt.metrics.rmse())
        rows_row = {"fold": fold, "mae": mae, "rmse": rmse,
                    "weights": single.reports_[0].estimator_.weights_.tolist()}
        print(f"  Fold {fold}: MAE={mae:.6f}  RMSE={rmse:.6f}", flush=True)
        if fold_states:
            state = fold_states[0]["base_state"]
        else:
            state = single.to_dict()
            state = {"base_state": state}
        fold_states.append(
            {"fold": fold, "mae": mae, "rmse": rmse,
             "report_dict": single.to_dict()}
        )

    fold_rmses = [fs["rmse"] for fs in fold_states]
    fold_maes  = [fs["mae"]  for fs in fold_states]
    print(f"\n=== 18_rmse_optimized_grouped ===")
    print(f"  Fold RMSEs: {[round(r,4) for r in fold_rmses]}")
    print(f"  Mean RMSE:  {np.mean(fold_rmses):.6f}  Std: {np.std(fold_rmses):.6f}")
    print(f"  Mean MAE:   {np.mean(fold_maes):.6f}   Std: {np.std(fold_maes):.6f}")
    print()
    print("vs 14_stable_clipped: RMSE=3.432985  MAE=2.538434")

    # ---- Assemble full CV report from individual fold reports ----
    # Reload all fold reports and merge into a full CrossValidationReport
    from skore import CrossValidationReport
    base = joblib.load(OUT / "18_rmse_optimized_grouped_fold_1_report.joblib")
    full_state = base.to_dict()
    all_fold_dicts = []
    for fold in range(1, 6):
        r = joblib.load(OUT / f"18_rmse_optimized_grouped_fold_{fold}_report.joblib")
        all_fold_dicts.append(r.to_dict()["estimator_reports"][0])
    full_state["split_indices"] = tuple(splits)
    full_state["estimator_reports"] = all_fold_dicts
    full_report = CrossValidationReport.from_dict(full_state)
    atomic_dump(full_report, OUT / "18_rmse_optimized_grouped_report.joblib")

    # ---- Publish to hub ----
    marker = OUT / "18_rmse_optimized_grouped_published.json"
    if not marker.exists():
        print("\nPublishing 18_rmse_optimized_grouped to hub...", flush=True)
        project.put("18_rmse_optimized_grouped", full_report)
        marker.write_text(
            json.dumps({"workspace": cfg["workspace"], "report_id": str(full_report.id)}),
            encoding="utf-8",
        )
    else:
        print("\nAlready published; skipping duplicate.", flush=True)
        print(json.loads(marker.read_text()))

    # ---- Submission CSV ----
    if args.prepare_csv:
        print("\nFitting full-training OLS blend...", flush=True)
        full_model = OLSBlendRegressor(inner_splits=4).fit(X, y)
        test  = pd.read_csv(ROOT / "data" / "X_test.csv")
        sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
        assert not set(X.patient_id) & set(test.patient_id)
        preds = full_model.predict(test)
        submission = sample[["Index"]].copy()
        submission["target"] = pd.Series(preds, index=test.Index).loc[sample.Index].to_numpy()
        assert np.isfinite(submission.target).all() and len(submission) == len(test)
        dest = ROOT / "submissions" / "18_rmse_optimized_grouped.csv"
        submission.to_csv(dest, index=False)
        saved = pd.read_csv(dest)
        np.testing.assert_array_equal(saved.Index, sample.Index)
        np.testing.assert_allclose(saved.target, submission.target, rtol=1e-12, atol=1e-12)
        atomic_dump(full_model, OUT / "18_rmse_optimized_grouped_full_model.joblib")
        print(f"Submission CSV: {dest}", flush=True)

    print(f"\nResults directory: {OUT}", flush=True)
    print("No Kaggle submission made.", flush=True)


if __name__ == "__main__":
    main()
