"""Evaluate and publish two fixed candidates; never submit to Kaggle.

Authored by Codex. Execute with the project's virtual environment in Bob.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from skore import evaluate, CrossValidationReport, Project, login
from parkinsons.hub import load_skore_credentials
from mae_patient_blend import CachedMAEPatientBlend, InverseBlendRegressor, atomic_dump


def check_prediction_contract(model, X):
    patients = X.patient_id.unique()[:3]
    subset = X[X.patient_id.isin(patients)].copy().reset_index(drop=True)
    first = model.predict(subset)
    shuffled = subset.sample(frac=1., random_state=42)
    reordered = pd.Series(model.predict(shuffled), index=shuffled.Index).loc[subset.Index].to_numpy()
    np.testing.assert_allclose(first, reordered, rtol=1e-7, atol=1e-7)
    for patient in patients:
        mask = subset.patient_id.eq(patient).to_numpy()
        np.testing.assert_allclose(model.predict(subset.loc[mask]), first[mask], rtol=1e-7, atol=1e-7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hub", action="store_true", help="Publish every completed candidate evaluation")
    parser.add_argument("--prepare-csv", action="store_true", help="Prepare only the selected candidate's CSV")
    args = parser.parse_args()
    output = ROOT / "data" / "day2_15_mae_intercept_stable_v2"
    output.mkdir(exist_ok=True)
    data_hash = hashlib.sha256(b"day2-ridge-v1")
    for name in ["X_train", "y_train", "X_test", "sample_submission"]:
        data_hash.update((ROOT / "data" / f"{name}.csv").read_bytes())
    # The old data fingerprint is evidence for the fitted experiment-12 caches.
    original_stamp = Path(r"C:\Users\boukr\bobathon-esilv\data\day2\ridge_input_sha256.txt")
    if data_hash.hexdigest() != original_stamp.read_text().strip():
        raise ValueError("Raw CSVs changed since experiment 12; cached models cannot be reused.")
    source_hash = hashlib.sha256()
    for name in ["mae_patient_blend", "patient_features", "response_features", "calibrated_features",
                 "flexible_response", "patient_curve", "orthogonal_patient_curve", "inverse_patient_curve", "residual_trajectory"]:
        source_hash.update((ROOT / "scripts" / f"{name}.py").read_bytes())
    signature = {"data": data_hash.hexdigest(), "source": source_hash.hexdigest()}
    manifest = output / "input_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != signature:
        raise ValueError("Inputs/source changed. Use a new experiment directory.")
    manifest.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    X = pd.read_csv(ROOT / "data" / "X_train.csv")
    labels = pd.read_csv(ROOT / "data" / "y_train.csv").set_index("Index")["target"]
    y = labels.loc[X.Index].to_numpy()
    assert X.Index.is_unique and labels.index.is_unique and np.isfinite(y).all()
    if y.min() < 0:
        raise ValueError("Target is not nonnegative; clipping requires review.")
    splits = list(GroupKFold(5).split(X, y, X.patient_id))
    reports, rows = {}, []
    variants = {"14_stable_clipped_inverse_blend_grouped": "baseline_clipped",
                "15_mae_intercept_grouped": "mae_intercept"}
    for key, variant in variants.items():
        fold_states, state = [], None
        for fold, (fit, val) in enumerate(splits, 1):
            assert not set(X.iloc[fit].patient_id) & set(X.iloc[val].patient_id)
            path = output / f"{key}_fold_{fold}_report.joblib"
            print(f"{key}: fold {fold}/5", flush=True)
            if path.exists():
                single = joblib.load(path)
                print("  Cached report reused", flush=True)
            else:
                single = evaluate(CachedMAEPatientBlend(str(ROOT), variant), X, y,
                                  splitter=[(fit, val)], n_jobs=1)
                check_prediction_contract(single.reports_[0].estimator_, X.iloc[val])
                atomic_dump(single, path)
            fold_report = single.reports_[0]
            np.testing.assert_array_equal(fold_report.X_test.Index, X.iloc[val].Index)
            np.testing.assert_allclose(fold_report.y_test, y[val])
            mae, rmse = float(fold_report.metrics.mae()), float(fold_report.metrics.rmse())
            rows.append({"model": key, "fold": fold, "mae": mae, "rmse": rmse})
            print(f"  Validation MAE {mae:.6f}; RMSE {rmse:.6f}", flush=True)
            current = single.to_dict()
            if state is None:
                state = current
            fold_states.append(current["estimator_reports"][0])
        state["split_indices"] = tuple(splits)
        state["estimator_reports"] = fold_states
        report = CrossValidationReport.from_dict(state)
        atomic_dump(report, output / f"{key}_report.joblib")
        reports[key] = report
    results = pd.DataFrame(rows)
    results.to_csv(output / "fold_scores.csv", index=False)
    summary = results.groupby("model")[["mae", "rmse"]].agg(["mean", "std"])
    print(summary.to_string(), flush=True)
    baseline = results[results.model.eq("14_stable_clipped_inverse_blend_grouped")].set_index("fold")
    candidate = results[results.model.eq("15_mae_intercept_grouped")].set_index("fold")
    delta = candidate.mae - baseline.mae
    print("Candidate minus clipped baseline MAE per fold:", delta.to_dict(), flush=True)
    winner = "15_mae_intercept_grouped" if delta.mean() < -.001 and (delta < 0).sum() >= 3 else "14_stable_clipped_inverse_blend_grouped"
    print(f"Selected: {winner}. Repeated development CV; not an independent final test.", flush=True)
    if args.hub:
        cfg = load_skore_credentials(ROOT / ".skore")
        login(mode="hub")
        project = Project(name="databoys", mode="hub", workspace=cfg["workspace"])
        for key, report in reports.items():
            marker = output / f"{key}_published.json"
            identity = {"workspace": cfg["workspace"], "report_id": report.id}
            if marker.exists() and json.loads(marker.read_text()) == identity:
                print(f"{key}: already published; skipping duplicate", flush=True)
                continue
            print(f"Publishing {key}; Skore may calculate additional report diagnostics", flush=True)
            project.put(key, report)
            marker.write_text(json.dumps(identity), encoding="utf-8")
    if args.prepare_csv:
        model = CachedMAEPatientBlend(str(ROOT), variants[winner]).fit(X, y)
        test = pd.read_csv(ROOT / "data" / "X_test.csv")
        sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
        assert not set(X.patient_id) & set(test.patient_id)
        assert test.Index.is_unique and sample.Index.is_unique
        check_prediction_contract(model, test)
        prediction = model.predict(test)
        submission = sample[["Index"]].copy()
        submission["target"] = pd.Series(prediction, index=test.Index).loc[sample.Index].to_numpy()
        assert np.isfinite(submission.target).all() and len(submission) == len(test)
        destination = ROOT / "submissions" / f"{winner}.csv"
        destination.parent.mkdir(exist_ok=True)
        submission.to_csv(destination, index=False)
        saved = pd.read_csv(destination)
        np.testing.assert_array_equal(saved.Index, sample.Index)
        np.testing.assert_allclose(saved.target, submission.target, rtol=1e-12, atol=1e-12)
        atomic_dump(model, output / f"{winner}_full_model.joblib")
        print(f"Validated CSV: {destination}", flush=True)
    print(f"Results: {output}. No Kaggle submission made.", flush=True)


if __name__ == "__main__":
    main()
