"""New forward-measurement strategy. Local reports and CSV; no Hub upload.

Authored by Codex for user execution in Bob. RMSE is the guide's selection
metric; MAE is also reported. Repeated development CV is not a final test.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
sys.path.insert(0, str(ROOT/"src"))
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from skore import evaluate, CrossValidationReport
from mae_patient_blend import atomic_dump
from forward_patient_posterior import ForwardPatientPosterior


def check_contract(model, X):
    subset = X[X.patient_id.isin(X.patient_id.unique()[:3])].reset_index(drop=True)
    prediction = model.predict(subset)
    shuffled = subset.sample(frac=1., random_state=42)
    ordered = pd.Series(model.predict(shuffled), index=shuffled.Index).loc[subset.Index].to_numpy()
    np.testing.assert_allclose(prediction, ordered, rtol=1e-7, atol=1e-7)
    for patient in subset.patient_id.unique():
        mask = subset.patient_id.eq(patient).to_numpy()
        np.testing.assert_allclose(model.predict(subset.loc[mask]), prediction[mask], rtol=1e-7, atol=1e-7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-csv", action="store_true")
    parser.add_argument("--screen", action="store_true", help="One fold only; cannot generate a selected submission")
    args = parser.parse_args()
    if args.screen and args.prepare_csv:
        parser.error("A screen is not full validation; remove --screen to prepare a CSV.")
    output = ROOT/"data"/"day2_16_forward_posterior_v1"
    output.mkdir(exist_ok=True)
    digest = hashlib.sha256(b"day2-ridge-v1")
    for name in ["X_train", "y_train", "X_test", "sample_submission"]:
        digest.update((ROOT/"data"/f"{name}.csv").read_bytes())
    expected = Path(r"C:\Users\boukr\bobathon-esilv\data\day2\ridge_input_sha256.txt").read_text().strip()
    if digest.hexdigest() != expected:
        raise ValueError("Raw data changed; experiment-12 caches are incompatible.")
    source = hashlib.sha256()
    for name in [ROOT/"scripts"/"forward_patient_posterior.py", ROOT/"scripts"/"mae_patient_blend.py", Path(__file__)]:
        source.update(name.read_bytes())
    identity = {"data": digest.hexdigest(), "source": source.hexdigest()}
    manifest = output/"manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError("Experiment code/data changed. Preserve the cache and use a new experiment version.")
    manifest.write_text(json.dumps(identity, indent=2), encoding="utf-8")
    X = pd.read_csv(ROOT/"data"/"X_train.csv")
    labels = pd.read_csv(ROOT/"data"/"y_train.csv").set_index("Index").target
    assert X.Index.is_unique and labels.index.is_unique
    y = labels.loc[X.Index].to_numpy()
    splits = list(GroupKFold(5).split(X, y, X.patient_id))
    baseline_path = ROOT/"data"/"day2_15_mae_intercept_stable_v2"/"14_stable_clipped_inverse_blend_grouped_report.joblib"
    if not baseline_path.exists():
        raise FileNotFoundError("Finish experiment 15's local evaluation first; its stable baseline report is required.")
    baseline = joblib.load(baseline_path)
    np.testing.assert_array_equal(baseline.X.Index, X.Index)
    np.testing.assert_allclose(baseline.y, y)
    rows, combined = [], {}
    number = 1 if args.screen else 5
    for fold, (fit, val) in enumerate(splits[:number], 1):
        assert not set(X.iloc[fit].patient_id) & set(X.iloc[val].patient_id)
        base_report = baseline.reports_[fold-1]
        np.testing.assert_array_equal(base_report.X_test.Index, X.iloc[val].Index)
        np.testing.assert_allclose(base_report.y_test, y[val])
        rows.append({"model":"14_stable_baseline", "fold":fold,
                     "mae":float(base_report.metrics.mae()), "rmse":float(base_report.metrics.rmse())})
    variants = {"16_forward_posterior":1., "17_forward_posterior_half":.5}
    report_states = {key: [] for key in variants}
    template_states = {}
    for fold, (fit, val) in enumerate(splits[:number], 1):
        first_fold_scores = []
        for key, weight in variants.items():
            print(f"\n{key}: patient fold {fold}/{number}", flush=True)
            path = output/f"{key}_fold_{fold}_report.joblib"
            if path.exists():
                single = joblib.load(path)
                print("  Saved evaluation reused", flush=True)
            else:
                single = evaluate(ForwardPatientPosterior(str(ROOT), mix_weight=weight), X, y,
                                  splitter=[(fit,val)], n_jobs=1)
                check_contract(single.reports_[0].estimator_, X.iloc[val])
                atomic_dump(single, path)
            r = single.reports_[0]
            np.testing.assert_array_equal(r.X_test.Index, X.iloc[val].Index)
            np.testing.assert_allclose(r.y_test, y[val])
            mae, rmse = float(r.metrics.mae()), float(r.metrics.rmse())
            rows.append({"model":key,"fold":fold,"mae":mae,"rmse":rmse})
            print(f"  VALIDATION RMSE {rmse:.6f}; MAE {mae:.6f}", flush=True)
            current = single.to_dict()
            if key not in template_states:
                template_states[key] = current
            report_states[key].append(current["estimator_reports"][0])
            first_fold_scores.append(rmse)
        if fold == 1 and not args.screen:
            base_first = next(row["rmse"] for row in rows if row["model"] == "14_stable_baseline" and row["fold"] == 1)
            if min(first_fold_scores) > base_first + .03:
                pd.DataFrame(rows).to_csv(output/"early_screen_scores.csv", index=False)
                print("Both fixed candidates are clearly worse on fold 1. Stopping the screen to avoid four more calibration fits.")
                print("No winning CSV or Hub upload. Keep the existing model 42; screen results are cached.")
                return
    for key in variants:
        state = template_states[key]
        state["split_indices"] = tuple(splits[:number])
        state["estimator_reports"] = report_states[key]
        combined[key] = CrossValidationReport.from_dict(state)
        atomic_dump(combined[key], output/f"{key}_report.joblib")
    frame = pd.DataFrame(rows)
    frame.to_csv(output/("screen_scores.csv" if args.screen else "fold_scores.csv"), index=False)
    means = frame.groupby("model")[["rmse","mae"]].mean()
    print("\n"+means.to_string(), flush=True)
    if args.screen:
        print("Screen only; no winner, CSV or Hub upload. Completed calibration is cached.")
        return
    pivot = frame.pivot(index="fold", columns="model", values="rmse")
    baseline_score = float(means.loc["14_stable_baseline","rmse"])
    eligible = [key for key in variants if means.loc[key,"rmse"] < baseline_score-.005
                and (pivot[key] < pivot["14_stable_baseline"]).sum() >= 3]
    if not eligible:
        print("Neither new candidate improved grouped RMSE reliably. Preserve model 42 and the stable baseline.")
        print("No new CSV generated; no Hub upload.")
        return
    winner = min(eligible, key=lambda key: means.loc[key,"rmse"])
    print(f"Selected {winner}; compare with baseline above. No guaranteed Kaggle score.", flush=True)
    if args.prepare_csv:
        model = ForwardPatientPosterior(str(ROOT), mix_weight=variants[winner]).fit(X,y)
        test = pd.read_csv(ROOT/"data"/"X_test.csv")
        sample = pd.read_csv(ROOT/"data"/"sample_submission.csv")
        assert test.Index.is_unique and sample.Index.is_unique and set(test.Index)==set(sample.Index)
        assert not set(X.patient_id)&set(test.patient_id)
        check_contract(model,test)
        prediction=model.predict(test)
        submission=sample[["Index"]].copy()
        submission["target"]=pd.Series(prediction,index=test.Index).loc[sample.Index].to_numpy()
        assert np.isfinite(submission.target).all()
        destination=ROOT/"submissions"/f"{winner}.csv"
        submission.to_csv(destination,index=False)
        saved=pd.read_csv(destination)
        np.testing.assert_array_equal(saved.Index,sample.Index)
        np.testing.assert_allclose(saved.target,submission.target,rtol=1e-12,atol=1e-12)
        atomic_dump(model,output/f"{winner}_full_model.joblib")
        print(f"Validated CSV: {destination}",flush=True)
    print(f"Local reports: {output}. No Hub upload or Kaggle submission.",flush=True)


if __name__=="__main__":
    main()
