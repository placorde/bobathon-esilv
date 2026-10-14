"""Predictor-only visit features; never uses target or identifier as model input."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PatientFeatures(TransformerMixin, BaseEstimator):
    """Build features from all available visits in the input prediction batch.

    This is offline prediction for unseen patients, not future-visit forecasting.
    Grouped CV must pass complete, disjoint patients to fit and validation.
    No training-patient table or outcome is retained by this transformer.
    """

    def __init__(self, pool_patients=True):
        self.pool_patients = pool_patients

    def fit(self, X, y=None):
        self._check(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    @staticmethod
    def _check(X):
        if "target" in X.columns:
            raise ValueError("Target must never enter the feature builder.")
        if "patient_id" not in X.columns:
            raise ValueError("patient_id is needed for grouping, then removed.")

    def transform(self, X):
        self._check(X)
        raw = X.copy()
        out = raw.drop(columns=["Index", "patient_id"], errors="ignore")
        duration = raw.age - raw.age_at_diagnosis
        out["years_since_diagnosis"] = duration
        out["on_off_gap"] = raw.off - raw.on
        for col in ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off"]:
            out[f"{col}_missing"] = raw[col].isna().astype(float)
        # Smooth timing/dose interactions provide candidate decay scales, not fitted constants.
        for state in ["on", "off"]:
            hours = raw[f"time_since_intake_{state}"].clip(lower=0)
            for scale in [1., 2., 4., 8.]:
                out[f"dose_remaining_{state}_{scale:g}h"] = raw.ledd * np.exp(-hours / scale)
        if self.pool_patients:
            group = raw.groupby("patient_id", sort=False)
            out["patient_visit_count"] = group.age.transform("size")
            age_mean = group.age.transform("mean")
            out["patient_age_offset"] = raw.age - age_mean
            out["patient_age_span"] = group.age.transform("max") - group.age.transform("min")
            for col in ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off"]:
                values = group[col]
                for statistic in ["mean", "median", "std", "min", "max", "count"]:
                    out[f"patient_{col}_{statistic}"] = values.transform(statistic)
                out[f"{col}_minus_patient_mean"] = raw[col] - out[f"patient_{col}_mean"]
            # Fit each patient's ON/OFF trajectory using observed scores and age only.
            # Validation/test trajectories never use targets or training-patient visits.
            ids = raw.patient_id
            for col in ["on", "off"]:
                available = raw[col].notna() & raw.age.notna()
                age_obs = raw.age.where(available)
                score_obs = raw[col].where(available)
                age_center = age_obs.groupby(ids, sort=False).transform("mean")
                score_center = score_obs.groupby(ids, sort=False).transform("mean")
                dx = age_obs - age_center
                dy = score_obs - score_center
                covariance = (dx * dy).groupby(ids, sort=False).transform("sum")
                variance = dx.pow(2).groupby(ids, sort=False).transform("sum")
                slope = covariance / variance.where(variance > 1e-8)
                out[f"patient_{col}_slope"] = slope
                # With only one observed score, use a flat trajectory; keep missing if none.
                trajectory = score_center + slope.fillna(0) * (raw.age - age_center)
                out[f"patient_{col}_trajectory"] = trajectory
                out[f"{col}_minus_trajectory"] = raw[col] - trajectory
        return out.replace([np.inf, -np.inf], np.nan)
