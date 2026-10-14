"""Fold-fitted score calibration followed by predictor-only patient pooling."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from patient_features import PatientFeatures


class CalibratedFeatures(TransformerMixin, BaseEstimator):
    """Learn measurement-minus-target bias from training patients only.

    Patient IDs never enter bias regression. transform receives no targets.
    Per-patient trajectories use all predictor visits in the current batch.
    """

    def __init__(self, alpha=10.):
        self.alpha = alpha

    @staticmethod
    def design(X, state):
        hours = X[f"time_since_intake_{state}"].clip(lower=0)
        dose = X.ledd / 100.
        table = X[["cohort", "rater_id"]].copy()
        table["dose"] = dose
        table["hours"] = hours
        table["hours_missing"] = hours.isna().astype(float)
        table["dose_missing"] = dose.isna().astype(float)
        for absorption in [.25, .5, 1.]:
            for decay in [1., 2., 4., 8., 16.]:
                table[f"concentration_{absorption:g}_{decay:g}"] = (
                    dose * (1 - np.exp(-hours / absorption)) * np.exp(-hours / decay)
                )
        return table

    def fit(self, X, y):
        if "target" in X.columns:
            raise ValueError("Target must be provided separately to fit, never as an input column.")
        y = np.asarray(y)
        self.calibrators_ = {}
        for state in ["on", "off"]:
            table = self.design(X, state)
            numeric = [col for col in table if col not in ["cohort", "rater_id"]]
            preprocessing = ColumnTransformer([
                ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                          StandardScaler()), numeric),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), ["cohort", "rater_id"]),
            ])
            calibrator = make_pipeline(preprocessing, Ridge(alpha=self.alpha, solver="lsqr"))
            mask = X[state].notna().to_numpy()
            # These targets are used only to fit an ordinary supervised regression.
            # Neither outcomes nor training-patient summaries are stored for transform.
            bias = X[state].to_numpy()[mask] - y[mask]
            calibrator.fit(table.iloc[np.flatnonzero(mask)], bias)
            self.calibrators_[state] = calibrator
        return self

    def transform(self, X):
        if "target" in X.columns:
            raise ValueError("Validation/prediction targets are prohibited.")
        builder = PatientFeatures(pool_patients=True)
        original = builder.transform(X)
        adjusted = X.copy()
        additions = {}
        for state in ["on", "off"]:
            correction = self.calibrators_[state].predict(self.design(X, state))
            additions[f"estimated_{state}_bias"] = correction
            adjusted[state] = X[state] - correction
            additions[f"corrected_{state}"] = adjusted[state]
        corrected_features = builder.transform(adjusted)
        related = [col for col in corrected_features if col in ["on_off_gap", "on", "off"]
                   or col.startswith("patient_on_") or col.startswith("patient_off_")
                   or col.startswith("on_minus_") or col.startswith("off_minus_")]
        return pd.concat([original, pd.DataFrame(additions, index=X.index),
                          corrected_features[related].add_prefix("calibrated_")], axis=1)
