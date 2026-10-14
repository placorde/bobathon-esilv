"""Predict a complete quadratic score trajectory for each unseen patient."""
import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from response_features import ResponseFeatures


def fit_curve(age, values):
    """Coefficients in centered age/5 coordinates; reduce degree for sparse input."""
    t = (np.asarray(age) - np.mean(age)) / 5.
    values = np.asarray(values)
    valid = np.isfinite(t) & np.isfinite(values)
    coefficients = np.zeros(3)
    if not valid.any():
        return np.full(3, np.nan)
    degree = min(2, len(np.unique(t[valid])) - 1)
    design = np.column_stack([t[valid] ** power for power in range(degree + 1)])
    coefficients[:degree + 1] = np.linalg.lstsq(design, values[valid], rcond=None)[0]
    return coefficients


class PatientCurveRegressor(RegressorMixin, BaseEstimator):
    """Training labels fit curve coefficients; prediction uses observed visits only.

    One row per training patient enters each coefficient model. Neither patient
    identifiers nor target-derived statistics enter the model's predictor table.
    Prediction requires all available visits for each test patient in one batch.
    """

    def __init__(self, residual=False):
        self.residual = residual

    def summarize(self, X, features):
        groups = X.patient_id
        table = features.select_dtypes(include="number").groupby(groups, sort=False).mean()
        for col in ["cohort", "gene"]:
            table[col] = X.groupby("patient_id", sort=False)[col].first().fillna("missing").astype(str)
        selected = ["on", "off", "ledd", "time_since_intake_on", "time_since_intake_off",
                    "response_on_corrected", "response_off_corrected",
                    "pooled_quadratic_score", "pooled_linear_score"]
        for col in selected:
            grouped = features[col].groupby(groups, sort=False)
            for statistic in ["min", "max", "std", "count"]:
                table[f"{col}_{statistic}_visits"] = grouped.agg(statistic)
        prior = []
        shape_rows = []
        for patient, pos in X.groupby("patient_id", sort=False).indices.items():
            ages = X.age.to_numpy()[pos]
            prior.append(fit_curve(ages, features.pooled_quadratic_score.to_numpy()[pos]))
            shapes = {}
            for col in ["on", "off", "ledd", "response_on_corrected", "response_off_corrected"]:
                coefficients = fit_curve(ages, features[col].to_numpy()[pos])
                for degree, value in enumerate(coefficients):
                    shapes[f"{col}_curve_coefficient_{degree}"] = value
            shape_rows.append(shapes)
        table = pd.concat([table, pd.DataFrame(shape_rows, index=table.index)], axis=1)
        table = table.replace([np.inf, -np.inf], np.nan)
        return table, np.asarray(prior)

    def fit(self, X, y):
        if "target" in X.columns:
            raise ValueError("Target cannot enter the predictor table.")
        self.features_ = ResponseFeatures(alpha=30., use_trajectory=True)
        features = self.features_.fit_transform(X, y)
        table, prior = self.summarize(X, features)
        coefficients, basis_weights = [], []
        target = np.asarray(y)
        for patient, pos in X.groupby("patient_id", sort=False).indices.items():
            coefficients.append(fit_curve(X.age.to_numpy()[pos], target[pos]))
            ages = X.age.to_numpy()[pos]
            t = (ages - np.mean(ages)) / 5.
            basis_weights.append([len(pos), max(float(np.sum(t*t)), .01),
                                  max(float(np.sum(t**4)), .001)])
        coefficients = np.asarray(coefficients)
        self.prior_fill_ = np.nanmedian(prior, axis=0)
        prior = np.where(np.isfinite(prior), prior, self.prior_fill_)
        outputs = coefficients - prior if self.residual else coefficients
        self.vectorizer_ = skrub.TableVectorizer()
        matrix = self.vectorizer_.fit_transform(table)
        self.models_ = []
        for column in range(3):
            model = HistGradientBoostingRegressor(max_iter=400, max_leaf_nodes=15,
                learning_rate=.04, l2_regularization=15., early_stopping=False, random_state=42)
            # Approximate each coefficient's contribution to visit-level squared error.
            weights = np.asarray(basis_weights)[:, column]
            model.fit(matrix, outputs[:, column], sample_weight=weights / np.mean(weights))
            self.models_.append(model)
        return self

    def predict(self, X):
        if "target" in X.columns:
            raise ValueError("Prediction must not receive targets.")
        features = self.features_.transform(X)
        table, prior = self.summarize(X, features)
        matrix = self.vectorizer_.transform(table)
        coefficients = np.column_stack([model.predict(matrix) for model in self.models_])
        if self.residual:
            coefficients += np.where(np.isfinite(prior), prior, self.prior_fill_)
        lookup = pd.DataFrame(coefficients, index=table.index)
        per_visit = lookup.loc[X.patient_id].to_numpy()
        age_center = X.groupby("patient_id", sort=False).age.transform("mean")
        t = (X.age.to_numpy() - age_center.to_numpy()) / 5.
        return per_visit[:, 0] + per_visit[:, 1]*t + per_visit[:, 2]*t*t
