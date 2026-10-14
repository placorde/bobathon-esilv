"""Fold-local nonlinear dose response and rater-specific score scaling."""
import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from response_features import ResponseFeatures
from calibrated_features import CalibratedFeatures
from patient_curve import PatientCurveRegressor, fit_curve


class FlexibleResponseFeatures(ResponseFeatures):
    def __init__(self, alpha=30., use_trajectory=True):
        self.alpha = alpha
        self.use_trajectory = use_trajectory

    @staticmethod
    def design(X, state):
        table = CalibratedFeatures.design(X, state)
        hours = X[f"time_since_intake_{state}"].clip(lower=0)
        dose = X.ledd.clip(lower=0)
        additions = {}
        # Candidate saturating dose curves, with scales in the dataset's dose units.
        for half_dose in [200., 500., 1000.]:
            saturated = dose / (dose + half_dose)
            for decay in [1., 2., 4., 8., 16.]:
                additions[f"concentration_saturating_{half_dose:g}_{decay:g}"] = (
                    saturated * (1 - np.exp(-hours / .5)) * np.exp(-hours / decay)
                )
        return pd.concat([table, pd.DataFrame(additions, index=X.index)], axis=1)

    def fit(self, X, y):
        # Learn the category vocabulary on training patients only.
        self.scaling_categories_ = {
            col: sorted(X[col].fillna("missing").astype(str).unique())
            for col in ["rater_id", "cohort", "gene"]
        }
        return super().fit(X, y)

    def response_design(self, X, state, latent):
        table = super().response_design(X, state, latent)
        latent = np.broadcast_to(np.asarray(latent), (len(X),))
        additions = {}
        for col, categories in self.scaling_categories_.items():
            observed = X[col].fillna("missing").astype(str)
            for index, category in enumerate(categories):
                additions[f"scale_{col}_{index}"] = (observed == category).to_numpy() * latent
        # No target-dependent values are supplied at inference: latent is then 0 or 1.
        return pd.concat([table, pd.DataFrame(additions, index=X.index)], axis=1)


class FlexiblePatientCurveRegressor(PatientCurveRegressor):
    """Same patient-curve target, using the expanded measurement correction."""

    def fit(self, X, y):
        if "target" in X.columns:
            raise ValueError("Targets must not enter the predictor table.")
        self.features_ = FlexibleResponseFeatures(alpha=30., use_trajectory=True)
        features = self.features_.fit_transform(X, y)
        table, prior = self.summarize(X, features)
        coefficients, basis_weights = [], []
        target = np.asarray(y)
        for patient, pos in X.groupby("patient_id", sort=False).indices.items():
            ages = X.age.to_numpy()[pos]
            coefficients.append(fit_curve(ages, target[pos]))
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
            model = HistGradientBoostingRegressor(max_iter=500, max_leaf_nodes=15,
                learning_rate=.04, l2_regularization=15., early_stopping=False, random_state=42)
            weights = np.asarray(basis_weights)[:, column]
            model.fit(matrix, outputs[:, column], sample_weight=weights / np.mean(weights))
            self.models_.append(model)
        return self
