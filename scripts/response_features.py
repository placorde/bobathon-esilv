"""Training-fold response calibration and weighted predictor-only trajectories."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from calibrated_features import CalibratedFeatures
from patient_features import PatientFeatures


class ResponseFeatures(CalibratedFeatures):
    """Estimate measurement = baseline bias + response factor * latent score.

    Outcomes enter fit only. Transform evaluates the fitted equation at latent
    values zero and one to recover its intercept and slope, then inverts it.
    This is an experimental approximation, not a clinical treatment model.
    """

    def __init__(self, alpha=30., use_trajectory=True):
        self.alpha = alpha
        self.use_trajectory = use_trajectory

    def response_design(self, X, state, latent):
        table = self.design(X, state)
        latent = np.broadcast_to(np.asarray(latent), (len(X),))
        table["latent_score"] = latent
        for col in list(table.columns):
            if col.startswith("concentration_"):
                table[f"score_times_{col}"] = table[col] * latent
        return table

    def fit(self, X, y):
        super().fit(X, y)
        y = np.asarray(y)
        self.response_models_ = {}
        for state in ["on", "off"]:
            table = self.response_design(X, state, y)
            numeric = [col for col in table if col not in ["cohort", "rater_id"]]
            preprocessing = ColumnTransformer([
                ("numeric", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                          StandardScaler()), numeric),
                ("categorical", OneHotEncoder(handle_unknown="ignore"), ["cohort", "rater_id"]),
            ])
            estimator = make_pipeline(preprocessing, Ridge(alpha=self.alpha, solver="lsqr"))
            # Exclude clipped zero scores from the uncensored measurement equation.
            available = (X[state].notna() & (X[state] > .1)).to_numpy()
            if available.sum() < 100:
                raise ValueError("Too few positive measured scores to fit response calibration.")
            estimator.fit(table.iloc[np.flatnonzero(available)], X[state].to_numpy()[available])
            self.response_models_[state] = estimator
        return self

    def transform(self, X):
        original = super().transform(X)
        adjusted = X.copy()
        extra = {}
        factors = {}
        for state in ["on", "off"]:
            model = self.response_models_[state]
            offset = model.predict(self.response_design(X, state, 0.))
            factor = model.predict(self.response_design(X, state, 1.)) - offset
            # Guard against ill-conditioned inversions; expose the raw factor as a feature.
            safe_factor = np.clip(factor, .2, 1.8)
            corrected = (X[state] - offset) / safe_factor
            adjusted[state] = corrected
            factors[state] = safe_factor
            extra[f"response_{state}_offset"] = offset
            extra[f"response_{state}_factor"] = factor
            extra[f"response_{state}_corrected"] = corrected
            extra[f"response_{state}_clipped_zero"] = (X[state] <= .1).astype(float)
        pooled = PatientFeatures().transform(adjusted)
        related = [col for col in pooled if col in ["on", "off", "on_off_gap"]
                   or col.startswith("patient_on_") or col.startswith("patient_off_")
                   or col.startswith("on_minus_") or col.startswith("off_minus_")]
        parts = [original, pd.DataFrame(extra, index=X.index), pooled[related].add_prefix("response_")]
        if self.use_trajectory:
            parts.append(self.trajectories(X, adjusted, factors))
        return pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)

    @staticmethod
    def trajectories(X, adjusted, factors):
        names = ["pooled_linear_score", "pooled_quadratic_score", "pooled_score_mean",
                 "pooled_linear_slope", "pooled_curve", "pooled_observations",
                 "pooled_linear_residual_rmse"]
        result = pd.DataFrame(np.nan, index=X.index, columns=names)
        age_all = X.age.to_numpy()
        corrected = {state: adjusted[state].to_numpy() for state in ["on", "off"]}
        observed = {state: X[state].to_numpy() for state in ["on", "off"]}
        for positions in X.groupby("patient_id", sort=False).indices.values():
            positions = np.asarray(positions)
            ages, values, weights = [], [], []
            for state in ["on", "off"]:
                valid = np.isfinite(corrected[state][positions]) & np.isfinite(age_all[positions])
                valid &= observed[state][positions] > .1
                chosen = positions[valid]
                ages.extend(age_all[chosen])
                values.extend(corrected[state][chosen])
                # Inversion amplifies score noise when the response factor is small.
                weights.extend(factors[state][chosen] ** 2)
            if not values:
                continue
            a, z, w = np.asarray(ages), np.asarray(values), np.asarray(weights)
            center = np.average(a, weights=w)
            t = (a - center) / 5.
            t_pred = (age_all[positions] - center) / 5.
            average = np.average(z, weights=w)
            outputs = {}
            for degree, penalty in [(1, .1), (2, 1.)]:
                design = np.column_stack([t ** power for power in range(degree + 1)])
                ridge_penalty = np.diag([1e-8] + [penalty] * degree)
                beta = np.linalg.solve(design.T @ (w[:, None] * design) + ridge_penalty,
                                       design.T @ (w * z))
                predict = np.column_stack([t_pred ** power for power in range(degree + 1)]) @ beta
                outputs[degree] = (predict, beta, design @ beta)
            result.iloc[positions, 0] = outputs[1][0]
            result.iloc[positions, 1] = outputs[2][0]
            result.iloc[positions, 2] = average
            result.iloc[positions, 3] = outputs[1][1][1] / 5.
            result.iloc[positions, 4] = outputs[2][1][2] / 25.
            result.iloc[positions, 5] = len(values)
            result.iloc[positions, 6] = np.sqrt(np.average((z - outputs[1][2]) ** 2, weights=w))
        return result
