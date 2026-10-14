"""Cross-fitted nonlinear measurement inversion, followed by patient curves."""
import numpy as np
import pandas as pd
import skrub
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from patient_features import PatientFeatures
from response_features import ResponseFeatures
from orthogonal_patient_curve import OrthogonalPatientCurve


class InverseMeasurements(TransformerMixin, BaseEstimator):
    """Learn target-minus-measurement directly; keep patient targets out of OOF features."""

    @staticmethod
    def _models(X, y):
        raw = PatientFeatures().fit_transform(X)
        models = {}
        for state in ["on", "off"]:
            available = X[state].notna().to_numpy()
            if available.sum() < 100:
                raise ValueError("Insufficient measured scores for inversion.")
            model = make_pipeline(skrub.TableVectorizer(), HistGradientBoostingRegressor(
                max_iter=220, max_leaf_nodes=15, min_samples_leaf=30,
                learning_rate=.05, l2_regularization=30., early_stopping=False,
                random_state=42))
            model.fit(raw.iloc[np.flatnonzero(available)],
                      np.asarray(y)[available]-X[state].to_numpy()[available])
            models[state] = model
        return models

    @staticmethod
    def _predictions(models, X):
        if "target" in X:
            raise ValueError("Targets cannot enter measurement inversion.")
        raw = PatientFeatures().transform(X)
        prediction = {}
        for state, model in models.items():
            available = X[state].notna().to_numpy()
            values = np.full(len(X), np.nan)
            if available.any():
                values[available] = (model.predict(raw.iloc[np.flatnonzero(available)])
                                     + X[state].to_numpy()[available])
            prediction[state] = values
        return prediction

    def fit_transform(self, X, y=None, **fit_params):
        if "target" in X or y is None:
            raise ValueError("Training targets must be supplied separately.")
        target = np.asarray(y, dtype=float)
        predictions = {state: np.full(len(X), np.nan) for state in ["on", "off"]}
        assigned = np.zeros(len(X), dtype=bool)
        for i, (fit, val) in enumerate(GroupKFold(3).split(X, target, X.patient_id), 1):
            if set(X.iloc[fit].patient_id) & set(X.iloc[val].patient_id):
                raise ValueError("Inner calibration patient leakage.")
            print(f"  Inner calibration {i}/3: training ON/OFF inverse models", flush=True)
            models = self._models(X.iloc[fit], target[fit])
            pred = self._predictions(models, X.iloc[val])
            for state in predictions:
                predictions[state][val] = pred[state]
            assigned[val] = True
        if not assigned.all():
            raise ValueError("Every training row must receive an out-of-fold prediction.")
        self.variances_ = {}
        for state in predictions:
            valid = X[state].notna().to_numpy()
            if not np.isfinite(predictions[state][valid]).all():
                raise ValueError("Non-finite out-of-fold inverse estimates.")
            self.variances_[state] = max(float(np.mean((target[valid]-predictions[state][valid])**2)), 1.)
        print("  Fitting inverse models on all outer-training patients", flush=True)
        self.models_ = self._models(X, target)
        self.feature_names_in_ = np.asarray(X.columns)
        return self._features(X, predictions)

    def fit(self, X, y):
        self.fit_transform(X, y)
        return self

    def transform(self, X):
        return self._features(X, self._predictions(self.models_, X))

    def _features(self, X, predictions):
        if "target" in X:
            raise ValueError("Targets are forbidden during prediction.")
        original = PatientFeatures().transform(X)
        adjusted = X.copy()
        additions, factors = {}, {}
        for state in ["on", "off"]:
            adjusted[state] = predictions[state]
            additions[f"response_{state}_corrected"] = predictions[state]
            additions[f"response_{state}_offset"] = X[state].to_numpy()-predictions[state]
            additions[f"response_{state}_factor"] = np.ones(len(X))
            factors[state] = np.full(len(X), 1./np.sqrt(self.variances_[state]))
        # These are inverse predictions, including predictions for measured zero
        # scores. The old trajectory routine's positivity mask must not remove them.
        mask = X.copy()
        for state in ["on", "off"]:
            mask[state] = np.where(X[state].notna(), 1., np.nan)
        # Preserve its regularization-to-weight ratio while using learned relative
        # precision: normalize factors to mean variance rather than tiny weights.
        variance_mean = np.mean(list(self.variances_.values()))
        factors = {state: values*np.sqrt(variance_mean) for state, values in factors.items()}
        trajectories = ResponseFeatures.trajectories(mask, adjusted, factors)
        result = pd.concat([original, pd.DataFrame(additions, index=X.index), trajectories], axis=1)
        return result.replace([np.inf, -np.inf], np.nan)


class InversePatientCurve(OrthogonalPatientCurve):
    """Orthogonal curve regression trained on cross-fitted inverse measurements."""

    def fit(self, X, y):
        if "target" in X:
            raise ValueError("Targets must be separate from predictors.")
        self.features_ = InverseMeasurements()
        features = self.features_.fit_transform(X, y)
        table, native_prior = self.summarize(X, features)
        self.prior_fill_ = np.nanmedian(native_prior, axis=0)
        if not np.isfinite(self.prior_fill_).all():
            raise ValueError("No finite training curve prior.")
        basis, prior, counts, active = self._geometry(X, native_prior)
        coefficients = []
        for b, pos in zip(basis, X.groupby("patient_id", sort=False).indices.values()):
            coefficients.append(np.mean(b*np.asarray(y)[pos, None], axis=0))
        outputs = np.asarray(coefficients)
        if self.residual:
            outputs -= prior
        self.vectorizer_ = skrub.TableVectorizer()
        matrix = self.vectorizer_.fit_transform(table)
        self.models_ = []
        print("  Fitting patient curve models on cross-fitted measurement features", flush=True)
        for column in range(3):
            weights = counts*active[:, column]
            if weights.sum() == 0:
                raise ValueError("Curve coordinate cannot be identified.")
            model = HistGradientBoostingRegressor(max_iter=400, max_leaf_nodes=15,
                learning_rate=.04, l2_regularization=20., early_stopping=False, random_state=42)
            model.fit(matrix, outputs[:, column], sample_weight=weights/np.mean(weights))
            self.models_.append(model)
        return self
