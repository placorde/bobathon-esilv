"""Patient-curve regression with orthogonal, visit-loss-aligned targets."""
import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from patient_curve import PatientCurveRegressor


def orthogonal_basis(age):
    """Each active column has mean square one; pairwise products average zero."""
    age = np.asarray(age, dtype=float)
    if not np.isfinite(age).all():
        raise ValueError("Finite ages are required for curve reconstruction.")
    t = age - age.mean()
    columns = [np.ones(len(age))]
    for raw in [t, t*t]:
        column = raw.copy()
        # Reorthogonalize to reduce cancellation for nearly repeated ages.
        for _ in range(2):
            for previous in columns:
                column -= np.mean(column * previous) * previous
        scale = np.sqrt(np.mean(column*column))
        columns.append(column/scale if scale > 1e-8 else np.zeros(len(age)))
    return np.column_stack(columns)


class OrthogonalPatientCurve(PatientCurveRegressor):
    """Same predictor summaries and booster; different curve coordinates.

    For active orthogonal coordinates, mean visit squared error equals the sum
    of coefficient squared errors, plus the target's projection residual. Weight
    training patients by visit count to match the visit-level competition loss.
    """

    def _geometry(self, X, native_prior):
        basis, prior, counts, active = [], [], [], []
        for row, pos in enumerate(X.groupby("patient_id", sort=False).indices.values()):
            ages = X.age.to_numpy(dtype=float)[pos]
            b = orthogonal_basis(ages)
            t = (ages - ages.mean()) / 5.
            native = np.where(np.isfinite(native_prior[row]), native_prior[row], self.prior_fill_)
            curve = native[0] + native[1]*t + native[2]*t*t
            basis.append(b)
            prior.append(np.mean(b*curve[:, None], axis=0))
            counts.append(len(pos))
            active.append(np.mean(b*b, axis=0) > .5)
        return basis, np.asarray(prior), np.asarray(counts), np.asarray(active)

    def fit_with_calibration(self, X, y, calibration):
        if "target" in X:
            raise ValueError("Targets must be separate from predictors.")
        self.features_ = calibration
        table, native_prior = self.summarize(X, calibration.transform(X))
        self.prior_fill_ = np.nanmedian(native_prior, axis=0)
        if not np.isfinite(self.prior_fill_).all():
            raise ValueError("No finite training prior for a curve coordinate.")
        basis, prior, counts, active = self._geometry(X, native_prior)
        coefficients = []
        target = np.asarray(y, dtype=float)
        projection_squared_error = 0.
        for b, pos in zip(basis, X.groupby("patient_id", sort=False).indices.values()):
            values = target[pos]
            coefficient = np.mean(b*values[:, None], axis=0)
            coefficients.append(coefficient)
            projection_squared_error += np.sum((values-b@coefficient)**2)
            # Check the exact loss decomposition with a fixed perturbation.
            shift = np.array([.7, -.3, .2]) * (np.mean(b*b, axis=0) > .5)
            actual = np.mean((values-b@(coefficient+shift))**2)
            expected = np.mean((values-b@coefficient)**2) + np.sum(shift*shift)
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
        outputs = np.asarray(coefficients)
        if self.residual:
            outputs = outputs - prior
        self.training_projection_rmse_ = float(np.sqrt(projection_squared_error/len(X)))
        self.vectorizer_ = skrub.TableVectorizer()
        matrix = self.vectorizer_.fit_transform(table)
        self.models_ = []
        for column in range(3):
            weights = counts * active[:, column]
            if weights.sum() == 0:
                raise ValueError("Training patients cannot identify this curve coordinate.")
            model = HistGradientBoostingRegressor(max_iter=500, max_leaf_nodes=15,
                learning_rate=.04, l2_regularization=15., early_stopping=False, random_state=42)
            model.fit(matrix, outputs[:, column], sample_weight=weights/np.mean(weights))
            self.models_.append(model)
        return self

    def fit(self, X, y):
        from flexible_response import FlexibleResponseFeatures
        return self.fit_with_calibration(X, y, FlexibleResponseFeatures().fit(X, y))

    def predict(self, X):
        if "target" in X:
            raise ValueError("Prediction cannot receive targets.")
        table, native_prior = self.summarize(X, self.features_.transform(X))
        basis, prior, _, active = self._geometry(X, native_prior)
        matrix = self.vectorizer_.transform(table)
        coefficients = np.column_stack([model.predict(matrix) for model in self.models_])
        if self.residual:
            coefficients += prior
        coefficients *= active
        prediction = np.empty(len(X))
        for i, (b, pos) in enumerate(zip(basis, X.groupby("patient_id", sort=False).indices.values())):
            prediction[pos] = b@coefficients[i]
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite predictions.")
        return prediction
