"""Robust inverse patient curve: InversePatientCurve with Huber-robust coefficients.

Identical to InversePatientCurve except the OrthogonalPatientCurve step
uses RobustOrthogonalPatientCurve (Huber IRLS) for the per-patient
coefficient targets. The inner 3-fold cross-fitted inversion is unchanged.
"""
import numpy as np
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from inverse_patient_curve import InverseMeasurements
from orthogonal_patient_curve import OrthogonalPatientCurve
from robust_patient_curve import RobustOrthogonalPatientCurve


class RobustInversePatientCurve(RobustOrthogonalPatientCurve):
    """InversePatientCurve with Huber-robust per-patient coefficient targets.

    Parameters
    ----------
    residual : bool, default True
    huber_delta : float, default 5.0
        Huber inflection point; choose within training patients only.
    """

    def __init__(self, residual=True, huber_delta=5.0):
        self.residual = residual
        self.huber_delta = huber_delta

    def fit(self, X, y):
        if "target" in X:
            raise ValueError("Targets must be separate from predictors.")
        self.features_ = InverseMeasurements()
        features = self.features_.fit_transform(X, y)
        table, native_prior = self.summarize(X, features)
        self.prior_fill_ = np.nanmedian(native_prior, axis=0)
        if not np.isfinite(self.prior_fill_).all():
            raise ValueError("No finite training curve prior.")
        basis_list, prior, counts, active = self._geometry(X, native_prior)
        target = np.asarray(y, dtype=float)
        coefficients = []
        from robust_patient_curve import fit_curve_robust
        for b, pos in zip(basis_list, X.groupby("patient_id", sort=False).indices.values()):
            values = target[pos]
            coefficients.append(fit_curve_robust(b, values, delta=self.huber_delta))
        outputs = np.asarray(coefficients)
        if self.residual:
            outputs -= prior
        self.vectorizer_ = skrub.TableVectorizer()
        matrix = self.vectorizer_.fit_transform(table)
        self.models_ = []
        for column in range(3):
            weights = counts * active[:, column]
            if weights.sum() == 0:
                raise ValueError("Curve coordinate cannot be identified.")
            model = HistGradientBoostingRegressor(
                max_iter=400, max_leaf_nodes=15,
                learning_rate=.04, l2_regularization=20.,
                early_stopping=False, random_state=42)
            model.fit(matrix, outputs[:, column],
                      sample_weight=weights / np.mean(weights))
            self.models_.append(model)
        return self
