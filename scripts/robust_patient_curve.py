"""Robust patient-curve fitting with Huber-weighted coefficient targets.

Identical to orthogonal_patient_curve.py except:
  - Per-patient curve coefficients are computed via iteratively-reweighted
    least squares (IRLS) with a Huber loss rather than plain OLS. This
    reduces the influence of noisy ON/OFF measurements on the per-patient
    quadratic target that feeds the HGBR step.
  - The Huber delta parameter is exposed and must be chosen inside
    training patients only.
  - All other hyperparameters, feature transformations, and HGBR settings
    are unchanged from OrthogonalPatientCurve.
"""
import numpy as np
import pandas as pd
import skrub
from sklearn.ensemble import HistGradientBoostingRegressor
from orthogonal_patient_curve import OrthogonalPatientCurve, orthogonal_basis


def _huber_weights(residuals, delta):
    """Huber IRLS weights: 1 where |r| <= delta, delta/|r| otherwise."""
    abs_r = np.abs(residuals)
    w = np.where(abs_r <= delta, 1.0, delta / np.maximum(abs_r, 1e-8))
    return w / w.mean()   # keep scale comparable across iterations


def fit_curve_robust(basis, values, delta=5.0, n_iter=5):
    """IRLS Huber fit of orthogonal coefficients to patient visit values.

    Parameters
    ----------
    basis : np.ndarray, shape (n_visits, 3)
        Orthogonal basis for this patient.
    values : np.ndarray, shape (n_visits,)
        Target values (true OFF) at each visit.
    delta : float
        Huber inflection point in target units.
    n_iter : int
        IRLS iterations (converges quickly; 5 is sufficient).

    Returns
    -------
    np.ndarray, shape (3,)
        Fitted orthogonal coefficients.
    """
    b = basis
    y = np.asarray(values, dtype=float)
    # Start with OLS
    coeffs = np.mean(b * y[:, None], axis=0)
    for _ in range(n_iter):
        residuals = y - b @ coeffs
        w = _huber_weights(residuals, delta)
        # Weighted least squares: (B^T W B) c = B^T W y
        bwb = b.T @ (w[:, None] * b)
        bwy = b.T @ (w * y)
        try:
            coeffs = np.linalg.solve(bwb, bwy)
        except np.linalg.LinAlgError:
            break   # fall back to previous coefficients
    return coeffs


class RobustOrthogonalPatientCurve(OrthogonalPatientCurve):
    """OrthogonalPatientCurve with Huber-robust per-patient coefficient targets.

    Parameters
    ----------
    residual : bool, default False
        Whether to fit residuals relative to the feature prior.
    huber_delta : float, default 5.0
        Huber inflection point in target (UPDRS) units.  Values between
        3 and 8 are reasonable; choose inside training patients only.
    """

    def __init__(self, residual=False, huber_delta=5.0):
        self.residual = residual
        self.huber_delta = huber_delta

    def fit_with_calibration(self, X, y, calibration):
        if "target" in X:
            raise ValueError("Targets must be separate from predictors.")
        self.features_ = calibration
        table, native_prior = self.summarize(X, calibration.transform(X))
        self.prior_fill_ = np.nanmedian(native_prior, axis=0)
        if not np.isfinite(self.prior_fill_).all():
            raise ValueError("No finite training prior for a curve coordinate.")
        basis_list, prior, counts, active = self._geometry(X, native_prior)
        target = np.asarray(y, dtype=float)
        coefficients = []
        for b, pos in zip(basis_list, X.groupby("patient_id", sort=False).indices.values()):
            values = target[pos]
            # Robust IRLS instead of plain OLS mean
            coefficients.append(fit_curve_robust(b, values, delta=self.huber_delta))
        outputs = np.asarray(coefficients)
        if self.residual:
            outputs = outputs - prior
        self.vectorizer_ = skrub.TableVectorizer()
        matrix = self.vectorizer_.fit_transform(table)
        self.models_ = []
        for column in range(3):
            weights = counts * active[:, column]
            if weights.sum() == 0:
                raise ValueError("Training patients cannot identify this curve coordinate.")
            model = HistGradientBoostingRegressor(
                max_iter=500, max_leaf_nodes=15,
                learning_rate=.04, l2_regularization=15.,
                early_stopping=False, random_state=42)
            model.fit(matrix, outputs[:, column],
                      sample_weight=weights / np.mean(weights))
            self.models_.append(model)
        return self

    def fit(self, X, y):
        from flexible_response import FlexibleResponseFeatures
        return self.fit_with_calibration(X, y, FlexibleResponseFeatures().fit(X, y))
