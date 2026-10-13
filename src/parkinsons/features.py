"""Feature functions and transformers for the Parkinson's UPDRS workspace.

Currently exports one stateless transformer for experiment 05:

- ``EngineerFeatures``: derives ``years_since_diagnosis``, clips ``off``
  at 108 (clinical scale maximum), and drops the now-redundant
  ``age_at_diagnosis`` and ``age`` columns.
"""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Clinical UPDRS motor score maximum (MDS-UPDRS Part III ceiling).
_UPDRS_MAX = 108.0


class EngineerFeatures(BaseEstimator, TransformerMixin):
    """Stateless feature transform for experiment 05.

    Operations applied in order:

    1. Derive ``years_since_diagnosis = age - age_at_diagnosis``.
    2. Drop ``age`` and ``age_at_diagnosis`` (collinear at r = 0.94).
    3. Clip ``off`` at ``_UPDRS_MAX`` (108) — removes one row whose
       value of 111 exceeds the clinical scale maximum.

    This transformer is **stateless**: ``fit`` is a no-op and
    ``transform`` applies the same deterministic operations at both
    fit and predict time, so there is no leakage risk.

    Parameters
    ----------
    clip_off : bool, default=True
        Whether to clip the ``off`` column at ``_UPDRS_MAX``. Set to
        ``False`` in ablation experiments to isolate the clipping effect.
    """

    def __init__(self, clip_off: bool = True) -> None:
        self.clip_off = clip_off

    def fit(self, X, y=None):  # noqa: ARG002
        """No-op — transformer is stateless."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply feature derivation and optional clipping.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain ``age``, ``age_at_diagnosis``, and (if
            ``clip_off=True``) ``off``.

        Returns
        -------
        pd.DataFrame
            Transformed frame with ``years_since_diagnosis`` added and
            ``age`` / ``age_at_diagnosis`` dropped.
        """
        X = X.copy()
        # 1. Derive years since PD diagnosis (NaN where age_at_diagnosis missing)
        X["years_since_diagnosis"] = X["age"] - X["age_at_diagnosis"]
        # 2. Drop collinear originals
        X = X.drop(columns=["age", "age_at_diagnosis"])
        # 3. Clip off at clinical maximum
        if self.clip_off and "off" in X.columns:
            X["off"] = X["off"].clip(upper=_UPDRS_MAX)
        # 4. Cast object columns to category dtype so HGBR treats them as
        #    native categoricals (object alone raises ValueError in sklearn).
        for col in X.select_dtypes(include="object").columns:
            X[col] = X[col].astype("category")
        return X
