"""Feature functions and transformers for the Parkinson's UPDRS workspace.

Exports:

- ``EngineerFeatures``: derives ``years_since_diagnosis``, clips ``off``
  at 108 (clinical scale maximum), and drops the now-redundant
  ``age_at_diagnosis`` and ``age`` columns (experiment 05).
- ``TemporalPatientFeatures``: computes per-patient expanding-window
  mean and linear trend of past ``off``/``on`` values plus visit count,
  using only strictly-prior visits (experiment 08).
- ``SelfContainedTemporalFeatures``: stateless variant — computes
  expanding stats purely from the rows present in the frame passed at
  transform time; no stored training history (experiment 08b).
"""

from __future__ import annotations

import numpy as np
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
        """Record input dimensionality so sklearn's check_is_fitted passes."""
        self.n_features_in_ = X.shape[1]
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


def _slope(ages: np.ndarray, values: np.ndarray) -> float:
    """Ordinary least-squares slope of ``values`` vs ``ages``.

    Returns ``NaN`` when fewer than two finite pairs are available.
    """
    mask = np.isfinite(values) & np.isfinite(ages)
    if mask.sum() < 2:
        return np.nan
    x = ages[mask]
    y = values[mask]
    x_c = x - x.mean()
    denom = (x_c * x_c).sum()
    if denom == 0:
        return np.nan
    return float((x_c * y).sum() / denom)


class TemporalPatientFeatures(BaseEstimator, TransformerMixin):
    """Expanding-window per-patient temporal features for experiment 08.

    For each visit, derives statistics computed exclusively from that
    patient's **prior** visits (strictly earlier ``age``), so no future
    information or target value leaks into the features.

    Features added
    --------------
    past_off_mean : float
        Expanding mean of ``off`` values over all preceding visits for
        this patient that have a non-NaN ``off``.
    past_off_trend : float
        OLS slope of ``off`` vs ``age`` over preceding non-NaN visits.
    past_on_mean : float
        Expanding mean of ``on`` values over preceding non-NaN visits.
    past_on_trend : float
        OLS slope of ``on`` vs ``age`` over preceding non-NaN visits.
    visit_count : int
        Number of preceding visits (regardless of NaN status).

    All five features are NaN for the first visit of a patient (no prior
    history). ``HistGradientBoostingRegressor`` handles NaN natively.

    Stateful behaviour
    ------------------
    ``fit_transform`` builds a per-patient state table from training
    rows and stores it in ``patient_history_``. ``transform`` initialises
    each patient from that stored state, then advances row-by-row through
    test rows sorted by ``age``. For patients not seen in training (cold
    start under GroupKFold), all five features are NaN.

    Parameters
    ----------
    age_col : str, default ``"age"``
        Column used to order visits within each patient.
    patient_col : str, default ``"patient_id"``
        Column identifying each patient.  Dropped from the output.
    off_col : str, default ``"off"``
        Column whose past values feed ``past_off_*`` features.
    on_col : str, default ``"on"``
        Column whose past values feed ``past_on_*`` features.
    """

    def __init__(
        self,
        age_col: str = "age",
        patient_col: str = "patient_id",
        off_col: str = "off",
        on_col: str = "on",
    ) -> None:
        self.age_col = age_col
        self.patient_col = patient_col
        self.off_col = off_col
        self.on_col = on_col

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_expanding(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute expanding stats for every row, using only prior rows."""
        age_col = self.age_col
        pid_col = self.patient_col
        off_col = self.off_col
        on_col = self.on_col

        # Sort so rows within each patient are in chronological order.
        X = X.sort_values([pid_col, age_col]).copy()

        past_off_mean = np.full(len(X), np.nan)
        past_off_trend = np.full(len(X), np.nan)
        past_on_mean = np.full(len(X), np.nan)
        past_on_trend = np.full(len(X), np.nan)
        visit_count = np.zeros(len(X), dtype=float)

        age_vals = X[age_col].to_numpy(dtype=float)
        off_vals = X[off_col].to_numpy(dtype=float)
        on_vals = X[on_col].to_numpy(dtype=float)
        pid_vals = X[pid_col].to_numpy()

        # Walk rows; accumulate per-patient history before computing.
        buf_pid: dict[object, dict] = {}

        for i, pid in enumerate(pid_vals):
            state = buf_pid.get(pid)
            if state is not None and state["count"] > 0:
                visit_count[i] = state["count"]
                offs = np.array(state["offs"])
                ons = np.array(state["ons"])
                ages_a = np.array(state["ages"])
                past_off_mean[i] = np.nanmean(offs) if np.any(np.isfinite(offs)) else np.nan
                past_off_trend[i] = _slope(ages_a, offs)
                past_on_mean[i] = np.nanmean(ons) if np.any(np.isfinite(ons)) else np.nan
                past_on_trend[i] = _slope(ages_a, ons)

            # Now add current row to the state (AFTER reading, so it's
            # strictly past for the next row).
            if state is None:
                buf_pid[pid] = {
                    "count": 0,
                    "ages": [],
                    "offs": [],
                    "ons": [],
                }
                state = buf_pid[pid]
            state["count"] += 1
            state["ages"].append(age_vals[i])
            state["offs"].append(off_vals[i])
            state["ons"].append(on_vals[i])

        X = X.copy()
        X["past_off_mean"] = past_off_mean
        X["past_off_trend"] = past_off_trend
        X["past_on_mean"] = past_on_mean
        X["past_on_trend"] = past_on_trend
        X["visit_count"] = visit_count
        return X.drop(columns=[pid_col])

    def _apply_from_history(
        self, X: pd.DataFrame, history: dict[object, dict]
    ) -> pd.DataFrame:
        """Apply stored per-patient history to a new (test) frame."""
        age_col = self.age_col
        pid_col = self.patient_col
        off_col = self.off_col
        on_col = self.on_col

        X = X.sort_values([pid_col, age_col]).copy()

        past_off_mean = np.full(len(X), np.nan)
        past_off_trend = np.full(len(X), np.nan)
        past_on_mean = np.full(len(X), np.nan)
        past_on_trend = np.full(len(X), np.nan)
        visit_count = np.zeros(len(X), dtype=float)

        age_vals = X[age_col].to_numpy(dtype=float)
        off_vals = X[off_col].to_numpy(dtype=float)
        on_vals = X[on_col].to_numpy(dtype=float)
        pid_vals = X[pid_col].to_numpy()

        # Copy stored history so we can extend per-test-row without
        # mutating the fitted state across repeated transform calls.
        live: dict[object, dict] = {
            pid: {
                "count": s["count"],
                "ages": list(s["ages"]),
                "offs": list(s["offs"]),
                "ons": list(s["ons"]),
            }
            for pid, s in history.items()
        }

        for i, pid in enumerate(pid_vals):
            state = live.get(pid)
            if state is not None and state["count"] > 0:
                visit_count[i] = state["count"]
                offs = np.array(state["offs"])
                ons = np.array(state["ons"])
                ages_a = np.array(state["ages"])
                past_off_mean[i] = np.nanmean(offs) if np.any(np.isfinite(offs)) else np.nan
                past_off_trend[i] = _slope(ages_a, offs)
                past_on_mean[i] = np.nanmean(ons) if np.any(np.isfinite(ons)) else np.nan
                past_on_trend[i] = _slope(ages_a, ons)
            # Advance the live state with current row for next visits.
            if state is None:
                live[pid] = {"count": 0, "ages": [], "offs": [], "ons": []}
                state = live[pid]
            state["count"] += 1
            state["ages"].append(age_vals[i])
            state["offs"].append(off_vals[i])
            state["ons"].append(on_vals[i])

        X = X.copy()
        X["past_off_mean"] = past_off_mean
        X["past_off_trend"] = past_off_trend
        X["past_on_mean"] = past_on_mean
        X["past_on_trend"] = past_on_trend
        X["visit_count"] = visit_count
        return X.drop(columns=[pid_col])

    # ------------------------------------------------------------------
    # sklearn API
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None) -> "TemporalPatientFeatures":  # noqa: ARG002
        """Store per-patient accumulated history from training rows."""
        pid_col = self.patient_col
        age_col = self.age_col
        off_col = self.off_col
        on_col = self.on_col

        X_sorted = X.sort_values([pid_col, age_col])
        pid_vals = X_sorted[pid_col].to_numpy()
        age_vals = X_sorted[age_col].to_numpy(dtype=float)
        off_vals = X_sorted[off_col].to_numpy(dtype=float)
        on_vals = X_sorted[on_col].to_numpy(dtype=float)

        buf: dict[object, dict] = {}
        for i, pid in enumerate(pid_vals):
            if pid not in buf:
                buf[pid] = {"count": 0, "ages": [], "offs": [], "ons": []}
            s = buf[pid]
            s["count"] += 1
            s["ages"].append(age_vals[i])
            s["offs"].append(off_vals[i])
            s["ons"].append(on_vals[i])

        self.patient_history_ = buf
        self.n_features_in_ = X.shape[1]
        return self

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:  # noqa: ARG002
        """Fit and return expanded frame in one pass (avoids double sort)."""
        pid_col = self.patient_col
        age_col = self.age_col
        off_col = self.off_col
        on_col = self.on_col

        result = self._compute_expanding(X)

        # Also populate patient_history_ from this pass.
        X_sorted = X.sort_values([pid_col, age_col])
        pid_vals = X_sorted[pid_col].to_numpy()
        age_vals = X_sorted[age_col].to_numpy(dtype=float)
        off_vals = X_sorted[off_col].to_numpy(dtype=float)
        on_vals = X_sorted[on_col].to_numpy(dtype=float)

        buf: dict[object, dict] = {}
        for i, pid in enumerate(pid_vals):
            if pid not in buf:
                buf[pid] = {"count": 0, "ages": [], "offs": [], "ons": []}
            s = buf[pid]
            s["count"] += 1
            s["ages"].append(age_vals[i])
            s["offs"].append(off_vals[i])
            s["ons"].append(on_vals[i])

        self.patient_history_ = buf
        self.n_features_in_ = X.shape[1]
        return result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply stored training history to X and return expanded frame."""
        return self._apply_from_history(X, self.patient_history_)


class SelfContainedTemporalFeatures(BaseEstimator, TransformerMixin):
    """Stateless expanding-window temporal features for experiment 08b.

    Computes per-patient expanding-window statistics using **only the rows
    present in the frame passed to transform** — no stored training history
    is referenced. The stat for visit ``i`` of patient ``p`` is computed
    from visits ``0..i-1`` of ``p`` within the current frame, sorted by
    ``age``.

    This makes the transformer safe under ``GroupKFold``: test patients
    have their own visits sorted chronologically, and each visit receives
    stats derived from earlier visits of the same patient in the test
    fold. First visits of any patient produce all-NaN stats (no prior
    rows), which ``HistGradientBoostingRegressor`` handles natively.

    Features added
    --------------
    past_off_mean : float
        Expanding mean of ``off`` values over strictly-prior visits of
        this patient in the current frame.
    past_off_trend : float
        OLS slope of ``off`` vs ``age`` over strictly-prior non-NaN visits.
    past_on_mean : float
        Expanding mean of ``on`` values over strictly-prior visits.
    past_on_trend : float
        OLS slope of ``on`` vs ``age`` over strictly-prior non-NaN visits.
    visit_count : int
        Number of strictly-prior visits in the current frame.

    Parameters
    ----------
    age_col : str, default ``"age"``
        Column used to order visits within each patient.
    patient_col : str, default ``"patient_id"``
        Column identifying each patient. Dropped from the output.
    off_col : str, default ``"off"``
    on_col : str, default ``"on"``
    """

    def __init__(
        self,
        age_col: str = "age",
        patient_col: str = "patient_id",
        off_col: str = "off",
        on_col: str = "on",
    ) -> None:
        self.age_col = age_col
        self.patient_col = patient_col
        self.off_col = off_col
        self.on_col = on_col

    def fit(self, X: pd.DataFrame, y=None) -> "SelfContainedTemporalFeatures":  # noqa: ARG002
        """No-op — transformer is stateless; records input width only."""
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute expanding stats within X and return the enriched frame."""
        # Delegate to the shared helper used by TemporalPatientFeatures.
        tf = TemporalPatientFeatures(
            age_col=self.age_col,
            patient_col=self.patient_col,
            off_col=self.off_col,
            on_col=self.on_col,
        )
        return tf._compute_expanding(X)
