"""Nonlinear observation likelihood anchored to a patient-disjoint fitted prior.

Training learns measured ON/OFF conditional on the known latent training target.
Inference never sees that target: it searches a latent patient trajectory using
only the supplied measurements, medication, demographics and the fitted prior.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from scipy.optimize import least_squares
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from mae_patient_blend import CachedMAEPatientBlend, atomic_dump, training_signature
from orthogonal_patient_curve import orthogonal_basis


def monotone_curve(age, prediction):
    return IsotonicRegression(increasing=True, y_min=0., y_max=132.,
                              out_of_bounds="clip").fit_transform(age, prediction)


class ForwardMeasurements(BaseEstimator):
    """Exclude ON/OFF predictors to prevent circular measurement reconstruction."""
    def __init__(self, max_iter=300):
        self.max_iter = max_iter

    def design(self, X, state, latent):
        if "target" in X:
            raise ValueError("Targets must be passed separately.")
        table = pd.DataFrame(index=X.index)
        table["latent_score"] = np.broadcast_to(np.asarray(latent, dtype=float), (len(X),))
        for name in ["age", "age_at_diagnosis", "sexM", "ledd", f"time_since_intake_{state}"]:
            table[name] = pd.to_numeric(X[name], errors="raise")
        table["duration"] = X.age - X.age_at_diagnosis
        table["log_dose"] = np.log1p(X.ledd.clip(lower=0))
        hours = X[f"time_since_intake_{state}"].clip(lower=0)
        table["dose_missing"] = X.ledd.isna().astype(float)
        table["hours_missing"] = hours.isna().astype(float)
        for half_dose in [200., 500., 1000.]:
            dose = X.ledd.clip(lower=0) / (X.ledd.clip(lower=0) + half_dose)
            for decay in [2., 8.]:
                table[f"concentration_{half_dose:g}_{decay:g}"] = dose * (1-np.exp(-hours/.5)) * np.exp(-hours/decay)
        for name in ["ledd", f"time_since_intake_{state}"]:
            groups = X[name].groupby(X.patient_id, sort=False)
            table[f"patient_{name}_mean"] = groups.transform("mean")
            table[f"patient_{name}_std"] = groups.transform("std")
            table[f"patient_{name}_count"] = groups.transform("count")
        for name, categories in self.categories_.items():
            observed = X[name].fillna("missing").astype(str)
            for index, category in enumerate(categories):
                table[f"{name}_category_{index}"] = observed.eq(category).astype(float)
        # Fixed precision prevents numerical-zero tree splits from changing with
        # patient batching. No measured on/off values enter this design.
        return table.replace([np.inf, -np.inf], np.nan).round(8)

    @staticmethod
    def bucket(X, state):
        return X.ledd.notna().astype(int).to_numpy()*2 + X[f"time_since_intake_{state}"].notna().astype(int).to_numpy()

    def fit(self, X, y):
        self.categories_ = {name: sorted(X[name].fillna("missing").astype(str).unique())
                            for name in ["cohort", "gene", "rater_id"]}
        self.models_, self.sigma_ = {}, {}
        for state in ["on", "off"]:
            available = X[state].notna().to_numpy()
            if available.sum() < 100:
                raise ValueError("Too few measured scores for calibration.")
            # Positive monotonicity is for latent severity, not visit age.
            model = HistGradientBoostingRegressor(max_iter=self.max_iter,
                max_leaf_nodes=31, min_samples_leaf=50, learning_rate=.06,
                l2_regularization=25., monotonic_cst={"latent_score": 1},
                early_stopping=False, random_state=42)
            design = self.design(X, state, np.asarray(y, dtype=float))
            print(f"  Learning nonlinear {state.upper()} measurement equation ({available.sum()} observations)", flush=True)
            model.fit(design.iloc[np.flatnonzero(available)], X[state].to_numpy()[available])
            residual = X[state].to_numpy()[available] - model.predict(design.iloc[np.flatnonzero(available)])
            # Training-only uncertainty estimates; conservative floor limits
            # overconfidence from in-sample calibration residuals.
            overall = max(float(np.sqrt(np.mean(residual**2))), 2.5)
            buckets = self.bucket(X, state)[available]
            scales = {}
            for bucket in range(4):
                selected = residual[buckets == bucket]
                scales[bucket] = max(float(np.sqrt(np.mean(selected**2))), 2.5) if len(selected) >= 100 else overall
            self.models_[state] = model
            self.sigma_[state] = scales
            print(f"  {state.upper()} uncertainty by medication availability: {scales}", flush=True)
        return self

    def grids(self, X):
        latent_grid = np.arange(0., 132.1, 4.)
        grids = {}
        for state, model in self.models_.items():
            design = self.design(X, state, 0.)
            result = np.empty((len(X), len(latent_grid)))
            for column, value in enumerate(latent_grid):
                design["latent_score"] = value
                result[:, column] = model.predict(design)
            if (np.diff(result, axis=1) < -1e-7).any():
                raise ValueError("Measurement equation is not monotonic in latent severity.")
            grids[state] = result
        return latent_grid, grids


class ForwardPatientPosterior(RegressorMixin, BaseEstimator):
    def __init__(self, project_root, mix_weight=1., max_iter=300):
        self.project_root = project_root
        self.mix_weight = mix_weight
        self.max_iter = max_iter

    def fit(self, X, y):
        root = Path(self.project_root)
        # This checks exact row/label membership before reusing fold-local models.
        self.prior_ = CachedMAEPatientBlend(str(root)).fit(X, y)
        signature = training_signature(X, y)
        directory = root / "data" / "day2_16_forward_posterior_v1"
        directory.mkdir(exist_ok=True)
        path = directory / f"forward_{signature[:20]}_{self.max_iter}.joblib"
        if path.exists():
            self.forward_ = joblib.load(path)
            print("  Reusing nonlinear measurement models", flush=True)
        else:
            order = np.argsort(X.Index.to_numpy(), kind="stable")
            self.forward_ = ForwardMeasurements(self.max_iter).fit(
                X.iloc[order].reset_index(drop=True), np.asarray(y)[order])
            atomic_dump(self.forward_, path)
        return self

    def predict(self, X):
        if "target" in X or not X.Index.is_unique:
            raise ValueError("Prediction requires unique row IDs and no target column.")
        order = np.argsort(X.Index.to_numpy(), kind="stable")
        frame = X.iloc[order].reset_index(drop=True)
        baseline = self.prior_.predict(frame)
        latent_grid, profiles = self.forward_.grids(frame)
        output = baseline.copy()
        measurements = {state: frame[state].to_numpy(dtype=float) for state in ["on", "off"]}
        buckets = {state: self.forward_.bucket(frame, state) for state in ["on", "off"]}
        for pos in frame.groupby("patient_id", sort=False).indices.values():
            age = frame.age.to_numpy(dtype=float)[pos]
            basis = orthogonal_basis(age)
            prior = monotone_curve(age, baseline[pos])
            chosen, observed, sigma, local_profiles = [], [], [], []
            for state in ["on", "off"]:
                valid = np.flatnonzero(np.isfinite(measurements[state][pos]))
                for row in valid:
                    absolute = pos[row]
                    chosen.append(row)
                    observed.append(measurements[state][absolute])
                    # Repeated measurements share unobserved patient response;
                    # modest correlation inflation avoids treating them as IID.
                    sigma.append(self.forward_.sigma_[state][buckets[state][absolute]] * np.sqrt(1.+.10*max(len(valid)-1, 0)))
                    local_profiles.append(profiles[state][absolute])
            if len(observed) < 2:
                output[pos] = prior
                continue
            chosen = np.asarray(chosen)
            observed, sigma, local_profiles = map(np.asarray, [observed, sigma, local_profiles])
            scales = np.array([4.5, 1.5, .75])

            def residual(parameters):
                latent = np.clip(prior + basis @ (parameters*scales), 0., 132.)
                expected = np.array([np.interp(latent[row], latent_grid, profile)
                                     for row, profile in zip(chosen, local_profiles)])
                # Integer measurements encode an interval; a zero is censored.
                error = expected - np.clip(expected, observed-.5, observed+.5)
                error = np.where(observed <= .1, np.maximum(expected-.5, 0.), error)
                z = error / sigma
                delta = 2.5
                robust = np.sign(z)*np.sqrt(2.*delta**2*(np.sqrt(1.+(z/delta)**2)-1.))
                declining = np.minimum(np.diff(latent[np.argsort(age)]), 0.)/.35
                return np.r_[robust, parameters, declining]

            start = np.zeros(3)
            fit = least_squares(residual, start, bounds=(-4., 4.), max_nfev=60,
                                ftol=1e-5, xtol=1e-5, gtol=1e-5)
            if np.isfinite(fit.x).all() and fit.cost <= .5*np.sum(residual(start)**2):
                updated = np.clip(prior+basis @ (fit.x*scales), 0., 132.)
            else:
                updated = prior
            mixed = (1.-self.mix_weight)*baseline[pos] + self.mix_weight*updated
            output[pos] = monotone_curve(age, mixed)
        prediction = np.empty(len(frame))
        prediction[order] = output
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite posterior prediction.")
        return prediction
