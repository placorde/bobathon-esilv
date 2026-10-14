"""MAE-focused patient intercept; reuse experiment 12's fitted components."""
from pathlib import Path
import copy
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from residual_trajectory import FittedBlend


class StableVectorizer(TransformerMixin, BaseEstimator):
    """Keep the fitted vocabulary; canonicalize numeric tree inputs."""
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        raise RuntimeError("This wrapper uses an already-fitted vectorizer.")

    def transform(self, X):
        matrix = self.vectorizer.transform(X)
        if isinstance(matrix, pd.DataFrame):
            matrix = matrix.copy()
            numeric = matrix.select_dtypes(include="number").columns
            matrix[numeric] = matrix[numeric].round(8)
            return matrix
        return np.round(matrix, 8)

    def get_feature_names_out(self, input_features=None):
        return self.vectorizer.get_feature_names_out(input_features)


class InverseBlendRegressor(RegressorMixin, BaseEstimator):
    """Compatibility name for reading experiment 12's __main__ pickle."""
    def fit(self, X, y):
        raise RuntimeError("Use CachedMAEPatientBlend for the new experiment.")

    def predict(self, X):
        return self.blend_final_.predict(X)


def atomic_dump(value, destination):
    destination = Path(destination)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(destination)


def training_signature(X, y):
    ordered = pd.DataFrame({"Index": X.Index.to_numpy(), "target": np.asarray(y, dtype=float)}).sort_values("Index")
    return hashlib.sha256(pd.util.hash_pandas_object(ordered, index=False).to_numpy().tobytes()).hexdigest()


class CachedMAEPatientBlend(RegressorMixin, BaseEstimator):
    """Change only B's intercept loss; A, C and B's shape models stay frozen.

    Cache reuse is allowed only for the exact experiment-12 training partitions.
    Patient IDs identify those partitions, never enter the learned predictors.
    """
    def __init__(self, project_root, variant="baseline_clipped"):
        self.project_root = project_root
        self.variant = variant

    def fit(self, X, y):
        if self.variant not in {"baseline_clipped", "mae_intercept"}:
            raise ValueError("Unknown variant.")
        root = Path(self.project_root)
        full = pd.read_csv(root / "data" / "X_train.csv")
        labels = pd.read_csv(root / "data" / "y_train.csv").set_index("Index")["target"]
        full_y = labels.loc[full.Index].to_numpy()
        signature = training_signature(X, y)
        if signature == training_signature(full, full_y):
            prefix = "full_train"
        else:
            matches = []
            for fold, (fit, val) in enumerate(GroupKFold(5).split(full, full_y, full.patient_id), 1):
                if signature == training_signature(full.iloc[fit], full_y[fit]):
                    matches.append(f"fold_{fold}")
                    assert not set(full.iloc[fit].patient_id) & set(full.iloc[val].patient_id)
            if len(matches) != 1:
                raise ValueError("Training rows/labels do not match an experiment-12 fold.")
            prefix = matches[0]
        expected = full.set_index("Index").loc[X.Index].reset_index()
        pd.testing.assert_frame_equal(X.reset_index(drop=True), expected[X.columns], check_dtype=False)
        cache = root / "data" / "day2_12_cache"
        print(f"  {prefix}: loading saved A, B and C (no calibration training)", flush=True)
        a = joblib.load(cache / f"{prefix}_flexible_patient_curve.joblib")
        b = joblib.load(cache / f"{prefix}_orthogonal_curve.joblib")
        c = joblib.load(cache / f"{prefix}_inverse_curve.joblib")
        # A/B calibration can yield 1e-12 differences depending on batch shape.
        # Their fitted trees may split at numerical-zero boundaries. Stabilize
        # the tree inputs without changing raw measurements or calibration fits.
        a.vectorizer_ = StableVectorizer(a.vectorizer_)
        b.vectorizer_ = StableVectorizer(b.vectorizer_)
        if self.variant == "mae_intercept":
            output = root / "data" / "day2_15_mae_intercept_stable_v2"
            output.mkdir(exist_ok=True)
            model_path = output / f"{prefix}_{signature[:16]}_intercept.joblib"
            if model_path.exists():
                intercept = joblib.load(model_path)
                print("  Reusing saved MAE intercept", flush=True)
            else:
                table, native_prior = b.summarize(X, b.features_.transform(X))
                _, prior, counts, _ = b._geometry(X, native_prior)
                target = np.asarray(y, dtype=float)
                means = np.asarray([target[pos].mean() for pos in X.groupby("patient_id", sort=False).indices.values()])
                outputs = means - prior[:, 0] if b.residual else means
                matrix = b.vectorizer_.transform(table)
                intercept = HistGradientBoostingRegressor(loss="absolute_error", max_iter=500,
                    max_leaf_nodes=15, learning_rate=.04, l2_regularization=15.,
                    early_stopping=False, random_state=42)
                print("  Training one MAE intercept booster; slopes, curvature and inverse models reused", flush=True)
                intercept.fit(matrix, outputs, sample_weight=counts / counts.mean())
                atomic_dump(intercept, model_path)
            b = copy.copy(b)
            b.models_ = [intercept, *b.models_[1:]]
        self.model_ = FittedBlend(FittedBlend(a, b, second_weight=.5), c, second_weight=1./3.)
        self.training_signature_ = signature
        return self

    def predict(self, X):
        if "target" in X.columns:
            raise ValueError("Targets cannot enter prediction.")
        if not X.Index.is_unique:
            raise ValueError("Duplicate row identifiers.")
        # Canonicalize numerical reductions before restoring the caller's order.
        # Index is used only for ordering/alignment, never as a learned feature.
        order = np.argsort(X.Index.to_numpy(), kind="stable")
        ordered = X.iloc[order].reset_index(drop=True)
        ordered_prediction = self.model_.predict(ordered)
        prediction = np.empty(len(X))
        prediction[order] = ordered_prediction
        if not np.isfinite(prediction).all():
            raise ValueError("Non-finite prediction.")
        # The target is nonnegative; this fixed constraint is tested on all folds.
        return np.maximum(prediction, 0.)
