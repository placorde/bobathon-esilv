"""Learn residuals around a fold-fitted patient trajectory."""
import numpy as np
import skrub
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from response_features import ResponseFeatures


class ResidualTrajectoryRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, max_leaf_nodes=15):
        self.max_leaf_nodes = max_leaf_nodes

    def prior(self, features):
        prior = features["pooled_quadratic_score"].to_numpy(dtype=float)
        return np.where(np.isfinite(prior), prior, self.fallback_)

    def fit(self, X, y):
        self.fallback_ = float(np.mean(y))
        self.features_ = ResponseFeatures(alpha=30., use_trajectory=True)
        features = self.features_.fit_transform(X, y)
        self.vectorizer_ = skrub.TableVectorizer()
        matrix = self.vectorizer_.fit_transform(features)
        self.regressor_ = HistGradientBoostingRegressor(
            max_iter=600, learning_rate=.04, max_leaf_nodes=self.max_leaf_nodes,
            l2_regularization=20., early_stopping=False, random_state=42)
        self.regressor_.fit(matrix, np.asarray(y) - self.prior(features))
        return self

    def predict(self, X):
        features = self.features_.transform(X)
        return self.prior(features) + self.regressor_.predict(self.vectorizer_.transform(features))


class FittedBlend(RegressorMixin, BaseEstimator):
    def __init__(self, first, second, second_weight=.5):
        self.first = first
        self.second = second
        self.second_weight = second_weight

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y):
        raise RuntimeError("This wrapper combines already fitted models only.")

    def predict(self, X):
        return ((1 - self.second_weight) * self.first.predict(X)
                + self.second_weight * self.second.predict(X))
