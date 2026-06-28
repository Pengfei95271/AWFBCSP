from abc import ABC, abstractmethod
import numpy as np

class BaseFeature(ABC):
    """Base class for feature extraction methods"""

    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, X, y=None):
        """Fit the feature extractor"""
        pass

    @abstractmethod
    def transform(self, X):
        """Transform data to features"""
        pass

    def fit_transform(self, X, y=None):
        """Fit and transform data"""
        return self.fit(X, y).transform(X)

    def _check_is_fitted(self):
        """Check if the extractor is fitted"""
        if not self.is_fitted:
            raise ValueError("Feature extractor not fitted. Call fit() first.")

    def _validate_input(self, X):
        """Validate input data"""
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array, got {X.ndim}D")
        return X