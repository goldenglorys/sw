"""
Shared utilities for the NIR classification pipeline.

Importing from here (rather than defining inline in each script) ensures
pickle can locate custom classes when loading saved models in a different script.
"""
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize


class SNVTransformer:
    """Row-wise Standard Normal Variate normalisation."""

    def fit(self, X: np.ndarray) -> "SNVTransformer":
        self._n_features = X.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        means = X.mean(axis=1, keepdims=True)
        stds  = X.std(axis=1, keepdims=True)
        stds  = np.where(stds == 0, 1.0, stds)
        return (X - means) / stds

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class PLSDAClassifier:
    """PCA → PLS-DA wrapper with scikit-learn-like predict API."""

    def __init__(self, n_pca: int, n_pls: int, n_classes: int):
        self.n_pca = n_pca
        self.n_pls = n_pls
        self.n_classes = n_classes
        self.pca = PCA(n_components=n_pca, random_state=42)
        self.pls = PLSRegression(n_components=n_pls, max_iter=500)

    def fit(self, X, y):
        X_pca = self.pca.fit_transform(X)
        Y_ohe = label_binarize(y, classes=np.arange(self.n_classes)).astype(float)
        self.pls.fit(X_pca, Y_ohe)
        return self

    def predict(self, X):
        X_pca = self.pca.transform(X)
        Y_hat = self.pls.predict(X_pca)
        return np.argmax(Y_hat, axis=1)

    def transform_pca(self, X):
        return self.pca.transform(X)
