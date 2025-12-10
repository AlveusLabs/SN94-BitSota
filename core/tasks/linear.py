import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_regression
from sklearn.preprocessing import StandardScaler

from .base import Task


class LinearRegressionTask(Task):
    """Synthetic linear regression task"""

    def __init__(self):
        super().__init__("LinearRegression", "regression")

    def load_data(
        self,
        n_features: int = 10,
        n_samples: int = 1000,
        noise: float = 0.1,
        train_split: float = 0.8,
    ):
        print("📥 generating linear regression data...")

        # Generate synthetic data
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features // 2,
            noise=noise,
            random_state=42,
        )

        # Normalize
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-6)
        y = (y - y.mean()) / (y.std() + 1e-6)

        print(f"✅ linear regression: {X.shape[0]} samples, {X.shape[1]} features")

        # train/val split
        n_train = int(train_split * len(X))
        indices = np.random.permutation(len(X))
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        self.X_train = X[train_idx].astype(np.float32)
        self.y_train = y[train_idx].astype(np.float32)
        self.X_val = X[val_idx].astype(np.float32)
        self.y_val = y[val_idx].astype(np.float32)
        self.input_dim = n_features

    def evaluate(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        # DON'T clip negative R² - we need the signal!
        if not np.all(np.isfinite(predictions)):
            return -99999999
        mae = np.mean(np.abs(predictions - labels))
        return -mae  # negative so higher is better (evolution maximizes)

    def get_task_description(self) -> str:
        return (
            f"Linear regression ({self.input_dim}D inputs) - predict continuous target"
        )

    def get_baseline_fitness(self) -> float:
        return -np.inf
