import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_regression
from sklearn.preprocessing import StandardScaler

from .base import Task


class MNISTBinaryTask(Task):
    """Binary classification on downscaled MNIST (0 vs 1)"""

    def __init__(self):
        super().__init__("MNISTBinary", "classification")

    def load_data(
        self, n_components: int = 16, n_samples: int = 2000, train_split: float = 0.5
    ):
        print("📥 loading MNIST...")

        # Fetch MNIST from OpenML
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X = mnist.data[:n_samples].astype(np.float32) / 255.0
        y = mnist.target[:n_samples].astype(np.int32)

        mask = (y == 0) | (y == 1)
        X = X[mask]
        y = y[mask]
        y = (y == 1).astype(np.float32)

        proj_matrix = np.random.randn(X.shape[1], n_components).astype(np.float32)
        proj_matrix /= np.linalg.norm(proj_matrix, axis=0)
        X_mini = X @ proj_matrix

        # Normalize
        X_mini = (X_mini - X_mini.mean(axis=0)) / (X_mini.std(axis=0) + 1e-6)

        print(f"✅ MNIST binary: {X.shape} -> {X_mini.shape}")
        print(f"   class balance: {y.mean():.2f}")

        n_train = int(train_split * len(X_mini))
        indices = np.random.permutation(len(X_mini))
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        self.X_train = X_mini[train_idx]
        self.y_train = y[train_idx]
        self.X_val = X_mini[val_idx]
        self.y_val = y[val_idx]
        self.input_dim = n_components

    def evaluate(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        pred_classes = (predictions > 0.5).astype(int)
        return np.mean(pred_classes == labels)

    def get_task_description(self) -> str:
        return f"Binary classification ({self.input_dim}D inputs) - MNIST digit 0 vs 1"

    def get_baseline_fitness(self) -> float:
        return -np.inf
