import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, make_regression
from sklearn.preprocessing import StandardScaler

from .base import Task


class TitanicTask(Task):
    """Binary classification on Titanic survival dataset"""

    def __init__(self):
        super().__init__("Titanic", "classification")

    def load_data(self, train_split: float = 0.8):
        print("📥 loading titanic dataset...")

        titanic = fetch_openml("titanic", version=1, as_frame=True, parser="auto")
        X_df = titanic.data
        y = titanic.target

        y = (y == "1").astype(np.float32).values  # ADD .values HERE

        # Feature engineering
        num_features = ["age", "fare"]
        cat_features = ["sex", "pclass", "embarked"]

        X_num = X_df[num_features].fillna(X_df[num_features].median())

        X_cat = pd.get_dummies(X_df[cat_features], drop_first=True)

        X = pd.concat([X_num, X_cat], axis=1).values.astype(np.float32)

        print(f"X shape: {X.shape}, y shape: {y.shape}")
        assert len(X) == len(y), f"length mismatch: X={len(X)}, y={len(y)}"

        # Normalize
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        print(f"✅ titanic: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"   survival rate: {y.mean():.2f}")

        # train/val split
        indices = np.random.permutation(len(X))
        n_train = int(train_split * len(X))
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        self.X_train = X[train_idx]
        self.y_train = y[train_idx]  # now numpy array indexing
        self.X_val = X[val_idx]
        self.y_val = y[val_idx]
        self.input_dim = X.shape[1]

    def evaluate(self, predictions: np.ndarray, labels: np.ndarray) -> float:
        pred_classes = (predictions > 0.5).astype(int)
        return np.mean(pred_classes == labels)

    def get_task_description(self) -> str:
        return f"Binary classification ({self.input_dim}D inputs) - Titanic survival prediction"

    def get_baseline_fitness(self) -> float:
        # Class imbalance baseline
        return -np.inf
        # return max(self.y_train.mean(), 1 - self.y_train.mean())
