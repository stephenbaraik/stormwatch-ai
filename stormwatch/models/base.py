"""
StormWatch AI - Base Model Interface
Abstract base class for all extreme weather models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


class BaseWeatherModel(ABC):
    """Abstract base class for all StormWatch AI models.

    Subclasses must implement:
    - ``build_pipeline()``: Return a sklearn Pipeline
    - ``get_feature_names()``: Return list of feature names
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: list[str] = []
        self._is_trained: bool = False

    @abstractmethod
    def build_pipeline(self) -> Pipeline:
        """Build and return the sklearn Pipeline for this model."""
        ...

    def get_feature_names(self) -> list[str]:
        """Return the expected feature names."""
        return self.feature_names

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Train the model and return evaluation metrics.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Dictionary of evaluation metrics
        """
        self.feature_names = list(X.columns)
        self.pipeline = self.build_pipeline()

        # Handle class imbalance by computing sample weights
        class_weights = self._compute_class_weights(y)

        self.pipeline.fit(X, y, **self._get_fit_kwargs(class_weights))
        self._is_trained = True

        # Evaluation
        y_pred = self.pipeline.predict(X)
        y_proba = self._predict_proba_safe(self.pipeline, X)

        metrics = self._compute_metrics(y, y_pred, y_proba)
        log = __import__("stormwatch.logger", fromlist=["get_logger"]).get_logger(
            __name__
        )
        log.info("Training complete: %s", metrics)

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Run inference on new data.

        Args:
            X: Feature matrix (must match training features)

        Returns:
            NumPy array of predictions
        """
        if not self._is_trained or self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = self._align_features(X)
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return prediction probabilities.

        Args:
            X: Feature matrix

        Returns:
            NumPy array of probabilities
        """
        if not self._is_trained or self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X = self._align_features(X)
        return self._predict_proba_safe(self.pipeline, X)

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Evaluate model on held-out data.

        Args:
            X: Test feature matrix
            y: Test target vector

        Returns:
            Dictionary of evaluation metrics
        """
        if not self._is_trained or self.pipeline is None:
            raise RuntimeError("Model not trained.")

        X = self._align_features(X)
        y_pred = self.pipeline.predict(X)
        y_proba = self._predict_proba_safe(self.pipeline, X)

        return self._compute_metrics(y, y_pred, y_proba)

    def is_trained(self) -> bool:
        """Check if model has been trained."""
        return self._is_trained

    def save(self, path: str) -> None:
        """Save trained model to disk via joblib."""
        if not self._is_trained:
            raise RuntimeError("Cannot save untrained model")
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> BaseWeatherModel:
        """Load a trained model from disk."""
        return joblib.load(path)

    def log_model(self, artifact_path: str = "model") -> str:
        """Log model to MLflow with signature, input example, and registry.

        Logs the trained sklearn Pipeline directly for native MLflow support.
        Returns the MLflow model URI.
        """
        import mlflow
        from mlflow.models.signature import infer_signature

        if not self._is_trained or self.pipeline is None:
            raise RuntimeError("Cannot log untrained model")

        sample = pd.DataFrame(
            {f: [0.0] for f in self.feature_names} if self.feature_names
            else {"_dummy": [0]}
        )
        for col in sample.columns:
            sample[col] = sample[col].astype(float)
        sample_input = sample.iloc[[0]]
        signature = infer_signature(
            sample_input,
            self.predict(sample_input),
        )

        # MLflow 3.x requires explicit trust for XGBoost sklearn wrappers
        trusted = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
        return mlflow.sklearn.log_model(
            sk_model=self.pipeline,
            artifact_path=artifact_path,
            signature=signature,
            input_example=sample_input,
            registered_model_name=None,
            skops_trusted_types=trusted,
        ).model_uri

    # ── Internal helpers ──

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Ensure X has the same columns as training data."""
        if not self.feature_names:
            return X

        missing = set(self.feature_names) - set(X.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")

        return X[self.feature_names]

    def _compute_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """Compute balanced class weights."""
        counts = y.value_counts()
        if len(counts) <= 1:
            return {0: 1.0}
        n_samples = len(y)
        n_classes = len(counts)
        return {cls: n_samples / (n_classes * count) for cls, count in counts.items()}

    def _get_fit_kwargs(self, class_weights: Dict[int, float]) -> Dict[str, Any]:
        """Return kwargs for pipeline.fit() - override for model specific params."""
        # Default: pass sample_weight based on class weights
        return {}

    def _predict_proba_safe(self, pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
        """Safely get prediction probabilities, returning zeros if not available."""
        try:
            return pipeline.predict_proba(X)
        except (AttributeError, NotImplementedError):
            return np.zeros((len(X), 2))

    def _compute_metrics(
        self, y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        metrics: Dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(
                precision_score(y_true, y_pred, zero_division=0, average="weighted")
            ),
            "recall": float(
                recall_score(y_true, y_pred, zero_division=0, average="weighted")
            ),
            "f1": float(f1_score(y_true, y_pred, zero_division=0, average="weighted")),
        }

        # AUC and log loss for binary classification
        if y_proba.shape[1] >= 2:
            n_classes = len(set(y_true))
            if n_classes == 2:
                try:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                except ValueError:
                    metrics["roc_auc"] = 0.5
                metrics["log_loss"] = float(log_loss(y_true, y_proba[:, 1]))
            elif n_classes > 2:
                try:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true, y_proba, multi_class="ovr")
                    )
                except ValueError:
                    metrics["roc_auc"] = 0.5
                metrics["log_loss"] = float(log_loss(y_true, y_proba))

        return metrics
