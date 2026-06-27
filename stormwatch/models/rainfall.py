"""
StormWatch AI - Extreme Rainfall Prediction Model
Binary classification: will precipitation exceed the 95th percentile?
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stormwatch.features.builder import RAINFALL_FEATURES, build_rainfall_features
from stormwatch.models.base import BaseWeatherModel


class ExtremeRainfallModel(BaseWeatherModel):
    """Predicts extreme rainfall events.

    Uses Random Forest with balanced weights to handle the
    natural imbalance of extreme precipitation events.

    Target: binary (1 = extreme rainfall, 0 = normal)
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.feature_names = [f for f in RAINFALL_FEATURES]

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=self.config.get("n_estimators", 200),
                    max_depth=self.config.get("max_depth", 12),
                    min_samples_split=self.config.get("min_samples_split", 5),
                    class_weight="balanced_subsample",
                    random_state=self.config.get("random_state", 42),
                    n_jobs=-1,
                ),
            ),
        ])

    def build_features(self, df: pd.DataFrame):
        """Build rainfall features from preprocessed DataFrame."""
        return build_rainfall_features(df)


class RainfallXGBModel(BaseWeatherModel):
    """XGBoost-based extreme rainfall model."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.feature_names = [f for f in RAINFALL_FEATURES]

    def build_pipeline(self) -> Pipeline:
        try:
            from xgboost import XGBClassifier
            return Pipeline([
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=self.config.get("n_estimators", 200),
                        max_depth=self.config.get("max_depth", 8),
                        learning_rate=self.config.get("learning_rate", 0.1),
                        scale_pos_weight=self.config.get("scale_pos_weight", 10),
                        eval_metric="logloss",
                        random_state=self.config.get("random_state", 42),
                        n_jobs=-1,
                    ),
                ),
            ])
        except ImportError:
            return ExtremeRainfallModel(self.config).build_pipeline()

    def build_features(self, df: pd.DataFrame):
        return build_rainfall_features(df)
