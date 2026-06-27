"""
StormWatch AI - Heatwave Prediction Model
Binary classification: will a heatwave occur given current conditions?
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stormwatch.features.builder import HEATWAVE_FEATURES, build_heatwave_features
from stormwatch.models.base import BaseWeatherModel


class HeatwavePredictionModel(BaseWeatherModel):
    """Predicts whether heatwave conditions will occur.

    Uses Gradient Boosting, which handles the temporal dependencies
    and interaction effects between temperature, humidity, and pressure.

    Target: binary (1 = heatwave occurring, 0 = normal)
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.feature_names = [f for f in HEATWAVE_FEATURES]

    def build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            (
                "classifier",
                GradientBoostingClassifier(
                    n_estimators=self.config.get("n_estimators", 150),
                    max_depth=self.config.get("max_depth", 5),
                    learning_rate=self.config.get("learning_rate", 0.1),
                    min_samples_split=self.config.get("min_samples_split", 10),
                    subsample=0.8,
                    random_state=self.config.get("random_state", 42),
                ),
            ),
        ])

    def build_features(self, df: pd.DataFrame):
        """Build heatwave features from preprocessed DataFrame."""
        return build_heatwave_features(df)


class HeatwaveXGBModel(BaseWeatherModel):
    """XGBoost-based heatwave model."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.feature_names = [f for f in HEATWAVE_FEATURES]

    def build_pipeline(self) -> Pipeline:
        try:
            from xgboost import XGBClassifier
            return Pipeline([
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=self.config.get("n_estimators", 200),
                        max_depth=self.config.get("max_depth", 6),
                        learning_rate=self.config.get("learning_rate", 0.08),
                        scale_pos_weight=self.config.get("scale_pos_weight", 5),
                        eval_metric="logloss",
                        random_state=self.config.get("random_state", 42),
                        n_jobs=-1,
                    ),
                ),
            ])
        except ImportError:
            return HeatwavePredictionModel(self.config).build_pipeline()

    def build_features(self, df: pd.DataFrame):
        return build_heatwave_features(df)
