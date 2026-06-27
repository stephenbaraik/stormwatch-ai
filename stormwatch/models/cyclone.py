"""
StormWatch AI - Cyclone Intensity Prediction Model
Multi-class classification of Saffir-Simpson category (0-5).
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stormwatch.features.builder import CYCLONE_FEATURES, build_cyclone_features
from stormwatch.models.base import BaseWeatherModel


class CycloneIntensityModel(BaseWeatherModel):
    """Predicts Saffir-Simpson cyclone intensity category (0-5).

    Uses Random Forest with balanced class weights to handle
    the natural imbalance in cyclone categories.
    """

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.feature_names = CYCLONE_FEATURES[:]

    def build_pipeline(self) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=self.config.get("n_estimators", 200),
                        max_depth=self.config.get("max_depth", 15),
                        min_samples_split=self.config.get("min_samples_split", 5),
                        class_weight="balanced",
                        random_state=self.config.get("random_state", 42),
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    def build_features(self, df: pd.DataFrame):
        """Build cyclone features from raw DataFrame."""
        return build_cyclone_features(df)


class CycloneIntensityXGB(BaseWeatherModel):
    """XGBoost-based cyclone intensity model (better performance)."""

    def __init__(self, config: Dict[str, Any] | None = None):
        super().__init__(config)
        self.feature_names = CYCLONE_FEATURES[:]

    def build_pipeline(self) -> Pipeline:
        try:
            from xgboost import XGBClassifier

            return Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "classifier",
                        XGBClassifier(
                            n_estimators=self.config.get("n_estimators", 200),
                            max_depth=self.config.get("max_depth", 8),
                            learning_rate=self.config.get("learning_rate", 0.1),
                            objective="multi:softprob",
                            num_class=6,  # categories 0-5
                            eval_metric="mlogloss",
                            random_state=self.config.get("random_state", 42),
                            n_jobs=-1,
                        ),
                    ),
                ]
            )
        except ImportError:
            # Fallback to RandomForest
            return CycloneIntensityModel(self.config).build_pipeline()

    def build_features(self, df: pd.DataFrame):
        return build_cyclone_features(df)
