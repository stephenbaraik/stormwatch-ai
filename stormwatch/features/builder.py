"""
StormWatch AI - Feature Builder Module
Provides feature engineering pipelines for each model.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from stormwatch.logger import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────
#  Cyclone features
# ──────────────────────────────────────────────

CYCLONE_FEATURES = [
    "lat_abs",
    "lon",
    "lat",
    "pressure_min",
    "dist_to_land",
    "year",
    "month",
    "dayofyear",
    "wind_kts",
]


def build_cyclone_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix and target for cyclone intensity prediction.

    Features:
    - Absolute latitude (correlates with cyclone strength)
    - Longitude
    - Pressure minimum
    - Distance to land
    - Temporal features (month, day of year)
    - Wind speed (as baseline feature)

    Target: Saffir-Simpson category (0-5)

    Returns:
        (X, y) where X is feature DataFrame, y is target Series
    """
    df = df.copy()

    # Available columns
    available = [c for c in CYCLONE_FEATURES if c in df.columns]

    if not available:
        log.error("No cyclone features available in DataFrame")
        return pd.DataFrame(), pd.Series(dtype="int64")

    X = df[available].copy()

    # Handle missing values
    for col in X.columns:
        if X[col].dtype in (np.float64, np.float32):
            X[col] = X[col].fillna(X[col].median())

    y = (
        df["category"].astype(int)
        if "category" in df.columns
        else pd.Series(dtype="int64")
    )

    # Ensure all categories 0-5 are present
    if not y.empty:
        log.info("Cyclone features: %d samples, %d features", len(X), X.shape[1])
        log.info("  Class distribution: %s", y.value_counts().to_dict())

    return X, y


# ──────────────────────────────────────────────
#  Heatwave features
# ──────────────────────────────────────────────

HEATWAVE_FEATURES = [
    "temp_max",
    "temp_max_lag_1",
    "temp_max_lag_3",
    "temp_max_roll_mean_3",
    "temp_max_roll_mean_7",
    "temp_min",
    "precipitation",
    "precipitation_lag_1",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "pressure_msl_mean",
    "month_sin",
    "month_cos",
    "month",
]


def build_heatwave_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix and target for heatwave prediction.

    Features:
    - Temperature (current and lagged)
    - Rolling temperature means
    - Humidity
    - Wind speed
    - Pressure
    - Seasonal features

    Target: heatwave_flag (binary)

    Returns:
        (X, y)
    """
    df = df.copy()

    available = [c for c in HEATWAVE_FEATURES if c in df.columns]
    X = df[available].copy() if available else pd.DataFrame()

    # Fill remaining NaN for numeric columns
    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)

    # Target
    target_col = "heatwave_flag"
    if target_col not in df.columns:
        log.error("Target column '%s' not found", target_col)
        return X, pd.Series(dtype="int64")

    y = df[target_col].astype(int)

    if not X.empty:
        log.info(
            "Heatwave features: %d samples, %d features, %d positive",
            len(X),
            X.shape[1],
            y.sum(),
        )

    return X, y


# ──────────────────────────────────────────────
#  Rainfall features
# ──────────────────────────────────────────────

RAINFALL_FEATURES = [
    "precipitation",
    "precipitation_lag_1",
    "precipitation_lag_3",
    "precipitation_roll_mean_3",
    "precipitation_roll_mean_7",
    "temp_max",
    "temp_max_roll_mean_3",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "pressure_msl_mean",
    "cloud_cover_mean",
    "month_sin",
    "month_cos",
    "month",
]


def build_rainfall_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix and target for extreme rainfall prediction.

    Features:
    - Precipitation (current and lagged)
    - Rolling precipitation statistics
    - Temperature
    - Humidity
    - Wind speed
    - Pressure
    - Cloud cover
    - Seasonal features

    Target: extreme_rainfall (binary)

    Returns:
        (X, y)
    """
    df = df.copy()

    available = [c for c in RAINFALL_FEATURES if c in df.columns]
    X = df[available].copy() if available else pd.DataFrame()

    for col in X.select_dtypes(include=[np.number]).columns:
        X[col] = X[col].fillna(X[col].median() if not X[col].isna().all() else 0)

    target_col = "extreme_rainfall"
    if target_col not in df.columns:
        log.error("Target column '%s' not found", target_col)
        return X, pd.Series(dtype="int64")

    y = df[target_col].astype(int)

    if not X.empty:
        log.info(
            "Rainfall features: %d samples, %d features, %d positive",
            len(X),
            X.shape[1],
            y.sum(),
        )

    return X, y


# ──────────────────────────────────────────────
#  Utility: get pipeline for scaling
# ──────────────────────────────────────────────


def get_preprocessing_pipeline() -> Pipeline:
    """Return a sklearn Pipeline for feature scaling."""
    return Pipeline([("scaler", StandardScaler())])
