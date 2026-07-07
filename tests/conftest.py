from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from stormwatch.data.preprocess import label_extreme_events, prepare_weather_features
from stormwatch.models.cyclone import CycloneIntensityModel
from stormwatch.models.heatwave import HeatwavePredictionModel
from stormwatch.models.rainfall import ExtremeRainfallModel


@pytest.fixture(scope="session")
def mock_weather_df() -> pd.DataFrame:
    """Minimal mock weather DataFrame matching the real Open-Meteo schema.

    Includes extreme temperature days to exercise heatwave detection.
    """
    n_days = 90  # 3 months per city
    cities = ["Mumbai", "Chennai", "Delhi"]
    rows: list[dict] = []

    rng = np.random.default_rng(42)
    for city in cities:
        for d in range(n_days):
            base_temp = 30.0 if city in ("Mumbai", "Chennai") else 28.0
            t = base_temp + 5 * np.sin(2 * np.pi * d / 365) + float(rng.normal(0, 1))

            # Inject a multi-day heatwave (days 20-23 for each city)
            if 20 <= d <= 23:
                t = 44.0 + float(rng.uniform(-1, 2))

            # Inject extreme rainfall (day 50 for each city)
            p = max(0, 10 + 8 * np.sin(2 * np.pi * (d + 30) / 365) + float(rng.exponential(3)))
            if d == 50:
                p = 120.0 + float(rng.uniform(0, 30))

            rows.append({
                "city": city,
                "time": pd.Timestamp("2024-01-01") + pd.Timedelta(days=d),
                "temperature_2m_max": round(t + float(rng.uniform(2, 5)), 1),
                "temperature_2m_min": round(t - float(rng.uniform(4, 8)), 1),
                "temperature_2m_mean": round(t - float(rng.uniform(1, 3)), 1),
                "precipitation_sum": round(p, 1),
                "rain_sum": round(p * 0.9, 1),
                "snowfall_sum": 0.0,
                "precipitation_hours": float(rng.poisson(max(1, p / 4))),
                "wind_speed_10m_max": round(float(rng.exponential(12)), 1),
                "wind_gusts_10m_max": round(float(rng.exponential(18)), 1),
                "wind_direction_10m_dominant": float(rng.uniform(0, 360)),
                "pressure_msl_mean": round(1013 + float(rng.normal(0, 5)), 1),
                "relative_humidity_2m_mean": float(rng.integers(40, 95)),
                "cloud_cover_mean": float(rng.integers(10, 95)),
                "shortwave_radiation_sum": float(rng.uniform(1000, 7000)),
                "et0_fao_evapotranspiration": round(float(rng.uniform(2, 8)), 2),
                "latitude": 19.1 if city == "Mumbai" else (13.1 if city == "Chennai" else 28.6),
                "longitude": 72.9 if city == "Mumbai" else (80.3 if city == "Chennai" else 77.2),
            })

    df = pd.DataFrame(rows)
    df = label_extreme_events(df)
    df = prepare_weather_features(df)
    return df


@pytest.fixture(scope="session")
def mock_cyclone_df() -> pd.DataFrame:
    """Minimal mock cyclone DataFrame matching IBTrACS schema."""
    n = 100
    rng = np.random.default_rng(42)
    wind = rng.exponential(40, n) + 25
    df = pd.DataFrame({
        "lat_abs": rng.uniform(5, 25, n),
        "lon": rng.uniform(60, 95, n),
        "lat": rng.uniform(-25, 25, n),
        "pressure_min": 1020 - wind * 0.5 + rng.normal(0, 5, n),
        "dist_to_land": rng.uniform(0, 500, n),
        "year": rng.integers(2010, 2025, n),
        "month": rng.integers(1, 13, n),
        "dayofyear": rng.integers(1, 366, n),
        "wind_kts": wind,
    })

    conditions = [
        (df["wind_kts"] < 34),
        (df["wind_kts"] >= 34) & (df["wind_kts"] < 64),
        (df["wind_kts"] >= 64) & (df["wind_kts"] < 83),
        (df["wind_kts"] >= 83) & (df["wind_kts"] < 96),
        (df["wind_kts"] >= 96) & (df["wind_kts"] < 113),
        (df["wind_kts"] >= 113),
    ]
    df["category"] = np.select(conditions, [0, 1, 2, 3, 4, 5], default=0)
    return df


@pytest.fixture
def cyclone_model() -> CycloneIntensityModel:
    from stormwatch.features.builder import CYCLONE_FEATURES

    model = CycloneIntensityModel()
    model.feature_names = list(CYCLONE_FEATURES)
    return model


@pytest.fixture
def heatwave_model() -> HeatwavePredictionModel:
    return HeatwavePredictionModel()


@pytest.fixture
def rainfall_model() -> ExtremeRainfallModel:
    return ExtremeRainfallModel()


@pytest.fixture
def trained_cyclone_model(mock_cyclone_df: pd.DataFrame) -> CycloneIntensityModel:
    from stormwatch.features.builder import build_cyclone_features
    X, y = build_cyclone_features(mock_cyclone_df)
    model = CycloneIntensityModel(config={"n_estimators": 20, "max_depth": 5})
    model.train(X, y)
    return model


@pytest.fixture
def trained_heatwave_model(mock_weather_df: pd.DataFrame) -> HeatwavePredictionModel:
    from stormwatch.features.builder import build_heatwave_features
    X, y = build_heatwave_features(mock_weather_df)
    model = HeatwavePredictionModel(config={"n_estimators": 20, "max_depth": 3})
    model.train(X, y)
    return model


@pytest.fixture
def trained_rainfall_model(mock_weather_df: pd.DataFrame) -> ExtremeRainfallModel:
    from stormwatch.features.builder import build_rainfall_features
    X, y = build_rainfall_features(mock_weather_df)
    model = ExtremeRainfallModel(config={"n_estimators": 20, "max_depth": 5})
    model.train(X, y)
    return model


@pytest.fixture
def sample_cyclone_features() -> Dict[str, Any]:
    return {
        "lat_abs": 15.5,
        "lon": 80.0,
        "lat": 15.5,
        "pressure_min": 980.0,
        "dist_to_land": 50.0,
        "year": 2024,
        "month": 10,
        "dayofyear": 285,
    }


@pytest.fixture
def sample_heatwave_features() -> Dict[str, Any]:
    return {
        "temp_max_lag_1": 42.0,
        "temp_max_lag_3": 40.0,
        "temp_max_lag_7": 38.0,
        "temp_max_roll_mean_3": 41.0,
        "temp_max_roll_mean_7": 39.5,
        "temp_min_lag_1": 28.0,
        "precipitation_lag_1": 0.0,
        "relative_humidity_2m_mean": 25.0,
        "wind_speed_10m_max": 15.0,
        "pressure_msl_mean": 1008.0,
        "month_sin": 0.5,
        "month_cos": 0.866,
        "month": 6,
    }


@pytest.fixture
def sample_rainfall_features() -> Dict[str, Any]:
    return {
        "precipitation_lag_1": 80.0,
        "precipitation_lag_3": 30.0,
        "precipitation_lag_7": 10.0,
        "precipitation_roll_mean_3": 45.0,
        "precipitation_roll_mean_7": 35.0,
        "temp_max_lag_1": 32.0,
        "temp_max_roll_mean_3": 31.0,
        "relative_humidity_2m_mean": 85.0,
        "wind_speed_10m_max": 25.0,
        "pressure_msl_mean": 1002.0,
        "cloud_cover_mean": 80.0,
        "month_sin": -0.5,
        "month_cos": 0.866,
        "month": 7,
    }
