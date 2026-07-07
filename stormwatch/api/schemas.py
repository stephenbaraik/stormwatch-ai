"""
StormWatch AI - API Schemas
Pydantic models for request/response validation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
#  Health
# ──────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    models_loaded: int = 0
    models: List[str] = []


# ──────────────────────────────────────────────
#  Cyclone Intensity  (8 features, match builder)
# ──────────────────────────────────────────────


class CycloneFeatures(BaseModel):
    lat_abs: float = Field(..., description="Absolute latitude (0-90)", ge=0, le=90)
    lon: float = Field(..., description="Longitude (-180 to 180)", ge=-180, le=180)
    lat: float = Field(..., description="Latitude (-90 to 90)", ge=-90, le=90)
    pressure_min: float = Field(
        ..., description="Minimum central pressure (hPa)", ge=850, le=1050
    )
    dist_to_land: float = Field(0, description="Distance to land (km)", ge=0)
    year: int = Field(2024, description="Year", ge=1900, le=2100)
    month: int = Field(..., description="Month (1-12)", ge=1, le=12)
    dayofyear: int = Field(..., description="Day of year (1-366)", ge=1, le=366)


class CyclonePrediction(BaseModel):
    category: int = Field(..., description="Predicted Saffir-Simpson category (0-5)")
    description: str = Field(..., description="Category description")
    probabilities: Dict[str, float] = Field(..., description="Probability per category")
    confidence: float = Field(
        ..., description="Prediction confidence (max probability)"
    )


# ──────────────────────────────────────────────
#  Heatwave  (13 features, match builder)
# ──────────────────────────────────────────────


class HeatwaveFeatures(BaseModel):
    temp_max_lag_1: float = Field(..., description="Max temperature 1 day ago (°C)")
    temp_max_lag_3: float = Field(0, description="Max temperature 3 days ago (°C)")
    temp_max_lag_7: float = Field(0, description="Max temperature 7 days ago (°C)")
    temp_max_roll_mean_3: float = Field(
        ..., description="3-day rolling mean of prior-day max temp (°C)"
    )
    temp_max_roll_mean_7: float = Field(
        0, description="7-day rolling mean of prior-day max temp (°C)"
    )
    temp_min_lag_1: float = Field(..., description="Min temperature 1 day ago (°C)")
    precipitation_lag_1: float = Field(0, description="Precipitation 1 day ago (mm)")
    relative_humidity_2m_mean: float = Field(
        ..., description="Relative humidity (%)", ge=0, le=100
    )
    wind_speed_10m_max: float = Field(..., description="Max wind speed (km/h)", ge=0)
    pressure_msl_mean: float = Field(..., description="Mean sea level pressure (hPa)")
    month_sin: float = Field(..., description="Month sin encoding")
    month_cos: float = Field(..., description="Month cos encoding")
    month: int = Field(..., description="Month (1-12)", ge=1, le=12)


class HeatwavePrediction(BaseModel):
    heatwave_probability: float = Field(
        ..., description="Probability of heatwave (0-1)", ge=0, le=1
    )
    is_heatwave: bool = Field(..., description="Binary prediction")
    severity: str = Field(
        ..., description="Severity level: none / watch / warning / severe"
    )
    confidence: float = Field(..., description="Prediction confidence")


# ──────────────────────────────────────────────
#  Extreme Rainfall  (14 features, match builder)
# ──────────────────────────────────────────────


class RainfallFeatures(BaseModel):
    precipitation_lag_1: float = Field(0, description="Precipitation 1 day ago (mm)")
    precipitation_lag_3: float = Field(0, description="Precipitation 3 days ago (mm)")
    precipitation_lag_7: float = Field(0, description="Precipitation 7 days ago (mm)")
    precipitation_roll_mean_3: float = Field(
        ..., description="3-day rolling mean of prior-day precipitation (mm)"
    )
    precipitation_roll_mean_7: float = Field(
        0, description="7-day rolling mean of prior-day precipitation (mm)"
    )
    temp_max_lag_1: float = Field(..., description="Max temperature 1 day ago (°C)")
    temp_max_roll_mean_3: float = Field(
        ..., description="3-day rolling mean of prior-day max temp (°C)"
    )
    relative_humidity_2m_mean: float = Field(
        ..., description="Relative humidity (%)", ge=0, le=100
    )
    wind_speed_10m_max: float = Field(..., description="Max wind speed (km/h)", ge=0)
    pressure_msl_mean: float = Field(..., description="Mean sea level pressure (hPa)")
    cloud_cover_mean: float = Field(..., description="Cloud cover (%)", ge=0, le=100)
    month_sin: float = Field(..., description="Month sin encoding")
    month_cos: float = Field(..., description="Month cos encoding")
    month: int = Field(..., description="Month (1-12)", ge=1, le=12)


class RainfallPrediction(BaseModel):
    extreme_rainfall_probability: float = Field(
        ..., description="Probability of extreme rainfall (0-1)", ge=0, le=1
    )
    is_extreme: bool = Field(..., description="Binary prediction")
    confidence: float = Field(..., description="Prediction confidence")


# ──────────────────────────────────────────────
#  Generic
# ──────────────────────────────────────────────


class PredictionResponse(BaseModel):
    model: str = Field(..., description="Model name")
    version: str = Field("1.0.0", description="Model version")
    prediction: dict
    warning: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# ──────────────────────────────────────────────
#  Monitoring
# ──────────────────────────────────────────────


class DriftReport(BaseModel):
    model_name: str
    drift_detected: bool
    drift_score: float
    metrics: Dict[str, float]
    recommendation: str


# ──────────────────────────────────────────────
#  Saffir-Simpson lookup
# ──────────────────────────────────────────────

SAFFIR_SIMPSON: Dict[int, str] = {
    0: "Tropical Depression",
    1: "Category 1",
    2: "Category 2",
    3: "Category 3",
    4: "Category 4",
    5: "Category 5",
}


def get_category_description(category: int) -> str:
    """Return human-readable Saffir-Simpson description."""
    return SAFFIR_SIMPSON.get(category, f"Unknown (category {category})")
