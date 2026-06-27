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
#  Cyclone Intensity
# ──────────────────────────────────────────────


class CycloneFeatures(BaseModel):
    lat_abs: float = Field(..., description="Absolute latitude (0-90)", ge=0, le=90)
    lon: float = Field(..., description="Longitude (-180 to 180)", ge=-180, le=180)
    lat: float = Field(..., description="Latitude (-90 to 90)", ge=-90, le=90)
    pressure_min: float = Field(..., description="Minimum pressure (hPa)", ge=850, le=1050)
    dist_to_land: float = Field(0, description="Distance to land (km)", ge=0)
    year: int = Field(2024, description="Year", ge=1900, le=2100)
    month: int = Field(..., description="Month (1-12)", ge=1, le=12)
    dayofyear: int = Field(..., description="Day of year (1-366)", ge=1, le=366)
    wind_kts: float = Field(..., description="Max sustained wind (knots)", ge=0)


class CyclonePrediction(BaseModel):
    category: int = Field(..., description="Predicted Saffir-Simpson category (0-5)")
    description: str = Field(..., description="Category description")
    probabilities: Dict[str, float] = Field(..., description="Probability per category")
    wind_kts: float = Field(..., description="Estimated wind speed (knots)")
    confidence: float = Field(..., description="Prediction confidence (max probability)")


# ──────────────────────────────────────────────
#  Heatwave
# ──────────────────────────────────────────────


class HeatwaveFeatures(BaseModel):
    temp_max: float = Field(..., description="Current max temperature (°C)")
    temp_max_lag_1: float = Field(..., description="Max temp 1 day ago (°C)")
    temp_max_lag_3: float = Field(..., description="Max temp 3 days ago (°C)")
    temp_max_roll_mean_3: float = Field(..., description="3-day rolling mean temp (°C)")
    temp_max_roll_mean_7: float = Field(..., description="7-day rolling mean temp (°C)")
    temp_min: float = Field(..., description="Min temperature (°C)")
    precipitation: float = Field(0, description="Precipitation (mm)")
    precipitation_lag_1: float = Field(0, description="Precipitation 1 day ago (mm)")
    relative_humidity_2m_mean: float = Field(..., description="Relative humidity (%)", ge=0, le=100)
    wind_speed_10m_max: float = Field(..., description="Max wind speed (km/h)", ge=0)
    pressure_msl_mean: float = Field(..., description="Mean sea level pressure (hPa)")
    month_sin: float = Field(..., description="Month sin encoding")
    month_cos: float = Field(..., description="Month cos encoding")
    month: int = Field(..., description="Month (1-12)", ge=1, le=12)


class HeatwavePrediction(BaseModel):
    heatwave_probability: float = Field(..., description="Probability of heatwave (0-1)", ge=0, le=1)
    is_heatwave: bool = Field(..., description="Binary prediction")
    severity: str = Field(..., description="Severity level: none / watch / warning / severe")
    confidence: float = Field(..., description="Prediction confidence")


# ──────────────────────────────────────────────
#  Extreme Rainfall
# ──────────────────────────────────────────────


class RainfallFeatures(BaseModel):
    precipitation: float = Field(..., description="Current precipitation (mm)")
    precipitation_lag_1: float = Field(0, description="Precipitation 1 day ago (mm)")
    precipitation_lag_3: float = Field(0, description="Precipitation 3 days ago (mm)")
    precipitation_roll_mean_3: float = Field(..., description="3-day rolling mean precipitation (mm)")
    precipitation_roll_mean_7: float = Field(..., description="7-day rolling mean precipitation (mm)")
    temp_max: float = Field(..., description="Max temperature (°C)")
    temp_max_roll_mean_3: float = Field(..., description="3-day rolling mean temp (°C)")
    relative_humidity_2m_mean: float = Field(..., description="Relative humidity (%)", ge=0, le=100)
    wind_speed_10m_max: float = Field(..., description="Max wind speed (km/h)", ge=0)
    pressure_msl_mean: float = Field(..., description="Mean sea level pressure (hPa)")
    cloud_cover_mean: float = Field(..., description="Cloud cover (%)", ge=0, le=100)
    month_sin: float = Field(..., description="Month sin encoding")
    month_cos: float = Field(..., description="Month cos encoding")
    month: int = Field(..., description="Month (1-12)", ge=1, le=12)


class RainfallPrediction(BaseModel):
    extreme_rainfall_probability: float = Field(..., description="Probability of extreme rainfall (0-1)", ge=0, le=1)
    is_extreme: bool = Field(..., description="Binary prediction")
    expected_precipitation: float = Field(..., description="Expected precipitation amount (mm)")
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
    1: "Tropical Storm",
    2: "Category 1 Hurricane",
    3: "Category 2 Hurricane",
    4: "Category 3 Hurricane",
    5: "Category 4-5 Hurricane",
}


def get_category_description(category: int) -> str:
    """Return human-readable Saffir-Simpson description."""
    return SAFFIR_SIMPSON.get(category, f"Unknown (category {category})")
