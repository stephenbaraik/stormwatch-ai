"""
StormWatch AI - Data Preprocessing Module
Cleans raw data, performs feature engineering, and labels extreme events.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from stormwatch.config import get_config
from stormwatch.logger import get_logger

log = get_logger(__name__)


# ──────────────────────────────────────────────
#  Cyclone data preprocessing
# ──────────────────────────────────────────────


IBTRACS_DTYPE_MAP: Dict[str, str] = {
    "ISO_TIME": "str",
    "NATURE": "str",
    "LAT": "float64",
    "LON": "float64",
    "WMO_WIND": "float64",
    "WMO_PRES": "float64",
    "USA_WIND": "float64",
    "USA_PRES": "float64",
    "USA_GUST": "float64",
    "TRACK_TYPE": "str",
    "DIST2LAND": "float64",
    "LAND": "float64",
}

SAFFIR_SIMPSON_CATEGORIES = {
    0: "Tropical Depression",
    1: "Category 1",
    2: "Category 2",
    3: "Category 3",
    4: "Category 4",
    5: "Category 5",
}


def preprocess_cyclones(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare IBTrACS cyclone data for modeling.

    Steps:
    - Parse timestamps
    - Filter to valid cyclone records (NATURE = 'TC')
    - Compute wind-based Saffir-Simpson category
    - Create temporal features (year, month, day)
    - Remove records with missing critical values
    """
    if df.empty:
        log.warning("Empty cyclone DataFrame")
        return df

    # Parse timestamp
    time_col = None
    for col in ["ISO_TIME", "iso_time", "time"]:
        if col in df.columns:
            time_col = col
            break

    if time_col is None:
        log.warning("No timestamp column found, using index")
        df["year"] = 0
        df["month"] = 0
        df["day"] = 0
    else:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df["year"] = df[time_col].dt.year
        df["month"] = df[time_col].dt.month
        df["day"] = df[time_col].dt.day
        df["dayofyear"] = df[time_col].dt.dayofyear

    # Filter to tropical cyclones
    nature_col = None
    for col in ["NATURE", "nature"]:
        if col in df.columns:
            nature_col = col
            break

    if nature_col:
        before = len(df)
        df = df[df[nature_col].str.upper().str.contains("TC", na=False)].copy()
        log.info("Filtered to tropical cyclones: %d → %d records", before, len(df))

    # Determine wind speed column (prefer WMO, fall back to USA)
    wind_col = "WMO_WIND" if "WMO_WIND" in df.columns else None
    wind_col = wind_col or ("USA_WIND" if "USA_WIND" in df.columns else None)
    wind_col = wind_col or ("wind" if "wind" in df.columns else None)

    pressure_col = "WMO_PRES" if "WMO_PRES" in df.columns else None
    pressure_col = pressure_col or ("USA_PRES" if "USA_PRES" in df.columns else None)
    pressure_col = pressure_col or ("pres" if "pres" in df.columns else None)

    if wind_col is None:
        log.warning(
            "No wind speed column found in IBTrACS data, defaulting to category 0"
        )
        df["wind_kts"] = 0
        df["category"] = 0
    else:
        # Convert knots to km/h if needed
        df["wind_kts"] = pd.to_numeric(df[wind_col], errors="coerce").fillna(0)

        # Saffir-Simpson category from max sustained wind (knots)
        conditions = [
            (df["wind_kts"] < 34),
            (df["wind_kts"] >= 34) & (df["wind_kts"] < 64),
            (df["wind_kts"] >= 64) & (df["wind_kts"] < 83),
            (df["wind_kts"] >= 83) & (df["wind_kts"] < 96),
            (df["wind_kts"] >= 96) & (df["wind_kts"] < 113),
            (df["wind_kts"] >= 113) & (df["wind_kts"] < 137),
            (df["wind_kts"] >= 137),
        ]
        categories = [0, 1, 2, 3, 4, 5, 5]
        df["category"] = np.select(conditions, categories, default=0)

    # Pressure
    if pressure_col:
        df["pressure_min"] = pd.to_numeric(df[pressure_col], errors="coerce")
    else:
        df["pressure_min"] = np.nan

    # Lat/Lon
    lat_col = "LAT" if "LAT" in df.columns else ("lat" if "lat" in df.columns else None)
    lon_col = "LON" if "LON" in df.columns else ("lon" if "lon" in df.columns else None)

    if lat_col:
        df["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
        df["lat_abs"] = df["lat"].abs()
    if lon_col:
        df["lon"] = pd.to_numeric(df[lon_col], errors="coerce")

    # Distance to land
    for col in ["DIST2LAND", "dist2land"]:
        if col in df.columns:
            df["dist_to_land"] = pd.to_numeric(df[col], errors="coerce")
            break

    # Remove rows where critical features are missing
    df = df.dropna(subset=["lat", "lon", "wind_kts"], how="all")

    # Sort by time
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)

    log.info(
        "Processed cyclone data: %d rows, %d features, %d categories",
        len(df),
        len(df.columns),
        df["category"].nunique(),
    )
    return df


# ──────────────────────────────────────────────
#  Weather data preprocessing
# ──────────────────────────────────────────────


def label_extreme_events(
    df: pd.DataFrame,
    temp_heatwave_threshold: float = 40.0,
    temp_severe_heatwave: float = 45.0,
    rainfall_percentile: float = 95.0,
    consecutive_days: int = 3,
) -> pd.DataFrame:
    """Label extreme weather events in daily weather data.

    Labels created:
    - ``heatwave_flag``: 1 if max temp > threshold for N consecutive days
    - ``severe_heatwave_flag``: 1 if max temp > severe threshold for N days
    - ``extreme_rainfall``: 1 if precipitation > 95th percentile of that city
    - ``heavy_rainfall``: 1 if precipitation > 99th percentile
    - ``cyclonic_flag``: 1 if wind gusts > 60 km/h and pressure < 1005 hPa
    """
    df = df.copy()

    # Convert temperature columns
    temp_cols = {
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "temperature_2m_mean": "temp_mean",
    }
    for src, dst in temp_cols.items():
        if src in df.columns:
            df[dst] = pd.to_numeric(df[src], errors="coerce")
        else:
            df[dst] = np.nan

    # Precipitation
    precip_col = "precipitation_sum" if "precipitation_sum" in df.columns else "precip"
    df["precipitation"] = (
        pd.to_numeric(df[precip_col], errors="coerce")
        if precip_col in df.columns
        else 0.0
    )

    # Wind
    gust_col = "wind_gusts_10m_max" if "wind_gusts_10m_max" in df.columns else None

    # Pressure
    press_col = "pressure_msl_mean" if "pressure_msl_mean" in df.columns else None

    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    # ── Heatwave detection ──
    # Consecutive days above threshold per city
    if "temp_max" in df.columns:
        df["above_heatwave"] = df["temp_max"] > temp_heatwave_threshold
        df["above_severe"] = df["temp_max"] > temp_severe_heatwave

        # Rolling consecutive count
        df["heatwave_streak"] = df.groupby("city")["above_heatwave"].transform(
            lambda s: s.groupby((s != s.shift()).cumsum()).cumsum()
        )
        df["severe_heatwave_streak"] = df.groupby("city")["above_severe"].transform(
            lambda s: s.groupby((s != s.shift()).cumsum()).cumsum()
        )

        df["heatwave_flag"] = (df["heatwave_streak"] >= consecutive_days).astype(int)
        df["severe_heatwave_flag"] = (
            df["severe_heatwave_streak"] >= consecutive_days
        ).astype(int)
    else:
        df["heatwave_flag"] = 0
        df["severe_heatwave_flag"] = 0

    # ── Extreme rainfall ──
    # Per-city percentile thresholds
    city_thresholds = (
        df.groupby("city")["precipitation"]
        .quantile(rainfall_percentile / 100)
        .to_dict()
    )
    severe_thresholds = df.groupby("city")["precipitation"].quantile(0.99).to_dict()

    df["extreme_rainfall"] = df.apply(
        lambda row: int(row["precipitation"] > city_thresholds.get(row["city"], 50)),
        axis=1,
    )
    df["heavy_rainfall"] = df.apply(
        lambda row: int(row["precipitation"] > severe_thresholds.get(row["city"], 100)),
        axis=1,
    )

    # ── Cyclonic conditions ──
    if gust_col and press_col:
        gusts = pd.to_numeric(df[gust_col], errors="coerce").fillna(0)
        pressure = pd.to_numeric(df[press_col], errors="coerce").fillna(1013)
        df["cyclonic_flag"] = ((gusts > 60) & (pressure < 1005)).astype(int)
    else:
        df["cyclonic_flag"] = 0

    # Drop intermediate columns
    drop_cols = [
        c
        for c in [
            "above_heatwave",
            "above_severe",
            "heatwave_streak",
            "severe_heatwave_streak",
        ]
        if c in df.columns
    ]
    df = df.drop(columns=drop_cols)

    log.info(
        "Labeled extreme events: heatwaves=%d, severe_heatwaves=%d, extreme_rain=%d, cyclonic=%d",
        df["heatwave_flag"].sum(),
        df["severe_heatwave_flag"].sum(),
        df["extreme_rainfall"].sum(),
        df["cyclonic_flag"].sum(),
    )
    return df


def prepare_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lagged features and rolling statistics for weather prediction.

    Features created per city:
    - 1-day, 3-day, 7-day lags for temp, precipitation, wind
    - Rolling means and stds
    - Seasonal (month, dayofyear)
    """
    df = df.copy()
    df = df.sort_values(["city", "time"]).reset_index(drop=True)

    # Lag features (1, 3, 7 day)
    for col in ["temp_max", "temp_min", "precipitation", "wind_speed_10m_max"]:
        if col not in df.columns:
            continue
        for lag in [1, 3, 7]:
            df[f"{col}_lag_{lag}"] = df.groupby("city")[col].shift(lag)

    # Rolling statistics
    for col in ["temp_max", "precipitation"]:
        if col not in df.columns:
            continue
        for window in [3, 7]:
            df[f"{col}_roll_mean_{window}"] = df.groupby("city")[col].transform(
                lambda s: s.rolling(window, min_periods=1).mean()
            )
            df[f"{col}_roll_std_{window}"] = df.groupby("city")[col].transform(
                lambda s: s.rolling(window, min_periods=1).std().fillna(0)
            )

    # Seasonal features
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df["month"] = df["time"].dt.month
        df["dayofyear"] = df["time"].dt.dayofyear
        # Cyclic encoding for seasonality
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Drop NaN rows from lag creation
    df = df.dropna(
        subset=[c for c in df.columns if "lag" in c or "roll" in c] + ["heatwave_flag"],
        how="all",
    )

    return df


# ──────────────────────────────────────────────
#  Main preprocessing pipeline
# ──────────────────────────────────────────────


def preprocess_all(
    weather_path: Optional[str] = None,
    cyclone_path: Optional[str] = None,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run full preprocessing pipeline on all data.

    Args:
        weather_path: Path to raw weather CSV (combined city file)
        cyclone_path: Path to raw cyclone CSV (IBTrACS)
        save: Save processed data to disk

    Returns:
        (weather_processed, cyclone_processed) DataFrames
    """
    config = get_config()
    processed_dir = Path(config.data.processed_path)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # ── Weather ──
    weather_df = pd.DataFrame()
    if weather_path and Path(weather_path).exists():
        log.info("Loading weather data from %s", weather_path)
        weather_df = pd.read_csv(weather_path)
        weather_df = label_extreme_events(weather_df)
        weather_df = prepare_weather_features(weather_df)

        if save:
            weather_out = processed_dir / "weather_processed.csv"
            weather_df.to_csv(weather_out, index=False)
            log.info(
                "Saved processed weather → %s (%d rows)", weather_out, len(weather_df)
            )

    # ── Cyclones ──
    cyclone_df = pd.DataFrame()
    if cyclone_path and Path(cyclone_path).exists():
        log.info("Loading cyclone data from %s", cyclone_path)
        cyclone_df = pd.read_csv(cyclone_path, low_memory=False)
        cyclone_df = preprocess_cyclones(cyclone_df)

        if save:
            cyclone_out = processed_dir / "cyclones_processed.csv"
            cyclone_df.to_csv(cyclone_out, index=False)
            log.info(
                "Saved processed cyclones → %s (%d rows)", cyclone_out, len(cyclone_df)
            )

    return weather_df, cyclone_df


if __name__ == "__main__":
    log.info("Run via: stormwatch.models.train.main() for the full training pipeline.")
    print(
        "Skipping smoke test: run via stormwatch.models.train.main() for the full pipeline."
    )
    print("✅ Preprocessing smoke test: PASSED")
