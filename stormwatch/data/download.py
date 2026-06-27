"""
StormWatch AI - Data Download Module
Downloads IBTrACS cyclone tracks and Open-Meteo historical weather data.

Uses the official openmeteo-requests client with:
- requests-cache: local HTTP cache so repeated chunk requests are instant
- retry-requests: automatic retry with exponential backoff on failures
"""

from __future__ import annotations

import csv
import io
import os
import time
import urllib.request
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from stormwatch.config import get_config
from stormwatch.logger import get_logger

log = get_logger(__name__)

# ──────────────────────────────────────────────
#  Open-Meteo client singleton
# ──────────────────────────────────────────────

_OPENMETEO_CLIENT: Any = None


def _get_openmeteo_client():
    """Lazy-init the openmeteo-requests client with caching and retry."""
    global _OPENMETEO_CLIENT
    if _OPENMETEO_CLIENT is not None:
        return _OPENMETEO_CLIENT

    import openmeteo_requests
    import requests_cache
    from retry_requests import retry

    cache_session = requests_cache.CachedSession(
        ".openmeteo_cache",
        expire_after=3600,           # cache valid for 1 hour
    )
    retry_session = retry(
        cache_session,
        retries=5,
        backoff_factor=0.5,          # 0.5s, 1s, 2s, 4s, 8s
    )
    _OPENMETEO_CLIENT = openmeteo_requests.Client(session=retry_session)
    return _OPENMETEO_CLIENT


# ──────────────────────────────────────────────
#  Indian cities for weather data
# ──────────────────────────────────────────────
# Selected to cover diverse climate zones across India

INDIAN_CITIES: List[Dict[str, Any]] = [
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "state": "Maharashtra", "zone": "coastal"},
    {"name": "Chennai", "lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu", "zone": "coastal"},
    {"name": "Kolkata", "lat": 22.5726, "lon": 88.3639, "state": "West Bengal", "zone": "coastal"},
    {"name": "Delhi", "lat": 28.7041, "lon": 77.1025, "state": "Delhi", "zone": "inland"},
    {"name": "Ahmedabad", "lat": 23.0225, "lon": 72.5714, "state": "Gujarat", "zone": "arid"},
    {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867, "state": "Telangana", "zone": "inland"},
    {"name": "Bengaluru", "lat": 12.9716, "lon": 77.5946, "state": "Karnataka", "zone": "inland"},
    {"name": "Kochi", "lat": 9.9312, "lon": 76.2673, "state": "Kerala", "zone": "coastal"},
    {"name": "Bhubaneswar", "lat": 20.2961, "lon": 85.8245, "state": "Odisha", "zone": "coastal"},
    {"name": "Jaipur", "lat": 26.9124, "lon": 75.7873, "state": "Rajasthan", "zone": "arid"},
    {"name": "Lucknow", "lat": 26.8467, "lon": 80.9462, "state": "Uttar Pradesh", "zone": "inland"},
    {"name": "Guwahati", "lat": 26.1445, "lon": 91.7362, "state": "Assam", "zone": "humid"},
    {"name": "Pune", "lat": 18.5204, "lon": 73.8567, "state": "Maharashtra", "zone": "inland"},
    {"name": "Visakhapatnam", "lat": 17.6868, "lon": 83.2185, "state": "Andhra Pradesh", "zone": "coastal"},
    {"name": "Surat", "lat": 21.1702, "lon": 72.8311, "state": "Gujarat", "zone": "coastal"},
]


def download_ibtracs(
    save_dir: Optional[str] = None,
    basin: str = "IO",
    force: bool = False,
) -> Optional[Path]:
    """Download IBTrACS cyclone track data for a given ocean basin.

    Args:
        save_dir: Directory to save the file (default: config data.raw_path)
        basin: Ocean basin code (IO = Indian Ocean, NA = North Atlantic, etc.)
        force: Re-download even if file exists

    Returns:
        Path to the downloaded file, or None on failure.
    """
    config = get_config()
    save_dir = save_dir or config.data.raw_path
    os.makedirs(save_dir, exist_ok=True)

    # IBTrACS v04r01 CSV files by basin
    basin_urls = {
        "IO": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.IO.list.v04r01.csv",
        "NA": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NA.list.v04r01.csv",
        "EP": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.EP.list.v04r01.csv",
        "WP": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.WP.list.v04r01.csv",
        "SP": "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.SP.list.v04r01.csv",
    }

    url = basin_urls.get(basin.upper())
    if not url:
        log.error("Unknown basin code: %s (use IO, NA, EP, WP, SP)", basin)
        return None

    filename = f"ibtracs_{basin}.csv"
    save_path = Path(save_dir) / filename

    if save_path.exists() and not force:
        log.info("IBTrACS data already exists at %s (use force=True to re-download)", save_path)
        return save_path

    log.info("Downloading IBTrACS %s basin data from NOAA...", basin)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as response:
            content = response.read()

        # Handle potential zip files
        if url.endswith(".zip") or content[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_files = [n for n in zf.namelist() if n.endswith(".csv")]
                if not csv_files:
                    log.error("No CSV found in zip archive")
                    return None
                with zf.open(csv_files[0]) as f:
                    df = pd.read_csv(f, encoding="latin1", low_memory=False)
        else:
            # Decode with latin1 to handle special characters
            decoded = content.decode("latin1")
            df = pd.read_csv(io.StringIO(decoded), low_memory=False)

        df.to_csv(save_path, index=False)
        log.info("Downloaded %s records to %s", len(df), save_path)
        return save_path

    except Exception as e:
        log.error("Failed to download IBTrACS data: %s", e)
        return None


# ──────────────────────────────────────────────
#  Yearly chunking for Open-Meteo API accounting
# ──────────────────────────────────────────────


def _year_chunks(start: str, end: str) -> List[Tuple[str, str]]:
    """Split a date range into year-sized chunks.

    Open-Meteo charges API calls per-variable-per-day-per-location.
    A 16-year request costs ~430 calls; yearly chunks cost ~39 each.
    This keeps us well under the 5 000 calls/hour free-tier limit.
    """
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    if s >= e:
        return []
    chunks: List[Tuple[str, str]] = []
    cursor = s
    while cursor < e:
        year_end = cursor.replace(month=12, day=31)
        chunk_end = min(year_end, e)
        if chunk_end >= cursor:
            chunks.append((cursor.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        cursor = datetime(chunk_end.year + 1, 1, 1)
    return chunks


# ──────────────────────────────────────────────
#  Weather variables requested from Open-Meteo
# ──────────────────────────────────────────────

DAILY_PARAMS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "pressure_msl_mean",
    "relative_humidity_2m_mean",
    "cloud_cover_mean",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]


def download_openmeteo_historical(
    lat: float,
    lon: float,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Download historical weather data from Open-Meteo Archive API.

    Uses the official ``openmeteo-requests`` client with:
    - Local HTTP cache (repeat chunk requests return instantly)
    - Automatic retry with exponential backoff (5 retries, 0.5× factor)
    - Yearly chunking to avoid the fractional API-call multiplier

    Args:
        lat: Latitude
        lon: Longitude
        start_date: Start date (YYYY-MM-DD, default from config)
        end_date: End date (YYYY-MM-DD, defaults to yesterday)

    Returns:
        DataFrame with daily weather data + lat/lon, or None on failure.
    """
    config = get_config()
    start_date = start_date or config.data.openmeteo.start_date
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    chunks = _year_chunks(start_date, end_date)
    if not chunks:
        log.info("Empty date range %s → %s — nothing to download", start_date, end_date)
        return None

    client = _get_openmeteo_client()
    url = "https://archive-api.open-meteo.com/v1/archive"
    all_dfs: List[pd.DataFrame] = []

    for i, (cs, ce) in enumerate(chunks):
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": cs,
            "end_date": ce,
            "daily": DAILY_PARAMS,
            "timezone": config.data.openmeteo.timezone,
        }
        try:
            responses = client.weather_api(url, params=params)
        except Exception as e:
            err_str = str(e).lower()
            if "limit" in err_str or "429" in err_str:
                log.warning(
                    "RATE LIMITED on %s..%s for lat=%s, lon=%s. "
                    "Sleeping 300s before next attempt.",
                    cs, ce, lat, lon,
                )
                time.sleep(300)
                try:
                    responses = client.weather_api(url, params=params)
                except Exception as e2:
                    log.warning("Chunk %s..%s still failed after rate-limit sleep: %s", cs, ce, lat, lon, e2)
                    continue
            else:
                log.warning("Chunk %s..%s failed for lat=%s, lon=%s: %s", cs, ce, lat, lon, e)
                continue

        response = responses[0]

        # Parse daily response
        daily = response.Daily()
        if daily is None:
            log.warning("No daily data for %s..%s (lat=%s, lon=%s)", cs, ce, lat, lon)
            continue

        # Build time index
        time_index = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left",
        )

        # Extract each variable by position
        data: Dict[str, Any] = {"time": time_index}
        for j, var_name in enumerate(DAILY_PARAMS):
            try:
                values = daily.Variables(j).ValuesAsNumpy()
                # Open-Meteo fills missing days with NaN
                data[var_name] = values
            except Exception:
                data[var_name] = np.full(len(time_index), np.nan)

        chunk_df = pd.DataFrame(data)
        all_dfs.append(chunk_df)

        if i < len(chunks) - 1:
            time.sleep(config.data.openmeteo.chunk_delay_seconds)

    if not all_dfs:
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    combined["latitude"] = lat
    combined["longitude"] = lon
    return combined


def download_all_weather_data(
    save_dir: Optional[str] = None,
    start_date: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Download historical weather data for all Indian cities.

    Args:
        save_dir: Directory to save individual city files
        start_date: Start date (YYYY-MM-DD, default from config)
        force: Re-download existing files

    Returns:
        Combined DataFrame with all cities' weather data.
    """
    config = get_config()
    save_dir = save_dir or config.data.raw_path
    start_date = start_date or config.data.openmeteo.start_date
    os.makedirs(save_dir, exist_ok=True)

    all_dfs: List[pd.DataFrame] = []

    for city in tqdm(INDIAN_CITIES, desc="Downloading weather data"):
        city_file = Path(save_dir) / f"weather_{city['name'].lower()}.csv"

        if city_file.exists() and not force:
            log.info("Using cached data for %s", city["name"])
            df = pd.read_csv(city_file)
            all_dfs.append(df)
            continue

        log.info("Fetching weather data for %s...", city["name"])

        # Retry with config-based backoff
        retries = config.data.openmeteo.retry_attempts
        for attempt in range(retries):
            df = download_openmeteo_historical(
                lat=city["lat"],
                lon=city["lon"],
                start_date=start_date,
            )
            if df is not None:
                break
            log.warning("Retry %d/%d for %s after %ds sleep",
                        attempt + 1, retries, city["name"],
                        config.data.openmeteo.retry_delay_seconds * (2**attempt))
            time.sleep(config.data.openmeteo.retry_delay_seconds * (2**attempt))

        if df is None:
            log.warning("Skipping %s after %d failed attempts", city["name"], retries)
            continue

        df["city"] = city["name"]
        df["state"] = city["state"]
        df["zone"] = city["zone"]
        df.to_csv(city_file, index=False)
        all_dfs.append(df)

        time.sleep(config.data.openmeteo.city_delay_seconds)

    if not all_dfs:
        log.error("No weather data downloaded")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_path = Path(save_dir) / "weather_all_cities.csv"
    combined.to_csv(combined_path, index=False)
    log.info("Combined weather data: %s rows, %s columns", len(combined), len(combined.columns))
    return combined


download_all = download_all_weather_data


if __name__ == "__main__":
    # Smoke test: download small weather sample for one city
    log.info("Running download smoke test...")
    df = download_openmeteo_historical(19.0760, 72.8777, start_date="2024-01-01", end_date="2024-01-10")
    if df is not None:
        print(df.head())
        print(f"✅ Weather download: {len(df)} days of data")
    else:
        print("⚠️  Weather download failed (expected if offline)")
