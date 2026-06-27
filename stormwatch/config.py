"""
StormWatch AI - Configuration Module
Loads and validates configuration from YAML file with environment overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel

# ──────────────────────────────────────────────
#  Pydantic models for typed config access
# ──────────────────────────────────────────────


class IBTrACSConfig(BaseModel):
    url: str = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.IO.list.v04r01.csv"
    filename: str = "ibtracs_cyclones.csv"
    basin: str = "IO"


class OpenMeteoConfig(BaseModel):
    start_date: str = "2010-01-01"
    timezone: str = "Asia/Kolkata"
    retry_attempts: int = 3
    retry_delay_seconds: int = 10
    city_delay_seconds: int = 20
    chunk_delay_seconds: float = 5.0


class DataConfig(BaseModel):
    raw_path: str = "data/raw"
    processed_path: str = "data/processed"
    external_path: str = "data/external"
    openmeteo: OpenMeteoConfig = OpenMeteoConfig()
    ibtracs: IBTrACSConfig = IBTrACSConfig()


class CycloneModelConfig(BaseModel):
    type: str = "multiclass"
    target: str = "category"
    test_size: float = 0.2
    random_state: int = 42
    hyperopt_evals: int = 30


class HeatwaveModelConfig(BaseModel):
    type: str = "binary"
    target: str = "heatwave_flag"
    test_size: float = 0.2
    random_state: int = 42
    hyperopt_evals: int = 30


class RainfallModelConfig(BaseModel):
    type: str = "binary"
    target: str = "extreme_rainfall"
    test_size: float = 0.2
    random_state: int = 42
    hyperopt_evals: int = 30


class TrainingConfig(BaseModel):
    mlflow_tracking_uri: str = "sqlite:///mlflow/mlflow.db"
    experiment_name: str = "stormwatch-ai"
    cv_folds: int = 5


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "StormWatch AI API"
    version: str = "1.0.0"
    model_path: str = "models/"


class MonitoringConfig(BaseModel):
    drift_interval_days: int = 7
    reference_window_days: int = 30


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


class AppConfig(BaseModel):
    project_name: str = "StormWatch AI"
    project_version: str = "1.0.0"
    data: DataConfig = DataConfig()
    cyclone_model: CycloneModelConfig = CycloneModelConfig()
    heatwave_model: HeatwaveModelConfig = HeatwaveModelConfig()
    rainfall_model: RainfallModelConfig = RainfallModelConfig()
    training: TrainingConfig = TrainingConfig()
    api: APIConfig = APIConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    logging: LoggingConfig = LoggingConfig()


# ──────────────────────────────────────────────
#  Singleton config loader
# ──────────────────────────────────────────────

_CONFIG: Optional[AppConfig] = None


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from YAML, with environment variable overrides.

    Environment variables override YAML values using the pattern
    ``STORMWATCH__<SECTION>__<KEY>`` (e.g. ``STORMWATCH__API__PORT=9000``).
    """
    global _CONFIG

    # Default config path
    if config_path is None:
        repo_root = Path(__file__).resolve().parent.parent
        config_path = str(repo_root / "configs" / "config.yaml")

    # Load from YAML
    raw: Dict[str, Any] = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}

    # Environment overrides
    for key, value in os.environ.items():
        if key.startswith("STORMWATCH__"):
            parts = key.replace("STORMWATCH__", "").lower().split("__")
            target = raw
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                target = target[part]
            target[parts[-1]] = _coerce_value(value)

    _CONFIG = AppConfig.model_validate(raw)
    return _CONFIG


def get_config() -> AppConfig:
    """Return the cached configuration, loading it if necessary."""
    if _CONFIG is None:
        return load_config()
    return _CONFIG


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────


def _coerce_value(value: str) -> Any:
    """Coerce environment variable strings to appropriate Python types."""
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value
