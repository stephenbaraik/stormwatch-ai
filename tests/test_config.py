from __future__ import annotations

import os
from pathlib import Path
from typing import Generator

import pytest

from stormwatch.config import (
    AppConfig,
    DataConfig,
    CycloneModelConfig,
    HeatwaveModelConfig,
    RainfallModelConfig,
    TrainingConfig,
    APIConfig,
    MonitoringConfig,
    LoggingConfig,
    _coerce_value,
    get_config,
    load_config,
)


@pytest.fixture(autouse=True)
def _reset_config() -> Generator[None, None, None]:
    from stormwatch import config as cfg
    cfg._CONFIG = None
    yield
    cfg._CONFIG = None


class TestAppConfig:
    def test_default_config_has_all_sections(self):
        cfg = AppConfig()
        assert cfg.project_name == "StormWatch AI"
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.cyclone_model, CycloneModelConfig)
        assert isinstance(cfg.heatwave_model, HeatwaveModelConfig)
        assert isinstance(cfg.rainfall_model, RainfallModelConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.api, APIConfig)
        assert isinstance(cfg.monitoring, MonitoringConfig)
        assert isinstance(cfg.logging, LoggingConfig)

    def test_load_config_from_yaml(self):
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        assert config_path.exists()
        cfg = load_config(str(config_path))
        assert cfg.project_name == "StormWatch AI"
        assert cfg.api.port == 8000
        assert cfg.logging.level == "INFO"

    def test_cyclone_model_config_defaults(self):
        cfg = CycloneModelConfig()
        assert cfg.type == "multiclass"
        assert cfg.target == "category"
        assert cfg.test_size == 0.2
        assert cfg.random_state == 42
        assert cfg.hyperopt_evals == 30

    def test_heatwave_model_config_defaults(self):
        cfg = HeatwaveModelConfig()
        assert cfg.type == "binary"
        assert cfg.target == "heatwave_flag"
        assert cfg.test_size == 0.2

    def test_training_config_defaults(self):
        cfg = TrainingConfig()
        assert "sqlite" in cfg.mlflow_tracking_uri
        assert cfg.experiment_name == "stormwatch-ai"
        assert cfg.cv_folds == 5

    def test_api_config_defaults(self):
        cfg = APIConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 8000
        assert cfg.title == "StormWatch AI API"
        assert cfg.model_path == "models/"

    def test_monitoring_config_defaults(self):
        cfg = MonitoringConfig()
        assert cfg.drift_interval_days == 7
        assert cfg.reference_window_days == 30

    def test_logging_config_defaults(self):
        cfg = LoggingConfig()
        assert cfg.level == "INFO"
        assert "%(asctime)s" in cfg.format

    def test_data_config_defaults(self):
        cfg = DataConfig()
        assert cfg.raw_path == "data/raw"
        assert cfg.processed_path == "data/processed"
        assert cfg.external_path == "data/external"


class TestConfigLoader:
    def test_get_config_loads_yaml(self):
        cfg = get_config()
        assert cfg.project_name == "StormWatch AI"
        assert isinstance(cfg, AppConfig)

    def test_load_config_is_cached(self):
        cfg1 = get_config()
        cfg2 = get_config()
        assert cfg1 is cfg2

    def test_load_config_with_missing_file(self, tmp_path):
        nonexistent = str(tmp_path / "nonexistent.yaml")
        cfg = load_config(nonexistent)
        assert isinstance(cfg, AppConfig)

    def test_environment_override(self, monkeypatch):
        monkeypatch.setenv("STORMWATCH__API__PORT", "9000")
        monkeypatch.setenv("STORMWATCH__LOGGING__LEVEL", "DEBUG")
        cfg = load_config()
        assert cfg.api.port == 9000
        assert cfg.logging.level == "DEBUG"

    def test_environment_override_bool(self, monkeypatch):
        monkeypatch.setenv("STORMWATCH__MONITORING__DRIFT_INTERVAL_DAYS", "14")
        cfg = load_config()
        assert cfg.monitoring.drift_interval_days == 14

    def test_environment_override_nested(self, monkeypatch):
        monkeypatch.setenv("STORMWATCH__DATA__IBTRACS__FILENAME", "test.csv")
        cfg = load_config()
        assert cfg.data.ibtracs.filename == "test.csv"

    def test_environment_override_true_values(self, monkeypatch):
        assert _coerce_value("true") is True
        assert _coerce_value("TRUE") is True
        assert _coerce_value("1") is True
        assert _coerce_value("yes") is True

    def test_environment_override_false_values(self, monkeypatch):
        assert _coerce_value("false") is False
        assert _coerce_value("FALSE") is False
        assert _coerce_value("0") is False
        assert _coerce_value("no") is False

    def test_environment_override_int(self):
        assert _coerce_value("42") == 42
        assert _coerce_value("-5") == -5

    def test_environment_override_float(self):
        assert _coerce_value("3.14") == 3.14

    def test_environment_override_string(self):
        assert _coerce_value("hello") == "hello"


class TestDataConfig:
    def test_ibtracs_config(self):
        cfg = DataConfig()
        assert cfg.ibtracs.basin == "NI"

    def test_ibtracs_url_default(self):
        cfg = DataConfig()
        assert "noaa.gov" in cfg.ibtracs.url
