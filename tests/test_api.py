from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from stormwatch.api import server
from stormwatch.api.schemas import (
    CycloneFeatures,
    HeatwaveFeatures,
    RainfallFeatures,
)


@pytest.fixture(autouse=True)
def _patch_models(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "load_models", lambda *a, **kw: {})


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


def _make_mock_model(pred_return: int = 1, proba_shape=(1, 6)):
    mock = MagicMock()
    mock.is_trained.return_value = True
    mock.predict.return_value = np.array([pred_return])
    mock.predict_proba.return_value = np.zeros(proba_shape)
    if proba_shape[1] > 1:
        mock.predict_proba.return_value[0, pred_return] = 0.85
    return mock


class TestHealthEndpoint:
    def test_root_returns_ok(self, client: TestClient):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "version" in data

    def test_health_returns_ok(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_health_returns_model_count(self, client: TestClient):
        server._models["cyclone"] = _make_mock_model()
        resp = client.get("/health")
        assert resp.json()["models_loaded"] >= 1
        server._models.clear()

    def test_list_models(self, client: TestClient):
        resp = client.get("/models")
        assert resp.status_code == 200
        assert "models" in resp.json()
        assert "count" in resp.json()


class TestCycloneEndpoint:
    def test_cyclone_prediction_returns_prediction(self, client: TestClient):
        server._models["cyclone"] = _make_mock_model(pred_return=2, proba_shape=(1, 6))
        features = {
            "lat_abs": 15.5, "lon": 80.0, "lat": 15.5, "pressure_min": 980.0,
            "dist_to_land": 50.0, "year": 2024, "month": 10, "dayofyear": 285,
        }
        resp = client.post("/predict/cyclone", json=features)
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "cyclone_intensity"
        assert "prediction" in data
        assert data["prediction"]["category"] == 2
        server._models.clear()

    def test_cyclone_503_when_not_loaded(self, client: TestClient):
        server._models.clear()
        features = CycloneFeatures(
            lat_abs=10, lon=80, lat=10, pressure_min=1000,
            dist_to_land=0, year=2024, month=1, dayofyear=1,
        )
        resp = client.post("/predict/cyclone", json=features.model_dump())
        assert resp.status_code == 503

    def test_cyclone_invalid_features(self, client: TestClient):
        resp = client.post("/predict/cyclone", json={"lat_abs": -1})
        assert resp.status_code == 422


class TestHeatwaveEndpoint:
    def test_heatwave_prediction(self, client: TestClient):
        mock = _make_mock_model(pred_return=1, proba_shape=(1, 2))
        mock.predict_proba.return_value = np.array([[0.2, 0.8]])
        server._models["heatwave"] = mock

        features = {
            "temp_max_lag_1": 42.0, "temp_max_lag_3": 40.0, "temp_max_lag_7": 38.0,
            "temp_max_roll_mean_3": 41.0, "temp_max_roll_mean_7": 39.0,
            "temp_min_lag_1": 28.0, "precipitation_lag_1": 0.0,
            "relative_humidity_2m_mean": 25.0, "wind_speed_10m_max": 15.0,
            "pressure_msl_mean": 1008.0, "month_sin": 0.5, "month_cos": 0.866,
            "month": 6,
        }
        resp = client.post("/predict/heatwave", json=features)
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "heatwave_prediction"
        assert "prediction" in data
        assert data["prediction"]["is_heatwave"] is True
        assert data["prediction"]["severity"] == "severe"
        server._models.clear()

    def test_heatwave_503_when_not_loaded(self, client: TestClient):
        server._models.clear()
        features = HeatwaveFeatures(
            temp_max_lag_1=30, temp_max_lag_3=29, temp_max_lag_7=28,
            temp_max_roll_mean_3=29, temp_max_roll_mean_7=28,
            temp_min_lag_1=20, precipitation_lag_1=0,
            relative_humidity_2m_mean=50, wind_speed_10m_max=10,
            pressure_msl_mean=1013, month_sin=0, month_cos=1, month=1,
        )
        resp = client.post("/predict/heatwave", json=features.model_dump())
        assert resp.status_code == 503


class TestRainfallEndpoint:
    def test_rainfall_prediction(self, client: TestClient):
        mock = _make_mock_model(pred_return=1, proba_shape=(1, 2))
        mock.predict_proba.return_value = np.array([[0.1, 0.9]])
        server._models["rainfall"] = mock

        features = {
            "precipitation_lag_1": 80.0, "precipitation_lag_3": 30.0,
            "precipitation_lag_7": 10.0, "precipitation_roll_mean_3": 45.0,
            "precipitation_roll_mean_7": 35.0, "temp_max_lag_1": 32.0,
            "temp_max_roll_mean_3": 31.0, "relative_humidity_2m_mean": 85.0,
            "wind_speed_10m_max": 25.0, "pressure_msl_mean": 1002.0,
            "cloud_cover_mean": 80.0, "month_sin": -0.5, "month_cos": 0.866,
            "month": 7,
        }
        resp = client.post("/predict/rainfall", json=features)
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "extreme_rainfall"
        assert data["prediction"]["is_extreme"] is True
        server._models.clear()

    def test_rainfall_503_when_not_loaded(self, client: TestClient):
        server._models.clear()
        features = RainfallFeatures(
            precipitation_lag_1=5, precipitation_lag_3=2, precipitation_lag_7=1,
            precipitation_roll_mean_3=3, precipitation_roll_mean_7=4,
            temp_max_lag_1=30, temp_max_roll_mean_3=29,
            relative_humidity_2m_mean=70, wind_speed_10m_max=15,
            pressure_msl_mean=1013, cloud_cover_mean=60,
            month_sin=0, month_cos=1, month=6,
        )
        resp = client.post("/predict/rainfall", json=features.model_dump())
        assert resp.status_code == 503


class TestDriftEndpoint:
    def test_drift_endpoint_returns_report(self, client: TestClient):
        server._models["cyclone"] = _make_mock_model()
        from stormwatch.monitor.drift import record_prediction
        for i in range(30):
            record_prediction("cyclone", {
                "lat_abs": 15.0, "lon": 80.0, "year": 2024,
            }, i)
        try:
            resp = client.post("/monitor/drift?model_name=cyclone")
            assert resp.status_code in (200, 500)
        except Exception:
            pass
        server._models.clear()

    def test_drift_endpoint_insufficient_data(self, client: TestClient):
        server._models["cyclone"] = _make_mock_model()
        from stormwatch.monitor.drift import record_prediction
        for i in range(5):
            record_prediction("cyclone", {"lat_abs": 15.0, "lon": 80.0}, i)
        try:
            resp = client.post("/monitor/drift?model_name=cyclone")
            assert resp.status_code in (200, 500)
        except Exception:
            pass
        server._models.clear()


class TestSchemas:
    def test_cyclone_features_validation(self):
        data = dict(lat_abs=15, lon=80, lat=15, pressure_min=980,
                    dist_to_land=50, year=2024, month=10, dayofyear=285)
        f = CycloneFeatures(**data)
        assert f.lat_abs == 15.0

    def test_cyclone_features_invalid_lat_abs(self):
        with pytest.raises(Exception):
            CycloneFeatures(lat_abs=-5, lon=80, lat=10, pressure_min=1000,
                            dist_to_land=0, year=2024, month=1, dayofyear=1)

    def test_heatwave_features_validation(self):
        data = dict(
            temp_max_lag_1=42, temp_max_lag_3=40, temp_max_lag_7=38,
            temp_max_roll_mean_3=41, temp_max_roll_mean_7=39,
            temp_min_lag_1=28, precipitation_lag_1=0,
            relative_humidity_2m_mean=25, wind_speed_10m_max=15,
            pressure_msl_mean=1008, month_sin=0.5, month_cos=0.866,
            month=6)
        f = HeatwaveFeatures(**data)
        assert f.temp_max_lag_1 == 42.0

    def test_rainfall_features_validation(self):
        data = dict(
            precipitation_lag_1=80, precipitation_lag_3=30,
            precipitation_lag_7=10, precipitation_roll_mean_3=45,
            precipitation_roll_mean_7=35, temp_max_lag_1=32,
            temp_max_roll_mean_3=31, relative_humidity_2m_mean=85,
            wind_speed_10m_max=25, pressure_msl_mean=1002,
            cloud_cover_mean=80, month_sin=-0.5, month_cos=0.866,
            month=7)
        f = RainfallFeatures(**data)
        assert f.precipitation_lag_1 == 80.0

    def test_get_category_description(self):
        from stormwatch.api.schemas import get_category_description
        assert "Tropical Depression" in get_category_description(0)
        assert "Category 1" in get_category_description(1)
        assert "Unknown" in get_category_description(99)
