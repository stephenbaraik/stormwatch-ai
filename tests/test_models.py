from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stormwatch.features.builder import (
    build_cyclone_features,
    build_heatwave_features,
    build_rainfall_features,
)


class TestCycloneIntensityModel:
    def test_build_pipeline(self, cyclone_model):
        pipeline = cyclone_model.build_pipeline()
        assert pipeline is not None
        steps = [name for name, _ in pipeline.steps]
        assert "scaler" in steps
        assert "classifier" in steps

    def test_train_returns_metrics(self, trained_cyclone_model):
        assert trained_cyclone_model.is_trained()

    def test_predict_returns_array(self, trained_cyclone_model, mock_cyclone_df):
        X, _ = build_cyclone_features(mock_cyclone_df)
        preds = trained_cyclone_model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X)
        assert all(0 <= p <= 5 for p in preds)

    def test_predict_proba_shape(self, trained_cyclone_model, mock_cyclone_df):
        X, _ = build_cyclone_features(mock_cyclone_df)
        proba = trained_cyclone_model.predict_proba(X)
        assert proba.ndim == 2
        assert proba.shape[0] == len(X)

    def test_predict_raises_if_not_trained(self, cyclone_model):
        X = pd.DataFrame({"lat_abs": [10.0], "lon": [80.0], "lat": [10.0],
                          "pressure_min": [1000], "dist_to_land": [0],
                          "year": [2024], "month": [1], "dayofyear": [1],
                          "wind_kts": [50]})
        with pytest.raises(RuntimeError, match="not trained"):
            cyclone_model.predict(X)

    def test_predict_missing_features(self, trained_cyclone_model):
        X_bad = pd.DataFrame({"lat_abs": [10.0]})
        with pytest.raises(ValueError, match="Missing features"):
            trained_cyclone_model.predict(X_bad)

    def test_evaluate_returns_metrics(self, trained_cyclone_model, mock_cyclone_df):
        X, y = build_cyclone_features(mock_cyclone_df)
        metrics = trained_cyclone_model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert all(isinstance(v, float) for v in metrics.values())

    def test_feature_names_match(self, cyclone_model):
        from stormwatch.features.builder import CYCLONE_FEATURES
        assert cyclone_model.feature_names == CYCLONE_FEATURES

    def test_build_features(self, mock_cyclone_df):
        from stormwatch.models.cyclone import CycloneIntensityModel
        model = CycloneIntensityModel()
        X, y = model.build_features(mock_cyclone_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert len(X) == len(y)


class TestHeatwavePredictionModel:
    def test_build_pipeline(self, heatwave_model):
        pipeline = heatwave_model.build_pipeline()
        steps = [name for name, _ in pipeline.steps]
        assert "scaler" in steps
        assert "classifier" in steps

    def test_train_and_predict(self, trained_heatwave_model, mock_weather_df):
        X, y = build_heatwave_features(mock_weather_df)
        preds = trained_heatwave_model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_binary(self, trained_heatwave_model, mock_weather_df):
        X, _ = build_heatwave_features(mock_weather_df)
        proba = trained_heatwave_model.predict_proba(X)
        assert proba.shape[1] == 2

    def test_predict_raises_if_not_trained(self, heatwave_model):
        X = pd.DataFrame({"temp_max": [30.0], "temp_max_lag_1": [29.0],
                          "temp_max_lag_3": [28.0], "temp_max_roll_mean_3": [29.0],
                          "temp_max_roll_mean_7": [28.5], "temp_min": [20.0],
                          "precipitation": [0], "precipitation_lag_1": [0],
                          "relative_humidity_2m_mean": [50], "wind_speed_10m_max": [10],
                          "pressure_msl_mean": [1013], "month_sin": [0], "month_cos": [1],
                          "month": [1]})
        with pytest.raises(RuntimeError, match="not trained"):
            heatwave_model.predict(X)

    def test_evaluate_metrics(self, trained_heatwave_model, mock_weather_df):
        X, y = build_heatwave_features(mock_weather_df)
        metrics = trained_heatwave_model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "roc_auc" in metrics

    def test_build_features(self, mock_weather_df):
        from stormwatch.models.heatwave import HeatwavePredictionModel
        model = HeatwavePredictionModel()
        X, y = model.build_features(mock_weather_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_target_is_binary(self, trained_heatwave_model, mock_weather_df):
        _, y = build_heatwave_features(mock_weather_df)
        assert y.dtype in (np.int64, np.int32, int)
        assert y.nunique() <= 2


class TestExtremeRainfallModel:
    def test_build_pipeline(self, rainfall_model):
        pipeline = rainfall_model.build_pipeline()
        steps = [name for name, _ in pipeline.steps]
        assert "scaler" in steps
        assert "classifier" in steps

    def test_train_and_predict(self, trained_rainfall_model, mock_weather_df):
        X, y = build_rainfall_features(mock_weather_df)
        preds = trained_rainfall_model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X)

    def test_predict_proba_shape(self, trained_rainfall_model, mock_weather_df):
        X, _ = build_rainfall_features(mock_weather_df)
        proba = trained_rainfall_model.predict_proba(X)
        assert proba.shape[1] == 2

    def test_predict_raises_if_not_trained(self, rainfall_model):
        X = pd.DataFrame({"precipitation": [10.0], "precipitation_lag_1": [5.0],
                          "precipitation_lag_3": [2.0], "precipitation_roll_mean_3": [3.0],
                          "precipitation_roll_mean_7": [4.0], "temp_max": [30.0],
                          "temp_max_roll_mean_3": [29.0], "relative_humidity_2m_mean": [70],
                          "wind_speed_10m_max": [15], "pressure_msl_mean": [1010],
                          "cloud_cover_mean": [60], "month_sin": [0], "month_cos": [1],
                          "month": [6]})
        with pytest.raises(RuntimeError, match="not trained"):
            rainfall_model.predict(X)

    def test_evaluate_metrics(self, trained_rainfall_model, mock_weather_df):
        X, y = build_rainfall_features(mock_weather_df)
        metrics = trained_rainfall_model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "roc_auc" in metrics

    def test_build_features(self, mock_weather_df):
        from stormwatch.models.rainfall import ExtremeRainfallModel
        model = ExtremeRainfallModel()
        X, y = model.build_features(mock_weather_df)
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_align_features_drops_extra_cols(self, trained_rainfall_model, mock_weather_df):
        X, y = build_rainfall_features(mock_weather_df)
        X_extra = X.copy()
        X_extra["extra_col"] = 1.0
        trained_rainfall_model.predict(X_extra)


class TestBaseModel:
    def test_is_trained_default_false(self, cyclone_model):
        assert not cyclone_model.is_trained()

    def test_get_feature_names(self, cyclone_model):
        names = cyclone_model.get_feature_names()
        assert isinstance(names, list)

    def test_compute_class_weights_balanced(self, cyclone_model):
        y = pd.Series([0, 0, 0, 1, 1, 2])
        weights = cyclone_model._compute_class_weights(y)
        assert len(weights) == 3
        for w in weights.values():
            assert w > 0
