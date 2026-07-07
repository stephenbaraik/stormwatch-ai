"""
StormWatch AI - Training Pipeline
End-to-end training with MLflow tracking and Hyperopt hyperparameter tuning.
Training requires real weather and cyclone data — no synthetic fallbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from sklearn.model_selection import cross_val_score, train_test_split

import mlflow
from stormwatch.config import get_config
from stormwatch.data.download import download_all_weather_data, download_ibtracs
from stormwatch.data.preprocess import (
    preprocess_all,
)
from stormwatch.features.builder import (
    build_cyclone_features,
    build_heatwave_features,
    build_rainfall_features,
)
from stormwatch.logger import get_logger
from stormwatch.models.cyclone import CycloneIntensityXGB
from stormwatch.models.heatwave import HeatwaveXGBModel
from stormwatch.models.rainfall import RainfallXGBModel

log = get_logger(__name__)


# ──────────────────────────────────────────────
#  Training runner
# ──────────────────────────────────────────────


def train_cyclone_model(
    df: pd.DataFrame,
    use_hyperopt: bool = True,
    experiment_name: Optional[str] = None,
) -> CycloneIntensityXGB:
    """Train the cyclone intensity model with MLflow tracking.

    Args:
        df: Processed cyclone DataFrame
        use_hyperopt: Run hyperparameter optimization
        experiment_name: MLflow experiment name (default from config)

    Returns:
        Trained CycloneIntensityXGB model
    """
    config = get_config()
    experiment_name = experiment_name or config.training.experiment_name
    mlflow.set_tracking_uri(config.training.mlflow_tracking_uri)
    mlflow.set_experiment(f"{experiment_name}_cyclone")

    X, y = build_cyclone_features(df)
    if X.empty:
        log.error("No cyclone features to train on")
        raise ValueError("Empty feature matrix")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="cyclone_intensity") as _:
        mlflow.log_params(
            {
                "model_type": "XGBoost",
                "n_features": X.shape[1],
                "n_train": len(X_train),
                "n_test": len(X_test),
                "classes": sorted(y.unique().tolist()),
            }
        )

        best_params = {}

        if use_hyperopt:
            log.info("Running Hyperopt for cyclone model...")
            best_params = _hyperopt_cyclone(X_train, y_train)
            best_params = _cast_hyperopt_params(
                best_params, ["max_depth", "n_estimators", "min_child_weight"]
            )
            mlflow.log_params({f"hyperopt_{k}": v for k, v in best_params.items()})
            log.info("Best params: %s", best_params)

        # Train with best params
        model = CycloneIntensityXGB(
            config={
                "random_state": 42,
                **best_params,
            }
        )
        train_metrics = model.train(X_train, y_train)
        test_metrics = model.evaluate(X_test, y_test)

        # Log metrics
        for prefix, metrics in [("train", train_metrics), ("test", test_metrics)]:
            for k, v in metrics.items():
                mlflow.log_metric(f"{prefix}_{k}", v)

        # Log feature importance
        _log_feature_importance(model, X.columns)

        # Save and register model
        _save_and_register(model, "cyclone", config)

        log.info(
            "Cyclone model trained: test_accuracy=%.3f, test_f1=%.3f",
            test_metrics.get("accuracy", 0),
            test_metrics.get("f1", 0),
        )

    return model


def train_heatwave_model(
    df: pd.DataFrame,
    use_hyperopt: bool = True,
    experiment_name: Optional[str] = None,
) -> HeatwaveXGBModel:
    """Train the heatwave prediction model with MLflow tracking."""
    config = get_config()
    experiment_name = experiment_name or config.training.experiment_name
    mlflow.set_tracking_uri(config.training.mlflow_tracking_uri)
    mlflow.set_experiment(f"{experiment_name}_heatwave")

    X, y = build_heatwave_features(df)
    if X.empty:
        raise ValueError("Empty heatwave feature matrix")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="heatwave_prediction") as _:
        mlflow.log_params(
            {
                "model_type": "XGBoost",
                "n_features": X.shape[1],
                "n_train": len(X_train),
                "n_test": len(X_test),
                "pos_ratio": float(y_train.mean()),
            }
        )

        best_params = {}
        if use_hyperopt:
            log.info("Running Hyperopt for heatwave model...")
            best_params = _hyperopt_heatwave(X_train, y_train)
            best_params = _cast_hyperopt_params(
                best_params, ["max_depth", "n_estimators", "scale_pos_weight"]
            )
            mlflow.log_params({f"hyperopt_{k}": v for k, v in best_params.items()})
            log.info("Best params: %s", best_params)

        model = HeatwaveXGBModel(
            config={
                "random_state": 42,
                "scale_pos_weight": int(
                    (y_train == 0).sum() / max((y_train == 1).sum(), 1)
                ),
                **best_params,
            }
        )
        train_metrics = model.train(X_train, y_train)
        test_metrics = model.evaluate(X_test, y_test)

        for prefix, metrics in [("train", train_metrics), ("test", test_metrics)]:
            for k, v in metrics.items():
                mlflow.log_metric(f"{prefix}_{k}", v)

        _log_feature_importance(model, X.columns)

        _save_and_register(model, "heatwave", config)

        log.info(
            "Heatwave model trained: test_accuracy=%.3f, test_auc=%.3f",
            test_metrics.get("accuracy", 0),
            test_metrics.get("roc_auc", 0),
        )

    return model


def train_rainfall_model(
    df: pd.DataFrame,
    use_hyperopt: bool = True,
    experiment_name: Optional[str] = None,
) -> RainfallXGBModel:
    """Train the extreme rainfall prediction model with MLflow tracking."""
    config = get_config()
    experiment_name = experiment_name or config.training.experiment_name
    mlflow.set_tracking_uri(config.training.mlflow_tracking_uri)
    mlflow.set_experiment(f"{experiment_name}_rainfall")

    X, y = build_rainfall_features(df)
    if X.empty:
        raise ValueError("Empty rainfall feature matrix")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="extreme_rainfall") as _:
        mlflow.log_params(
            {
                "model_type": "XGBoost",
                "n_features": X.shape[1],
                "n_train": len(X_train),
                "n_test": len(X_test),
                "pos_ratio": float(y_train.mean()),
            }
        )

        best_params = {}
        if use_hyperopt:
            log.info("Running Hyperopt for rainfall model...")
            best_params = _hyperopt_rainfall(X_train, y_train)
            best_params = _cast_hyperopt_params(
                best_params, ["max_depth", "n_estimators", "scale_pos_weight"]
            )
            mlflow.log_params({f"hyperopt_{k}": v for k, v in best_params.items()})

        model = RainfallXGBModel(
            config={
                "random_state": 42,
                "scale_pos_weight": int(
                    (y_train == 0).sum() / max((y_train == 1).sum(), 1)
                ),
                **best_params,
            }
        )
        train_metrics = model.train(X_train, y_train)
        test_metrics = model.evaluate(X_test, y_test)

        for prefix, metrics in [("train", train_metrics), ("test", test_metrics)]:
            for k, v in metrics.items():
                mlflow.log_metric(f"{prefix}_{k}", v)

        _log_feature_importance(model, X.columns)

        _save_and_register(model, "rainfall", config)

        log.info(
            "Rainfall model trained: test_accuracy=%.3f, test_auc=%.3f",
            test_metrics.get("accuracy", 0),
            test_metrics.get("roc_auc", 0),
        )

    return model


# ──────────────────────────────────────────────
#  Real data download & preprocessing
# ──────────────────────────────────────────────


def _ensure_data_downloaded(
    download_weather: bool = True,
    download_cyclone: bool = True,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download real data from Open-Meteo and IBTrACS, then preprocess.

    Args:
        download_weather: Download weather data for 15 Indian cities
        download_cyclone: Download IBTrACS cyclone data for Indian Ocean
        force: Re-download even if cached files exist

    Returns:
        (weather_processed, cyclone_processed) DataFrames
    """
    config = get_config()
    weather_df: pd.DataFrame = pd.DataFrame()
    cyclone_df: pd.DataFrame = pd.DataFrame()

    # ── Weather data ──
    combined_path = Path(config.data.raw_path) / "weather_all_cities.csv"
    if download_weather and (not combined_path.exists() or force):
        log.info(
            "Downloading real weather data from Open-Meteo for %d cities...",
            len(download_all_weather_data.__code__.co_consts),
        )
        raw_weather = download_all_weather_data(force=force)
        if raw_weather.empty:
            log.warning("Open-Meteo download returned no data")
        else:
            log.info("Downloaded %d rows of weather data", len(raw_weather))
    elif combined_path.exists():
        log.info("Using cached weather data from %s", combined_path)
    else:
        log.info("Weather download skipped")

    # ── Cyclone data (IBTrACS) ──
    cyclone_path = (
        Path(config.data.raw_path) / f"ibtracs_{config.data.ibtracs.basin}.csv"
    )
    if download_cyclone and (not cyclone_path.exists() or force):
        log.info(
            "Downloading IBTrACS cyclone data (%s basin)...", config.data.ibtracs.basin
        )
        result = download_ibtracs(basin=config.data.ibtracs.basin, force=force)
        if result is None:
            log.warning("IBTrACS download failed")
    elif cyclone_path.exists():
        log.info("Using cached cyclone data from %s", cyclone_path)
    else:
        log.info("Cyclone download skipped")

    # ── Preprocess everything available ──
    weather_df, cyclone_df = preprocess_all(
        weather_path=str(combined_path) if combined_path.exists() else None,
        cyclone_path=str(cyclone_path) if cyclone_path.exists() else None,
        save=True,
    )

    if weather_df.empty:
        log.warning("No weather data available after preprocessing")
    if cyclone_df.empty:
        log.warning("No cyclone data available after preprocessing")

    return weather_df, cyclone_df


# ──────────────────────────────────────────────
#  Full training run
# ──────────────────────────────────────────────


def train_all(
    weather_df: Optional[pd.DataFrame] = None,
    cyclone_df: Optional[pd.DataFrame] = None,
    use_hyperopt: bool = False,
) -> Dict[str, Any]:
    """Train all three models end-to-end on real data only.

    Downloads real Open-Meteo and IBTrACS data if not provided.
    Raises ValueError if no real data is available — no synthetic fallbacks.

    Args:
        weather_df: Preprocessed weather DataFrame (auto-download if None)
        cyclone_df: Preprocessed cyclone DataFrame (auto-download if None)
        use_hyperopt: Run Hyperopt HP tuning (takes longer)

    Returns:
        Dict of model_name -> test_metrics
    """
    results: Dict[str, Any] = {}

    # Download real data if not provided
    if weather_df is None and cyclone_df is None:
        log.info(
            "No data provided. Downloading real weather data from Open-Meteo / IBTrACS..."
        )
        weather_df, cyclone_df = _ensure_data_downloaded()

    # ── Cyclone model ──
    if cyclone_df is not None and not cyclone_df.empty:
        log.info("Training cyclone model on real data...")
        model = train_cyclone_model(cyclone_df, use_hyperopt=use_hyperopt)
        results["cyclone"] = model.evaluate(*build_cyclone_features(cyclone_df))
    else:
        log.warning("Skipping cyclone model: no data available")

    # ── Heatwave model ──
    if weather_df is not None and not weather_df.empty:
        log.info("Training heatwave model on real data...")
        model = train_heatwave_model(weather_df, use_hyperopt=use_hyperopt)
        X_h, y_h = build_heatwave_features(weather_df)
        results["heatwave"] = model.evaluate(X_h, y_h)
    else:
        log.warning("Skipping heatwave model: no data available")

    # ── Rainfall model ──
    if weather_df is not None and not weather_df.empty:
        log.info("Training rainfall model on real data...")
        model = train_rainfall_model(weather_df, use_hyperopt=use_hyperopt)
        X_r, y_r = build_rainfall_features(weather_df)
        results["rainfall"] = model.evaluate(X_r, y_r)
    else:
        log.warning("Skipping rainfall model: no data available")

    return results


# ──────────────────────────────────────────────
#  Hyperopt objective functions
# ──────────────────────────────────────────────


def _hyperopt_cyclone(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Hyperparameter optimization for cyclone model."""

    def objective(params: Dict) -> Dict[str, Any]:
        params["max_depth"] = int(params["max_depth"])
        params["n_estimators"] = int(params["n_estimators"])
        try:
            model = CycloneIntensityXGB(config={"random_state": 42, **params})
            scores = cross_val_score(
                model.build_pipeline(), X, y, cv=3, scoring="f1_weighted"
            )
            return {"loss": -scores.mean(), "status": STATUS_OK}
        except Exception:
            return {"loss": 1.0, "status": STATUS_OK}

    space = {
        "max_depth": hp.quniform("max_depth", 4, 12, 1),
        "n_estimators": hp.quniform("n_estimators", 100, 300, 50),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
        "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
        "subsample": hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials,
        rstate=np.random.default_rng(42),
    )

    return best


def _hyperopt_heatwave(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Hyperparameter optimization for heatwave model."""

    def objective(params: Dict) -> Dict[str, Any]:
        params["max_depth"] = int(params["max_depth"])
        params["n_estimators"] = int(params["n_estimators"])
        params["scale_pos_weight"] = int(params["scale_pos_weight"])
        try:
            model = HeatwaveXGBModel(config={"random_state": 42, **params})
            scores = cross_val_score(
                model.build_pipeline(), X, y, cv=3, scoring="roc_auc"
            )
            return {"loss": -scores.mean(), "status": STATUS_OK}
        except Exception:
            return {"loss": 1.0, "status": STATUS_OK}

    pos_weight = int((y == 0).sum() / max((y == 1).sum(), 1))
    space = {
        "max_depth": hp.quniform("max_depth", 3, 10, 1),
        "n_estimators": hp.quniform("n_estimators", 100, 300, 50),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
        "scale_pos_weight": hp.quniform(
            "scale_pos_weight", max(1, pos_weight // 2), pos_weight * 2, 1
        ),
        "subsample": hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials,
        rstate=np.random.default_rng(42),
    )
    return best


def _hyperopt_rainfall(X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
    """Hyperparameter optimization for rainfall model."""

    def objective(params: Dict) -> Dict[str, Any]:
        params["max_depth"] = int(params["max_depth"])
        params["n_estimators"] = int(params["n_estimators"])
        params["scale_pos_weight"] = int(params["scale_pos_weight"])
        try:
            model = RainfallXGBModel(config={"random_state": 42, **params})
            scores = cross_val_score(
                model.build_pipeline(), X, y, cv=3, scoring="roc_auc"
            )
            return {"loss": -scores.mean(), "status": STATUS_OK}
        except Exception:
            return {"loss": 1.0, "status": STATUS_OK}

    pos_weight = int((y == 0).sum() / max((y == 1).sum(), 1))
    space = {
        "max_depth": hp.quniform("max_depth", 3, 10, 1),
        "n_estimators": hp.quniform("n_estimators", 100, 300, 50),
        "learning_rate": hp.uniform("learning_rate", 0.01, 0.3),
        "scale_pos_weight": hp.quniform(
            "scale_pos_weight", max(1, pos_weight // 2), pos_weight * 2, 1
        ),
        "subsample": hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
    }

    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=20,
        trials=trials,
        rstate=np.random.default_rng(42),
    )
    return best


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────


def _cast_hyperopt_params(
    params: Dict[str, Any], int_keys: list[str]
) -> Dict[str, Any]:
    """Cast hyperopt float params to native Python ints.

    hp.quniform returns np.float64 values, but XGBoost requires int for
    n_estimators, max_depth, min_child_weight, and scale_pos_weight.
    """
    result = dict(params)
    for k in int_keys:
        if k in result:
            result[k] = int(result[k])
    return result


def _log_feature_importance(model: Any, feature_names: pd.Index) -> None:
    """Log feature importance to MLflow."""
    try:
        pipe = model.pipeline
        if pipe is None:
            return

        classifier = pipe.named_steps.get("classifier")
        if classifier is None:
            return

        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
            for name, imp in zip(feature_names, importances):
                mlflow.log_metric(f"feat_imp_{name}", float(imp))
    except Exception as e:
        log.warning("Failed to log feature importance: %s", e)


def _save_and_register(model: Any, name: str, config: Any) -> None:
    """Save model to disk (fallback) and log to MLflow registry.

    Steps:
    1. Save .pkl to disk for local fallback loading
    2. Log model to MLflow with signature and input example
    3. Register model in MLflow registry
    4. Set alias 'Production' on the new version
    """
    import mlflow
    from mlflow import MlflowClient

    # Save to disk as local fallback
    models_dir = Path(config.api.model_path)
    models_dir.mkdir(parents=True, exist_ok=True)
    disk_path = models_dir / f"{name}_model.pkl"
    model.save(str(disk_path))
    log.info("Saved %s model to %s", name, disk_path)

    # Log to MLflow with signature
    try:
        uri = config.training.mlflow_tracking_uri
        mlflow.set_tracking_uri(uri)
        mlflow.set_registry_uri(uri)

        # Ensure artifact root is set for local filesystem storage
        import os

        artifact_root = os.path.abspath("./mlruns")
        os.makedirs(artifact_root, exist_ok=True)

        model.log_model("model")
        run_id = mlflow.active_run().info.run_id
        registered_name = f"stormwatch-{name}"

        client = MlflowClient(tracking_uri=config.training.mlflow_tracking_uri)
        try:
            client.create_registered_model(registered_name)
        except Exception:
            pass

        mv = client.create_model_version(
            name=registered_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id,
        )
        client.set_registered_model_alias(registered_name, "Production", mv.version)
        log.info(
            "Registered %s v%d as '%s' (Production alias)",
            registered_name,
            mv.version,
            mv.version,
        )
    except Exception as e:
        log.warning("Failed to register model in MLflow: %s", e)
        log.info("Disk fallback available at %s", disk_path)


# ──────────────────────────────────────────────
#  CLI entry point
# ──────────────────────────────────────────────


def main() -> None:
    """Run full training pipeline from CLI.

    Downloads real Open-Meteo weather data and IBTrACS cyclone data.
    Requires real data — no synthetic fallback.
    """
    import argparse

    parser = argparse.ArgumentParser(description="StormWatch AI Training Pipeline")
    parser.add_argument(
        "--hyperopt",
        action="store_true",
        help="Run hyperparameter optimization (takes longer)",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download data even if cached files exist",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("StormWatch AI - Training Pipeline")
    log.info("=" * 60)

    if args.force_download:
        log.info("Force-downloading fresh data...")
        _ensure_data_downloaded(force=True)
    results = train_all(use_hyperopt=args.hyperopt)

    log.info("=" * 60)
    log.info("Training Complete!")
    for model_name, metrics in results.items():
        acc = metrics.get("accuracy", 0)
        auc = metrics.get("roc_auc", 0)
        log.info("  %s: accuracy=%.3f, roc_auc=%.3f", model_name, acc, auc)

    log.info("=" * 60)
    log.info("View MLflow UI: mlflow ui --backend-store-uri ./mlflow")


if __name__ == "__main__":
    main()
