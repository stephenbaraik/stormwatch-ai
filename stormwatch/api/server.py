"""
StormWatch AI - FastAPI Server
Production-grade REST API for extreme weather predictions.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader

from stormwatch.api.schemas import (
    CycloneFeatures,
    CyclonePrediction,
    DriftReport,
    ErrorResponse,
    HealthResponse,
    HeatwaveFeatures,
    HeatwavePrediction,
    PredictionResponse,
    RainfallFeatures,
    RainfallPrediction,
    get_category_description,
)
from stormwatch.config import get_config
from stormwatch.logger import get_logger

log = get_logger(__name__)

# ──────────────────────────────────────────────
#  API key auth dependency
# ──────────────────────────────────────────────

import os

API_KEY_NAME = "X-API-Key"
_api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def require_api_key(api_key: str | None = Security(_api_key_header)):
    expected = os.getenv("STORMWATCH_API_KEY", "")
    if not expected:
        return True
    if api_key and api_key == expected:
        return True
    raise HTTPException(status_code=401, detail="Invalid or missing API key")

# ──────────────────────────────────────────────
#  Global model cache
# ──────────────────────────────────────────────

_models: Dict[str, object] = {}


def load_models(model_dir: Optional[str] = None) -> Dict[str, object]:
    """Load trained models: MLflow registry first, disk fallback second.

    MLflow registry requires a reachable tracking server and registered
    models with Production aliases. Falls back to .pkl files on disk.
    """
    config = get_config()
    model_dir = model_dir or config.api.model_path

    loaded: Dict[str, object] = {}

    for model_name in ["cyclone", "heatwave", "rainfall"]:
        model = _load_from_mlflow(model_name)
        if model is not None:
            loaded[model_name] = model
            log.info("Loaded %s from MLflow registry", model_name)
            continue

        model = _load_from_disk(model_name, model_dir)
        if model is not None:
            loaded[model_name] = model
            log.info("Loaded %s from disk", model_name)

    if not loaded:
        log.warning("No models available — train models first with 'make train'")
    return loaded


def _load_from_mlflow(model_name: str) -> object | None:
    """Try loading a model from the MLflow registry (Production alias)."""
    try:
        import mlflow
        cfg = get_config()
        uri = cfg.training.mlflow_tracking_uri
        mlflow.set_tracking_uri(uri)
        model_uri = f"models:/stormwatch-{model_name}@Production"
        pipeline = mlflow.sklearn.load_model(model_uri)

        class _PipelineWrapper:
            def __init__(self, pipe):
                self._pipe = pipe
            def predict(self, X):
                return self._pipe.predict(X)
            def predict_proba(self, X):
                return self._pipe.predict_proba(X)
            def is_trained(self):
                return True

        return _PipelineWrapper(pipeline)
    except Exception:
        return None


def _load_from_disk(model_name: str, model_dir: str) -> object | None:
    """Try loading a model from a .pkl file on disk."""
    p = Path(model_dir) / f"{model_name}_model.pkl"
    if not p.exists():
        return None
    try:
        model = joblib.load(str(p))
        if hasattr(model, "predict") and hasattr(model, "is_trained"):
            if model.is_trained():
                return model
    except Exception as e:
        log.warning("Failed to load %s from disk: %s", p.name, e)
    return None


# ──────────────────────────────────────────────
#  Lifecycle
# ──────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    global _models
    _models = load_models()
    log.info("StormWatch AI API ready with %d models", len(_models))
    yield
    _models.clear()
    log.info("StormWatch AI API shutdown")


# ──────────────────────────────────────────────
#  App
# ──────────────────────────────────────────────

config = get_config()

app = FastAPI(
    title=config.api.title,
    version=config.api.version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────


@app.get("/", response_model=HealthResponse, tags=["System"])
def root():
    """Root endpoint - API health check."""
    return HealthResponse(
        status="ok",
        version=config.api.version,
        models_loaded=len(_models),
        models=list(_models.keys()),
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        version=config.api.version,
        models_loaded=len(_models),
        models=list(_models.keys()),
    )


@app.get("/models", tags=["System"])
def list_models():
    """List all loaded models and their status."""
    return {
        "models": {
            name: {
                "loaded": True,
                "type": type(model).__name__,
            }
            for name, model in _models.items()
        },
        "count": len(_models),
    }


@app.post(
    "/predict/cyclone",
    response_model=PredictionResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["Predictions"],
)
def predict_cyclone(features: CycloneFeatures, _=Depends(require_api_key)):
    """Predict cyclone intensity (Saffir-Simpson category 0-5)."""
    model = _models.get("cyclone")
    if model is None:
        raise HTTPException(status_code=503, detail="Cyclone model not loaded")

    try:
        # Build feature DataFrame
        X = pd.DataFrame([features.model_dump()])

        # Get prediction
        category = int(model.predict(X)[0])
        proba = model.predict_proba(X)

        # Build probabilities dict
        n_classes = proba.shape[1] if proba.ndim > 1 else 6
        if proba.ndim > 1 and proba.shape[1] > 1:
            probabilities = {str(i): float(proba[0, i]) for i in range(n_classes)}
        else:
            probabilities = {str(i): 0.0 for i in range(n_classes)}
            probabilities[str(category)] = 1.0

        confidence = float(max(probabilities.values()))

        prediction = CyclonePrediction(
            category=category,
            description=get_category_description(category),
            probabilities=probabilities,
            confidence=confidence,
        )

        return PredictionResponse(
            model="cyclone_intensity",
            prediction=prediction.model_dump(),
        )

    except Exception as e:
        log.error("Cyclone prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/heatwave",
    response_model=PredictionResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["Predictions"],
)
def predict_heatwave(features: HeatwaveFeatures, _=Depends(require_api_key)):
    """Predict heatwave probability."""
    model = _models.get("heatwave")
    if model is None:
        raise HTTPException(status_code=503, detail="Heatwave model not loaded")

    try:
        X = pd.DataFrame([features.model_dump()])
        proba = model.predict_proba(X)
        probability = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
        is_heatwave = probability >= 0.5

        # Severity levels
        if probability < 0.3:
            severity = "none"
        elif probability < 0.5:
            severity = "watch"
        elif probability < 0.75:
            severity = "warning"
        else:
            severity = "severe"

        prediction = HeatwavePrediction(
            heatwave_probability=round(probability, 3),
            is_heatwave=is_heatwave,
            severity=severity,
            confidence=round(max(probability, 1 - probability), 3),
        )

        return PredictionResponse(
            model="heatwave_prediction",
            prediction=prediction.model_dump(),
        )

    except Exception as e:
        log.error("Heatwave prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/predict/rainfall",
    response_model=PredictionResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["Predictions"],
)
def predict_rainfall(features: RainfallFeatures, _=Depends(require_api_key)):
    """Predict extreme rainfall probability."""
    model = _models.get("rainfall")
    if model is None:
        raise HTTPException(status_code=503, detail="Rainfall model not loaded")

    try:
        X = pd.DataFrame([features.model_dump()])
        proba = model.predict_proba(X)
        probability = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
        is_extreme = probability >= 0.5

        prediction = RainfallPrediction(
            extreme_rainfall_probability=round(probability, 3),
            is_extreme=is_extreme,
            confidence=round(max(probability, 1 - probability), 3),
        )

        return PredictionResponse(
            model="extreme_rainfall",
            prediction=prediction.model_dump(),
        )

    except Exception as e:
        log.error("Rainfall prediction failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/monitor/drift", response_model=DriftReport, tags=["Monitoring"])
def check_drift(model_name: str = "cyclone", _=Depends(require_api_key)):
    """Check for data drift in model predictions."""
    from stormwatch.monitor.drift import run_drift_check

    try:
        return run_drift_check(model_name, _models.get(model_name))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────


def main() -> None:
    """Run the API server with uvicorn."""
    import uvicorn

    uvicorn.run(
        "stormwatch.api.server:app",
        host=config.api.host,
        port=config.api.port,
        reload=True,
        log_level="info",
    )


if __name__ == "__main__":
    main()
