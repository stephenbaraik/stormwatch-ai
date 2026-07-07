.PHONY: help setup clean lint test run data train api docker-build docker-up mlflow-ui mlflow-serve promote-model figures

help:
	@echo "StormWatch AI - Makefile"
	@echo "======================="
	@echo "setup        : Create virtualenv and install dependencies"
	@echo "data         : Download and process datasets"
	@echo "train        : Train all models with MLflow tracking"
	@echo "api          : Run FastAPI server locally"
	@echo "monitor      : Run drift monitoring"
	@echo "lint         : Run ruff + mypy"
	@echo "test         : Run pytest suite"
	@echo "figures      : Regenerate report figures"
	@echo "clean        : Remove cache files"
	@echo "mlflow-ui    : Start MLflow tracking UI"
	@echo "mlflow-serve : Serve a model from MLflow registry"
	@echo "promote-model: Promote model version in MLflow registry"
	@echo "docker-build : Build Docker images"
	@echo "docker-up    : Start full stack with Docker Compose"

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

data:
	python -m stormwatch.data.download
	python -m stormwatch.data.preprocess

train:
	python -m stormwatch.models.train

api:
	uvicorn stormwatch.api.server:app --reload --host 0.0.0.0 --port 8000

pipeline:
	python -m stormwatch.data.pipeline

pipeline-force:
	python -m stormwatch.data.pipeline --force

monitor:
	python -m stormwatch.monitor.drift

pipeline-status:
	python -m stormwatch.monitor.pipeline_status

figures:
	python scripts/generate_figures.py

lint:
	ruff check stormwatch/ tests/
	mypy stormwatch/

test:
	pytest tests/ -v --cov=stormwatch

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db --port 5001 --host 127.0.0.1

mlflow-serve:
	@echo "Usage: make mlflow-serve MODEL=stormwatch-cyclone VERSION=1"
	mlflow models serve -m "models:/$(MODEL)/$(VERSION)" --port 5002 --host 127.0.0.1

promote-model:
	@echo "Usage: make promote-model MODEL=stormwatch-cyclone VERSION=1 STAGE=Production"
	python -c "from mlflow import MlflowClient; c = MlflowClient(); c.transition_model_version_stage(name='$(MODEL)', version='$(VERSION)', stage='$(STAGE)')"

docker-build:
	docker compose build

docker-up:
	docker compose up -d