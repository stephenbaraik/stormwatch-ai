.PHONY: help setup clean lint test run data train api docker-build docker-up

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
	@echo "clean        : Remove cache files"
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

lint:
	ruff check stormwatch/ tests/
	mypy stormwatch/

test:
	pytest tests/ -v --cov=stormwatch

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d
