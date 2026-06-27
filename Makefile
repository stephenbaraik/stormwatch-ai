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
	python -m src.data.download
	python -m src.data.preprocess

train:
	python -m src.train

api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

monitor:
	python -m src.monitoring.drift_detection

lint:
	ruff check src/ tests/
	mypy src/

test:
	pytest tests/ -v --cov=src

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache .mypy_cache

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d
