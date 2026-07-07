FROM python:3.13-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p models data

COPY requirements/base.txt requirements/
RUN pip install --no-cache-dir -r requirements/base.txt

COPY stormwatch/ stormwatch/
COPY configs/ configs/
# Models are loaded at runtime from MLflow registry (Production alias)
# with automatic disk fallback if .pkl files are present in models/

EXPOSE 8000

ENV STORMWATCH__TRAINING__MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-sqlite:///mlflow/mlflow.db}

CMD ["uvicorn", "stormwatch.api.server:app", "--host", "0.0.0.0", "--port", "8000"]