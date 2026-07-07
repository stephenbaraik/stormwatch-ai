# StormWatch AI

<p align="center">
  <em>Extreme Weather Early Warning System for the Indian Subcontinent</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/tests-80%20passed-brightgreen" alt="Tests">
  <img src="https://img.shields.io/badge/python-3.13-blue" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/data-real-important" alt="Real Data">
</p>

---

StormWatch AI is an end-to-end machine learning system that predicts three classes of extreme weather events across 15 Indian cities using XGBoost models trained on **16+ years of real meteorological data**. No synthetic data. Every prediction is verifiable against historical records.

| Model | Task | Accuracy | ROC-AUC |
|---|---|---|---|
| **Cyclone Intensity** | Saffir-Simpson category (0–5) | **98.5%** | — |
| **Heatwave Detection** | Next-day heatwave flag | **99.0%** | **0.997** |
| **Extreme Rainfall** | Next-day 95th-percentile exceedance | **88.3%** | **0.881** |

> **90,138 daily weather records** · **15 Indian cities** · **2009–2026** · **57,632 real cyclone tracks** (NOAA IBTrACS, North Indian Ocean basin)  
> See the [full report](docs/StormWatch_AI_Report.docx) for methodology, findings, and the label-leakage case study.

## Architecture

```
 Open-Meteo Archive API          NOAA IBTrACS
         │                            │
         ▼                            ▼
   ┌──────────┐               ┌──────────────┐
   │  Weather │               │   Cyclone    │
   │  90,138  │               │   57,632     │
   │   rows   │               │   records    │
   └────┬─────┘               └──────┬───────┘
        │                            │
        ▼                            ▼
 ┌──────────────────────────────────────────┐
 │            Preprocessing                  │
 │  · Extreme event labeling                │
 │  · Lag features (1, 3, 7 day)            │
 │  · Rolling statistics (prior-day only)    │
 │  · Cyclic month encoding                  │
 └────────────────────┬─────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ Cyclone │ │Heatwave │ │Rainfall │
    │  XGBoost│ │ XGBoost │ │ XGBoost │
    │ 8 feat. │ │13 feat. │ │14 feat. │
    └────┬────┘ └────┬────┘ └────┬────┘
         │           │           │
         ▼           ▼           ▼
    ┌─────────────────────────────────────┐
    │         FastAPI Server               │
    │  /predict/cyclone                    │
    │  /predict/heatwave                   │
    │  /predict/rainfall                   │
    │  /monitor/drift                      │
    └─────────────────────────────────────┘
         │
         ▼
    ┌──────────┐    ┌──────────────┐
    │ MLflow   │    │  Monitoring   │
    │ Registry │    │  KS-test      │
    │ SQLite   │    │  drift detect │
    └──────────┘    └──────────────┘
```

## Key Findings

**Label leakage was discovered and corrected across all three models.** Each model had been inadvertently fed the same variable used to define its own label — e.g., the heatwave model was given today's temperature to predict whether today was a heatwave day. After removing same-day leaky features, shifting targets to the next day, and recomputing rolling statistics from prior-day values only, the models now produce genuine forecasts:

- **Rainfall accuracy dropped from an artificial 99.7% to an honest 88.3%** — next-day precipitation is genuinely one of the hardest short-term forecasting targets, and a model that claims otherwise is a bug, not a result.
- **Cyclone remained at 98.5%** — minimum central pressure is a legitimate physical proxy for storm intensity, not definitional leakage (verified via ablation study: the model loses only 2.7% accuracy when pressure is removed).
- **Heatwave stayed at 99.0%** — extreme temperatures are highly autocorrelated day-to-day; yesterday's heat genuinely predicts tomorrow's.
- **Hyperopt tuning found no improvement over default XGBoost parameters** across all three models (20 trials, TPE, 3-fold CV). Defaults were already effectively optimal.

Read the [full report](docs/StormWatch_AI_Report.docx) for detailed methodology, the pressure_min leakage audit, hyperopt evaluation results, confusion matrices, feature importance analysis, and all 11 figures.

## Quick Start

```bash
# Clone and set up
git clone https://github.com/stephenbaraik/stormwatch-ai.git
cd stormwatch-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements/base.txt

# Set environment (optional — needed for Supabase data pull)
cp .env.example .env   # add SUPABASE_URL + SUPABASE_SERVICE_KEY

# Pull real data and train
python scripts/pull_supabase_weather.py
python -m stormwatch.models.train

# Start the API
uvicorn stormwatch.api.server:app --reload

# Open Swagger UI: http://localhost:8000/docs
```

## API

FastAPI with interactive Swagger docs at `/docs`. All prediction endpoints require an `X-API-Key` header (configurable via `STORMWATCH_API_KEY` env var; disabled when unset).

```bash
# Cyclone intensity prediction
curl -X POST http://localhost:8000/predict/cyclone \
  -H "Content-Type: application/json" \
  -d '{
    "lat_abs": 15.0, "lon": 75.0, "lat": 15.0,
    "pressure_min": 970.0, "dist_to_land": 50.0,
    "year": 2024, "month": 6, "dayofyear": 180
  }'
```

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check + loaded model count |
| `GET` | `/models` | List loaded models |
| `POST` | `/predict/cyclone` | Cyclone intensity (8 features, category 0–5) |
| `POST` | `/predict/heatwave` | Heatwave probability (13 features) |
| `POST` | `/predict/rainfall` | Extreme rainfall probability (14 features) |
| `POST` | `/monitor/drift` | Statistical drift check (KS-test) |

## Tech Stack

| Layer | Technology |
|---|---|
| Models | XGBoost 3.3 · scikit-learn 1.9 |
| API | FastAPI 0.139 · Pydantic v2 · Uvicorn |
| Experiment tracking | MLflow 3.14 (SQLite backend + model registry) |
| Monitoring | Kolmogorov-Smirnov drift detection |
| ETL | Pandas + PySpark (partitioned Parquet) |
| Data sources | Open-Meteo Archive API · NOAA IBTrACS · Supabase PostgreSQL |
| Deployment | Docker · Docker Compose · GitHub Actions CI/CD |
| Testing | pytest (80 tests, 100% passing) |

## Data Coverage

15 cities spanning four Indian climate zones:

| Zone | Cities |
|---|---|
| Coastal | Mumbai · Chennai · Kolkata · Kochi · Bhubaneswar · Visakhapatnam · Surat |
| Inland | Delhi · Hyderabad · Bengaluru · Lucknow · Pune |
| Arid | Ahmedabad · Jaipur |
| Humid | Guwahati |

The dataset spans 31 December 2009 to 25 June 2026 — 90,138 daily records with 18 meteorological variables per observation. Cyclone training data is sourced from 57,632 NOAA IBTrACS track records filtered to tropical storm (TS) and mixed/subtropical (MX) classifications.

## Project Structure

```
stormwatch-ai/
├── stormwatch/           # Main package
│   ├── api/              # FastAPI server + Pydantic schemas
│   ├── data/             # Download, preprocessing, PySpark ETL
│   ├── features/         # Feature engineering pipelines
│   ├── models/           # XGBoost classifiers + training orchestration
│   └── monitor/          # KS-test drift detection
├── tests/                # 80 tests, 100% passing
├── models/               # Trained .pkl artifacts (disk fallback)
├── configs/              # YAML configuration
├── docs/                 # Report + 11 regenerated figures
├── requirements/         # Pinned dependencies (base + dev)
├── scripts/              # Figure generation, data pull utilities
└── Dockerfile            # Multi-service Compose deployment
```

## License

MIT
