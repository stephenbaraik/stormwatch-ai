# StormWatch AI — Extreme Weather Early Warning System

[![Data Pipeline](https://github.com/stephenbaraik/stormwatch-ai/actions/workflows/data-pipeline.yml/badge.svg)](https://github.com/stephenbaraik/stormwatch-ai/actions/workflows/data-pipeline.yml)

StormWatch AI downloads **real weather data** from the [Open-Meteo Archive API](https://archive-api.open-meteo.com/), ingests it into **Supabase** via an overnight cron pipeline, and trains **three XGBoost models** to predict extreme weather events across 15 Indian cities.

**No synthetic data.** Every prediction is backed by 16+ years of historical meteorological records.

| Model | Task | Accuracy | ROC-AUC |
|-------|------|----------|---------|
| **Cyclone Intensity** | Saffir-Simpson category (0–5) | **98.9%** | — |
| **Heatwave Detection** | Heatwave flag (binary) | **99.4%** | **0.9982** |
| **Extreme Rainfall** | 95th percentile exceedance (binary) | **97.5%** | **0.9744** |

---

## Table of Contents

- [Architecture](#architecture)
- [Models](#models)
- [Cities Covered](#cities-covered)
- [Data Pipeline](#data-pipeline)
- [PySpark ETL](#pyspark-etl)
- [API](#api)
- [Setup](#setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)

---

## Architecture

```
Open-Meteo Archive API
        │
        ▼
┌──────────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Data Pipeline   │────▶│   Supabase   │────▶│  XGBoost Models  │
│ (GitHub Actions) │     │ (PostgreSQL) │     │  3 classifiers   │
│  cron @ 2 AM IST │     │  weather_data│     │                  │
│  yearly chunks   │     │  download_   │     │  cyclone         │
│  5s/chunk delay  │     │  batches     │     │  heatwave        │
│  20s/city delay  │     │              │     │  extreme_rainfall│
└──────────────────┘     └──────────────┘     └──────────────────┘
        │                                              │
        ▼                                              ▼
┌──────────────────┐                          ┌──────────────────┐
│  PySpark ETL     │                          │  FastAPI Server  │
│  Window functions│                          │  /predict/*      │
│  Parquet output  │                          │  /monitor/drift  │
└──────────────────┘                          │  /health         │
                                              └──────────────────┘
```

### Rate Limit Protection

Open-Meteo enforces a **5,000 API calls per hour** limit. Each variable-day-location combination counts as one call. The pipeline avoids hitting this with three safeguards:

| Measure | Value | Why |
|---|---|---|
| **Yearly chunking** | 16 single-year chunks per city | A single 16-year request for 15 variables costs ~430 calls; yearly chunks cost ~39 each. Under the 5,000/hr limit for all 15 cities. |
| **Chunk delay** | 5 seconds between chunks | Prevents burst throttling within a city's download. |
| **City delay** | 20 seconds between cities | Spreads load evenly across the full city list. |
| **Retry backoff** | Exponential: 10s, 20s, 40s | If a chunk fails, waits longer before retrying. |
| **Total call estimate** | ~240 calls per full run | Well within the 5,000/hour budget. |

All delays are configurable in [`configs/config.yaml`](#configuration).

---

## Models

Three XGBoost classifiers trained exclusively on real data from Supabase:

### 1. Cyclone Intensity (Multi-class)

Predicts Saffir-Simpson category (0–5) from atmospheric features.

| Feature | Source |
|---|---|
| `lat`, `lon` | Coordinates |
| `wind_max` | Max wind speed |
| `pressure_min` | Min pressure |
| `wind_gust` | Wind gust speed |
| `pressure_trend` | Pressure change |
| `wind_trend` | Wind speed change |
| `year`, `month` | Temporal features |
| `lat_abs` | Absolute latitude |

### 2. Heatwave Prediction (Binary)

Flags whether a day qualifies as a heatwave based on temperature thresholds.

**Labeling rule:** A day is flagged as a heatwave when the mean temperature exceeds the 90th percentile of the trailing 30-day window.

### 3. Extreme Rainfall (Binary)

Flags extreme precipitation events.

**Labeling rule:** A day is flagged as extreme when precipitation exceeds the 95th percentile of the trailing 30-day window.

> **Training requires real data in Supabase.** Run the data pipeline first (see [Data Pipeline](#data-pipeline)).

---

## Cities Covered

15 cities spanning India's climate zones:

| City | State | Zone | Coordinates |
|---|---|---|---|
| Mumbai | Maharashtra | Coastal | 19.08°N, 72.88°E |
| Chennai | Tamil Nadu | Coastal | 13.08°N, 80.27°E |
| Kolkata | West Bengal | Coastal | 22.57°N, 88.36°E |
| Delhi | Delhi | Inland | 28.70°N, 77.10°E |
| Ahmedabad | Gujarat | Arid | 23.02°N, 72.57°E |
| Hyderabad | Telangana | Inland | 17.39°N, 78.49°E |
| Bengaluru | Karnataka | Inland | 12.97°N, 77.59°E |
| Kochi | Kerala | Coastal | 9.93°N, 76.27°E |
| Bhubaneswar | Odisha | Coastal | 20.30°N, 85.82°E |
| Jaipur | Rajasthan | Arid | 26.91°N, 75.79°E |
| Lucknow | Uttar Pradesh | Inland | 26.85°N, 80.95°E |
| Guwahati | Assam | Humid | 26.14°N, 91.74°E |
| Pune | Maharashtra | Inland | 18.52°N, 73.86°E |
| Visakhapatnam | Andhra Pradesh | Coastal | 17.69°N, 83.22°E |
| Surat | Gujarat | Coastal | 21.17°N, 72.83°E |

---

## Data Pipeline

The pipeline downloads 15 weather variables from Open-Meteo's archive API for each city from 2010-01-01 to yesterday:

`temperature_2m_max`, `temperature_2m_min`, `temperature_2m_mean`, `precipitation_sum`, `rain_sum`, `snowfall_sum`, `precipitation_hours`, `wind_speed_10m_max`, `wind_gusts_10m_max`, `wind_direction_10m_dominant`, `pressure_msl_mean`, `relative_humidity_2m_mean`, `cloud_cover_mean`, `shortwave_radiation_sum`, `et0_fao_evapotranspiration`

### Flow

1. **Download** — Open-Meteo archive API, yearly chunks with pacing delays
2. **Preprocess** — Label extreme events (heatwave/rainfall/cyclone thresholds), build feature columns
3. **Upload** — Upsert to Supabase `weather_data` table with batch tracking
4. **CSV backup** — Individual city files saved as GitHub Actions artifacts (7-day retention)

### Schedule

| Trigger | Time | Mechanism |
|---|---|---|
| **Daily cron** | 2:00 AM IST (20:30 UTC) | GitHub Actions scheduled workflow |
| **Manual** | Any time | GitHub → Actions → Data Pipeline → Run workflow |

### Supabase Schema

Two tables in the `public` schema:

**`weather_data`** — One row per city per day with all 15 weather variables, coordinates, extreme event flags, and batch metadata. Unique constraint on `(city, time)` for safe re-runs.

**`download_batches`** — Tracks each pipeline run: start time, completion time, cities processed, rows ingested, status, and error messages.

---

## PySpark ETL

An **Apache PySpark** ETL layer sits alongside the pandas pipeline, demonstrating distributed data processing. It reads the same city CSVs and produces partitioned Parquet output with identical feature engineering:

```
CSVs (14 cities, 84K rows)
        │
        ▼
┌───────────────────────────────┐
│  PySpark ETL (spark_etl.py)  │
│  1. Read CSVs + rename cols  │
│  2. Seasonal sin/cos encode  │
│  3. Heatwave streak (Window) │
│  4. Percentile thresholds    │
│  5. Lag features (1, 3, 7)   │
│  6. Rolling mean/std (3, 7)  │
│  7. Write partitioned Parquet│
└───────────────────────────────┘
        │
        ▼
weather_pyspark.parquet/ (partitioned by city)
```

### Requirements

- **JDK 21** (Spark 4.x is incompatible with JDK 24+)
- PySpark (included in `requirements.txt`)

### Running

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
python -m stormwatch.data.spark_etl
# Output: data/processed/weather_pyspark.parquet/
```

### Training on PySpark Output

```python
import pandas as pd
from stormwatch.models.train import train_heatwave_model, train_rainfall_model

df = pd.read_parquet("data/processed/weather_pyspark.parquet")
hw_model = train_heatwave_model(df, use_hyperopt=False)
rf_model = train_rainfall_model(df, use_hyperopt=False)
```

---

## API

FastAPI server with interactive docs at `/docs`.

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Health check + loaded models |
| GET | `/models` | List loaded models |
| POST | `/predict/cyclone` | Cyclone intensity (category 0–5) |
| POST | `/predict/heatwave` | Heatwave probability + severity |
| POST | `/predict/rainfall` | Extreme rainfall probability |
| POST | `/monitor/drift` | Data drift check |

### Prediction Response Format

```json
{
  "model": "cyclone_intensity",
  "prediction": {
    "category": 3,
    "description": "Category 3 (111-129 mph)",
    "probabilities": {
      "0": 0.02, "1": 0.08, "2": 0.15,
      "3": 0.55, "4": 0.18, "5": 0.02
    },
    "wind_kts": 90.0,
    "confidence": 0.55
  }
}
```

---

## Setup

### Prerequisites

- Python 3.13+
- A [Supabase](https://supabase.com) project (free tier works)
- JDK 21 (for PySpark ETL — optional)
- Node.js 18+ (for PDF report generation — optional)
- GitHub account (for Actions cron)

### Local Setup

```bash
# Clone and enter
git clone https://github.com/stephenbaraik/stormwatch-ai.git
cd stormwatch-ai

# Create virtualenv and install
make setup

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Supabase URL and service role key
```

### Supabase Setup

1. Create a project at [supabase.com](https://supabase.com)
2. Go to **SQL Editor**, paste and run [`stormwatch/database/schema.sql`](stormwatch/database/schema.sql)
3. Get your credentials from **Project Settings → API**:
   - `SUPABASE_URL` — Project URL (e.g. `https://xxxx.supabase.co`)
   - `SUPABASE_SERVICE_KEY` — `service_role` secret (not the anon key)

### GitHub Secrets

For the cron pipeline to upload to Supabase, set these in your repo:
**Settings → Secrets and variables → Actions → New repository secret**

- `SUPABASE_URL` — Your Supabase project URL
- `SUPABASE_SERVICE_KEY` — Your service role key

---

## Usage

### Run the Data Pipeline

Downloads weather data, preprocesses it, and uploads to Supabase:

```bash
# Full pipeline (download → preprocess → supabase → CSV)
make pipeline

# Or with explicit options
python -m stormwatch.data.pipeline \
  --start-date 2010-01-01 \
  --output-dir data/raw

# Skip Supabase upload (CSV only)
python -m stormwatch.data.pipeline --no-upload
```

Manual trigger from GitHub:
1. Go to your repo → **Actions** → **Data Pipeline**
2. Click **Run workflow**
3. (Optional) Set a custom start date or check "Full refresh"

### Train Models

Requires data in Supabase first:

```bash
make train
```

Or step by step:

```bash
# Train all three models
python -m stormwatch.models.train

# Train individual models
python -c "from stormwatch.models.train import train_all; train_all()"
```

### Run the API

```bash
make api
# Or manually:
uvicorn stormwatch.api.server:app --reload --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

### Monitor Drift

```bash
make monitor
```

### Generate Report Figures

```bash
python scripts/generate_figures.py
# Output: docs/figures/ (11 PNG visualizations)
```

### Generate PDF Report

Converts the markdown report to a styled PDF with embedded figures:

```bash
npm install          # one-time: install marked + puppeteer
node scripts/convert_to_pdf.mjs
# Output: docs/end_to_end_report.pdf
```

---

## Project Structure

```
stormwatch-ai/
├── .github/workflows/       # CI/CD
│   ├── ci.yml               # Lint → Test → Docker build
│   └── data-pipeline.yml    # Daily cron @ 2 AM IST
├── configs/
│   └── config.yaml          # Runtime configuration
├── data/
│   ├── raw/                 # Downloaded CSV files (gitignored)
│   └── processed/           # PySpark Parquet output (gitignored)
├── docs/
│   ├── end_to_end_report.md # Full ML report (1000+ lines)
│   └── figures/             # 11 generated visualizations
├── models/                  # Trained model pickles (gitignored)
├── scripts/
│   ├── convert_to_pdf.mjs   # Markdown → PDF (Puppeteer)
│   ├── generate_figures.py  # Report figure generation
│   └── check_supabase.py    # Supabase connectivity check
├── stormwatch/              # Main package
│   ├── api/                 # FastAPI server + request/response schemas
│   │   ├── schemas.py
│   │   └── server.py
│   ├── data/                # Data pipeline
│   │   ├── download.py      # Open-Meteo + IBTrACS downloaders
│   │   ├── pipeline.py      # Orchestrator (download → preprocess → upload)
│   │   ├── preprocess.py    # Extreme event labeling + feature engineering
│   │   └── spark_etl.py     # PySpark ETL: Window features + Parquet
│   ├── database/            # Supabase client + schema
│   │   ├── schema.sql
│   │   └── supabase_client.py
│   ├── features/            # Feature engineering
│   │   └── builder.py
│   ├── models/              # XGBoost model definitions + training
│   │   ├── base.py
│   │   ├── cyclone.py
│   │   ├── heatwave.py
│   │   ├── rainfall.py
│   │   └── train.py
│   ├── monitor/             # Data drift monitoring
│   │   └── drift.py
│   ├── config.py            # Pydantic-typed config loader
│   └── logger.py            # Logging setup
├── tests/                   # Test suite (80 tests)
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_config.py
│   ├── test_models.py
│   └── test_monitor.py
├── .env.example             # Environment template
├── .gitignore
├── docker-compose.yml
├── Dockerfile
├── Makefile                 # Common commands
├── package.json             # Node deps (marked, puppeteer)
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## Configuration

All tunable parameters in [`configs/config.yaml`](configs/config.yaml):

```yaml
data:
  openmeteo:
    start_date: "2010-01-01"
    timezone: "Asia/Kolkata"
    retry_attempts: 3
    retry_delay_seconds: 10
    city_delay_seconds: 20      # ⬅ Pacing between cities
    chunk_delay_seconds: 5.0    # ⬅ Pacing between yearly chunks

models:
  cyclone_intensity:
    type: "multiclass"
    target: "category"
    hyperopt_evals: 50
  heatwave_prediction:
    type: "binary"
    target: "heatwave_occurred"
    hyperopt_evals: 50
  extreme_rainfall:
    type: "binary"
    target: "extreme_rainfall"
    hyperopt_evals: 50

training:
  mlflow_tracking_uri: "sqlite:///mlflow/mlflow.db"
  experiment_name: "stormwatch-ai"
  cv_folds: 5
```

Environment variables override config values using the pattern:
```bash
STORMWATCH__DATA__OPENMETEO__CHUNK_DELAY_SECONDS=10
```

---

## Testing

```bash
# Run all tests
make test

# Or manually
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=stormwatch
```

The test suite uses mock data fixtures (no external API calls). 80+ tests covering:
- Data preprocessing and extreme event labeling
- Model training and prediction interfaces
- API endpoints (request/response serialization)
- Config loading and validation
- Drift monitoring logic

---

## Deployment

### Docker

```bash
# Build and run the full stack
make docker-up

# Or manually
docker compose up -d
```

### GitHub Actions (Data Pipeline — already configured)

The pipeline runs daily at 2:00 AM IST. Monitor runs at:
- GitHub → Actions → Data Pipeline

### API Deploy (Render / Railway / Fly)

The FastAPI server (`stormwatch.api.server:app`) can be deployed to any platform. Ensure:
- Environment variables `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` are set
- Trained model pickles are available in the `models/` directory

---

## Monitoring

The drift detection module uses the **Kolmogorov-Smirnov two-sample test** (`scipy.stats.ks_2samp`) to compare recent predictions against a reference window. Alerts fire when ≥1/3 of features show statistically significant drift (p < 0.05).

```bash
# Run drift check
python -m stormwatch.monitor.drift

# Via API
curl -X POST "http://localhost:8000/monitor/drift?model_name=cyclone"
```

All predictions are logged to SQLite (`mlflow/monitor.db`) for drift analysis.

---

## License

MIT
