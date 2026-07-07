# Session Handoff — StormWatch AI (July 2026, pass 3)

## What's done since last handoff

### Pass 2: Figure regeneration (all stale figures → live computation)
- `scripts/generate_figures.py` rewritten — computes all metrics live from real data/models, no hardcoded values
- `FEATURE_NAMES` updated to match post-leakage-fix `builder.py`
- Zone mapping added from `download.py` for EDA figures
- All 11 figures regenerated with real test-set metrics

### Pass 3: Industry readiness — remove all hardcoded/fake data
- **API schemas** (`schemas.py`): Removed leaked fields (`wind_kts`, same-day `temp_max`/`temp_min`/`precipitation`), added missing lags, fixed Saffir-Simpson labels to match standard convention
- **API server** (`server.py`): Removed `_category_to_wind()` hardcoded lookup, removed `wind_kts` and `expected_precipitation` from responses, added `X-API-Key` auth middleware on prediction/monitoring endpoints
- **Config** (`config.yaml`): `features.cyclone` synced with builder
- **Dependencies**: All pinned to exact versions in `requirements/base.txt`; root `requirements.txt` delegates via `-r`
- **Tests**: All fixtures/tests updated to new schemas; 80/80 pass
- **Documentation**: All curl examples, response samples, drift reports updated. All "stale" figure warnings removed.

### Pass 4: Hyperopt evaluation + pressure_min audit
- **Hyperopt**: Ran 20-trial TPE tuning on all 3 models. Result: defaults already optimal (cyclone: 0.1% worse, heatwave: 0.1% worse, rainfall: +0.013 AUC but −13.1% accuracy). Baseline defaults retained. Findings documented in §7.5.
- **`pressure_min` audit**: −0.80 correlation with category, 95.7% accuracy alone, model without it drops only 2.7%. Verified category derivation uses only `wind_kts`, not pressure. Concluded legitimate physical signal. Documented in §7.6.

### Report version: 1.2.0

## Remaining (future sessions)
- MLflow model registry instead of baked `.pkl` in Docker
- SHAP analysis for feature interpretability
- Severity calibration (Platt scaling)
- Rate limiting on API
- Automated retraining pipeline
- Expand to 100+ cities, additional hazards, forecast mode

## Key files changed this pass
- `stormwatch/api/schemas.py` — schema rewrite
- `stormwatch/api/server.py` — auth, removed hardcoded lookups
- `stormwatch/models/train.py` — hyperopt int cast fix
- `scripts/generate_figures.py` — live computation
- `tests/conftest.py`, `tests/test_api.py`, `tests/test_config.py`, `tests/test_models.py`
- `configs/config.yaml` — feature list sync
- `requirements/base.txt` — pinned versions
- `requirements.txt` — delegates to base
- `.env.example` — added `STORMWATCH_API_KEY`
- `docs/end_to_end_report.md` — v1.2.0
