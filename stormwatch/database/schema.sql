-- StormWatch AI — Supabase Schema
-- Run this in the Supabase SQL editor to set up your database.
-- Safe to re-run (uses IF NOT EXISTS / idempotent indexes).

-- Weather data from Open-Meteo archive API
CREATE TABLE IF NOT EXISTS weather_data (
    id BIGSERIAL PRIMARY KEY,
    city TEXT NOT NULL,
    time TIMESTAMPTZ NOT NULL,
    temperature_2m_max DOUBLE PRECISION,
    temperature_2m_min DOUBLE PRECISION,
    temperature_2m_mean DOUBLE PRECISION,
    precipitation_sum DOUBLE PRECISION,
    rain_sum DOUBLE PRECISION,
    snowfall_sum DOUBLE PRECISION,
    precipitation_hours DOUBLE PRECISION,
    wind_speed_10m_max DOUBLE PRECISION,
    wind_gusts_10m_max DOUBLE PRECISION,
    wind_direction_10m_dominant DOUBLE PRECISION,
    pressure_msl_mean DOUBLE PRECISION,
    relative_humidity_2m_mean DOUBLE PRECISION,
    cloud_cover_mean DOUBLE PRECISION,
    shortwave_radiation_sum DOUBLE PRECISION,
    et0_fao_evapotranspiration DOUBLE PRECISION,
    latitude DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    heatwave_flag INTEGER DEFAULT 0,
    extreme_rainfall INTEGER DEFAULT 0,
    cyclonic_flag INTEGER DEFAULT 0,
    batch_id TEXT,
    ingested_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(city, time)
);

CREATE INDEX IF NOT EXISTS idx_weather_city_time ON weather_data(city, time);
CREATE INDEX IF NOT EXISTS idx_weather_ingested ON weather_data(ingested_at);

-- Download batch tracking
CREATE TABLE IF NOT EXISTS download_batches (
    id BIGSERIAL PRIMARY KEY,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    cities_count INTEGER DEFAULT 0,
    rows_ingested INTEGER DEFAULT 0,
    status TEXT DEFAULT 'running',
    error_message TEXT
);
