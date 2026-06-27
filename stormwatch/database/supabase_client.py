"""Supabase client wrapper for StormWatch AI data persistence.

Usage:
    1. Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env or environment
    2. Run the SQL in db_schema.sql via Supabase SQL editor to create tables
    3. Use upload_weather_batch() to persist downloaded data
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from stormwatch.logger import get_logger

log = get_logger(__name__)

# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────


@dataclass
class SupabaseConfig:
    url: str = ""
    service_key: str = ""

    @classmethod
    def from_env(cls) -> SupabaseConfig:
        return cls(
            url=os.environ.get("SUPABASE_URL", ""),
            service_key=os.environ.get("SUPABASE_SERVICE_KEY", ""),
        )

    @property
    def configured(self) -> bool:
        return bool(self.url and self.service_key)


# ──────────────────────────────────────────────
#  Client
# ──────────────────────────────────────────────


class SupabaseClient:
    """Thin wrapper around the Supabase Python client.

    Lazily connects on first use. All public methods accept and return
    standard Python types (dicts, DataFrames) — no Supabase-specific
    types leak to callers.
    """

    def __init__(self, config: Optional[SupabaseConfig] = None):
        self._config = config or SupabaseConfig.from_env()
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client
        if not self._config.configured:
            raise RuntimeError(
                "Supabase not configured. Set SUPABASE_URL and SUPABASE_SERVICE_KEY."
            )
        from supabase import create_client

        self._client = create_client(self._config.url, self._config.service_key)
        return self._client

    # ── Schema management ──

    def ensure_tables(self) -> bool:
        """Check that required tables exist; schema must be applied manually.

        Run the SQL in stormwatch/database/schema.sql via Supabase SQL Editor
        before running the pipeline for the first time.
        """
        try:
            result = (
                self._get_client()
                .table("download_batches")
                .select("id")
                .limit(1)
                .execute()
            )
            if result.data is not None:
                return True
        except Exception:
            pass
        log.warning(
            "Tables not found. Run stormwatch/database/schema.sql "
            "in your Supabase SQL Editor before running the pipeline."
        )
        return False

    # ── Batch tracking ──

    def create_batch(self) -> int:
        """Insert a new download batch record and return its id."""
        client = self._get_client()
        result = (
            client.table("download_batches")
            .insert({"started_at": datetime.now(timezone.utc).isoformat()})
            .execute()
        )
        if not result.data:
            raise RuntimeError("Failed to create batch record")
        return result.data[0]["id"]

    def complete_batch(self, batch_id: int, rows_ingested: int) -> None:
        """Mark a batch as completed."""
        client = self._get_client()
        client.table("download_batches").update(
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "rows_ingested": rows_ingested,
                "status": "completed",
            }
        ).eq("id", batch_id).execute()

    def fail_batch(self, batch_id: int, error: str) -> None:
        """Mark a batch as failed."""
        client = self._get_client()
        client.table("download_batches").update(
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "status": "failed",
                "error_message": error[:500],
            }
        ).eq("id", batch_id).execute()

    def get_latest_batch(self) -> Optional[dict]:
        """Return the most recent batch record, or None."""
        client = self._get_client()
        result = (
            client.table("download_batches")
            .select("*")
            .order("started_at", desc=True)
            .limit(1)
            .execute()
        )
        return result.data[0] if result.data else None

    # ── Data ingestion ──

    def upload_weather_data(self, df: pd.DataFrame, batch_id: int) -> int:
        """Upsert weather data rows into Supabase.

        Uses the city+time unique constraint so re-runs are safe.
        Returns the number of rows upserted.
        """
        client = self._get_client()
        records = df.to_dict(orient="records")

        # Add batch metadata
        for r in records:
            r["batch_id"] = str(batch_id)

        # Upsert in chunks to avoid payload limits
        chunk_size = 500
        total = 0
        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]
            result = (
                client.table("weather_data")
                .upsert(chunk, ignore_duplicates=False)
                .execute()
            )
            total += len(result.data) if result.data else 0

        log.info("Upserted %d / %d rows to Supabase", total, len(records))
        return total

    def get_ingested_cities(self) -> list[str]:
        """Return list of city names that already have data in Supabase."""
        client = self._get_client()
        result = client.table("weather_data").select("city").execute()
        if not result.data:
            return []
        return list({r["city"] for r in result.data})

    def get_city_date_range(self, city: str) -> tuple[Optional[str], Optional[str]]:
        """Return ``(min_date, max_date)`` for a city's ingested data."""
        client = self._get_client()
        result = (
            client.table("weather_data")
            .select("time")
            .eq("city", city)
            .order("time")
            .execute()
        )
        if not result.data:
            return None, None
        return result.data[0]["time"], result.data[-1]["time"]

    def get_city_max_date(self, city: str) -> Optional[str]:
        """Return the most recent date with data for a city, or None if none."""
        client = self._get_client()
        result = (
            client.table("weather_data")
            .select("time")
            .eq("city", city)
            .order("time", desc=True)
            .limit(1)
            .execute()
        )
        if not result.data:
            return None
        return result.data[0]["time"]

    def get_city_record_count(self, city: str) -> int:
        """Return the number of rows ingested for a city."""
        client = self._get_client()
        result = (
            client.table("weather_data")
            .select("id", count="exact")
            .eq("city", city)
            .limit(0)
            .execute()
        )
        return result.count if hasattr(result, "count") else 0


# ──────────────────────────────────────────────
#  SQL schema (for manual execution in Supabase SQL editor)
# ──────────────────────────────────────────────


_SCHEMA_SQL = """
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
"""
