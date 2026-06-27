"""StormWatch AI — Data Pipeline Orchestrator

Downloads real weather data from Open-Meteo (respecting API rate limits),
preprocesses it, and uploads to Supabase. Designed to run as a scheduled
job (GitHub Actions cron, Render Cron, etc.) that increments data overnight.

Typical run (~28 min for 15 cities × 16 years):
    .venv/bin/python -m stormwatch.data.pipeline

Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env or environment variables
to enable cloud persistence. Without them the script only saves to CSV.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from stormwatch.config import get_config
from stormwatch.data.download import INDIAN_CITIES, download_openmeteo_historical
from stormwatch.data.preprocess import label_extreme_events, prepare_weather_features
from stormwatch.logger import get_logger

log = get_logger(__name__)


def download_city_weather(
    city: dict,
    start_date: str,
    end_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch historical weather for one city from Open-Meteo.

    Args:
        city: City dict with 'name', 'lat', 'lon'
        start_date: YYYY-MM-DD start
        end_date: Optional YYYY-MM-DD end (defaults to today)

    Returns:
        DataFrame with weather data + city metadata, or None.
    """
    config = get_config()
    retries = config.data.openmeteo.retry_attempts

    for attempt in range(retries):
        df = download_openmeteo_historical(
            lat=city["lat"],
            lon=city["lon"],
            start_date=start_date,
            end_date=end_date,
        )
        if df is not None:
            df["city"] = city["name"]
            df["state"] = city["state"]
            df["zone"] = city["zone"]
            return df

        delay = config.data.openmeteo.retry_delay_seconds * (2**attempt)
        log.warning("Retry %d/%d for %s after %ds", attempt + 1, retries, city["name"], delay)
        time.sleep(delay)

    log.warning("Skipping %s after %d failed attempts", city["name"], retries)
    return None


def run_ingest_batch(
    start_date: str = "2010-01-01",
    cities: Optional[list[dict]] = None,
    upload: bool = True,
    save_csv: bool = True,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Download one batch of weather data for all cities.

    Respects Open-Meteo rate limits via yearly chunking and configurable
    delays between cities and chunks (set in config.yaml).

    Args:
        start_date: YYYY-MM-DD start (default 2010-01-01 for full archive)
        cities: List of city dicts (defaults to INDIAN_CITIES)
        upload: Upload to Supabase (requires credentials)
        save_csv: Save individual city CSVs to output_dir
        output_dir: Directory for CSV output (default from config)

    Returns:
        Combined DataFrame of all successfully downloaded data.
    """
    config = get_config()
    output_dir = output_dir or config.data.raw_path
    cities = cities or INDIAN_CITIES
    os.makedirs(output_dir, exist_ok=True)

    # ── Optional Supabase client ──
    supabase = None
    if upload:
        try:
            from stormwatch.database.supabase_client import SupabaseClient

            supabase = SupabaseClient()
            if not supabase._config.configured:
                log.warning("Supabase not configured — skipping upload")
                supabase = None
        except Exception as exc:
            log.warning("Failed to init Supabase client: %s — skipping upload", exc)
            supabase = None

    batch_id: Optional[int] = None
    if supabase:
        try:
            batch_id = supabase.create_batch()
            log.info("Created batch id=%d", batch_id)
        except Exception as exc:
            log.warning("Failed to create batch record: %s", exc)

    # ── Download loop ──
    all_dfs: list[pd.DataFrame] = []
    total_rows = 0

    for i, city in enumerate(cities):
        log.info("[%d/%d] Downloading %s...", i + 1, len(cities), city["name"])
        df = download_city_weather(city, start_date=start_date)

        if df is None:
            continue

        # Preprocess: label extreme events + build feature columns
        try:
            df = label_extreme_events(df)
            df = prepare_weather_features(df)
        except Exception as exc:
            log.warning("Preprocessing failed for %s: %s", city["name"], exc)

        # Save locally
        if save_csv:
            city_path = Path(output_dir) / f"weather_{city['name'].lower()}.csv"
            df.to_csv(city_path, index=False)
            log.info("Saved %s (%d rows)", city_path.name, len(df))

        # Upload to Supabase
        if supabase and batch_id is not None:
            try:
                n = supabase.upload_weather_data(df, batch_id)
                total_rows += n
            except Exception as exc:
                log.warning("Supabase upload failed for %s: %s", city["name"], exc)

        all_dfs.append(df)

        # Pacing delay between cities (configurable in config.yaml)
        if i < len(cities) - 1:
            time.sleep(config.data.openmeteo.city_delay_seconds)

    # ── Finalize batch ──
    if supabase and batch_id is not None:
        try:
            supabase.complete_batch(batch_id, total_rows)
            log.info("Batch %d completed: %d rows ingested", batch_id, total_rows)
        except Exception as exc:
            try:
                supabase.fail_batch(batch_id, str(exc))
            except Exception:
                pass
            log.warning("Failed to finalize batch: %s", exc)

    if not all_dfs:
        log.error("No data downloaded for any city")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    log.info(
        "Batch complete: %d cities, %d total rows",
        len(all_dfs),
        len(combined),
    )
    return combined


def main() -> None:
    """CLI entry point for the data pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="StormWatch AI — Data Pipeline (download → preprocess → upload)"
    )
    parser.add_argument(
        "--start-date",
        default="2010-01-01",
        help="Start date YYYY-MM-DD (default: 2010-01-01)",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Skip Supabase upload (local CSV only)",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip saving local CSV files",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for CSV files (default: data/raw)",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("StormWatch AI — Data Pipeline")
    log.info("=" * 60)
    log.info("Start date: %s", args.start_date)
    log.info("Upload to Supabase: %s", not args.no_upload)
    log.info("Save CSV: %s", not args.no_csv)
    log.info("=" * 60)

    start = datetime.now(timezone.utc)
    try:
        df = run_ingest_batch(
            start_date=args.start_date,
            upload=not args.no_upload,
            save_csv=not args.no_csv,
            output_dir=args.output_dir,
        )
        elapsed = (datetime.now(timezone.utc) - start).total_seconds()
        if not df.empty:
            log.info(
                "Pipeline complete: %d rows from %d cities in %.1f minutes",
                len(df),
                df["city"].nunique(),
                elapsed / 60,
            )
        else:
            log.warning("Pipeline finished but no data was collected")
    except Exception as exc:
        log.error("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
