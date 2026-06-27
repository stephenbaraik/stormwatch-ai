"""StormWatch AI — Data Pipeline Orchestrator

Downloads real weather data from Open-Meteo (respecting API rate limits),
preprocesses it, and uploads directly to Supabase. Designed to run as a
scheduled job (GitHub Actions cron, Render Cron, etc.).

Resume logic
------------
Before downloading a city, the pipeline queries Supabase for the most recent
date already stored. If data exists, it only downloads *new* days (the gap
between the stored max date and yesterday). If a chunk is rate-limited
mid-run, the unique ``(city, time)`` constraint means the next run picks up
exactly where it left off — no duplicates, no gaps.

Typical run
-----------
- First run (16 years × 15 cities):  ~28 min
- Daily incremental (1 new day × 15 cities):  ~2 min

Usage
-----
    .venv/bin/python -m stormwatch.data.pipeline
    .venv/bin/python -m stormwatch.data.pipeline --force   # full re-download

Set SUPABASE_URL and SUPABASE_SERVICE_KEY in .env or environment variables
to enable cloud persistence. Without them the script falls back to CSV only.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from stormwatch.config import get_config
from stormwatch.data.download import INDIAN_CITIES, download_openmeteo_historical
from stormwatch.data.preprocess import label_extreme_events, prepare_weather_features
from stormwatch.logger import get_logger

log = get_logger(__name__)


def _get_supabase_client():
    """Try to initialise the Supabase client, returning None if unavailable."""
    try:
        from stormwatch.database.supabase_client import SupabaseClient

        client = SupabaseClient()
        if not client._config.configured:
            log.warning("Supabase not configured — skipping upload (set SUPABASE_URL + SUPABASE_SERVICE_KEY)")
            return None
        return client
    except Exception as exc:
        log.warning("Failed to init Supabase client: %s — skipping upload", exc)
        return None


def _resolve_date_range(
    city: dict,
    start_date: str,
    supabase,
    force: bool,
) -> tuple[str, str]:
    """Determine the date range to download for a city.

    Args:
        city: City dict
        start_date: Fallback start date if no data in Supabase
        supabase: SupabaseClient or None
        force: If True, ignore existing data and re-download full range

    Returns:
        (effective_start, effective_end) — the date range to fetch.
    """
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    if force:
        log.info("  └─ Force mode — downloading full range %s → %s", start_date, yesterday)
        return start_date, yesterday

    if supabase is None:
        log.info("  └─ No Supabase — downloading %s → %s", start_date, yesterday)
        return start_date, yesterday

    # Check what's already in Supabase
    max_date = supabase.get_city_max_date(city["name"])
    if max_date is None:
        log.info("  └─ No existing data — downloading %s → %s", start_date, yesterday)
        return start_date, yesterday

    # Compute next day after max_date
    max_dt = datetime.fromisoformat(max_date) if "T" in max_date else datetime.strptime(max_date, "%Y-%m-%d")
    next_day = (max_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    if next_day >= yesterday:
        log.info("  └─ Already up to date (max=%s) — nothing to download", max_date[:10])
        return "", ""

    count = supabase.get_city_record_count(city["name"])
    log.info(
        "  └─ Resume from %s → %s (%d existing rows, fetching %s onward)",
        start_date, yesterday, count, next_day,
    )
    return next_day, yesterday


def _process_city_data(df: pd.DataFrame, city: dict) -> pd.DataFrame:
    """Add city metadata columns and run preprocessing."""
    df["city"] = city["name"]
    df["state"] = city["state"]
    df["zone"] = city["zone"]

    try:
        df = label_extreme_events(df)
        df = prepare_weather_features(df)
    except Exception as exc:
        log.warning("  └─ Preprocessing failed: %s", exc)

    return df


def run_ingest_batch(
    start_date: str = "2010-01-01",
    cities: Optional[list[dict]] = None,
    upload: bool = True,
    save_csv: bool = True,
    output_dir: Optional[str] = None,
    force: bool = False,
) -> pd.DataFrame:
    """Download weather data for all cities and ingest it into Supabase.

    Each city independently resumes from its last stored date in Supabase.
    If a city is already fully up to date, it is skipped.

    Args:
        start_date: Fallback start YYYY-MM-DD (used when no data exists yet)
        cities: List of city dicts (defaults to INDIAN_CITIES)
        upload: Upload to Supabase (requires credentials)
        save_csv: Save individual city CSVs to ``output_dir``
        output_dir: Directory for CSV output (default from config)
        force: Re-download everything even if already in Supabase

    Returns:
        Combined DataFrame of all newly downloaded data.
    """
    config = get_config()
    output_dir = output_dir or config.data.raw_path
    cities = cities or INDIAN_CITIES
    os.makedirs(output_dir, exist_ok=True)

    # ── Optional Supabase client ──
    supabase = _get_supabase_client() if upload else None

    batch_id: Optional[int] = None
    if supabase:
        try:
            batch_id = supabase.create_batch()
            log.info("Created Supabase batch id=%d", batch_id)
        except Exception as exc:
            log.warning("Failed to create batch record: %s", exc)

    # ── Download loop ──
    all_dfs: list[pd.DataFrame] = []
    total_rows = 0
    skipped_cities = 0
    rate_limited_cities = 0

    for i, city in enumerate(cities):
        log.info("[%d/%d] %s — checking...", i + 1, len(cities), city["name"])

        # Determine what date range needs fetching
        effective_start, effective_end = _resolve_date_range(city, start_date, supabase, force)
        if not effective_start:
            log.info("  └─ Skipping (fully up to date)")
            skipped_cities += 1
            continue

        # Download (includes yearly chunking + client-side cache + retry)
        df = download_openmeteo_historical(
            lat=city["lat"],
            lon=city["lon"],
            start_date=effective_start,
            end_date=effective_end,
        )

        if df is None or df.empty:
            # Check if it was a rate-limit issue by whether Supabase has *any* data
            if supabase and supabase.get_city_record_count(city["name"]) == 0:
                log.warning("  └─ No data and nothing in Supabase — rate-limited? Will retry next run")
                rate_limited_cities += 1
            else:
                log.warning("  └─ No new data to download")
            continue

        # Add metadata + preprocess
        df = _process_city_data(df, city)
        log.info("  └─ Downloaded %d rows (%s → %s)", len(df), effective_start, effective_end)

        # Save CSV backup (before rename to keep original column names)
        if save_csv:
            city_path = Path(output_dir) / f"weather_{city['name'].lower()}.csv"
            df.to_csv(city_path, index=False)
            log.info("     Saved %s", city_path.name)

        # Upload to Supabase immediately (per-city, so partial progress is saved)
        if supabase and batch_id is not None:
            try:
                n = supabase.upload_weather_data(df, batch_id)
                total_rows += n
                log.info("     Uploaded %d rows to Supabase", n)
            except Exception as exc:
                log.warning("     Supabase upload failed: %s", exc)
                rate_limited_cities += 1

        all_dfs.append(df)

        # Pacing delay between cities (configurable in config.yaml)
        if i < len(cities) - 1:
            time.sleep(config.data.openmeteo.city_delay_seconds)

    # ── Finalize batch ──
    if supabase and batch_id is not None and total_rows > 0:
        try:
            supabase.complete_batch(batch_id, total_rows)
            log.info("Batch %d completed: %d rows ingested", batch_id, total_rows)
        except Exception as exc:
            try:
                supabase.fail_batch(batch_id, str(exc))
            except Exception:
                pass
            log.warning("Failed to finalize batch: %s", exc)

    # ── Summary ──
    summary = (
        f"Skipped (up to date): {skipped_cities} | "
        f"New rows: {total_rows} | "
        f"Rate-limited/partial: {rate_limited_cities}"
    )
    log.info("Pipeline summary — %s", summary)

    if not all_dfs:
        if skipped_cities == len(cities):
            log.info("All cities are fully up to date — nothing to do.")
        else:
            log.error("No data downloaded for any city")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    log.info("Cities with new data: %d | Combined rows: %d", len(all_dfs), len(combined))
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
        help="Fallback start date YYYY-MM-DD (default: 2010-01-01). "
             "Only used when a city has no existing data in Supabase.",
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download all cities from --start-date, ignoring Supabase state",
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("StormWatch AI — Data Pipeline")
    log.info("=" * 60)
    log.info("Start date: %s", args.start_date)
    log.info("Upload to Supabase: %s", not args.no_upload)
    log.info("Save CSV: %s", not args.no_csv)
    log.info("Force re-download: %s", args.force)
    log.info("=" * 60)

    start = datetime.now(timezone.utc)
    try:
        df = run_ingest_batch(
            start_date=args.start_date,
            upload=not args.no_upload,
            save_csv=not args.no_csv,
            output_dir=args.output_dir,
            force=args.force,
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
            log.info("Pipeline finished — no new data (%.1f s)", elapsed)
    except Exception as exc:
        log.error("Pipeline failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
