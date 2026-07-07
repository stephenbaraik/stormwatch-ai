"""Pull all real weather rows from Supabase and write the combined CSV
that stormwatch.data.preprocess.preprocess_all() expects.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from stormwatch.database.supabase_client import SupabaseClient

load_dotenv()

OUT_PATH = Path("data/raw/weather_all_cities.csv")


def main() -> None:
    client = SupabaseClient()._get_client()

    cols = ",".join(SupabaseClient._SUPABASE_WEATHER_COLS - {"batch_id", "ingested_at"})
    page_size = 1000
    rows = []
    start = 0
    while True:
        result = (
            client.table("weather_data")
            .select(cols)
            .range(start, start + page_size - 1)
            .execute()
        )
        batch = result.data
        if not batch:
            break
        rows.extend(batch)
        print(f"fetched {len(rows)} rows...")
        if len(batch) < page_size:
            break
        start += page_size

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print(df.groupby("city")["time"].agg(["count", "min", "max"]))


if __name__ == "__main__":
    main()
