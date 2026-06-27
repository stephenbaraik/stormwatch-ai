"""Verify Supabase connectivity and table access."""
import os
import sys

import requests

url = os.environ.get("SUPABASE_URL", "")
key = os.environ.get("SUPABASE_SERVICE_KEY", "")

errors = 0

if not url:
    print("ERROR: SUPABASE_URL is not set")
    errors += 1
if not key:
    print("ERROR: SUPABASE_SERVICE_KEY is not set")
    errors += 1

if errors:
    sys.exit(1)

headers = {
    "apikey": key,
    "Authorization": f"Bearer {key}",
    "Accept": "application/json",
}

for table in ("download_batches", "weather_data"):
    r = requests.get(
        f"{url}/rest/v1/{table}",
        params={"select": "id", "limit": 1},
        headers=headers,
    )
    print(f"{table}: HTTP {r.status_code}", end="")
    if r.status_code == 200:
        print(f" OK — {r.json()}")
    else:
        print(f" FAIL — {r.text[:200]}")
        errors += 1

sys.exit(errors)
