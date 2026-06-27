"""CLI tool to check data pipeline health.

Usage:
    python -m stormwatch.monitor.pipeline_status

Shows:
- Last 5 pipeline runs (batch tracking)
- Per-city data coverage
- Whether each city is up to date (within 2 days)
- Any batches that failed
"""

from __future__ import annotations

from datetime import datetime, timezone

from stormwatch.database.supabase_client import SupabaseClient
from stormwatch.logger import get_logger

log = get_logger(__name__)


def _fmt(dt_str: str | None) -> str:
    if not dt_str:
        return "-"
    return dt_str[:19] if "T" in dt_str else dt_str[:10]


def show_pipeline_status():
    """Query Supabase and print a human-readable pipeline health report."""
    try:
        client = SupabaseClient()
        _ = client._get_client()
    except Exception as exc:
        log.error("Cannot connect to Supabase: %s", exc)
        log.info("Set SUPABASE_URL and SUPABASE_SERVICE_KEY in your environment.")
        return

    print("=" * 70)
    print("  StormWatch AI — Pipeline Health")
    print("=" * 70)

    # ── Last 5 batches ──
    try:
        batches = (
            client._get_client()
            .table("download_batches")
            .select("*")
            .order("started_at", desc=True)
            .limit(5)
            .execute()
        )
    except Exception:
        batches = type("obj", (object,), {"data": []})()

    print(f"\n  Last {len(batches.data)} pipeline runs:")
    if not batches.data:
        print("    (no runs yet)")
    for b in batches.data:
        status = b["status"]
        icon = "✅" if status == "completed" else "❌" if status == "failed" else "⏳"
        dur = ""
        if b.get("started_at") and b.get("completed_at"):
            try:
                s = datetime.fromisoformat(b["started_at"])
                e = datetime.fromisoformat(b["completed_at"])
                dur = f" ({(e - s).total_seconds() / 60:.1f} min)"
            except Exception:
                pass
        print(
            f"    {icon}  Batch #{b['id']}: {status}  "
            f"| {b['rows_ingested']} rows{dur}"
        )
        if b.get("error_message"):
            print(f"       Error: {b['error_message'][:200]}")

    # ── Per-city coverage ──
    print(f"\n  City coverage:")
    try:
        cities_data = (
            client._get_client()
            .table("weather_data")
            .select("city")
            .execute()
        )
        city_names = sorted(set(r["city"] for r in cities_data.data))
    except Exception:
        city_names = []

    if not city_names:
        print("    (no data ingested yet)")
    else:
        print(f"    {'City':<20} {'Rows':>7} {'Earliest':<14} {'Latest':<14} {'Status'}")
        print(f"    {'─'*20} {'─'*7} {'─'*14} {'─'*14} {'─'*10}")
        now = datetime.now(timezone.utc)
        for city in city_names:
            min_d, max_d = client.get_city_date_range(city)
            cnt = client.get_city_record_count(city)
            status = "🟢 OK" if max_d else "⚪ empty"
            if max_d:
                try:
                    max_dt = datetime.fromisoformat(max_d.replace("Z", "+00:00"))
                    days_old = (now - max_dt).days
                    if days_old > 7:
                        status = "🔴 stale"
                    elif days_old > 2:
                        status = "🟡 aging"
                except Exception:
                    pass
            print(f"    {city:<20} {cnt:>7} {str(min_d)[:10] if min_d else '-':<14} {str(max_d)[:10] if max_d else '-':<14} {status}")

    total_rows = sum(client.get_city_record_count(c) for c in city_names) if city_names else 0
    today_rows = 0
    try:
        result = (
            client._get_client()
            .table("weather_data")
            .select("id", count="exact")
            .gte("ingested_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT00:00:00Z"))
            .limit(0)
            .execute()
        )
        today_rows = result.count if hasattr(result, "count") else 0
    except Exception:
        pass

    print(f"\n  Totals:  {total_rows} rows across {len(city_names)} cities")
    print(f"  Today:   {today_rows} rows ingested")
    print("=" * 70)


if __name__ == "__main__":
    show_pipeline_status()
