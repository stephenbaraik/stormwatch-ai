"""Apply StormWatch AI database schema to a Supabase Postgres instance.

Usage:
    # Via direct database URL (recommended for CI / local)
    export SUPABASE_DB_URL="postgresql://postgres:PASSWORD@db.PROJECT_REF.supabase.co:5432/postgres"
    python -m stormwatch.database.migrate

    # Fallback: copy the SQL from schema.sql into the Supabase SQL editor
    # (Settings → SQL Editor in the Supabase dashboard)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from stormwatch.database.supabase_client import SupabaseClient, SupabaseConfig
from stormwatch.logger import get_logger

log = get_logger(__name__)

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def _read_schema() -> str | None:
    if not _SCHEMA_PATH.exists():
        log.error("Schema file not found: %s", _SCHEMA_PATH)
        return None
    return _SCHEMA_PATH.read_text()


def apply_via_db_url(db_url: str) -> bool:
    try:
        import psycopg2
    except ImportError:
        log.error(
            "psycopg2 is required for direct DB migration. "
            "Install it with: pip install psycopg2-binary"
        )
        return False

    sql = _read_schema()
    if not sql:
        return False

    log.info("Connecting to database via SUPABASE_DB_URL...")
    try:
        conn = psycopg2.connect(db_url, connect_timeout=15)
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        log.info("Schema applied successfully.")
        conn.close()
        return True
    except Exception as exc:
        log.error("Failed to apply schema via DB URL: %s", exc)
        return False


def apply_via_exec_sql(client: SupabaseClient) -> bool:
    sql = _read_schema()
    if not sql:
        return False
    try:
        client._get_client().rpc("exec_sql", {"query": sql}).execute()
        log.info("Schema applied via exec_sql RPC.")
        return True
    except Exception as exc:
        err_msg = str(exc)
        if "PGRST202" in err_msg:
            log.info("exec_sql function not available — skipping RPC method.")
        else:
            log.warning("exec_sql RPC failed: %s", exc)
        return False


def apply_via_pooler(client: SupabaseClient) -> bool:
    import psycopg2

    project_ref = client._config.url.replace("https://", "").split(".")[0]
    service_key = client._config.service_key
    if not project_ref or not service_key:
        return False

    sql = _read_schema()
    if not sql:
        return False

    # Try connection pooler with JWT auth (service_role key as password)
    pooler_host = "aws-0-ap-south-1.pooler.supabase.com"
    try:
        conn = psycopg2.connect(
            host=pooler_host,
            port=6543,
            dbname="postgres",
            user=f"postgres.{project_ref}",
            password=service_key,
            connect_timeout=10,
        )
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        log.info("Schema applied via connection pooler (JWT auth).")
        conn.close()
        return True
    except Exception:
        return False


def _tables_exist(client: SupabaseClient) -> bool:
    """Check if the required DB tables already exist via the REST API."""
    try:
        result = (
            client._get_client()
            .table("download_batches")
            .select("id")
            .limit(1)
            .execute()
        )
        if result.data is not None:
            return True
    except Exception:
        pass
    return False


def migrate() -> bool:
    """Run database migrations. Tries multiple methods in order.

    Returns True if schema was applied successfully, False otherwise.
    """
    config = SupabaseConfig.from_env()

    if config.configured:
        if _tables_exist(SupabaseClient(config)):
            return True

    db_url = os.environ.get("SUPABASE_DB_URL")
    if db_url:
        return apply_via_db_url(db_url)

    if not config.configured:
        log.warning("Supabase not configured — skipping migration.")
        return False

    client = SupabaseClient(config)

    if apply_via_exec_sql(client):
        return True

    try:
        if apply_via_pooler(client):
            return True
    except ImportError:
        pass

    # Nothing worked
    log.warning(
        "\n===== Manual Schema Setup Required =====\n"
        "Could not auto-apply the database schema.\n\n"
        "Option 1: Set SUPABASE_DB_URL and re-run:\n"
        '  export SUPABASE_DB_URL="postgresql://postgres:PASS@db.PROJECT_REF.supabase.co:5432/postgres"\n'
        "  python -m stormwatch.database.migrate\n\n"
        "Option 2: Run the SQL manually:\n"
        "  1. Go to your Supabase dashboard → SQL Editor\n"
        "  2. Paste and run the contents of:\n"
        "     stormwatch/database/schema.sql\n"
        "=========================================="
    )
    return False


if __name__ == "__main__":
    success = migrate()
    sys.exit(0 if success else 1)
