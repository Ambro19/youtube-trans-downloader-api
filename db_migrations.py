# # backend/db_migrations.py
# from sqlalchemy import text
# from sqlalchemy.engine import Engine

from __future__ import annotations
import logging
from sqlalchemy.engine import Engine

log = logging.getLogger("youtube_trans_downloader.migrations")

def _sqlite_has_column(conn, table: str, column: str) -> bool:
    rows = conn.exec_driver_sql(f"PRAGMA table_info('{table}')").fetchall()
    # PRAGMA table_info: (cid, name, type, notnull, dflt_value, pk)
    return any(r[1] == column for r in rows)

def _create_index(conn, name: str, table: str, cols: list[str]) -> None:
    cols_sql = ", ".join(cols)
    conn.exec_driver_sql(f"CREATE INDEX IF NOT EXISTS {name} ON {table} ({cols_sql})")

def run_startup_migrations(engine: Engine) -> None:
    """
    Idempotent migrations & pragmas for SQLite.
    Safe to run at every startup.
    """
    with engine.begin() as conn:
        # SQLite safety/health
        conn.exec_driver_sql("PRAGMA foreign_keys = ON;")
        conn.exec_driver_sql("PRAGMA journal_mode = WAL;")

        # --- Add subscriptions.stripe_customer_id (missing in your DB) ---
        if not _sqlite_has_column(conn, "subscriptions", "stripe_customer_id"):
            log.info("Adding subscriptions.stripe_customer_id …")
            conn.exec_driver_sql("ALTER TABLE subscriptions ADD COLUMN stripe_customer_id TEXT")

            # Backfill from users.stripe_customer_id when available
            if _sqlite_has_column(conn, "users", "stripe_customer_id"):
                conn.exec_driver_sql("""
                    UPDATE subscriptions
                    SET stripe_customer_id = (
                        SELECT u.stripe_customer_id
                        FROM users AS u
                        WHERE u.id = subscriptions.user_id
                    )
                """)

        # Ensure related column exists as well (idempotent)
        if not _sqlite_has_column(conn, "subscriptions", "stripe_subscription_id"):
            log.info("Adding subscriptions.stripe_subscription_id …")
            conn.exec_driver_sql("ALTER TABLE subscriptions ADD COLUMN stripe_subscription_id TEXT")

        # Helpful idempotent indexes
        _create_index(conn, "idx_subscriptions_user_id", "subscriptions", ["user_id"])
        _create_index(conn, "idx_subscriptions_customer_id", "subscriptions", ["stripe_customer_id"])
        _create_index(conn, "idx_users_username", "users", ["username"])
        _create_index(conn, "idx_users_email", "users", ["email"])

        log.info("✅ DB migrations completed")
