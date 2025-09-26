# backend/db_migrations.py
from sqlalchemy import text
from sqlalchemy.engine import Engine

def run_startup_migrations(engine: Engine) -> None:
    """
    Idempotent, SQLite-safe migrations for indexes + pragmas.
    Won't error if an index already exists.
    """
    with engine.begin() as conn:
        # Better concurrency for SQLite
        conn.execute(text("PRAGMA journal_mode=WAL"))

        # Users table hardening (names kept to avoid conflicts)
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_users_username ON users (username)"))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email ON users (email)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_users_stripe_customer_id ON users (stripe_customer_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_users_created_at ON users (created_at)"))

        # If you have a downloads/activity table, uncomment or add accordingly:
        # conn.execute(text("CREATE INDEX IF NOT EXISTS ix_activity_user_created ON activity (user_id, created_at)"))
        # conn.execute(text("CREATE INDEX IF NOT EXISTS ix_downloads_user_created ON downloads (user_id, created_at)"))
