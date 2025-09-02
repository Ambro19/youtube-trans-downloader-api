# backend/alembic/env.py
from __future__ import annotations

import os
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import Engine
from logging.config import fileConfig

# --- Load backend/.env so DATABASE_URL works even from CLI ---
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")
except Exception:
    pass

# --- Alembic config object ---
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# --- Import your models to get target_metadata ---
# Make sure "backend" root is on sys.path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import models  # <-- your backend/models.py
target_metadata = models.Base.metadata


# ---- Resolve DB URL from env (fallback to a sane default) --------------------
def get_url() -> str:
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    # Fallback (SQLite file in backend/)
    return "sqlite:///./youtube_trans_downloader.db"


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    # Inject the URL so alembic.ini doesn't need a hardcoded value
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_url()

    # SQLite needs special connect args; others are fine with defaults.
    url = configuration["sqlalchemy.url"]
    is_sqlite = url.startswith("sqlite")
    connect_args = {"check_same_thread": False} if is_sqlite else {}

    engine: Engine = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
        connect_args=connect_args,
    )

    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
