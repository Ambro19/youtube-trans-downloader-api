# backend/alembic/env.py
from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, create_engine, pool

# Import your models so Alembic can autogenerate from Base.metadata
# IMPORTANT: path must match your project structure
import models  # <- has Base = declarative_base()

# This is the Alembic Config object, which provides access to the values within the .ini file.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set the target metadata for 'autogenerate' support
target_metadata = models.Base.metadata

# ---- Resolve DB URL from env (fallback to a sane default) --------------------
def get_url() -> str:
    # Single source of truth – prefer env var
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    # Fallback (SQLite file in backend/)
    return "sqlite:///./youtube_trans_downloader.db"

# ---- Offline mode ------------------------------------------------------------
def run_migrations_offline() -> None:
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=url.startswith("sqlite"),  # needed for SQLite DDL changes
    )
    with context.begin_transaction():
        context.run_migrations()

# ---- Online mode -------------------------------------------------------------
def run_migrations_online() -> None:
    url = get_url()

    # Option A: set the URL on the Alembic config then use engine_from_config
    config.set_main_option("sqlalchemy.url", url)

    if url.startswith("sqlite"):
        # SQLite needs special handling (no connection pooling, batch mode)
        connectable = create_engine(url, poolclass=pool.NullPool)
        with connectable.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
                render_as_batch=True,
            )
            with context.begin_transaction():
                context.run_migrations()
    else:
        # Non-SQLite (Postgres/MySQL/etc.)
        connectable = engine_from_config(
            config.get_section(config.config_ini_section),
            prefix="sqlalchemy.",
            poolclass=pool.NullPool,
        )
        with connectable.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                compare_type=True,
            )
            with context.begin_transaction():
                context.run_migrations()

# Entrypoint
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
