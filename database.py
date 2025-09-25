# backend/database.py
from datetime import datetime
import os
from typing import Generator
from sqlalchemy import text
from sqlalchemy.orm import Session
from models import (
    Base,
    engine,
    SessionLocal,
    get_db as _models_get_db,
    initialize_database as _models_initialize_database,
)

__all__ = ["Base", "engine", "SessionLocal", "get_db", "initialize_database", "check_database_health", "backup_database"]

def get_db() -> Generator[Session, None, None]:
    yield from _models_get_db()

def initialize_database() -> bool:
    return _models_initialize_database()

def check_database_health():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow().isoformat()}

def backup_database(backup_path: str | None = None) -> bool:
    url = os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")
    if not url.startswith("sqlite"):
        return False
    try:
        import shutil
        db_file = url.replace("sqlite:///", "").replace("./", "")
        if not backup_path:
            stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{db_file}.{stamp}.backup"
        shutil.copy2(db_file, backup_path)
        return True
    except Exception:
        return False

########################## ==== BACKUP FILE ==== ###################

# # backend/database.py — normalized, single source of truth via models.py
# from datetime import datetime
# import os
# from typing import Generator

# from sqlalchemy import text
# from sqlalchemy.orm import Session

# # Re-export from models (single source of truth)
# from models import (
#     Base,
#     engine,
#     SessionLocal,
#     get_db as _models_get_db,
#     initialize_database as _models_initialize_database,
# )

# # Public re-exports so existing imports keep working
# __all__ = [
#     "Base",
#     "engine",
#     "SessionLocal",
#     "get_db",
#     "initialize_database",
#     "check_database_health",
#     "backup_database",
# ]

# def get_db() -> Generator[Session, None, None]:
#     # passthrough (keeps old imports working)
#     yield from _models_get_db()

# def initialize_database() -> bool:
#     # single call site for table create + light runtime checks
#     return _models_initialize_database()

# def check_database_health():
#     """Ping DB and return a tiny health payload."""
#     try:
#         with engine.connect() as conn:
#             conn.execute(text("SELECT 1"))
#         return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
#     except Exception as e:
#         return {"status": "unhealthy", "error": str(e), "timestamp": datetime.utcnow().isoformat()}

# def backup_database(backup_path: str | None = None) -> bool:
#     """SQLite-only convenience backup."""
#     url = os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")
#     if not url.startswith("sqlite"):
#         return False
#     try:
#         import shutil
#         db_file = url.replace("sqlite:///", "").replace("./", "")
#         if not backup_path:
#             stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
#             backup_path = f"{db_file}.{stamp}.backup"
#         shutil.copy2(db_file, backup_path)
#         return True
#     except Exception:
#         return False

