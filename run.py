# backend/run.py
"""
Production-safe runner for FastAPI.
- Binds to 0.0.0.0 so Render can route traffic.
- Uses reload only outside production.
- Logs DB driver based on DATABASE_URL.
"""
import os
import sys
import logging
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
log = logging.getLogger("youtube_trans_downloader")


def _db_driver_from_env() -> str:
    url = (os.getenv("DATABASE_URL") or "").lower()
    if url.startswith("postgres://") or url.startswith("postgresql://"):
        return "postgres"
    if url.startswith("sqlite://"):
        return "sqlite"
    return "unknown"


def _local_ip_hint() -> str:
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _check_files():
    required = ["main.py", "models.py"]
    missing = [p for p in required if not (ROOT / p).exists()]
    for p in required:
        if (ROOT / p).exists():
            log.info("✅ Found %s", p)
    if missing:
        log.error("❌ Missing required files: %s", missing)
        sys.exit(1)


def main():
    env = os.getenv("ENVIRONMENT", "production").lower()
    port = int(os.getenv("PORT", "8000"))
    reload = env != "production"

    log.info("🟣 Environment: %s", env)
    log.info("🗄️  Database driver (from env): %s", _db_driver_from_env())

    _check_files()

    try:
        from main import app  # noqa: F401
        log.info("✅ Application imported successfully")
    except Exception:
        log.exception("❌ Failed to import FastAPI application")
        sys.exit(1)

    import uvicorn

    local_ip = _local_ip_hint()
    log.info("🌐 Will listen on 0.0.0.0:%d (Render needs this).", port)
    log.info("📱 LAN hint: http://%s:%d", local_ip, port)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
        log_level=os.getenv("UVICORN_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()

#=======================End run Module=========================
