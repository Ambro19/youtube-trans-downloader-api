# from datetime import datetime
# import os
# from typing import Optional

# from sqlalchemy import (
#     create_engine, Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
# )
# from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

# # --- SQLAlchemy base/engine/session -----------------------------------------
# DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")

# engine = create_engine(
#     DATABASE_URL,
#     connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
#     echo=False,
# )
# SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
# Base = declarative_base()

# # --- Models ------------------------------------------------------------------
# class User(Base):
#     __tablename__ = "users"
#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String(100), unique=True, index=True, nullable=False)
#     email = Column(String(255), unique=True, index=True, nullable=False)
#     hashed_password = Column(String(255), nullable=False)

#     created_at = Column(DateTime, default=datetime.utcnow)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

#     is_active = Column(Boolean, default=True)
#     is_verified = Column(Boolean, default=False)

#     # subscription + usage fields expected by app
#     subscription_tier = Column(String(50), default="free")   # free | pro | premium
#     stripe_customer_id = Column(String(255), nullable=True)
#     # add missing field so /cancel_subscription works reliably
#     stripe_subscription_id = Column(String(255), nullable=True)

#     usage_clean_transcripts = Column(Integer, default=0)
#     usage_unclean_transcripts = Column(Integer, default=0)
#     usage_audio_downloads = Column(Integer, default=0)
#     usage_video_downloads = Column(Integer, default=0)
#     usage_reset_date = Column(DateTime, default=datetime.utcnow)

#     subscriptions = relationship("Subscription", back_populates="user")
#     downloads = relationship("TranscriptDownload", back_populates="user")

#     def __repr__(self):
#         return f"<User id={self.id} username={self.username!r} tier={self.subscription_tier!r}>"


# class Subscription(Base):
#     __tablename__ = "subscriptions"
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
#     tier = Column(String(50), nullable=False, default="free")
#     status = Column(String(50), default="active")

#     created_at = Column(DateTime, default=datetime.utcnow)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
#     expires_at = Column(DateTime, nullable=True)
#     cancelled_at = Column(DateTime, nullable=True)

#     stripe_subscription_id = Column(String(255), nullable=True)
#     stripe_payment_intent_id = Column(String(255), nullable=True)
#     stripe_customer_id = Column(String(255), nullable=True)

#     price_paid = Column(Float, nullable=True)
#     currency = Column(String(10), default="usd")
#     extra_data = Column(Text, nullable=True)  # JSON string

#     user = relationship("User", back_populates="subscriptions")


# class TranscriptDownload(Base):
#     __tablename__ = "transcript_downloads"
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
#     youtube_id = Column(String(20), index=True, nullable=False)

#     # clean_transcripts | unclean_transcripts | audio_downloads | video_downloads
#     transcript_type = Column(String(50), nullable=False)
#     quality = Column(String(20), nullable=True)
#     file_format = Column(String(10), nullable=True)
#     file_size = Column(Integer, nullable=True)

#     processing_time = Column(Float, nullable=True)
#     status = Column(String(20), default="completed")
#     error_message = Column(Text, nullable=True)

#     language = Column(String(10), default="en")
#     video_title = Column(String(500), nullable=True)
#     video_uploader = Column(String(255), nullable=True)
#     video_duration = Column(Integer, nullable=True)

#     created_at = Column(DateTime, default=datetime.utcnow, index=True)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

#     user = relationship("User", back_populates="downloads")


# # --- Helpers used by main.py -------------------------------------------------
# def get_db() -> Session:
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


# def _ensure_runtime_columns():
#     """
#     Lightweight 'migration': add missing columns on SQLite for dev machines.
#     In production you should use Alembic; this keeps local dev from breaking.
#     """
#     if not DATABASE_URL.startswith("sqlite"):
#         return
#     from sqlalchemy import inspect, text

#     insp = inspect(engine)
#     cols = {c["name"] for c in insp.get_columns("users")}
#     needed = set()

#     if "stripe_subscription_id" not in cols:
#         needed.add(("stripe_subscription_id", "TEXT"))
#     # add more columns here if you evolve the schema

#     if not needed:
#         return

#     with engine.connect() as conn:
#         for name, sqltype in needed:
#             try:
#                 conn.execute(text(f'ALTER TABLE users ADD COLUMN "{name}" {sqltype}'))
#             except Exception:
#                 pass
#         conn.commit()


# def initialize_database() -> bool:
#     try:
#         Base.metadata.create_all(bind=engine)
#         _ensure_runtime_columns()
#         return True
#     except Exception:
#         return False


# def create_download_record_safe(
#     db: Session, user_id: int, download_type: str, youtube_id: str, **kw
# ) -> Optional[TranscriptDownload]:
#     try:
#         rec = TranscriptDownload(
#             user_id=user_id,
#             youtube_id=youtube_id,
#             transcript_type=download_type,
#             quality=kw.get("quality", "default"),
#             file_format=kw.get("file_format", "txt"),
#             file_size=kw.get("file_size", 0),
#             processing_time=kw.get("processing_time", 0.0),
#             video_title=kw.get("video_title"),
#             video_uploader=kw.get("video_uploader"),
#             video_duration=kw.get("video_duration"),
#             status="completed",
#             created_at=datetime.utcnow(),
#         )
#         db.add(rec)
#         db.commit()
#         db.refresh(rec)
#         return rec
#     except Exception:
#         db.rollback()
#         return None


# __all__ = [
#     "User",
#     "Subscription",
#     "TranscriptDownload",
#     "engine",
#     "SessionLocal",
#     "Base",
#     "get_db",
#     "initialize_database",
#     "create_download_record_safe",
# ]

##################################################################
##################################################################

from datetime import datetime
import os
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Boolean, Text, Float,
    ForeignKey, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from sqlalchemy import inspect

# --- SQLAlchemy base/engine/session -----------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# --- Models ------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)

    # subscription + usage fields expected by app
    subscription_tier = Column(String(50), default="free")   # free | pro | premium
    stripe_customer_id = Column(String(255), nullable=True)
    # used by cancellation / portal flows
    stripe_subscription_id = Column(String(255), nullable=True)

    usage_clean_transcripts = Column(Integer, default=0)
    usage_unclean_transcripts = Column(Integer, default=0)
    usage_audio_downloads = Column(Integer, default=0)
    usage_video_downloads = Column(Integer, default=0)
    usage_reset_date = Column(DateTime, default=datetime.utcnow)

    subscriptions = relationship("Subscription", back_populates="user")
    downloads = relationship("TranscriptDownload", back_populates="user")

    def __repr__(self):
        return f"<User id={self.id} username={self.username!r} tier={self.subscription_tier!r}>"


class Subscription(Base):
    __tablename__ = "subscriptions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    tier = Column(String(50), nullable=False, default="free")
    status = Column(String(50), default="active")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)

    stripe_subscription_id = Column(String(255), nullable=True)
    stripe_payment_intent_id = Column(String(255), nullable=True)
    stripe_customer_id = Column(String(255), nullable=True)

    price_paid = Column(Float, nullable=True)
    currency = Column(String(10), default="usd")
    extra_data = Column(Text, nullable=True)  # JSON string

    user = relationship("User", back_populates="subscriptions")


class TranscriptDownload(Base):
    __tablename__ = "transcript_downloads"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    youtube_id = Column(String(20), index=True, nullable=False)

    # clean_transcripts | unclean_transcripts | audio_downloads | video_downloads
    transcript_type = Column(String(50), nullable=False)
    quality = Column(String(20), nullable=True)
    file_format = Column(String(10), nullable=True)
    file_size = Column(Integer, nullable=True)

    processing_time = Column(Float, nullable=True)
    status = Column(String(20), default="completed")
    error_message = Column(Text, nullable=True)

    language = Column(String(10), default="en")
    video_title = Column(String(500), nullable=True)
    video_uploader = Column(String(255), nullable=True)
    video_duration = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="downloads")


# --- Helpers used by main.py -------------------------------------------------
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _ensure_runtime_columns() -> None:
    """
    Lightweight 'migration' for the users table (SQLite dev envs).
    Adds any newly introduced nullable columns if they don't exist.
    """
    if not DATABASE_URL.startswith("sqlite"):
        return

    insp = inspect(engine)
    try:
        user_cols = {c["name"] for c in insp.get_columns("users")}
    except Exception:
        return

    to_add = []
    if "stripe_subscription_id" not in user_cols:
        to_add.append(("users", "stripe_subscription_id", "TEXT"))

    if not to_add:
        return

    with engine.begin() as conn:
        for table, col, sqltype in to_add:
            try:
                conn.execute(text(f'ALTER TABLE {table} ADD COLUMN "{col}" {sqltype}'))
            except Exception:
                # ignore if another process added it already
                pass


def ensure_transcript_downloads_columns() -> None:
    """
    Idempotent column adder for the transcript_downloads table (SQLite dev envs).
    Ensures new optional columns exist so queries don't break when the model evolves.
    """
    if not DATABASE_URL.startswith("sqlite"):
        return

    insp = inspect(engine)
    try:
        cols = {c["name"] for c in insp.get_columns("transcript_downloads")}
    except Exception:
        # Table might not be created yet; create_all() will run before this anyway.
        return

    # Only nullable/optional columns (safe ALTER TABLE in SQLite)
    desired = [
        ("quality", "TEXT"),
        ("file_format", "TEXT"),
        ("file_size", "INTEGER"),
        ("processing_time", "REAL"),
        ("status", "TEXT"),
        ("error_message", "TEXT"),
        ("language", "TEXT"),
        ("video_title", "TEXT"),
        ("video_uploader", "TEXT"),
        ("video_duration", "INTEGER"),
        ("updated_at", "DATETIME"),
    ]

    to_add = [(name, ddl) for (name, ddl) in desired if name not in cols]
    if not to_add:
        return

    with engine.begin() as conn:
        for name, ddl in to_add:
            try:
                conn.execute(text(f'ALTER TABLE transcript_downloads ADD COLUMN "{name}" {ddl}'))
            except Exception:
                # ignore if another process added it already
                pass


def initialize_database() -> bool:
    try:
        Base.metadata.create_all(bind=engine)
        # Ensure dev SQLite has any newer columns we rely on
        _ensure_runtime_columns()
        ensure_transcript_downloads_columns()
        return True
    except Exception:
        return False


def create_download_record_safe(
    db: Session, user_id: int, download_type: str, youtube_id: str, **kw
) -> Optional[TranscriptDownload]:
    try:
        rec = TranscriptDownload(
            user_id=user_id,
            youtube_id=youtube_id,
            transcript_type=download_type,
            quality=kw.get("quality", "default"),
            file_format=kw.get("file_format", "txt"),
            file_size=kw.get("file_size", 0),
            processing_time=kw.get("processing_time", 0.0),
            video_title=kw.get("video_title"),
            video_uploader=kw.get("video_uploader"),
            video_duration=kw.get("video_duration"),
            status=kw.get("status", "completed"),
            error_message=kw.get("error_message"),
            language=kw.get("language", "en"),
            created_at=datetime.utcnow(),
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec
    except Exception:
        db.rollback()
        return None


__all__ = [
    "User",
    "Subscription",
    "TranscriptDownload",
    "engine",
    "SessionLocal",
    "Base",
    "get_db",
    "initialize_database",
    "create_download_record_safe",
]
