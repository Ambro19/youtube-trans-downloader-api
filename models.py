# backend/models.py
from datetime import datetime
import os
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

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

    # subscription + usage fields expected by main.py
    subscription_tier = Column(String(50), default="free")   # free | pro | premium
    stripe_customer_id = Column(String(255), nullable=True)

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

    transcript_type = Column(String(50), nullable=False)  # clean_transcripts | unclean_transcripts | audio_downloads | video_downloads
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

def initialize_database() -> bool:
    try:
        Base.metadata.create_all(bind=engine)
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
            status="completed",
            created_at=datetime.utcnow(),
        )
        db.add(rec); db.commit(); db.refresh(rec)
        return rec
    except Exception:
        db.rollback()
        return None

__all__ = [
    "User", "Subscription", "TranscriptDownload",
    "engine", "SessionLocal", "Base",
    "get_db", "initialize_database", "create_download_record_safe",
]



