# backend/models.py
import os
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_downloader.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Subscription fields
    subscription_tier = Column(String, default="free")  # free, pro, premium
    stripe_customer_id = Column(String, nullable=True, unique=True)
    
    # Usage tracking fields
    usage_clean_transcripts = Column(Integer, default=0)
    usage_unclean_transcripts = Column(Integer, default=0)
    usage_audio_downloads = Column(Integer, default=0)
    usage_video_downloads = Column(Integer, default=0)
    usage_reset_date = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to subscriptions
    subscriptions = relationship("Subscription", back_populates="user")
    transcript_downloads = relationship("TranscriptDownload", back_populates="user")

class Subscription(Base):
    __tablename__ = "subscriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    tier = Column(String, nullable=False)  # free, pro, premium
    status = Column(String, default="active")  # active, cancelled, expired, paused
    stripe_subscription_id = Column(String, nullable=True, unique=True)
    stripe_customer_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    cancelled_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    extra_data = Column(Text, nullable=True)  # JSON string for additional metadata
    
    # Relationship back to user
    user = relationship("User", back_populates="subscriptions")

class TranscriptDownload(Base):
    __tablename__ = "transcript_downloads"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    youtube_id = Column(String, nullable=False, index=True)
    transcript_type = Column(String, nullable=False)  # clean_transcripts, unclean_transcripts, audio_downloads, video_downloads
    quality = Column(String, nullable=True)  # For audio/video quality settings
    file_format = Column(String, nullable=True)  # txt, srt, vtt, mp3, mp4, etc.
    file_size = Column(Integer, nullable=True)  # File size in bytes
    file_path = Column(String, nullable=True)  # Local file path (if stored)
    processing_time = Column(Float, nullable=True)  # Processing time in seconds
    status = Column(String, default="completed")  # completed, failed, processing
    language = Column(String, default="en")  # Language code
    created_at = Column(DateTime, default=datetime.utcnow)
    error_message = Column(Text, nullable=True)  # Error details if failed
    
    # Relationship back to user
    user = relationship("User", back_populates="transcript_downloads")

def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def initialize_database():
    """Create database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
        raise