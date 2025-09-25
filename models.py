# Modern SQLAlchemy base + helpful compound index.
# backend/models.py
import os
from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text, ForeignKey, create_engine, Index
from sqlalchemy.orm import sessionmaker, relationship, declarative_base

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")

engine = create_engine(
    DATABASE_URL,
    future=True,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    is_active = Column(Boolean, default=True)

    subscription_tier = Column(String, default="free")
    stripe_customer_id = Column(String, nullable=True, unique=True)

    usage_clean_transcripts = Column(Integer, default=0)
    usage_unclean_transcripts = Column(Integer, default=0)
    usage_audio_downloads = Column(Integer, default=0)
    usage_video_downloads = Column(Integer, default=0)
    usage_reset_date = Column(DateTime, default=datetime.utcnow)

    subscriptions = relationship("Subscription", back_populates="user")
    transcript_downloads = relationship("TranscriptDownload", back_populates="user")

class Subscription(Base):
    __tablename__ = "subscriptions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    tier = Column(String, nullable=False)
    status = Column(String, default="active")
    stripe_subscription_id = Column(String, nullable=True, unique=True)
    stripe_customer_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    cancelled_at = Column(DateTime, nullable=True)
    expires_at = Column(DateTime, nullable=True)
    extra_data = Column(Text, nullable=True)
    user = relationship("User", back_populates="subscriptions")

class TranscriptDownload(Base):
    __tablename__ = "transcript_downloads"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    youtube_id = Column(String, nullable=False, index=True)
    transcript_type = Column(String, nullable=False)
    quality = Column(String, nullable=True)
    file_format = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)
    file_path = Column(String, nullable=True)
    processing_time = Column(Float, nullable=True)
    status = Column(String, default="completed")
    language = Column(String, default="en")
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    error_message = Column(Text, nullable=True)
    user = relationship("User", back_populates="transcript_downloads")

    __table_args__ = (
        Index("ix_transcripts_user_created", "user_id", "created_at"),
    )

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def initialize_database():
    try:
        db_path = DATABASE_URL.replace("sqlite:///", "").replace("./", "")
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            print(f"📁 Created database directory: {db_dir}")
        Base.metadata.create_all(bind=engine)
        print(f"✅ Database tables created successfully at: {db_path}")
        return True
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
        raise

#################################################################

# # backend/models.py - FIXED: Standardized database name
# import os
# from datetime import datetime
# from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float, Text, ForeignKey, create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker, relationship

# # FIXED: Use consistent database name throughout the application
# DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")

# engine = create_engine(
#     DATABASE_URL,
#     connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
# )

# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# class User(Base):
#     __tablename__ = "users"

#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String, unique=True, index=True, nullable=False)
#     email = Column(String, unique=True, index=True, nullable=False)
#     hashed_password = Column(String, nullable=False)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     is_active = Column(Boolean, default=True)
    
#     # Subscription fields
#     subscription_tier = Column(String, default="free")  # free, pro, premium
#     stripe_customer_id = Column(String, nullable=True, unique=True)
    
#     # Usage tracking fields
#     usage_clean_transcripts = Column(Integer, default=0)
#     usage_unclean_transcripts = Column(Integer, default=0)
#     usage_audio_downloads = Column(Integer, default=0)
#     usage_video_downloads = Column(Integer, default=0)
#     usage_reset_date = Column(DateTime, default=datetime.utcnow)
    
#     # Relationship to subscriptions
#     subscriptions = relationship("Subscription", back_populates="user")
#     transcript_downloads = relationship("TranscriptDownload", back_populates="user")

# class Subscription(Base):
#     __tablename__ = "subscriptions"
    
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
#     tier = Column(String, nullable=False)  # free, pro, premium
#     status = Column(String, default="active")  # active, cancelled, expired, paused
#     stripe_subscription_id = Column(String, nullable=True, unique=True)
#     stripe_customer_id = Column(String, nullable=True)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
#     cancelled_at = Column(DateTime, nullable=True)
#     expires_at = Column(DateTime, nullable=True)
#     extra_data = Column(Text, nullable=True)  # JSON string for additional metadata
    
#     # Relationship back to user
#     user = relationship("User", back_populates="subscriptions")

# class TranscriptDownload(Base):
#     __tablename__ = "transcript_downloads"

#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
#     youtube_id = Column(String, nullable=False, index=True)
#     transcript_type = Column(String, nullable=False)  # clean_transcripts, unclean_transcripts, audio_downloads, video_downloads
#     quality = Column(String, nullable=True)  # For audio/video quality settings
#     file_format = Column(String, nullable=True)  # txt, srt, vtt, mp3, mp4, etc.
#     file_size = Column(Integer, nullable=True)  # File size in bytes
#     file_path = Column(String, nullable=True)  # Local file path (if stored)
#     processing_time = Column(Float, nullable=True)  # Processing time in seconds
#     status = Column(String, default="completed")  # completed, failed, processing
#     language = Column(String, default="en")  # Language code
#     created_at = Column(DateTime, default=datetime.utcnow)
#     error_message = Column(Text, nullable=True)  # Error details if failed
    
#     # Relationship back to user
#     user = relationship("User", back_populates="transcript_downloads")

# def get_db():
#     """Dependency to get database session"""
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# def initialize_database():
#     """Create database tables"""
#     try:
#         # Check if database file exists and create directory if needed
#         db_path = DATABASE_URL.replace("sqlite:///", "").replace("./", "")
#         db_dir = os.path.dirname(db_path)
#         if db_dir and not os.path.exists(db_dir):
#             os.makedirs(db_dir)
#             print(f"📁 Created database directory: {db_dir}")
        
#         Base.metadata.create_all(bind=engine)
#         print(f"✅ Database tables created successfully at: {db_path}")
#         return True
#     except Exception as e:
#         print(f"❌ Error creating database tables: {e}")
#         raise