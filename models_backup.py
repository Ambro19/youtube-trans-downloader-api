"""
Enhanced Models with consistent database naming
==============================================

This file contains all database models with proper schema definitions
for tracking transcripts, audio, and video downloads.

FIXED: Uses the correct database name (youtube_trans_downloader.db)
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
from typing import Dict, Any
import os
import logging

logger = logging.getLogger("youtube_trans_downloader")

# Database Configuration - FIXED: Use your existing database name
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# =============================================================================
# USER MODEL
# =============================================================================

class User(Base):
    """Enhanced User model with subscription and usage tracking"""
    __tablename__ = "users"
    
    # Basic user info
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Subscription info
    subscription_tier = Column(String(20), default="free")  # free, pro, premium
    stripe_customer_id = Column(String(100), nullable=True)
    stripe_subscription_id = Column(String(100), nullable=True)
    subscription_status = Column(String(20), default="inactive")  # active, inactive, canceled
    subscription_start_date = Column(DateTime, nullable=True)
    subscription_end_date = Column(DateTime, nullable=True)
    
    # Usage tracking (reset monthly)
    usage_clean_transcripts = Column(Integer, default=0)
    usage_unclean_transcripts = Column(Integer, default=0)
    usage_audio_downloads = Column(Integer, default=0)
    usage_video_downloads = Column(Integer, default=0)
    usage_reset_date = Column(DateTime, default=datetime.utcnow)
    
    # Additional fields
    is_active = Column(Boolean, default=True)
    last_login = Column(DateTime, nullable=True)
    total_downloads = Column(Integer, default=0)

    def get_plan_limits(self) -> Dict[str, Any]:
        """Get usage limits based on subscription tier"""
        limits = {
            "free": {
                "clean_transcripts": 5,
                "unclean_transcripts": 3,
                "audio_downloads": 2,
                "video_downloads": 1
            },
            "pro": {
                "clean_transcripts": 100,
                "unclean_transcripts": 50,
                "audio_downloads": 50,
                "video_downloads": 20
            },
            "premium": {
                "clean_transcripts": float('inf'),
                "unclean_transcripts": float('inf'),
                "audio_downloads": float('inf'),
                "video_downloads": float('inf')
            }
        }
        return limits.get(self.subscription_tier, limits["free"])

    def increment_usage(self, usage_type: str):
        """Increment usage counter for a specific type"""
        current_usage = getattr(self, f"usage_{usage_type}", 0)
        setattr(self, f"usage_{usage_type}", current_usage + 1)
        self.total_downloads += 1

    def reset_monthly_usage(self):
        """Reset monthly usage counters"""
        self.usage_clean_transcripts = 0
        self.usage_unclean_transcripts = 0
        self.usage_audio_downloads = 0
        self.usage_video_downloads = 0
        self.usage_reset_date = datetime.utcnow()

    def can_download(self, download_type: str) -> bool:
        """Check if user can perform a specific download type"""
        limits = self.get_plan_limits()
        current_usage = getattr(self, f"usage_{download_type}", 0)
        limit = limits.get(download_type, 0)
        
        if limit == float('inf'):
            return True
        return current_usage < limit

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', tier='{self.subscription_tier}')>"

# =============================================================================
# DOWNLOAD TRACKING MODEL
# =============================================================================

class TranscriptDownload(Base):
    """Enhanced download tracking for all content types"""
    __tablename__ = "transcript_downloads"
    
    # Basic tracking
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    youtube_id = Column(String(20), nullable=False, index=True)
    transcript_type = Column(String(20), nullable=False)  # clean, unclean, audio, video
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # File details (these columns may not exist in old databases - handle gracefully)
    file_size = Column(Integer, nullable=True)  # File size in bytes
    processing_time = Column(Float, nullable=True)  # Processing time in seconds
    download_method = Column(String(20), nullable=True)  # api, ytdlp, etc.
    quality = Column(String(20), nullable=True)  # high, medium, low, 720p, etc.
    language = Column(String(10), default="en")
    file_format = Column(String(10), nullable=True)  # mp3, mp4, txt, srt, vtt
    
    # Download URLs and expiration
    download_url = Column(Text, nullable=True)  # URL for file download
    expires_at = Column(DateTime, nullable=True)  # When download URL expires
    
    # Status tracking
    status = Column(String(20), default="completed")  # pending, completed, failed, expired
    error_message = Column(Text, nullable=True)  # Error details if failed
    
    # Additional metadata
    video_title = Column(String(200), nullable=True)
    video_duration = Column(Integer, nullable=True)  # Duration in seconds
    ip_address = Column(String(45), nullable=True)  # User IP for analytics
    user_agent = Column(String(500), nullable=True)  # Browser info

    def __repr__(self):
        return f"<TranscriptDownload(id={self.id}, user_id={self.user_id}, youtube_id='{self.youtube_id}', type='{self.transcript_type}')>"

# =============================================================================
# PAYMENT TRACKING MODEL
# =============================================================================

class PaymentRecord(Base):
    """Track payment transactions"""
    __tablename__ = "payment_records"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    stripe_payment_intent_id = Column(String(100), unique=True, nullable=False)
    amount = Column(Float, nullable=False)  # Amount in USD
    currency = Column(String(3), default="usd")
    plan_type = Column(String(20), nullable=False)  # pro, premium
    status = Column(String(20), nullable=False)  # pending, succeeded, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Stripe details
    stripe_customer_id = Column(String(100), nullable=True)
    payment_method = Column(String(50), nullable=True)  # card, etc.
    
    def __repr__(self):
        return f"<PaymentRecord(id={self.id}, user_id={self.user_id}, amount=${self.amount}, status='{self.status}')>"

# =============================================================================
# SYSTEM ANALYTICS MODEL
# =============================================================================

class SystemAnalytics(Base):
    """Track system usage analytics"""
    __tablename__ = "system_analytics"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Daily counters
    total_users = Column(Integer, default=0)
    new_users = Column(Integer, default=0)
    active_users = Column(Integer, default=0)
    
    # Download counters
    transcript_downloads = Column(Integer, default=0)
    audio_downloads = Column(Integer, default=0)
    video_downloads = Column(Integer, default=0)
    
    # Subscription counters
    free_users = Column(Integer, default=0)
    pro_users = Column(Integer, default=0)
    premium_users = Column(Integer, default=0)
    
    # Revenue
    daily_revenue = Column(Float, default=0.0)
    
    def __repr__(self):
        return f"<SystemAnalytics(date={self.date.strftime('%Y-%m-%d')}, users={self.active_users})>"

# =============================================================================
# DATABASE UTILITY FUNCTIONS
# =============================================================================

def get_db():
    """Database session dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables(engine_instance=None):
    """Create all database tables"""
    if engine_instance is None:
        engine_instance = engine
    
    try:
        Base.metadata.create_all(bind=engine_instance)
        logger.info("✅ All database tables created successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating database tables: {e}")
        return False

def drop_all_tables(engine_instance=None):
    """Drop all database tables (use with caution!)"""
    if engine_instance is None:
        engine_instance = engine
    
    try:
        Base.metadata.drop_all(bind=engine_instance)
        logger.info("✅ All database tables dropped successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error dropping database tables: {e}")
        return False

def reset_database(engine_instance=None):
    """Reset database by dropping and recreating all tables"""
    if engine_instance is None:
        engine_instance = engine
    
    logger.info("🔄 Resetting database...")
    
    if drop_all_tables(engine_instance):
        if create_tables(engine_instance):
            logger.info("✅ Database reset completed successfully")
            return True
    
    logger.error("❌ Database reset failed")
    return False

def upgrade_database_schema():
    """Upgrade existing database to new schema (migration)"""
    try:
        # This will add new columns if they don't exist
        # SQLite will ignore columns that already exist
        with engine.connect() as conn:
            # Add new columns to transcript_downloads table
            try:
                conn.execute("ALTER TABLE transcript_downloads ADD COLUMN file_size INTEGER")
            except:
                pass  # Column already exists
            
            try:
                conn.execute("ALTER TABLE transcript_downloads ADD COLUMN processing_time REAL")
            except:
                pass
            
            try:
                conn.execute("ALTER TABLE transcript_downloads ADD COLUMN download_method VARCHAR(20)")
            except:
                pass
            
            try:
                conn.execute("ALTER TABLE transcript_downloads ADD COLUMN quality VARCHAR(20)")
            except:
                pass
            
            try:
                conn.execute("ALTER TABLE transcript_downloads ADD COLUMN language VARCHAR(10) DEFAULT 'en'")
            except:
                pass
            
            try:
                conn.execute("ALTER TABLE transcript_downloads ADD COLUMN file_format VARCHAR(10)")
            except:
                pass
            
            try:
                conn.execute("ALTER TABLE transcript_downloads ADD COLUMN download_url TEXT")
            except:
                pass
            
            try:
                conn.execute("ALTER TABLE transcript_downloads ADD COLUMN expires_at DATETIME")
            except:
                pass
            
            try:
                conn.execute("ALTER TABLE transcript_downloads ADD COLUMN status VARCHAR(20) DEFAULT 'completed'")
            except:
                pass
            
            # Add new columns to users table
            try:
                conn.execute("ALTER TABLE users ADD COLUMN usage_audio_downloads INTEGER DEFAULT 0")
            except:
                pass
            
            try:
                conn.execute("ALTER TABLE users ADD COLUMN usage_video_downloads INTEGER DEFAULT 0")
            except:
                pass
            
            try:
                conn.execute("ALTER TABLE users ADD COLUMN total_downloads INTEGER DEFAULT 0")
            except:
                pass
            
            conn.commit()
        
        logger.info("✅ Database schema upgraded successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error upgrading database schema: {e}")
        return False

# =============================================================================
# SAFE DOWNLOAD RECORD CREATION
# =============================================================================

def create_download_record_safe(db, user_id, youtube_id, transcript_type, **kwargs):
    """
    Safely create a download record, handling missing columns gracefully
    """
    try:
        # Create basic record with required fields only
        download_record = TranscriptDownload(
            user_id=user_id,
            youtube_id=youtube_id,
            transcript_type=transcript_type,
            created_at=datetime.utcnow()
        )
        
        # Add optional fields only if they exist in the database schema
        optional_fields = {
            'file_size': kwargs.get('file_size'),
            'processing_time': kwargs.get('processing_time'),
            'download_method': kwargs.get('download_method'),
            'quality': kwargs.get('quality'),
            'language': kwargs.get('language', 'en'),
            'file_format': kwargs.get('file_format'),
            'download_url': kwargs.get('download_url'),
            'expires_at': kwargs.get('expires_at'),
            'status': kwargs.get('status', 'completed'),
            'error_message': kwargs.get('error_message'),
            'video_title': kwargs.get('video_title'),
            'video_duration': kwargs.get('video_duration'),
            'ip_address': kwargs.get('ip_address'),
            'user_agent': kwargs.get('user_agent')
        }
        
        # Try to set each optional field
        for field_name, field_value in optional_fields.items():
            if field_value is not None:
                try:
                    setattr(download_record, field_name, field_value)
                except AttributeError:
                    # Column doesn't exist in database yet
                    logger.debug(f"Column {field_name} not available, skipping")
                    pass
        
        db.add(download_record)
        return download_record
        
    except Exception as e:
        logger.error(f"Error creating download record: {e}")
        return None

# =============================================================================
# INITIALIZATION
# =============================================================================

def initialize_database():
    """Initialize database with proper error handling"""
    try:
        logger.info(f"📊 Initializing database: {DATABASE_URL}")
        
        # Try to upgrade existing schema first
        if upgrade_database_schema():
            logger.info("✅ Database schema upgrade completed")
        
        # Create any missing tables
        if create_tables():
            logger.info("✅ Database initialization completed")
            return True
        else:
            logger.error("❌ Database initialization failed")
            return False
            
    except Exception as e:
        logger.error(f"❌ Critical database error: {e}")
        return False

# Run initialization when module is imported
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_database()
else:
    # Auto-initialize when imported
    initialize_database()
