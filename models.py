# models.py - COMPLETE DATABASE MODELS with Subscription and TranscriptDownload
# 🔥 FIXES:
# - ✅ Added missing Subscription model
# - ✅ Added missing TranscriptDownload model for history tracking
# - ✅ Updated User model with subscription fields
# - ✅ Complete database schema for all functionality

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy import create_engine
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create declarative base
Base = declarative_base()

# =============================================================================
# USER MODEL
# =============================================================================

class User(Base):
    __tablename__ = "users"
    
    # Primary fields
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Account status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # 🔥 Subscription fields
    subscription_tier = Column(String(50), default='free')  # 'free', 'pro', 'premium'
    stripe_customer_id = Column(String(255), nullable=True)
    
    # 🔥 Usage tracking fields (monthly)
    usage_clean_transcripts = Column(Integer, default=0)
    usage_unclean_transcripts = Column(Integer, default=0)
    usage_audio_downloads = Column(Integer, default=0)
    usage_video_downloads = Column(Integer, default=0)
    usage_reset_date = Column(DateTime, default=datetime.utcnow)
    
    # Additional fields
    full_name = Column(String(255), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    
    # 🔥 Relationships
    subscriptions = relationship("Subscription", back_populates="user")
    downloads = relationship("TranscriptDownload", back_populates="user")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}', tier='{self.subscription_tier}')>"

# =============================================================================
# SUBSCRIPTION MODEL
# =============================================================================

class Subscription(Base):
    __tablename__ = "subscriptions"
    
    # Primary fields
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Subscription details
    tier = Column(String(50), nullable=False)  # 'free', 'pro', 'premium'
    status = Column(String(50), default='active')  # 'active', 'cancelled', 'expired', 'pending'
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    
    # Payment details
    stripe_subscription_id = Column(String(255), nullable=True)
    stripe_payment_intent_id = Column(String(255), nullable=True)
    stripe_customer_id = Column(String(255), nullable=True)
    
    # Pricing
    price_paid = Column(Float, nullable=True)
    currency = Column(String(10), default='usd')
    
    # Additional data (renamed from metadata to avoid SQLAlchemy reserved word)
    extra_data = Column(Text, nullable=True)  # JSON string for additional data
    
    # 🔥 Relationships
    user = relationship("User", back_populates="subscriptions")
    
    def __repr__(self):
        return f"<Subscription(id={self.id}, user_id={self.user_id}, tier='{self.tier}', status='{self.status}')>"

# =============================================================================
# TRANSCRIPT DOWNLOAD MODEL (for history tracking)
# =============================================================================

class TranscriptDownload(Base):
    __tablename__ = "transcript_downloads"
    
    # Primary fields
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
    # Download details
    youtube_id = Column(String(20), nullable=False, index=True)  # YouTube video ID
    transcript_type = Column(String(50), nullable=False)  # 'clean_transcripts', 'unclean_transcripts', 'audio_downloads', 'video_downloads'
    
    # File details
    quality = Column(String(20), nullable=True)  # 'high', 'medium', 'low' for audio; '1080p', '720p', etc. for video
    file_format = Column(String(10), nullable=True)  # 'txt', 'srt', 'vtt', 'mp3', 'mp4'
    file_size = Column(Integer, nullable=True)  # Size in bytes
    
    # Processing details
    processing_time = Column(Float, nullable=True)  # Time taken in seconds
    status = Column(String(20), default='completed')  # 'completed', 'failed', 'processing'
    error_message = Column(Text, nullable=True)
    
    # Metadata
    language = Column(String(10), default='en')
    video_title = Column(String(500), nullable=True)
    video_uploader = Column(String(255), nullable=True)
    video_duration = Column(Integer, nullable=True)  # Duration in seconds
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 🔥 Relationships
    user = relationship("User", back_populates="downloads")
    
    def __repr__(self):
        return f"<TranscriptDownload(id={self.id}, user_id={self.user_id}, youtube_id='{self.youtube_id}', type='{self.transcript_type}')>"

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================

# Database URL - using SQLite for development
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
    echo=False  # Set to True for SQL query logging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# =============================================================================
# DATABASE UTILITY FUNCTIONS
# =============================================================================

def get_db() -> Session:
    """
    Dependency to get database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def initialize_database():
    """
    Initialize database and create all tables
    """
    try:
        logger.info("📊 Initializing database...")
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        
        logger.info("✅ All database tables created successfully")
        logger.info("✅ Database initialization completed")
        
        # Log table information
        tables = Base.metadata.tables.keys()
        logger.info(f"📋 Created tables: {', '.join(tables)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        return False

def create_download_record_safe(db: Session, user_id: int, download_type: str, youtube_id: str, **kwargs):
    """
    Safely create a download record with error handling
    """
    try:
        download_record = TranscriptDownload(
            user_id=user_id,
            youtube_id=youtube_id,
            transcript_type=download_type,
            quality=kwargs.get('quality', 'default'),
            file_format=kwargs.get('file_format', 'txt'),
            file_size=kwargs.get('file_size', 0),
            processing_time=kwargs.get('processing_time', 0),
            video_title=kwargs.get('video_title', None),
            video_uploader=kwargs.get('video_uploader', None),
            video_duration=kwargs.get('video_duration', None),
            status='completed',
            created_at=datetime.utcnow()
        )
        
        db.add(download_record)
        db.commit()
        db.refresh(download_record)
        
        logger.info(f"✅ Download record created: {download_type} for video {youtube_id} by user {user_id}")
        return download_record
        
    except Exception as e:
        logger.error(f"❌ Error creating download record: {e}")
        db.rollback()
        return None

def get_user_stats(db: Session, user_id: int) -> dict:
    """
    Get user statistics for downloads
    """
    try:
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            return {}
        
        # Get download counts
        total_downloads = db.query(TranscriptDownload).filter(TranscriptDownload.user_id == user_id).count()
        
        # Get downloads by type
        transcript_downloads = db.query(TranscriptDownload).filter(
            TranscriptDownload.user_id == user_id,
            TranscriptDownload.transcript_type.in_(['clean_transcripts', 'unclean_transcripts'])
        ).count()
        
        audio_downloads = db.query(TranscriptDownload).filter(
            TranscriptDownload.user_id == user_id,
            TranscriptDownload.transcript_type == 'audio_downloads'
        ).count()
        
        video_downloads = db.query(TranscriptDownload).filter(
            TranscriptDownload.user_id == user_id,
            TranscriptDownload.transcript_type == 'video_downloads'
        ).count()
        
        return {
            'total_downloads': total_downloads,
            'transcript_downloads': transcript_downloads,
            'audio_downloads': audio_downloads,
            'video_downloads': video_downloads,
            'subscription_tier': user.subscription_tier,
            'member_since': user.created_at,
            'usage': {
                'clean_transcripts': user.usage_clean_transcripts or 0,
                'unclean_transcripts': user.usage_unclean_transcripts or 0,
                'audio_downloads': user.usage_audio_downloads or 0,
                'video_downloads': user.usage_video_downloads or 0
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting user stats: {e}")
        return {}

# =============================================================================
# DATABASE MIGRATION HELPERS
# =============================================================================

def add_missing_columns():
    """
    Add missing columns to existing tables (for migrations)
    """
    try:
        from sqlalchemy import text
        
        with engine.connect() as conn:
            # Check if subscription_tier exists in users table
            try:
                result = conn.execute(text("PRAGMA table_info(users)"))
                columns = [row[1] for row in result.fetchall()]
                
                if 'subscription_tier' not in columns:
                    logger.info("Adding missing subscription columns to users table...")
                    conn.execute(text("ALTER TABLE users ADD COLUMN subscription_tier VARCHAR(50) DEFAULT 'free'"))
                    conn.execute(text("ALTER TABLE users ADD COLUMN stripe_customer_id VARCHAR(255)"))
                    conn.execute(text("ALTER TABLE users ADD COLUMN usage_clean_transcripts INTEGER DEFAULT 0"))
                    conn.execute(text("ALTER TABLE users ADD COLUMN usage_unclean_transcripts INTEGER DEFAULT 0"))
                    conn.execute(text("ALTER TABLE users ADD COLUMN usage_audio_downloads INTEGER DEFAULT 0"))
                    conn.execute(text("ALTER TABLE users ADD COLUMN usage_video_downloads INTEGER DEFAULT 0"))
                    conn.execute(text("ALTER TABLE users ADD COLUMN usage_reset_date DATETIME"))
                    conn.commit()
                    logger.info("✅ Missing columns added successfully")
                    
            except Exception as e:
                logger.warning(f"Column migration warning: {e}")
                
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")

# =============================================================================
# INITIALIZE ON IMPORT
# =============================================================================

if __name__ == "__main__":
    # Initialize database when run directly
    initialize_database()
    add_missing_columns()
    logger.info("🔥 Database models initialized successfully")
else:
    # Add missing columns when imported
    try:
        add_missing_columns()
    except Exception as e:
        logger.warning(f"Migration warning on import: {e}")

# Export commonly used items
__all__ = [
    'User', 
    'Subscription', 
    'TranscriptDownload', 
    'get_db', 
    'initialize_database', 
    'create_download_record_safe',
    'get_user_stats',
    'engine',
    'SessionLocal',
    'Base'
]
