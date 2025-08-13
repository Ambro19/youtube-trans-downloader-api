# Complete database.py - Enhanced with Audio/Video Support
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_transcript_downloader.db")

# Create engine with appropriate settings
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False},
        echo=False  # Set to True for SQL debugging
    )
else:
    # For PostgreSQL/MySQL in production
    engine = create_engine(DATABASE_URL, echo=False)

# Session configuration
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()

# =============================================================================
# DATABASE MODELS
# =============================================================================

class Subscription(Base):
    """Enhanced subscription tracking table"""
    __tablename__ = "subscriptions"
   
    # Core subscription fields
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    tier = Column(String(20), nullable=False)  # free, pro, premium
    start_date = Column(DateTime, nullable=False)
    expiry_date = Column(DateTime, nullable=False)
    payment_id = Column(String(255), nullable=True)  # Stripe payment intent ID
    auto_renew = Column(Boolean, default=True)
   
    # Enhanced subscription tracking
    stripe_subscription_id = Column(String(255), nullable=True, index=True)
    stripe_price_id = Column(String(255), nullable=True)
    status = Column(String(20), default='active')  # active, cancelled, past_due, etc.
    current_period_start = Column(DateTime, nullable=True)
    current_period_end = Column(DateTime, nullable=True)
    cancel_at_period_end = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Subscription(user_id={self.user_id}, tier='{self.tier}', status='{self.status}')>"

class TranscriptDownload(Base):
    """Enhanced download tracking table with audio/video support"""
    __tablename__ = "transcript_downloads"
   
    # Core download fields
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    youtube_id = Column(String(20), nullable=False, index=True)
    transcript_type = Column(String(20), nullable=False)  # clean, unclean, audio, video
    created_at = Column(DateTime, nullable=False)
   
    # Enhanced download tracking
    file_size = Column(Integer, nullable=True)  # Size in bytes
    processing_time = Column(Integer, nullable=True)  # Time in milliseconds
    download_method = Column(String(50), nullable=True)  # youtube-transcript-api, yt-dlp, etc.
    quality = Column(String(20), nullable=True)  # high, medium, low, 1080p, 720p, etc.
    language = Column(String(10), default='en')
    
    # File information
    file_format = Column(String(10), nullable=True)  # txt, srt, vtt, mp3, mp4
    download_url = Column(String(500), nullable=True)  # Cloud storage URL
    expires_at = Column(DateTime, nullable=True)  # Download link expiration

    def __repr__(self):
        return f"<TranscriptDownload(id={self.id}, user_id={self.user_id}, youtube_id='{self.youtube_id}', type='{self.transcript_type}')>"

class PaymentHistory(Base):
    """Payment history tracking table"""
    __tablename__ = "payment_history"
   
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    stripe_payment_intent_id = Column(String(255), nullable=False, index=True)
    stripe_customer_id = Column(String(255), nullable=False, index=True)
    amount = Column(Integer, nullable=False)  # Amount in cents
    currency = Column(String(3), default='usd')
    status = Column(String(20), nullable=False)  # succeeded, failed, pending
    subscription_tier = Column(String(20), nullable=False)
    created_at = Column(DateTime, nullable=False)
    payment_metadata = Column(Text, nullable=True)  # JSON string for additional data

    def __repr__(self):
        return f"<PaymentHistory(user_id={self.user_id}, amount={self.amount}, status='{self.status}')>"

class DownloadStats(Base):
    """Daily download statistics for analytics"""
    __tablename__ = "download_stats"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=False, index=True)
    
    # Download counts by type
    total_downloads = Column(Integer, default=0)
    transcript_downloads = Column(Integer, default=0)
    audio_downloads = Column(Integer, default=0)
    video_downloads = Column(Integer, default=0)
    
    # Download counts by tier
    free_tier_downloads = Column(Integer, default=0)
    pro_tier_downloads = Column(Integer, default=0)
    premium_tier_downloads = Column(Integer, default=0)
    
    # Popular formats
    txt_downloads = Column(Integer, default=0)
    srt_downloads = Column(Integer, default=0)
    vtt_downloads = Column(Integer, default=0)
    mp3_downloads = Column(Integer, default=0)
    mp4_downloads = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<DownloadStats(date={self.date.strftime('%Y-%m-%d')}, total={self.total_downloads})>"

class UserActivity(Base):
    """User activity tracking for analytics and monitoring"""
    __tablename__ = "user_activity"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    activity_type = Column(String(50), nullable=False)  # login, download, upgrade, etc.
    activity_details = Column(Text, nullable=True)  # JSON string with additional details
    ip_address = Column(String(45), nullable=True)  # IPv4 or IPv6
    user_agent = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)

    def __repr__(self):
        return f"<UserActivity(user_id={self.user_id}, type='{self.activity_type}')>"

class SystemSettings(Base):
    """System-wide settings and configuration"""
    __tablename__ = "system_settings"
    
    id = Column(Integer, primary_key=True, index=True)
    setting_key = Column(String(100), nullable=False, unique=True, index=True)
    setting_value = Column(Text, nullable=True)
    setting_type = Column(String(20), default='string')  # string, integer, boolean, json
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<SystemSettings(key='{self.setting_key}', value='{self.setting_value[:50]}')>"

# =============================================================================
# DATABASE UTILITY FUNCTIONS
# =============================================================================

def get_db():
    """Database dependency for FastAPI"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all database tables with error handling"""
    try:
        # Import User model from models.py to ensure it's included
        from models import User, SubscriptionHistory, Download, UsageAnalytics
        
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
        
        # Initialize default system settings
        _initialize_system_settings()
        
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {str(e)}")
        return False

def _initialize_system_settings():
    """Initialize default system settings"""
    try:
        db = SessionLocal()
        
        default_settings = [
            {
                'setting_key': 'max_file_size_mb',
                'setting_value': '500',
                'setting_type': 'integer',
                'description': 'Maximum file size for downloads in MB'
            },
            {
                'setting_key': 'download_retention_days',
                'setting_value': '7',
                'setting_type': 'integer',
                'description': 'Number of days to keep download files available'
            },
            {
                'setting_key': 'rate_limit_per_hour',
                'setting_value': '100',
                'setting_type': 'integer',
                'description': 'Maximum API requests per hour per user'
            },
            {
                'setting_key': 'maintenance_mode',
                'setting_value': 'false',
                'setting_type': 'boolean',
                'description': 'Enable maintenance mode to disable new downloads'
            },
            {
                'setting_key': 'supported_languages',
                'setting_value': '["en", "es", "fr", "de", "it", "pt", "ru", "ja", "ko", "zh"]',
                'setting_type': 'json',
                'description': 'List of supported transcript languages'
            }
        ]
        
        for setting in default_settings:
            existing = db.query(SystemSettings).filter(
                SystemSettings.setting_key == setting['setting_key']
            ).first()
            
            if not existing:
                db_setting = SystemSettings(**setting)
                db.add(db_setting)
        
        db.commit()
        db.close()
        print("✅ System settings initialized")
        
    except Exception as e:
        print(f"⚠️ Warning: Could not initialize system settings: {str(e)}")

def check_database_health():
    """Check database connection and basic health"""
    try:
        db = SessionLocal()
        
        # Test basic query
        db.execute("SELECT 1")
        
        # Check if core tables exist
        from models import User
        user_count = db.query(User).count()
        
        db.close()
        
        return {
            "status": "healthy",
            "user_count": user_count,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

def migrate_database():
    """Run database migrations for existing installations"""
    try:
        from models import add_audio_video_columns
        
        print("🔄 Running database migrations...")
        
        # Add audio/video columns to existing users table
        success = add_audio_video_columns(engine)
        
        if success:
            print("✅ Database migration completed successfully")
        else:
            print("⚠️ Database migration completed with warnings")
            
        return success
        
    except Exception as e:
        print(f"❌ Database migration failed: {str(e)}")
        return False

def backup_database(backup_path: str = None):
    """Create a backup of the database (SQLite only)"""
    if not DATABASE_URL.startswith("sqlite"):
        print("⚠️ Database backup only supported for SQLite")
        return False
    
    try:
        import shutil
        from pathlib import Path
        
        # Extract database file path from URL
        db_file = DATABASE_URL.replace("sqlite:///", "").replace("./", "")
        
        if not backup_path:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_path = f"backup_{timestamp}_{db_file}"
        
        # Create backup
        shutil.copy2(db_file, backup_path)
        
        print(f"✅ Database backed up to: {backup_path}")
        return True
        
    except Exception as e:
        print(f"❌ Database backup failed: {str(e)}")
        return False

def get_database_stats():
    """Get database statistics for monitoring"""
    try:
        db = SessionLocal()
        
        # Import models
        from models import User
        
        stats = {
            "total_users": db.query(User).count(),
            "active_users": db.query(User).filter(User.is_active == True).count(),
            "total_downloads": db.query(TranscriptDownload).count(),
            "recent_downloads": db.query(TranscriptDownload).filter(
                TranscriptDownload.created_at >= datetime.utcnow().replace(day=1)
            ).count(),
            "subscribed_users": db.query(User).filter(
                User.subscription_tier.in_(['pro', 'premium'])
            ).count()
        }
        
        # Download type breakdown
        download_types = db.query(
            TranscriptDownload.transcript_type,
            db.func.count(TranscriptDownload.id).label('count')
        ).group_by(TranscriptDownload.transcript_type).all()
        
        stats["download_types"] = {dt[0]: dt[1] for dt in download_types}
        
        db.close()
        
        return stats
        
    except Exception as e:
        print(f"❌ Error getting database stats: {str(e)}")
        return {}

# =============================================================================
# TESTING UTILITIES
# =============================================================================

def create_test_user(db: SessionLocal):
    """Create a test user for development"""
    try:
        from models import User
        
        test_user = User(
            username="testuser",
            email="test@example.com",
            hashed_password="$2b$12$test_hash_for_development",
            subscription_tier="pro",
            created_at=datetime.utcnow()
        )
        
        db.add(test_user)
        db.commit()
        db.refresh(test_user)
        
        print(f"✅ Test user created: {test_user.username}")
        return test_user
        
    except Exception as e:
        print(f"❌ Error creating test user: {str(e)}")
        return None

# =============================================================================
# INITIALIZATION
# =============================================================================

if __name__ == "__main__":
    """Initialize database when run directly"""
    print("🔄 Initializing YouTube Transcript Downloader Database...")
    
    success = create_tables()
    
    if success:
        print("🎉 Database initialization completed successfully!")
        
        # Show database stats
        stats = get_database_stats()
        if stats:
            print(f"📊 Database Stats: {stats}")
    else:
        print("❌ Database initialization failed!")
        exit(1)
