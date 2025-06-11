# database.py - Enhanced with subscription fields

from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")

# Create engine
if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Enhanced User model with subscription fields
class User(Base):
    __tablename__ = "users"

    # Primary key and basic fields
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Subscription fields (these should now exist after migration)
    subscription_tier = Column(String(20), default='free', nullable=False)
    subscription_status = Column(String(20), default='inactive', nullable=False)
    subscription_id = Column(String(255), nullable=True)  # Stripe subscription ID
    subscription_current_period_end = Column(DateTime, nullable=True)
    stripe_customer_id = Column(String(255), nullable=True)

    # Usage tracking fields (reset monthly)
    usage_clean_transcripts = Column(Integer, default=0, nullable=False)
    usage_unclean_transcripts = Column(Integer, default=0, nullable=False)
    usage_audio_downloads = Column(Integer, default=0, nullable=False)
    usage_video_downloads = Column(Integer, default=0, nullable=False)
    usage_reset_date = Column(DateTime, default=datetime.utcnow, nullable=True)

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}', tier='{self.subscription_tier}')>"

    def is_subscription_active(self) -> bool:
        """Check if user has an active subscription"""
        if self.subscription_tier == 'free':
            return True
        
        if not self.subscription_current_period_end:
            return False
            
        return (
            self.subscription_status in ['active', 'trialing'] and
            self.subscription_current_period_end > datetime.utcnow()
        )

    def get_plan_limits(self) -> dict:
        """Get the usage limits for the user's current plan"""
        limits = {
            'free': {
                'clean_transcripts': 5,
                'unclean_transcripts': 3,
                'audio_downloads': 2,
                'video_downloads': 1
            },
            'pro': {
                'clean_transcripts': 100,
                'unclean_transcripts': 50,
                'audio_downloads': 50,
                'video_downloads': 20
            },
            'premium': {
                'clean_transcripts': float('inf'),
                'unclean_transcripts': float('inf'),
                'audio_downloads': float('inf'),
                'video_downloads': float('inf')
            }
        }
        return limits.get(self.subscription_tier, limits['free'])

    def can_perform_action(self, action_type: str) -> bool:
        """Check if user can perform the specified action based on limits"""
        # Reset usage if it's a new month
        if self.usage_reset_date and self.usage_reset_date.month != datetime.utcnow().month:
            self.reset_monthly_usage()

        limits = self.get_plan_limits()
        current_usage = getattr(self, f'usage_{action_type}', 0)
        limit = limits.get(action_type, 0)

        if limit == float('inf'):
            return True

        return current_usage < limit

    def reset_monthly_usage(self):
        """Reset monthly usage counters"""
        self.usage_clean_transcripts = 0
        self.usage_unclean_transcripts = 0
        self.usage_audio_downloads = 0
        self.usage_video_downloads = 0
        self.usage_reset_date = datetime.utcnow()

# Subscription model (keep your existing one if you have it)
class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    tier = Column(String(20), nullable=False)  # 'free', 'pro', 'premium'
    start_date = Column(DateTime, default=datetime.utcnow)
    expiry_date = Column(DateTime, nullable=False)
    payment_id = Column(String(255), nullable=True)  # Stripe subscription ID
    auto_renew = Column(Boolean, default=True)

    def __repr__(self):
        return f"<Subscription(user_id={self.user_id}, tier='{self.tier}', expiry='{self.expiry_date}')>"

# TranscriptDownload model (keep your existing one)
class TranscriptDownload(Base):
    __tablename__ = "transcript_downloads"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    youtube_id = Column(String(255), nullable=False)
    transcript_type = Column(String(20), nullable=False)  # 'clean', 'unclean', 'audio', 'video'
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<TranscriptDownload(user_id={self.user_id}, youtube_id='{self.youtube_id}', type='{self.transcript_type}')>"

# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create tables function
def create_tables():
    """Create all tables in the database"""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        raise

# Initialize database
if __name__ == "__main__":
    create_tables()


#=================================================

# # database.py

# from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, create_engine
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker, relationship
# from datetime import datetime
# import os
# from dotenv import load_dotenv
# import logging

# # Load environment variables
# load_dotenv()

# # Configure logging
# logger = logging.getLogger("youtube_trans_downloader.database")

# # Create SQLAlchemy base
# Base = declarative_base()

# # Define database models
# class User(Base):
#     __tablename__ = "users"
   
#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String, unique=True, index=True)
#     hashed_password = Column(String)
#     email = Column(String, unique=True, index=True)
#     created_at = Column(DateTime, default=datetime.now)
   
#     # Relationships
#     subscriptions = relationship("Subscription", back_populates="user", cascade="all, delete-orphan")
#     downloads = relationship("TranscriptDownload", back_populates="user", cascade="all, delete-orphan")

# class Subscription(Base):
#     __tablename__ = "subscriptions"
   
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
#     tier = Column(String)  # "free", "pro", "premium"  
#     start_date = Column(DateTime, default=datetime.now)
#     expiry_date = Column(DateTime)
#     payment_id = Column(String, nullable=True)
#     auto_renew = Column(Boolean, default=False)
   
#     # Relationship
#     user = relationship("User", back_populates="subscriptions")

# class TranscriptDownload(Base):
#     __tablename__ = "transcript_downloads"
   
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
#     youtube_id = Column(String)
#     transcript_type = Column(String)  # "clean" or "unclean"
#     created_at = Column(DateTime, default=datetime.now)
   
#     # Relationship
#     user = relationship("User", back_populates="downloads")

# # Database configuration
# def get_database_url():
#     """Get database URL from environment variables, with fallback to SQLite"""
#     env = os.getenv("ENV", "development")
    
#     # For production, use PostgreSQL
#     if env == "production":
#         # Construct PostgreSQL connection string
#         db_user = os.getenv("DB_USER")
#         db_password = os.getenv("DB_PASSWORD")
#         db_host = os.getenv("DB_HOST", "localhost")
#         db_port = os.getenv("DB_PORT", "5432")
#         db_name = os.getenv("DB_NAME", "youtube_trans_db")
        
#         if db_user and db_password:
#             return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
#         else:
#             logger.warning("PostgreSQL credentials not found, falling back to SQLite")
#             return os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")
    
#     # For development or testing, use SQLite by default
#     return os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")

# # Get the appropriate database URL
# SQLALCHEMY_DATABASE_URL = get_database_url()
# logger.info(f"Using database: {SQLALCHEMY_DATABASE_URL.split('@')[0].split(':')[0]}://****")

# # Configure engine based on database type
# if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
#     engine = create_engine(
#         SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}  # SQLite specific
#     )
# else:
#     # For PostgreSQL, MySQL, etc.
#     engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_size=5, max_overflow=10)

# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# # Function to get DB session
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # Create all tables if they don't exist
# def create_tables():
#     try:
#         Base.metadata.create_all(bind=engine)
#         logger.info("Database tables created successfully")
#     except Exception as e:
#         logger.error(f"Error creating database tables: {str(e)}")
#         raise