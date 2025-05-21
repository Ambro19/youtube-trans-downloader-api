from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("youtube_trans_downloader.database")

# Create SQLAlchemy base
Base = declarative_base()

# Define database models
class User(Base):
    __tablename__ = "users"
   
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=datetime.now)
   
    # Relationships
    subscriptions = relationship("Subscription", back_populates="user", cascade="all, delete-orphan")
    downloads = relationship("TranscriptDownload", back_populates="user", cascade="all, delete-orphan")

class Subscription(Base):
    __tablename__ = "subscriptions"
   
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    tier = Column(String)  # "free", "basic", "premium"
    start_date = Column(DateTime, default=datetime.now)
    expiry_date = Column(DateTime)
    payment_id = Column(String, nullable=True)
    auto_renew = Column(Boolean, default=False)
   
    # Relationship
    user = relationship("User", back_populates="subscriptions")

class TranscriptDownload(Base):
    __tablename__ = "transcript_downloads"
   
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    youtube_id = Column(String)
    transcript_type = Column(String)  # "clean" or "unclean"
    created_at = Column(DateTime, default=datetime.now)
   
    # Relationship
    user = relationship("User", back_populates="downloads")

# Database configuration
def get_database_url():
    """Get database URL from environment variables, with fallback to SQLite"""
    env = os.getenv("ENV", "development")
    
    # For production, use PostgreSQL
    if env == "production":
        # Construct PostgreSQL connection string
        db_user = os.getenv("DB_USER")
        db_password = os.getenv("DB_PASSWORD")
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "youtube_trans_db")
        
        if db_user and db_password:
            return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
        else:
            logger.warning("PostgreSQL credentials not found, falling back to SQLite")
            return os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")
    
    # For development or testing, use SQLite by default
    return os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")

# Get the appropriate database URL
SQLALCHEMY_DATABASE_URL = get_database_url()
logger.info(f"Using database: {SQLALCHEMY_DATABASE_URL.split('@')[0].split(':')[0]}://****")

# Configure engine based on database type
if SQLALCHEMY_DATABASE_URL.startswith("sqlite"):
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}  # SQLite specific
    )
else:
    # For PostgreSQL, MySQL, etc.
    engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_size=5, max_overflow=10)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Create all tables if they don't exist
def create_tables():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error creating database tables: {str(e)}")
        raise 