from sqlalchemy import Column, Integer, String, DateTime, Boolean, ForeignKey, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime

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
    subscriptions = relationship("Subscription", back_populates="user")
    downloads = relationship("TranscriptDownload", back_populates="user")

class Subscription(Base):
    __tablename__ = "subscriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
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
    user_id = Column(Integer, ForeignKey("users.id"))
    youtube_id = Column(String)
    transcript_type = Column(String)  # "clean" or "unclean"
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationship
    user = relationship("User", back_populates="downloads")

# Database connection setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./youtube_trans_downloader.db"  # Use SQLite for development

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}  # Needed for SQLite
)

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
    Base.metadata.create_all(bind=engine)    