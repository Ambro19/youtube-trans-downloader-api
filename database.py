# database.py - Simple version that matches your main.py exactly

from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
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

# Simple User model (matches your main.py expectations)
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    stripe_customer_id = Column(String(255), nullable=True)  # For Stripe integration

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"

# Subscription model (exactly what your main.py expects)
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

# TranscriptDownload model (exactly what your main.py expects)
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

