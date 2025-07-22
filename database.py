#======= the very last one: DO NOT CANGE IT ==============
# Complete Fixed database.py - ALL columns included

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from sqlalchemy import create_engine
from models import Base, User, SubscriptionHistory  # etc.

import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)


Base = declarative_base()


class Subscription(Base):
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

class TranscriptDownload(Base):
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
    quality = Column(String(20), nullable=True)  # high, medium, low
    language = Column(String(10), default='en')

class PaymentHistory(Base):
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
    payment_metadata = Column(Text, nullable=True)  # JSON string for additional data (NOT 'metadata'!)

# Database setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_transcript.db")

if DATABASE_URL.startswith("sqlite"):
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all database tables with ALL columns"""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {str(e)}")
        return False
