#=========================================== NEW VERSION OF database.py ================================

# Enhanced database.py - Add this to your User model

from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_database, text
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False)
    
    # 🔧 ADDED: Stripe customer integration
    stripe_customer_id = Column(String(255), nullable=True, index=True)
    stripe_subscription_id = Column(String(255), nullable=True, index=True)
    
    # 🔧 ADDED: User preferences and metadata
    full_name = Column(String(100), nullable=True)
    phone_number = Column(String(20), nullable=True)
    is_active = Column(Boolean, default=True)
    email_verified = Column(Boolean, default=False)
    last_login = Column(DateTime, nullable=True)

class Subscription(Base):
    __tablename__ = "subscriptions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    tier = Column(String(20), nullable=False)  # free, pro, premium
    start_date = Column(DateTime, nullable=False)
    expiry_date = Column(DateTime, nullable=False)
    payment_id = Column(String(255), nullable=True)  # Stripe payment intent ID
    auto_renew = Column(Boolean, default=True)
    
    # 🔧 ADDED: Enhanced subscription tracking
    stripe_subscription_id = Column(String(255), nullable=True, index=True)
    stripe_price_id = Column(String(255), nullable=True)
    status = Column(String(20), default='active')  # active, cancelled, past_due, etc.
    current_period_start = Column(DateTime, nullable=True)
    current_period_end = Column(DateTime, nullable=True)
    cancel_at_period_end = Column(Boolean, default=False)

class TranscriptDownload(Base):
    __tablename__ = "transcript_downloads"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    youtube_id = Column(String(20), nullable=False, index=True)
    transcript_type = Column(String(20), nullable=False)  # clean, unclean, audio, video
    created_at = Column(DateTime, nullable=False)
    
    # 🔧 ADDED: Enhanced download tracking
    file_size = Column(Integer, nullable=True)  # Size in bytes
    processing_time = Column(Integer, nullable=True)  # Time in milliseconds
    download_method = Column(String(50), nullable=True)  # youtube-transcript-api, yt-dlp, etc.
    quality = Column(String(20), nullable=True)  # high, medium, low
    language = Column(String(10), default='en')

# 🔧 ADDED: Payment tracking table
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
    metadata = Column(Text, nullable=True)  # JSON string for additional data

# Database setup functions remain the same
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_transcript.db")

if DATABASE_URL.startswith("sqlite"):
    from sqlalchemy import create_engine
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    from sqlalchemy import create_engine
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def create_tables():
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {str(e)}")
        return False

#========= OLD VERSION OF database.py ===============

# # database.py - Simple version that matches your main.py exactly

# from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, Text
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker
# from datetime import datetime
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # Database configuration
# DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")

# # Create engine
# if DATABASE_URL.startswith("sqlite"):
#     engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
# else:
#     engine = create_engine(DATABASE_URL)

# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
# Base = declarative_base()

# # Simple User model (matches your main.py expectations)
# class User(Base):
#     __tablename__ = "users"

#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String(50), unique=True, index=True, nullable=False)
#     email = Column(String(255), unique=True, index=True, nullable=False)
#     hashed_password = Column(String(255), nullable=False)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     stripe_customer_id = Column(String(255), nullable=True)  # For Stripe integration

#     def __repr__(self):
#         return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"

# # Subscription model (exactly what your main.py expects)
# class Subscription(Base):
#     __tablename__ = "subscriptions"

#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, nullable=False)
#     tier = Column(String(20), nullable=False)  # 'free', 'pro', 'premium'
#     start_date = Column(DateTime, default=datetime.utcnow)
#     expiry_date = Column(DateTime, nullable=False)
#     payment_id = Column(String(255), nullable=True)  # Stripe subscription ID
#     auto_renew = Column(Boolean, default=True)

#     def __repr__(self):
#         return f"<Subscription(user_id={self.user_id}, tier='{self.tier}', expiry='{self.expiry_date}')>"

# # TranscriptDownload model (exactly what your main.py expects)
# class TranscriptDownload(Base):
#     __tablename__ = "transcript_downloads"

#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, nullable=False)
#     youtube_id = Column(String(255), nullable=False)
#     transcript_type = Column(String(20), nullable=False)  # 'clean', 'unclean', 'audio', 'video'
#     created_at = Column(DateTime, default=datetime.utcnow)

#     def __repr__(self):
#         return f"<TranscriptDownload(user_id={self.user_id}, youtube_id='{self.youtube_id}', type='{self.transcript_type}')>"

# # Database session dependency
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# # Create tables function
# def create_tables():
#     """Create all tables in the database"""
#     try:
#         Base.metadata.create_all(bind=engine)
#         print("✅ Database tables created successfully")
#     except Exception as e:
#         print(f"❌ Error creating tables: {e}")
#         raise

# # Initialize database
# if __name__ == "__main__":
#     create_tables()

