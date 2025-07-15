# from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float
# from sqlalchemy.ext.declarative import declarative_base
# from datetime import datetime
# import bcrypt
# import json

# Base = declarative_base()

# class User(Base):
#     __tablename__ = "users"

#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String(50), unique=True, index=True)
#     email = Column(String(100), unique=True, index=True)
#     hashed_password = Column(String(255))
#     created_at = Column(DateTime, default=datetime.utcnow)
    
#     # Stripe integration
#     stripe_customer_id = Column(String(255), index=True)
#     stripe_subscription_id = Column(String(255), index=True)
    
#     # User details
#     full_name = Column(String(100))
#     phone_number = Column(String(20))
#     is_active = Column(Boolean, default=True)
#     email_verified = Column(Boolean, default=False)
#     last_login = Column(DateTime)
    
#     # Subscription info
#     subscription_tier = Column(String(20), default='free')
#     subscription_status = Column(String(20), default='inactive')
#     current_period_end = Column(DateTime)
    
#     # Usage tracking
#     usage_clean_transcripts = Column(Integer, default=0)
#     usage_unclean_transcripts = Column(Integer, default=0)
#     usage_audio_downloads = Column(Integer, default=0)
#     usage_video_downloads = Column(Integer, default=0)
#     usage_reset_date = Column(DateTime, default=datetime.utcnow)
    
#     # Preferences
#     timezone = Column(String(50), default='UTC')
#     language = Column(String(10), default='en')
#     notification_preferences = Column(Text)

#     def set_password(self, password: str):
#         salt = bcrypt.gensalt()
#         self.hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

#     def verify_password(self, password: str) -> bool:
#         return bcrypt.checkpw(
#             password.encode('utf-8'), 
#             self.hashed_password.encode('utf-8')
#         )

#     def is_subscription_active(self) -> bool:
#         if self.subscription_tier == 'free':
#             return True
#         return (
#             self.subscription_status in ['active', 'trialing'] and
#             self.current_period_end and
#             self.current_period_end > datetime.utcnow()
#         )

#     def get_plan_limits(self) -> dict:
#         return {
#             'free': {
#                 'clean_transcripts': 5,
#                 'unclean_transcripts': 3,
#                 'audio_downloads': 2,
#                 'video_downloads': 1
#             },
#             'pro': {
#                 'clean_transcripts': 100,
#                 'unclean_transcripts': 50,
#                 'audio_downloads': 50,
#                 'video_downloads': 20
#             },
#             'premium': {
#                 'clean_transcripts': float('inf'),
#                 'unclean_transcripts': float('inf'),
#                 'audio_downloads': float('inf'),
#                 'video_downloads': float('inf')
#             }
#         }.get(self.subscription_tier)

#     def can_perform_action(self, action_type: str) -> bool:
#         if self.usage_reset_date.month != datetime.utcnow().month:
#             self.reset_monthly_usage()
            
#         limits = self.get_plan_limits()
#         current_usage = getattr(self, f'usage_{action_type}', 0)
#         limit = limits.get(action_type, 0)
        
#         return current_usage < limit if limit != float('inf') else True

#     def reset_monthly_usage(self):
#         for usage_type in ['clean_transcripts', 'unclean_transcripts', 'audio_downloads', 'video_downloads']:
#             setattr(self, f'usage_{usage_type}', 0)
#         self.usage_reset_date = datetime.utcnow()

#     def increment_usage(self, action_type: str):
#         current = getattr(self, f'usage_{action_type}', 0)
#         setattr(self, f'usage_{action_type}', current + 1)

#     def to_dict(self):
#         return {
#             'id': self.id,
#             'username': self.username,
#             'email': self.email,
#             'subscription_tier': self.subscription_tier,
#             'subscription_status': self.subscription_status,
#             'current_period_end': self.current_period_end.isoformat() if self.current_period_end else None,
#             'usage': {
#                 'clean_transcripts': self.usage_clean_transcripts,
#                 'unclean_transcripts': self.usage_unclean_transcripts,
#                 'audio_downloads': self.usage_audio_downloads,
#                 'video_downloads': self.usage_video_downloads
#             }
#         }

# class SubscriptionHistory(Base):
#     __tablename__ = "subscription_history"

#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, index=True)
#     action = Column(String(50))  # upgrade, downgrade, cancel, renew
#     from_tier = Column(String(20))
#     to_tier = Column(String(20))
#     amount = Column(Float)
#     stripe_subscription_id = Column(String(255))
#     stripe_payment_intent_id = Column(String(255))
#     created_at = Column(DateTime, default=datetime.utcnow)
#     history_meta_data = Column(Text)  # Changed from metadata to meta_data

#     def set_meta_data(self, data: dict):
#         self.history_meta_data = json.dumps(data)

#     def get_meta_data(self) -> dict:
#         return json.loads(self.history_meta_data) if self.history_meta_data else {}

# def create_tables(engine):
#     Base.metadata.create_all(bind=engine)

#====================================
# # models.py - Updated User Model with Subscription Fields
# # models.py

# from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float
# from sqlalchemy.ext.declarative import declarative_base
# from datetime import datetime
# import bcrypt

# Base = declarative_base()

# class User(Base):
#     __tablename__ = "users"

#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String(50), unique=True, index=True, nullable=False)
#     email = Column(String(255), unique=True, index=True, nullable=False)
#     full_name = Column(String(255), nullable=True)
#     hashed_password = Column(String(255), nullable=False)
#     is_active = Column(Boolean, default=True)
#     is_verified = Column(Boolean, default=False)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
#     last_login = Column(DateTime, nullable=True)
#     subscription_tier = Column(String(20), default='free', nullable=False)
#     subscription_status = Column(String(20), default='inactive', nullable=False)
#     subscription_id = Column(String(255), nullable=True)
#     subscription_current_period_end = Column(DateTime, nullable=True)
#     stripe_customer_id = Column(String(255), nullable=True)
#     usage_clean_transcripts = Column(Integer, default=0, nullable=False)
#     usage_unclean_transcripts = Column(Integer, default=0, nullable=False)
#     usage_audio_downloads = Column(Integer, default=0, nullable=False)
#     usage_video_downloads = Column(Integer, default=0, nullable=False)
#     usage_reset_date = Column(DateTime, default=datetime.utcnow, nullable=False)
#     timezone = Column(String(50), default='UTC')
#     language = Column(String(10), default='en')
#     notification_preferences = Column(Text, nullable=True)

#     def __repr__(self):
#         return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"

#     def set_password(self, password: str):
#         salt = bcrypt.gensalt()
#         self.hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

#     def verify_password(self, password: str) -> bool:
#         return bcrypt.checkpw(password.encode('utf-8'), self.hashed_password.encode('utf-8'))

#     def is_subscription_active(self) -> bool:
#         if self.subscription_tier == 'free':
#             return True
#         if not self.subscription_current_period_end:
#             return False
#         return (
#             self.subscription_status in ['active', 'trialing'] and
#             self.subscription_current_period_end > datetime.utcnow()
#         )

#     def get_plan_limits(self) -> dict:
#         limits = {
#             'free': {
#                 'clean_transcripts': 5,
#                 'unclean_transcripts': 3,
#                 'audio_downloads': 2,
#                 'video_downloads': 1
#             },
#             'pro': {
#                 'clean_transcripts': 100,
#                 'unclean_transcripts': 50,
#                 'audio_downloads': 50,
#                 'video_downloads': 20
#             },
#             'premium': {
#                 'clean_transcripts': float('inf'),
#                 'unclean_transcripts': float('inf'),
#                 'audio_downloads': float('inf'),
#                 'video_downloads': float('inf')
#             }
#         }
#         return limits.get(self.subscription_tier, limits['free'])

#     def get_current_usage(self) -> dict:
#         return {
#             'clean_transcripts': self.usage_clean_transcripts,
#             'unclean_transcripts': self.usage_unclean_transcripts,
#             'audio_downloads': self.usage_audio_downloads,
#             'video_downloads': self.usage_video_downloads
#         }

#     def can_perform_action(self, action_type: str) -> bool:
#         if self.usage_reset_date.month != datetime.utcnow().month:
#             self.reset_monthly_usage()
#         limits = self.get_plan_limits()
#         current_usage = getattr(self, f'usage_{action_type}', 0)
#         limit = limits.get(action_type, 0)
#         if limit == float('inf'):
#             return True
#         return current_usage < limit

#     def reset_monthly_usage(self):
#         self.usage_clean_transcripts = 0
#         self.usage_unclean_transcripts = 0
#         self.usage_audio_downloads = 0
#         self.usage_video_downloads = 0
#         self.usage_reset_date = datetime.utcnow()

#     def increment_usage(self, action_type: str):
#         current_usage = getattr(self, f'usage_{action_type}', 0)
#         setattr(self, f'usage_{action_type}', current_usage + 1)

#     def to_dict(self) -> dict:
#         return {
#             'id': self.id,
#             'email': self.email,
#             'full_name': self.full_name,
#             'is_active': self.is_active,
#             'is_verified': self.is_verified,
#             'subscription_tier': self.subscription_tier,
#             'subscription_status': self.subscription_status,
#             'created_at': self.created_at.isoformat() if self.created_at else None,
#             'last_login': self.last_login.isoformat() if self.last_login else None,
#         }

# class TranscriptDownload(Base):
#     __tablename__ = "transcript_downloads"

#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, nullable=False, index=True)
#     youtube_id = Column(String(20), nullable=False, index=True)
#     transcript_type = Column(String(20), nullable=False)  # clean, unclean, audio, video
#     created_at = Column(DateTime, nullable=False)
#     file_size = Column(Integer, nullable=True)  # Size in bytes
#     processing_time = Column(Integer, nullable=True)  # Time in milliseconds
#     download_method = Column(String(50), nullable=True)  # youtube-transcript-api, yt-dlp, etc.
#     quality = Column(String(20), nullable=True)  # high, medium, low
#     language = Column(String(10), default='en')

# class SubscriptionHistory(Base):
#     __tablename__ = "subscription_history"

#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, nullable=False)
#     action = Column(String(50), nullable=False)  # 'upgraded', 'downgraded', 'cancelled', 'renewed'
#     from_tier = Column(String(20), nullable=True)
#     to_tier = Column(String(20), nullable=True)
#     amount = Column(Float, nullable=True)
#     stripe_subscription_id = Column(String(255), nullable=True)
#     stripe_payment_intent_id = Column(String(255), nullable=True)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     history_metadata = Column(Text, nullable=True)  # JSON string for additional data

#     def __repr__(self):
#         return f"<SubscriptionHistory(user_id={self.user_id}, action='{self.action}', from='{self.from_tier}', to='{self.to_tier}')>"

# def create_tables(engine):
#     """Create all tables in the database"""
#     Base.metadata.create_all(bind=engine)

#===========================================

#======= the very last one: DO NOT CANGE IT, PLEASE ==============
# # models.py - Updated User Model with Subscription Fields

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime
import bcrypt

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    # Primary key and basic fields
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Subscription fields
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
    usage_reset_date = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Additional user preferences
    timezone = Column(String(50), default='UTC')
    language = Column(String(10), default='en')
    notification_preferences = Column(Text, nullable=True)  # JSON string

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', tier='{self.subscription_tier}')>"

    def set_password(self, password: str):
        """Hash and set the user's password"""
        salt = bcrypt.gensalt()
        self.hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def verify_password(self, password: str) -> bool:
        """Verify the user's password"""
        return bcrypt.checkpw(
            password.encode('utf-8'), 
            self.hashed_password.encode('utf-8')
        )

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

    def get_current_usage(self) -> dict:
        """Get the user's current usage for this month"""
        return {
            'clean_transcripts': self.usage_clean_transcripts,
            'unclean_transcripts': self.usage_unclean_transcripts,
            'audio_downloads': self.usage_audio_downloads,
            'video_downloads': self.usage_video_downloads
        }

    def can_perform_action(self, action_type: str) -> bool:
        """Check if user can perform the specified action based on limits"""
        # Reset usage if it's a new month
        if self.usage_reset_date.month != datetime.utcnow().month:
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

    def increment_usage(self, action_type: str):
        """Increment usage counter for the specified action"""
        current_usage = getattr(self, f'usage_{action_type}', 0)
        setattr(self, f'usage_{action_type}', current_usage + 1)

    def to_dict(self) -> dict:
        """Convert user to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'email': self.email,
            'full_name': self.full_name,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'subscription_tier': self.subscription_tier,
            'subscription_status': self.subscription_status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
        }

# Additional model for storing subscription history (optional)
class SubscriptionHistory(Base):
    __tablename__ = "subscription_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    action = Column(String(50), nullable=False)  # 'upgraded', 'downgraded', 'cancelled', 'renewed'
    from_tier = Column(String(20), nullable=True)
    to_tier = Column(String(20), nullable=True)
    amount = Column(Float, nullable=True)
    stripe_subscription_id = Column(String(255), nullable=True)
    stripe_payment_intent_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    history_metadata = Column(Text, nullable=True)  # JSON string for additional data

    def __repr__(self):
        return f"<SubscriptionHistory(user_id={self.user_id}, action='{self.action}', from='{self.from_tier}', to='{self.to_tier}')>"

# Create tables function
def create_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(bind=engine)

# ===================

# #========= IMPORTANT NOTE: THIS MAIN.PY WORKS FINE EXCEPT ONE WORKING EXAMPLE LINK SO KEEP IT!! ==========
# #========= DESACTIVATED 7/9/25 @ 10:10 PM =========

# # main.py (advanced, usage tracked on User model, for youtube-transcript-api only)
# from fastapi import FastAPI, HTTPException, Depends, status
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from sqlalchemy.orm import Session
# from datetime import datetime, timedelta
# from typing import Optional
# import os
# import jwt
# from jwt.exceptions import PyJWTError
# from pydantic import BaseModel
# from passlib.context import CryptContext
# import logging
# from dotenv import load_dotenv
# import re


# import subprocess
# import json


# from database import engine, SessionLocal, get_db  # Make sure this points to your db
# from models import User, create_tables  # Use your models.py User model!
# #from database import SessionLocal, get_db  # Make sure this points to your db

# load_dotenv()

# # --- Logging ---
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("youtube_trans_downloader.main")

# # --- App & CORS ---
# app = FastAPI(title="YouTubeTransDownloader API", version="2.0.0 (user-usage)")
# ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# allowed_origins = [
#     "http://localhost:3000", "http://127.0.0.1:3000", FRONTEND_URL
# ] if ENVIRONMENT != "production" else [
#     "https://youtube-trans-downloader-api.onrender.com", FRONTEND_URL
# ]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=allowed_origins, allow_credentials=True,
#     allow_methods=["*"], allow_headers=["*"],
# )

# # --- Security ---
# SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # --- Helper Functions ---

# def get_user(db: Session, username: str) -> Optional[User]:
#     return db.query(User).filter(User.username == username).first()

# # def get_user_by_email(db: Session, email: str) -> Optional[User]:
# #     return db.query(User).filter(User.email == email).first()
# def get_user_by_username(db: Session, username: str):
#     return db.query(User).filter(User.username == username).first()


# def verify_password(plain_password, hashed_password):
#     return pwd_context.verify(plain_password, hashed_password)

# def get_password_hash(password):
#     return pwd_context.hash(password)

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
#     to_encode = data.copy()
#     expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")
#         if username is None:
#             raise credentials_exception
#     except PyJWTError:
#         raise credentials_exception
#     user = get_user(db, username)
#     if user is None:
#         raise credentials_exception
#     return user

# # --- Pydantic Models ---

# class UserCreate(BaseModel):
#     username: str
#     email: str
#     password: str

# class UserResponse(BaseModel):
#     id: int
#     username: str = None
#     email: str
#     created_at: Optional[datetime] = None
#     class Config:
#         #orm_mode = True
#         from_attributes = True

# class Token(BaseModel):
#     access_token: str
#     token_type: str

# class TranscriptRequest(BaseModel):
#     youtube_id: str
#     clean_transcript: bool = True

# # --- Transcript logic (youtube-transcript-api only, no yt-dlp) ---

# def extract_youtube_video_id(youtube_id_or_url: str) -> str:
#     # Accept raw ID, youtu.be/..., youtube.com/watch?v=...
#     patterns = [
#         r'(?:youtube\.com\/watch\?v=)([^&\n?#]+)',
#         r'(?:youtu\.be\/)([^&\n?#]+)',
#         r'(?:youtube\.com\/embed\/)([^&\n?#]+)',
#         r'(?:youtube\.com\/shorts\/)([^&\n?#]+)',
#         r'[?&]v=([^&\n?#]+)'
#     ]
#     for pattern in patterns:
#         match = re.search(pattern, youtube_id_or_url)
#         if match:
#             return match.group(1)[:11]
#     return youtube_id_or_url.strip()[:11]

# def get_transcript_youtube_api(video_id: str, clean: bool = True) -> str:
#     try:
#         from youtube_transcript_api import YouTubeTranscriptApi
#         transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
#         if clean:
#             text = " ".join([seg['text'].replace('\n', ' ') for seg in transcript])
#             return " ".join(text.split())
#         else:
#             lines = []
#             for seg in transcript:
#                 t = int(seg['start'])
#                 timestamp = f"[{t//60:02d}:{t%60:02d}]"
#                 #lines.append(f"{timestamp} {seg['text'].replace('\n', ' ')}")
#                 text_clean = seg['text'].replace('\n', ' ')
#             return "\n".join(lines)
#     except Exception as e:
#         # Try yt-dlp fallback!
#         print(f"Transcript API failed: {e} - trying yt-dlp fallback...")
#         yt_dlp_transcript = get_transcript_with_ytdlp(video_id, clean=clean)
#         if yt_dlp_transcript:
#             return yt_dlp_transcript
#         return None

# def get_transcript_with_ytdlp(video_id, clean=True):
#     """
#     Use yt-dlp to extract captions as a fallback.
#     Returns transcript as a string, or None.
#     """
#     try:
#         # Ensure yt-dlp is installed and in your PATH or venv
#         # Try to get the English auto-generated captions
#         cmd = [
#             "yt-dlp",
#             "--skip-download",
#             "--write-auto-subs",
#             "--sub-lang", "en",
#             "--sub-format", "json3",
#             "--output", "%(id)s",
#             f"https://www.youtube.com/watch?v={video_id}"
#         ]
#         subprocess.run(cmd, check=True, capture_output=True)
#         json_path = f"{video_id}.en.json3"
#         if not os.path.exists(json_path):
#             return None
#         with open(json_path, encoding="utf8") as f:
#             data = json.load(f)
#         # Parse the JSON3 captions (Google format)
#         text_blocks = []
#         for event in data.get("events", []):
#             if "segs" in event and "tStartMs" in event:
#                 text = "".join([seg.get("utf8", "") for seg in event["segs"]]).strip()
#                 if text:
#                     if clean:
#                         text_blocks.append(text)
#                     else:
#                         start_sec = int(event["tStartMs"] // 1000)
#                         timestamp = f"[{start_sec // 60:02d}:{start_sec % 60:02d}]"
#                         text_blocks.append(f"{timestamp} {text}")
#         os.remove(json_path)  # Clean up
#         return "\n".join(text_blocks) if text_blocks else None
#     except Exception as e:
#         print("yt-dlp fallback error:", e)
#         return None

# def get_demo_content(clean=True):
#     if clean:
#         return "This is demo transcript content. The real YouTube transcript could not be downloaded."
#     else:
#         return "[00:00] This is demo transcript content. The real YouTube transcript could not be downloaded."

# # --- Usage keys for User model (matches your models.py) ---
# USAGE_KEYS = {
#     True: "clean_transcripts",
#     False: "unclean_transcripts"
# }

# # --- FastAPI Endpoints ---

# @app.on_event("startup")
# def startup():
#     create_tables(engine)

# @app.get("/")
# def root():
#     return {"message": "YouTube Transcript Downloader API", "status": "running"}

# @app.post("/register")
# def register(user: UserCreate, db: Session = Depends(get_db)):
#     # Check for existing username/email
#     if db.query(User).filter(User.username == user.username).first():
#         raise HTTPException(status_code=400, detail="Username already exists.")
#     if db.query(User).filter(User.email == user.email).first():
#         raise HTTPException(status_code=400, detail="Email already exists.")
#     user_obj = User(
#         username=user.username,
#         email=user.email,
#         hashed_password=get_password_hash(user.password),
#         created_at=datetime.utcnow()
#     )
#     db.add(user_obj)
#     db.commit()
#     db.refresh(user_obj)
#     return {"message": "User registered successfully."}


# @app.post("/token")
# def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
#     user = db.query(User).filter(User.username == form_data.username).first()
#     if not user or not verify_password(form_data.password, user.hashed_password):
#         raise HTTPException(status_code=401, detail="Incorrect username or password")
#     # Create access token and return (add your create_access_token logic here)
#     access_token = create_access_token(
#         data={"sub": user.username},
#         expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     )
#     return {"access_token": access_token, "token_type": "bearer"}

# @app.get("/users/me", response_model=UserResponse)
# def read_users_me(current_user: User = Depends(get_current_user)):
#     return current_user

# @app.post("/download_transcript/")
# def download_transcript(
#     request: TranscriptRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     video_id = extract_youtube_video_id(request.youtube_id)
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")

#     # --- Usage logic using models.py advanced fields ---
#     usage_key = USAGE_KEYS[request.clean_transcript]
#     if user.usage_reset_date.month != datetime.utcnow().month:
#         user.reset_monthly_usage()
#         db.commit()
#     # Check subscription and usage limit
#     plan_limits = user.get_plan_limits()
#     current_usage = getattr(user, f"usage_{usage_key}", 0)
#     allowed = plan_limits[usage_key]
#     if allowed != float('inf') and current_usage >= allowed:
#         raise HTTPException(
#             status_code=403,
#             detail=f"Monthly limit reached for {usage_key.replace('_',' ')}. Please upgrade your plan."
#         )

#     transcript = get_transcript_youtube_api(video_id, clean=request.clean_transcript)
#     if not transcript or len(transcript.strip()) < 10:
#         #transcript = get_demo_content(clean=request.clean_transcript)
#         raise HTTPException(status_code=404, detail="No transcript/captions found for this video.")

#     # Increment usage counter, save
#     user.increment_usage(usage_key)
#     db.commit()
#     logger.info(f"User {user.username} downloaded transcript for {video_id} ({usage_key})")
#     return {
#         "transcript": transcript,
#         "youtube_id": video_id,
#         "message": "Transcript downloaded successfully"
#     }

# #======
# @app.get("/subscription_status/")
# async def get_subscription_status_ultra_safe(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     try:
#         subscription = db.query(Subscription).filter(
#             Subscription.user_id == current_user.id
#         ).first()
#         if not subscription:
#             tier = "free"
#             status = "inactive"
#             expiry_date = None
#         else:
#             if hasattr(subscription, 'expiry_date') and subscription.expiry_date and subscription.expiry_date < datetime.now():
#                 tier = "free"
#                 status = "expired"
#                 expiry_date = subscription.expiry_date
#             else:
#                 tier = subscription.tier if subscription.tier else "free"
#                 status = "active" if tier != "free" else "inactive"
#                 expiry_date = subscription.expiry_date if hasattr(subscription, 'expiry_date') else None
#         # Usage logic...
#         month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
#         def get_safe_usage(transcript_type):
#             try:
#                 return db.query(TranscriptDownload).filter(
#                     TranscriptDownload.user_id == current_user.id,
#                     TranscriptDownload.transcript_type == transcript_type,
#                     TranscriptDownload.created_at >= month_start
#                 ).count()
#             except Exception:
#                 return 0
#         clean_usage = get_safe_usage("clean")
#         unclean_usage = get_safe_usage("unclean")
#         audio_usage = get_safe_usage("audio")
#         video_usage = get_safe_usage("video")
#         limits = SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS["free"])
#         json_limits = {k: ('unlimited' if v == float('inf') else v) for k, v in limits.items()}
#         return {
#             "tier": tier,
#             "status": status,
#             "usage": {
#                 "clean_transcripts": clean_usage,
#                 "unclean_transcripts": unclean_usage,
#                 "audio_downloads": audio_usage,
#                 "video_downloads": video_usage,
#             },
#             "limits": json_limits,
#             "subscription_id": subscription.payment_id if subscription and hasattr(subscription, 'payment_id') else None,
#             "stripe_customer_id": getattr(current_user, 'stripe_customer_id', None),
#             "current_period_end": expiry_date.isoformat() if expiry_date else None
#         }
#     except Exception as e:
#         logger.error(f"❌ Error getting subscription status: {str(e)}")
#         return {
#             "tier": "free",
#             "status": "inactive", 
#             "usage": {
#                 "clean_transcripts": 0,
#                 "unclean_transcripts": 0,
#                 "audio_downloads": 0,
#                 "video_downloads": 0,
#             },
#             "limits": {
#                 "clean_transcripts": 5,
#                 "unclean_transcripts": 3,
#                 "audio_downloads": 2,
#                 "video_downloads": 1
#             },
#             "subscription_id": None,
#             "stripe_customer_id": None,
#             "current_period_end": None
#         }

# @app.get("/health/")
# def health():
#     return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

# @app.get("/test_videos")
# def get_test_videos():
#     return {
#         "videos": [
#             {"id": "dQw4w9WgXcQ", "title": "Rick Astley - Never Gonna Give You Up"},
#             {"id": "eYDS3T1egng", "title": "General-Purpose vs Special-Purpose Computers: What's the Difference?"}
#         ]
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

