from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta, timezone
from typing import Optional, List
import jwt
from jwt.exceptions import PyJWTError
from pydantic import BaseModel
from passlib.context import CryptContext
import stripe
import os
import logging
from dotenv import load_dotenv
import re
import sqlite3

import warnings
warnings.filterwarnings("ignore", message=".*bcrypt.*")

# Import from database.py
from database import get_db, User, Subscription, TranscriptDownload, create_tables

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("youtube_trans_downloader.main")

# Stripe configuration
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
DOMAIN = os.getenv("DOMAIN", "https://youtube-trans-downloader-api.onrender.com")

# Database path for direct SQLite operations
DATABASE_PATH = "youtube_trans_downloader.db"

# Create FastAPI app
app = FastAPI(
    title="YouTubeTransDownloader API", 
    description="API for downloading and processing YouTube video transcripts",
    version="1.0.0"
)

# Environment-aware configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Configure CORS based on environment
if ENVIRONMENT == "production":
    allowed_origins = [
        "http://localhost:8000",
        "https://youtube-trans-downloader-api.onrender.com",
        FRONTEND_URL
    ]
    logger.info(f"🌍 Production mode - CORS origins: {allowed_origins}")
else:
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        FRONTEND_URL
    ]
    logger.info(f"🔧 Development mode - CORS origins: {allowed_origins}")

# CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Authentication setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Constants
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# Enhanced subscription limits with detailed feature sets
SUBSCRIPTION_LIMITS = {
    "free": {
        "transcript": 5, "audio": 2, "video": 1, "clean": 5, "unclean": 3,
        "clean_transcripts": 5, "unclean_transcripts": 3, 
        "audio_downloads": 2, "video_downloads": 1,
        "monthly_downloads": 5,
        "bulk_downloads": False,
        "priority_processing": False,
        "advanced_formats": False,
        "api_access": False,
        "export_formats": ["txt"],
        "concurrent_downloads": 1
    },
    "trial": {
        "transcript": 10, "audio": 5, "video": 2, "clean": 10, "unclean": 5,
        "clean_transcripts": 10, "unclean_transcripts": 5,
        "audio_downloads": 5, "video_downloads": 2,
        "monthly_downloads": 10,
        "bulk_downloads": False,
        "priority_processing": False,
        "advanced_formats": False,
        "api_access": False,
        "export_formats": ["txt"],
        "concurrent_downloads": 1
    },
    "pro": {
        "transcript": 100, "audio": 50, "video": 20, "clean": 100, "unclean": 50,
        "clean_transcripts": 100, "unclean_transcripts": 50,
        "audio_downloads": 50, "video_downloads": 20,
        "monthly_downloads": 100,
        "bulk_downloads": True,
        "priority_processing": True,
        "advanced_formats": True,
        "api_access": False,
        "export_formats": ["txt", "srt", "vtt", "json"],
        "concurrent_downloads": 3
    },
    "premium": {
        "transcript": float('inf'), "audio": float('inf'), "video": float('inf'), 
        "clean": float('inf'), "unclean": float('inf'),
        "clean_transcripts": float('inf'), "unclean_transcripts": float('inf'),
        "audio_downloads": float('inf'), "video_downloads": float('inf'),
        "monthly_downloads": float('inf'),
        "bulk_downloads": True,
        "priority_processing": True,
        "advanced_formats": True,
        "api_access": True,
        "export_formats": ["txt", "srt", "vtt", "json", "xml", "docx"],
        "concurrent_downloads": 10
    }
}

# Price ID mapping
PRICE_ID_MAP = {
    "pro": os.getenv("PRO_PRICE_ID"),
    "premium": os.getenv("PREMIUM_PRICE_ID")
}

@app.on_event("startup")
async def startup_event():
    """Enhanced startup with environment validation"""
    try:
        logger.info("🚀 Starting YouTube Transcript Downloader API...")
        logger.info(f"🌍 Environment: {ENVIRONMENT}")
        logger.info(f"🔗 Domain: {DOMAIN}")
        
        # Validate critical environment variables
        required_vars = {
            "SECRET_KEY": "JWT secret key",
            "STRIPE_SECRET_KEY": "Stripe secret key",
        }
        
        missing_vars = []
        for var, description in required_vars.items():
            value = os.getenv(var)
            if not value:
                missing_vars.append(f"{var} ({description})")
            else:
                logger.info(f"✅ {var}: {value[:8]}..." if len(value) > 8 else f"✅ {var}: SET")
        
        if missing_vars:
            logger.error(f"❌ Missing required environment variables:")
            for var in missing_vars:
                logger.error(f"   - {var}")
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
        
        # Initialize database
        create_tables()
        logger.info("✅ Database initialized successfully")
        logger.info("🎉 Application startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

#===============================================#
# PYDANTIC MODELS: USER ACCOUNT RELATED CLASSES #
#===============================================#

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class TranscriptRequest(BaseModel):
    youtube_id: str
    clean_transcript: bool = False

class CreatePaymentIntentRequest(BaseModel):
    price_id: str

class ConfirmPaymentRequest(BaseModel):
    payment_intent_id: str

class SubscriptionRequest(BaseModel):
    token: Optional[str] = None
    subscription_tier: str

class SubscriptionResponse(BaseModel):
    tier: str
    status: str
    expiry_date: Optional[str] = None
    limits: dict
    usage: Optional[dict] = None
    remaining: Optional[dict] = None
    
    class Config:
        from_attributes = True

#==================================================#
# HELPER FUNCTIONS: USER ACCOUNT RELATED FUNCTIONS #
#==================================================#

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
        
    user = get_user(db, username)
    if user is None:
        raise credentials_exception
    return user

def get_subscription_limits(tier: str) -> dict:
    """Get subscription limits for a given tier"""
    return SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS["free"])

# Enhanced Stripe integration functions for main.py
def get_or_create_stripe_customer(user, db: Session):
    """
    Enhanced Stripe customer creation with better tracking
    This ensures customers appear in your Stripe dashboard automatically
    """
    try:
        # Check if user already has a Stripe customer ID
        if hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
            try:
                # Verify the customer still exists in Stripe
                customer = stripe.Customer.retrieve(user.stripe_customer_id)
                logger.info(f"✅ Found existing Stripe customer: {customer.id} for user {user.username}")
                return customer
            except stripe.error.InvalidRequestError:
                logger.info(f"📝 Stripe customer {user.stripe_customer_id} not found, creating new one")
                pass
        
        # Create new Stripe customer
        logger.info(f"🔄 Creating new Stripe customer for user: {user.username}")
        customer = stripe.Customer.create(
            email=user.email,
            name=user.username,
            description=f"YouTube Transcript Downloader user: {user.username}",
            metadata={
                'user_id': str(user.id),
                'username': user.username,
                'signup_date': user.created_at.isoformat() if user.created_at else None,
                'app_name': 'YouTube Transcript Downloader'
            }
        )
        
        # Save the Stripe customer ID to our database
        if hasattr(user, 'stripe_customer_id'):
            user.stripe_customer_id = customer.id
            db.commit()
            db.refresh(user)
            logger.info(f"✅ Stripe customer created and saved: {customer.id}")
        
        return customer
        
    except Exception as e:
        logger.error(f"❌ Error creating Stripe customer for user {user.username}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create payment customer. Please try again."
        )

#=================================#
# ENHANCED SUBSCRIPTION FUNCTIONS #
#=================================#

# FIXED SUBSCRIPTION FUNCTION - Resolves datetime timezone issues
def check_subscription_limit(user_id: int, transcript_type: str = None, db: Session = None) -> dict:
    """
    FIXED: Enhanced subscription limit checker with timezone handling
    """
    try:
        if db:
            # Get user with subscription details
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return {"allowed": False, "error": "User not found"}
            
            subscription = db.query(Subscription).filter(Subscription.user_id == user_id).first()
            
            # Determine subscription tier and status
            if not subscription:
                subscription_tier = "free"
                subscription_status = "inactive"
                start_date = None
                end_date = None
            else:
                subscription_tier = subscription.tier
                subscription_status = getattr(subscription, 'status', 'active')
                start_date = subscription.start_date
                end_date = subscription.expiry_date
        else:
            # Fallback to direct database access
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT subscription_tier, subscription_status, subscription_start_date, subscription_end_date 
                FROM users WHERE id = ?
            """, (user_id,))
            
            user_data = cursor.fetchone()
            if not user_data:
                return {"allowed": False, "error": "User not found"}
            
            subscription_tier, subscription_status, start_date, end_date = user_data
            conn.close()
        
        # FIXED: Parse subscription dates with proper timezone handling
        now = datetime.now()  # Use naive datetime consistently
        start_dt = None
        end_dt = None
        
        if start_date:
            try:
                if isinstance(start_date, str):
                    # Remove timezone info to make it naive
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '').replace('+00:00', ''))
                else:
                    # Convert datetime to naive if needed
                    start_dt = start_date.replace(tzinfo=None) if start_date.tzinfo else start_date
            except:
                start_dt = None
        
        if end_date:
            try:
                if isinstance(end_date, str):
                    # Remove timezone info to make it naive
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '').replace('+00:00', ''))
                else:
                    # Convert datetime to naive if needed
                    end_dt = end_date.replace(tzinfo=None) if end_date.tzinfo else end_date
            except:
                end_dt = None
        
        # Check if subscription is active (all naive datetimes now)
        is_subscription_active = (
            subscription_status == 'active' and 
            start_dt and start_dt <= now and 
            (not end_dt or end_dt > now)
        )
        
        # Get current month's usage
        current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        if db:
            # Use SQLAlchemy for usage queries
            downloads_this_month = db.query(TranscriptDownload).filter(
                TranscriptDownload.user_id == user_id,
                TranscriptDownload.created_at >= current_month_start
            ).count()
            
            # Get breakdown by type if needed
            if transcript_type:
                type_downloads = db.query(TranscriptDownload).filter(
                    TranscriptDownload.user_id == user_id,
                    TranscriptDownload.transcript_type == transcript_type,
                    TranscriptDownload.created_at >= current_month_start
                ).count()
            else:
                type_downloads = downloads_this_month
        else:
            # Direct database query fallback
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT COUNT(*) FROM transcript_downloads 
                WHERE user_id = ? AND created_at >= ?
            """, (user_id, current_month_start.isoformat()))
            
            downloads_this_month = cursor.fetchone()[0] or 0
            type_downloads = downloads_this_month
            conn.close()
        
        # Get subscription limits
        limits = get_subscription_limits(subscription_tier)
        monthly_limit = limits['monthly_downloads']
        
        # Calculate usage metrics
        usage_percentage = (downloads_this_month / monthly_limit * 100) if monthly_limit != float('inf') and monthly_limit > 0 else 0
        remaining_downloads = max(0, monthly_limit - downloads_this_month) if monthly_limit != float('inf') else float('inf')
        
        # Determine access status
        if not is_subscription_active:
            if subscription_tier in ['free', 'trial']:
                allowed = downloads_this_month < monthly_limit
                reason = "Free tier limit" if not allowed else "Free tier access"
            else:
                allowed = False
                reason = "Subscription expired or inactive"
        else:
            allowed = downloads_this_month < monthly_limit or monthly_limit == float('inf')
            reason = "Subscription active" if allowed else "Monthly limit exceeded"
        
        logger.info(f"✅ Subscription check: {subscription_tier} tier, {downloads_this_month}/{monthly_limit} downloads, allowed={allowed}")
        
        return {
            "allowed": allowed,
            "reason": reason,
            "subscription": {
                "tier": subscription_tier,
                "status": subscription_status,
                "is_active": is_subscription_active,
                "start_date": start_date.isoformat() if start_date and hasattr(start_date, 'isoformat') else str(start_date),
                "end_date": end_date.isoformat() if end_date and hasattr(end_date, 'isoformat') else str(end_date),
                "days_remaining": (end_dt - now).days if end_dt and end_dt > now else None
            },
            "usage": {
                "downloads_this_month": downloads_this_month,
                "monthly_limit": monthly_limit if monthly_limit != float('inf') else 'unlimited',
                "remaining_downloads": remaining_downloads if remaining_downloads != float('inf') else 'unlimited',
                "usage_percentage": round(usage_percentage, 1),
                "type_specific_downloads": type_downloads
            },
            "limits": limits,
            "features": {
                "bulk_downloads": limits.get('bulk_downloads', False),
                "priority_processing": limits.get('priority_processing', False),
                "advanced_formats": limits.get('advanced_formats', False),
                "api_access": limits.get('api_access', False)
            }
        }
        
    except Exception as e:
        logger.error(f"Error checking subscription limit: {e}")
        return {
            "allowed": True,  # Allow on error to avoid blocking users
            "error": str(e),
            "usage": {"downloads_this_month": 0, "monthly_limit": 5}
        }

def get_subscription_status_enhanced(user_id: int, db: Session = None) -> dict:
    """
    Get comprehensive subscription status with usage analytics
    This is a NEW function that provides detailed subscription insights
    """
    try:
        if db:
            # Use SQLAlchemy ORM
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return {"error": "User not found"}
            
            subscription = db.query(Subscription).filter(Subscription.user_id == user_id).first()
            
            # Get user details
            email = user.email
            user_created = user.created_at
            
            # Get subscription details
            if not subscription:
                subscription_tier = "free"
                subscription_status = "inactive"
                start_date = None
                end_date = None
            else:
                subscription_tier = subscription.tier
                subscription_status = getattr(subscription, 'status', 'active')
                start_date = subscription.start_date
                end_date = subscription.expiry_date
        else:
            # Direct database access fallback
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT subscription_tier, subscription_status, subscription_start_date, 
                       subscription_end_date, email, created_at
                FROM users WHERE id = ?
            """, (user_id,))
            
            user_data = cursor.fetchone()
            if not user_data:
                return {"error": "User not found"}
            
            subscription_tier, subscription_status, start_date, end_date, email, user_created = user_data
            conn.close()
        
        # Parse dates
        now = datetime.now(timezone.utc)
        start_dt = None
        end_dt = None
        user_created_dt = None
        
        if start_date:
            try:
                if isinstance(start_date, str):
                    start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                else:
                    start_dt = start_date
            except:
                start_dt = None
        
        if end_date:
            try:
                if isinstance(end_date, str):
                    end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                else:
                    end_dt = end_date
            except:
                end_dt = None
        
        if user_created:
            try:
                if isinstance(user_created, str):
                    user_created_dt = datetime.fromisoformat(user_created.replace('Z', '+00:00'))
                else:
                    user_created_dt = user_created
            except:
                user_created_dt = None
        
        # Calculate subscription metrics
        is_active = (
            subscription_status == 'active' and 
            start_dt and start_dt <= now and 
            (not end_dt or end_dt > now)
        )
        
        days_remaining = None
        if end_dt and end_dt > now:
            days_remaining = (end_dt - now).days
        
        days_since_start = None
        if start_dt:
            days_since_start = (now - start_dt).days
        
        # Get current month usage
        current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        if db:
            # SQLAlchemy queries
            current_month_downloads = db.query(TranscriptDownload).filter(
                TranscriptDownload.user_id == user_id,
                TranscriptDownload.created_at >= current_month_start
            ).count()
            
            total_downloads = db.query(TranscriptDownload).filter(
                TranscriptDownload.user_id == user_id
            ).count()
            
            # Get first and last download dates
            first_download = db.query(TranscriptDownload).filter(
                TranscriptDownload.user_id == user_id
            ).order_by(TranscriptDownload.created_at.asc()).first()
            
            last_download = db.query(TranscriptDownload).filter(
                TranscriptDownload.user_id == user_id
            ).order_by(TranscriptDownload.created_at.desc()).first()
            
            first_ever_download = first_download.created_at if first_download else None
            last_download_date = last_download.created_at if last_download else None
            
        else:
            # Direct SQL queries
            conn = sqlite3.connect(DATABASE_PATH)
            cursor = conn.cursor()
            
            # Current month usage
            cursor.execute("""
                SELECT COUNT(*) FROM transcript_downloads 
                WHERE user_id = ? AND created_at >= ?
            """, (user_id, current_month_start.isoformat()))
            current_month_downloads = cursor.fetchone()[0] or 0
            
            # Total downloads
            cursor.execute("""
                SELECT COUNT(*), MIN(created_at), MAX(created_at)
                FROM transcript_downloads WHERE user_id = ?
            """, (user_id,))
            
            total_data = cursor.fetchone()
            total_downloads = total_data[0] if total_data else 0
            first_ever_download = total_data[1] if total_data else None
            last_download_date = total_data[2] if total_data else None
            
            conn.close()
        
        # Get subscription limits and features
        limits = get_subscription_limits(subscription_tier)
        
        # Calculate usage percentage
        monthly_limit = limits['monthly_downloads']
        usage_percentage = (current_month_downloads / monthly_limit * 100) if monthly_limit != float('inf') and monthly_limit > 0 else 0
        
        # Calculate user engagement metrics
        account_age_days = (now - user_created_dt).days if user_created_dt else 0
        avg_downloads_per_day = total_downloads / max(account_age_days, 1)
        
        # Format file sizes helper
        def format_size(size_bytes):
            if not size_bytes:
                return "0 B"
            if size_bytes < 1024:
                return f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
        
        return {
            "user_id": user_id,
            "email": email,
            "account_created": user_created.isoformat() if hasattr(user_created, 'isoformat') else str(user_created),
            "account_age_days": account_age_days,
            "subscription": {
                "tier": subscription_tier,
                "status": subscription_status,
                "is_active": is_active,
                "start_date": start_date.isoformat() if hasattr(start_date, 'isoformat') else str(start_date),
                "end_date": end_date.isoformat() if hasattr(end_date, 'isoformat') else str(end_date),
                "days_remaining": days_remaining,
                "days_since_start": days_since_start,
                "auto_renewal": subscription_status == 'active'
            },
            "current_month": {
                "downloads": current_month_downloads,
                "usage_percentage": round(usage_percentage, 1),
                "remaining_downloads": max(0, monthly_limit - current_month_downloads) if monthly_limit != float('inf') else 'unlimited'
            },
            "lifetime_stats": {
                "total_downloads": total_downloads,
                "first_download": first_ever_download.isoformat() if hasattr(first_ever_download, 'isoformat') else str(first_ever_download) if first_ever_download else None,
                "last_download": last_download_date.isoformat() if hasattr(last_download_date, 'isoformat') else str(last_download_date) if last_download_date else None,
                "avg_downloads_per_day": round(avg_downloads_per_day, 2)
            },
            "limits": {k: (v if v != float('inf') else 'unlimited') for k, v in limits.items()},
            "features": {
                "bulk_downloads": limits.get('bulk_downloads', False),
                "priority_processing": limits.get('priority_processing', False),
                "advanced_formats": limits.get('advanced_formats', False),
                "api_access": limits.get('api_access', False),
                "export_formats": limits.get('export_formats', ['txt']),
                "concurrent_downloads": limits.get('concurrent_downloads', 1)
            },
            "status_summary": {
                "can_download": is_active and (current_month_downloads < monthly_limit or monthly_limit == float('inf')),
                "limit_reached": current_month_downloads >= monthly_limit and monthly_limit != float('inf'),
                "needs_upgrade": subscription_tier == 'free' and current_month_downloads >= monthly_limit,
                "expires_soon": days_remaining is not None and days_remaining <= 7,
                "heavy_user": avg_downloads_per_day > 5,
                "new_user": account_age_days <= 30
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting subscription status: {e}")
        return {"error": str(e)}

#===========================================================
# ENHANCED TRANSCRIPT FUNCTIONS
#===========================================================

# FIXED TRANSCRIPT EXTRACTION - Simplified and more reliable
def get_youtube_transcript_corrected(video_id: str, clean: bool = True) -> str:
    """
    FIXED: Simplified YouTube transcript extraction focusing on reliability
    """
    logger.info(f"🎯 FIXED transcript extraction for: {video_id}")
    
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # METHOD 1: Simple direct extraction with just English
        try:
            logger.info("🔄 Trying simple English extraction...")
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            
            if transcript_list and len(transcript_list) > 0:
                logger.info(f"✅ SUCCESS: Got {len(transcript_list)} transcript segments")
                return format_transcript_enhanced(transcript_list, clean)
                
        except Exception as e:
            logger.info(f"📝 Direct English extraction failed: {str(e)}")
        
        # METHOD 2: List transcripts and find English
        try:
            logger.info("🔄 Trying transcript list method...")
            transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Find English transcript (manual or auto-generated)
            english_transcript = None
            
            # Look for manual English transcript first
            for transcript in transcript_list_obj:
                if transcript.language_code == 'en':
                    english_transcript = transcript
                    logger.info(f"🎯 Found manual English transcript")
                    break
            
            # If no manual transcript, try auto-generated
            if not english_transcript:
                try:
                    english_transcript = transcript_list_obj.find_generated_transcript(['en'])
                    logger.info("🤖 Found auto-generated English transcript")
                except:
                    pass
            
            # Fetch and format the transcript
            if english_transcript:
                transcript_data = english_transcript.fetch()
                if transcript_data and len(transcript_data) > 0:
                    logger.info(f"✅ LIST SUCCESS: Got {len(transcript_data)} segments")
                    return format_transcript_enhanced(transcript_data, clean)
                        
        except Exception as e:
            logger.info(f"📝 List method failed: {str(e)}")
        
        # METHOD 3: Try any available language and translate
        try:
            logger.info("🔄 Trying any available transcript...")
            transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Get any available transcript
            for transcript in transcript_list_obj:
                try:
                    logger.info(f"🔄 Trying transcript in: {transcript.language_code}")
                    transcript_data = transcript.fetch()
                    if transcript_data and len(transcript_data) > 0:
                        logger.info(f"✅ SUCCESS with {transcript.language_code}: {len(transcript_data)} segments")
                        return format_transcript_enhanced(transcript_data, clean)
                except Exception as inner_e:
                    logger.info(f"📝 Failed with {transcript.language_code}: {str(inner_e)}")
                    continue
                        
        except Exception as e:
            logger.info(f"📝 Any language method failed: {str(e)}")
        
        # If all methods fail, log the video ID and return demo content
        logger.warning(f"⚠️ All extraction methods failed for {video_id}")
        logger.info(f"🔄 Falling back to demo content for testing")
        return get_demo_content(clean)
        
    except Exception as e:
        logger.error(f"💥 Critical error in transcript extraction for {video_id}: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"Unable to extract transcript for video {video_id}. The video may not have captions enabled, be private, or be unavailable."
        )

# FIXED FORMAT FUNCTION - Better error handling
def format_transcript_enhanced(transcript_list: list, clean: bool = True) -> str:
    """
    FIXED: Enhanced transcript formatting with better error handling
    """
    if not transcript_list:
        logger.warning("📝 Empty transcript list received")
        raise Exception("Empty transcript data")
    
    logger.info(f"🔄 Formatting {len(transcript_list)} transcript segments (clean={clean})")
    
    try:
        if clean:
            # Enhanced clean format - create readable paragraph text
            texts = []
            for item in transcript_list:
                text = item.get('text', '').strip()
                if text:
                    # Clean and normalize the text
                    text = clean_transcript_text(text)
                    if text and len(text) > 1:  # Only add meaningful text
                        texts.append(text)
            
            if not texts:
                logger.warning("📝 No valid text found in transcript segments")
                raise Exception("No valid text content found")
            
            # Join all text segments into one continuous string
            result = ' '.join(texts)
            
            # Final cleanup for better readability
            result = ' '.join(result.split())  # Normalize whitespace
            result = result.replace(' .', '.').replace(' ,', ',')  # Fix punctuation spacing
            
            # Validate that we have meaningful content
            if len(result) < 10:  # Too short, likely invalid
                logger.warning(f"📝 Transcript too short: {len(result)} characters")
                raise Exception("Transcript too short or invalid")
                
            logger.info(f"✅ Clean transcript formatted: {len(result)} characters")
            return result
        else:
            # Enhanced timestamped format - each line has [MM:SS] timestamp
            lines = []
            for item in transcript_list:
                start = item.get('start', 0)
                text = item.get('text', '').strip()
                if text:
                    # Clean the text but preserve timestamps
                    text = clean_transcript_text(text)
                    if text and len(text) > 1:
                        # Convert seconds to MM:SS format
                        minutes = int(start // 60)
                        seconds = int(start % 60)
                        timestamp = f"[{minutes:02d}:{seconds:02d}]"
                        lines.append(f"{timestamp} {text}")
            
            if not lines:
                logger.warning("📝 No valid timestamped lines created")
                raise Exception("No valid timestamped content")
            
            # Validate that we have enough content
            if len(lines) < 3:  # Too few lines, likely invalid
                logger.warning(f"📝 Too few transcript lines: {len(lines)}")
                # Don't fail completely, just warn
            
            result = '\n'.join(lines)
            logger.info(f"✅ Timestamped transcript formatted: {len(lines)} lines")
            return result
            
    except Exception as e:
        logger.error(f"❌ Error formatting transcript: {str(e)}")
        raise Exception(f"Failed to format transcript: {str(e)}")

# IMPROVED CLEAN FUNCTION
def clean_transcript_text(text: str) -> str:
    """
    IMPROVED: Clean transcript text with better handling
    """
    if not text:
        return ""
    
    try:
        # Fix common HTML entities
        text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('\n', ' ').replace('\r', ' ')
        
        # Remove HTML/XML tags (like <c>, <i>, etc.)
        import re
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove common transcript artifacts
        text = re.sub(r'\[.*?\]', '', text)  # Remove [Music], [Applause], etc.
        text = re.sub(r'\(.*?\)', '', text)  # Remove (inaudible), etc.
        
        # Normalize whitespace (replace multiple spaces with single space)
        text = ' '.join(text.split())
        
        # Remove leading/trailing punctuation artifacts that don't belong
        text = text.strip('.,!?;: -')
        
        return text.strip()
    except Exception as e:
        logger.warning(f"📝 Error cleaning text: {str(e)}")
        return text.strip() if text else ""

# SIMPLIFIED DEMO CONTENT
def get_demo_content(clean: bool) -> str:
    """
    IMPROVED: Better demo content that clearly indicates it's a fallback
    """
    demo_text = """[DEMO CONTENT - Transcript extraction failed] 
    
    This is demonstration content because the actual YouTube transcript could not be extracted. 
    The video may not have captions available, may be private, or there might be a temporary issue with YouTube's transcript service.
    
    Please try:
    1. A different video with confirmed captions
    2. One of the working example videos provided
    3. Checking if the video has captions enabled on YouTube
    
    If you continue to see this message, please contact support."""
    
    if clean:
        return demo_text
    else:
        # Add timestamps for demonstration
        lines = [
            "[00:00] [DEMO CONTENT - Transcript extraction failed]",
            "[00:05] This is demonstration content because the actual YouTube transcript",
            "[00:10] could not be extracted. The video may not have captions available,",
            "[00:15] may be private, or there might be a temporary issue",
            "[00:20] with YouTube's transcript service.",
            "[00:25] Please try a different video with confirmed captions",
            "[00:30] or one of the working example videos provided."
        ]
        return '\n'.join(lines)

#=======================================================================
# API ENDPOINTS: ALL THE ENDPOINTS OF THE YOUTUBE TRANS DOWNLOADER API #
#=======================================================================

@app.get("/")
async def root():
    return {"message": "YouTube Transcript Downloader API", "status": "running", "version": "1.0.0"}

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user(db, user_data.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    email_exists = get_user_by_email(db, user_data.email)
    if email_exists:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        created_at=datetime.now()
    )
    
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        logger.info(f"User registered successfully: {user_data.username}")
        return new_user
    except Exception as e:
        db.rollback()
        logger.error(f"Error registering user: {str(e)}")
        raise HTTPException(status_code=500, detail="Error registering user")

@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in successfully: {form_data.username}")
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/download_transcript/")
async def download_transcript_corrected(
    request: TranscriptRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Enhanced transcript downloader with robust extraction methods
    """
    video_id = request.youtube_id.strip()
    
    # Extract video ID from various YouTube URL formats
    if 'youtube.com' in video_id or 'youtu.be' in video_id:
        patterns = [
            r'(?:youtube\.com\/watch\?v=)([^&\n?#]+)',      # Standard watch URL
            r'(?:youtu\.be\/)([^&\n?#]+)',                  # Short URL
            r'(?:youtube\.com\/embed\/)([^&\n?#]+)',        # Embed URL
            r'(?:youtube\.com\/shorts\/)([^&\n?#]+)',       # Shorts URL
            r'[?&]v=([^&\n?#]+)'                            # Video parameter
        ]
        
        for pattern in patterns:
            match = re.search(pattern, video_id)
            if match:
                video_id = match.group(1)[:11]  # YouTube IDs are 11 characters
                logger.info(f"✅ Extracted video ID: {video_id}")
                break
    
    # Validate video ID format
    if not video_id or len(video_id) != 11:
        raise HTTPException(
            status_code=400, 
            detail="Invalid YouTube video ID. Please provide a valid 11-character video ID or full YouTube URL."
        )
    
    logger.info(f"🎯 ENHANCED transcript request for: {video_id}")
    
    # Check subscription limits based on transcript type
    transcript_type = "clean" if request.clean_transcript else "unclean"
    limit_check = check_subscription_limit(user.id, transcript_type, db)
    
    if not limit_check.get("allowed", False):
        raise HTTPException(
            status_code=403, 
            detail=f"You've reached your monthly limit for {transcript_type} transcripts. Please upgrade your plan."
        )
    
    # Extract transcript using enhanced method with multiple fallbacks
    try:
        transcript_text = get_youtube_transcript_corrected(video_id, clean=request.clean_transcript)
        
        # Validate that we got meaningful content
        if not transcript_text or len(transcript_text.strip()) < 10:
            raise HTTPException(
                status_code=404,
                detail=f"No transcript content found for video {video_id}."
            )
        
        # Record successful download for usage tracking
        new_download = TranscriptDownload(
            user_id=user.id,
            youtube_id=video_id,
            transcript_type=transcript_type,
            created_at=datetime.now()
        )
        
        db.add(new_download)
        db.commit()
        
        logger.info(f"🎉 ENHANCED SUCCESS: {user.username} downloaded {len(transcript_text)} chars for {video_id}")
        
        return {
            "transcript": transcript_text,
            "youtube_id": video_id,
            "message": "Transcript downloaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"💥 Enhanced extraction failed: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract transcript for video {video_id}. Error: {str(e)}"
        )

@app.get("/subscription_status/")
async def get_subscription_status_ultra_safe(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Ultra-safe subscription status - handles all database issues gracefully"""
    try:
        # Get basic subscription info
        subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id
        ).first()
        
        # Determine current subscription tier
        if not subscription:
            tier = "free"
            status = "inactive"
            expiry_date = None
        else:
            # Check if subscription is expired
            if hasattr(subscription, 'expiry_date') and subscription.expiry_date and subscription.expiry_date < datetime.now():
                tier = "free"
                status = "expired"
                expiry_date = subscription.expiry_date
            else:
                tier = subscription.tier if subscription.tier else "free"
                status = "active" if tier != "free" else "inactive"
                expiry_date = subscription.expiry_date if hasattr(subscription, 'expiry_date') else None
        
        # Calculate usage for current month using safe method
        month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Safe usage calculation with multiple fallbacks
        def get_safe_usage(transcript_type):
            try:
                # Method 1: Try ORM query
                return db.query(TranscriptDownload).filter(
                    TranscriptDownload.user_id == current_user.id,
                    TranscriptDownload.transcript_type == transcript_type,
                    TranscriptDownload.created_at >= month_start
                ).count()
            except Exception:
                try:
                    # Method 2: Raw SQL fallback
                    result = db.execute(
                        "SELECT COUNT(*) FROM transcript_downloads WHERE user_id = ? AND transcript_type = ? AND created_at >= ?",
                        (current_user.id, transcript_type, month_start.isoformat())
                    )
                    return result.scalar() or 0
                except Exception:
                    # Method 3: Ultimate fallback
                    logger.warning(f"All usage queries failed for {transcript_type}, returning 0")
                    return 0
        
        # Get usage for all types safely
        clean_usage = get_safe_usage("clean")
        unclean_usage = get_safe_usage("unclean")
        audio_usage = get_safe_usage("audio")
        video_usage = get_safe_usage("video")
        
        # Get limits based on current tier
        limits = SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS["free"])
        
        # Convert infinity to string for JSON serialization
        json_limits = {}
        for key, value in limits.items():
            if value == float('inf'):
                json_limits[key] = 'unlimited'
            else:
                json_limits[key] = value
        
        logger.info(f"✅ Safe subscription status for {current_user.username}: tier={tier}, status={status}")
        
        return {
            "tier": tier,
            "status": status,
            "usage": {
                "clean_transcripts": clean_usage,
                "unclean_transcripts": unclean_usage,
                "audio_downloads": audio_usage,
                "video_downloads": video_usage,
            },
            "limits": json_limits,
            "subscription_id": subscription.payment_id if subscription and hasattr(subscription, 'payment_id') else None,
            "stripe_customer_id": getattr(current_user, 'stripe_customer_id', None),
            "current_period_end": expiry_date.isoformat() if expiry_date else None
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting subscription status: {str(e)}")
        # Return safe defaults on any error
        return {
            "tier": "free",
            "status": "inactive", 
            "usage": {
                "clean_transcripts": 0,
                "unclean_transcripts": 0,
                "audio_downloads": 0,
                "video_downloads": 0,
            },
            "limits": {
                "clean_transcripts": 5,
                "unclean_transcripts": 3,
                "audio_downloads": 2,
                "video_downloads": 1
            },
            "subscription_id": None,
            "stripe_customer_id": None,
            "current_period_end": None
        }

@app.get("/subscription_status_enhanced/")
async def get_subscription_status_enhanced_endpoint(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    NEW ENDPOINT: Enhanced subscription status with detailed analytics
    """
    try:
        enhanced_status = get_subscription_status_enhanced(current_user.id, db)
        return enhanced_status
    except Exception as e:
        logger.error(f"❌ Error getting enhanced subscription status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve enhanced subscription status"
        )

@app.post("/confirm_payment/")
async def confirm_payment_endpoint(
    request: ConfirmPaymentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enhanced payment confirmation with complete Stripe integration"""
    try:
        # Retrieve payment intent from Stripe
        intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
        
        if intent.status != 'succeeded':
            raise HTTPException(status_code=400, detail=f"Payment not completed. Status: {intent.status}")

        # Get plan details
        plan_type = intent.metadata.get('plan_type', 'pro')
        price_id = intent.metadata.get('price_id')
        
        # Create or update subscription record
        user_subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id
        ).first()

        if not user_subscription:
            user_subscription = Subscription(
                user_id=current_user.id,
                tier=plan_type,
                start_date=datetime.utcnow(),
                expiry_date=datetime.utcnow() + timedelta(days=30),
                payment_id=request.payment_intent_id,
                auto_renew=True,
                stripe_price_id=price_id,
                status='active',
                current_period_start=datetime.utcnow(),
                current_period_end=datetime.utcnow() + timedelta(days=30)
            )
            db.add(user_subscription)
        else:
            user_subscription.tier = plan_type
            user_subscription.start_date = datetime.utcnow()
            user_subscription.expiry_date = datetime.utcnow() + timedelta(days=30)
            user_subscription.payment_id = request.payment_intent_id
            user_subscription.auto_renew = True
            user_subscription.stripe_price_id = price_id
            user_subscription.status = 'active'
            user_subscription.current_period_start = datetime.utcnow()
            user_subscription.current_period_end = datetime.utcnow() + timedelta(days=30)

        # Update user's Stripe customer ID if not already set
        if hasattr(current_user, 'stripe_customer_id') and not current_user.stripe_customer_id:
            current_user.stripe_customer_id = intent.customer

        db.commit()
        db.refresh(user_subscription)

        logger.info(f"🎉 Payment confirmed: User {current_user.username} upgraded to {plan_type}")

        return {
            'success': True,
            'subscription_tier': user_subscription.tier,
            'expires_at': user_subscription.expiry_date.isoformat(),
            'status': 'active',
            'payment_id': request.payment_intent_id,
            'customer_id': intent.customer
        }

    except Exception as e:
        logger.error(f"❌ Payment confirmation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to confirm payment: {str(e)}")

#========================
# Health check endpoints
# =======================
@app.get("/health/")
async def health_check():
    return {
        "status": "healthy",
        "stripe_configured": bool(os.getenv("STRIPE_SECRET_KEY")),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok", "version": "1.0.0"}

#======================
# Library info endpoint
#======================
@app.get("/library_info")
async def library_info():
    try:
        from youtube_transcript_api import __version__
        return {"youtube_transcript_api_version": __version__}
    except:
        return {"youtube_transcript_api_version": "unknown"}

@app.get("/test_videos")
async def get_test_videos():
    """Get list of verified working video IDs for testing"""
    working_videos = [
        {
            "id": "dQw4w9WgXcQ",
            "title": "Rick Astley - Never Gonna Give You Up",
            "description": "Classic music video with reliable captions"
        },
        {
            "id": "jNQXAC9IVRw", 
            "title": "Me at the zoo",
            "description": "First YouTube video ever uploaded"
        }
    ]
    
    return {
        "message": "These video IDs have been verified to work for transcript extraction",
        "videos": working_videos,
        "usage": "Use any of these video IDs to test your transcript downloader",
        "note": "These examples include demo content fallback for testing"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)



######################################################################
### IMPORTANT MESSAGE: DO NOT ALTER THIS MAIN.PY ANYMORE-- THANKS! ###
######################################################################   

# from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Request, Response
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from sqlalchemy.orm import Session
# from datetime import datetime, timedelta
# from typing import Optional, List
# import jwt
# from jwt.exceptions import PyJWTError
# from pydantic import BaseModel
# from passlib.context import CryptContext
# import stripe
# import os
# import logging
# from dotenv import load_dotenv
# import re

# import warnings
# warnings.filterwarnings("ignore", message=".*bcrypt.*")

# # Import from database.py
# from database import get_db, User, Subscription, TranscriptDownload, create_tables

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("youtube_trans_downloader.main")

# # Stripe configuration
# stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
# endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
# DOMAIN = os.getenv("DOMAIN", "https://youtube-trans-downloader-api.onrender.com")

# # Create FastAPI app
# app = FastAPI(
#     title="YouTubeTransDownloader API", 
#     description="API for downloading and processing YouTube video transcripts",
#     version="1.0.0"
# )

# # Environment-aware configuration
# ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# # Configure CORS based on environment
# if ENVIRONMENT == "production":
#     allowed_origins = [
#         "http://localhost:8000",
#         "https://youtube-trans-downloader-api.onrender.com",
#         FRONTEND_URL
#     ]
#     logger.info(f"🌍 Production mode - CORS origins: {allowed_origins}")
# else:
#     allowed_origins = [
#         "http://localhost:3000",
#         "http://127.0.0.1:3000",
#         FRONTEND_URL
#     ]
#     logger.info(f"🔧 Development mode - CORS origins: {allowed_origins}")

# # CORS MIDDLEWARE
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=allowed_origins,
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
#     allow_headers=["*"],
# )

# # Authentication setup
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # Constants
# SECRET_KEY = os.getenv("SECRET_KEY")
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# # Enhanced subscription limits
# SUBSCRIPTION_LIMITS = {
#     "free": {
#         "transcript": 5, "audio": 2, "video": 1, "clean": 5, "unclean": 3,
#         "clean_transcripts": 5, "unclean_transcripts": 3, 
#         "audio_downloads": 2, "video_downloads": 1
#     },
#     "pro": {
#         "transcript": 100, "audio": 50, "video": 20, "clean": 100, "unclean": 50,
#         "clean_transcripts": 100, "unclean_transcripts": 50,
#         "audio_downloads": 50, "video_downloads": 20
#     },
#     "premium": {
#         "transcript": float('inf'), "audio": float('inf'), "video": float('inf'), 
#         "clean": float('inf'), "unclean": float('inf'),
#         "clean_transcripts": float('inf'), "unclean_transcripts": float('inf'),
#         "audio_downloads": float('inf'), "video_downloads": float('inf')
#     }
# }

# # Price ID mapping
# PRICE_ID_MAP = {
#     "pro": os.getenv("PRO_PRICE_ID"),
#     "premium": os.getenv("PREMIUM_PRICE_ID")
# }

# @app.on_event("startup")
# async def startup_event():
#     """Enhanced startup with environment validation"""
#     try:
#         logger.info("🚀 Starting YouTube Transcript Downloader API...")
#         logger.info(f"🌍 Environment: {ENVIRONMENT}")
#         logger.info(f"🔗 Domain: {DOMAIN}")
        
#         # Validate critical environment variables
#         required_vars = {
#             "SECRET_KEY": "JWT secret key",
#             "STRIPE_SECRET_KEY": "Stripe secret key",
#         }
        
#         missing_vars = []
#         for var, description in required_vars.items():
#             value = os.getenv(var)
#             if not value:
#                 missing_vars.append(f"{var} ({description})")
#             else:
#                 logger.info(f"✅ {var}: {value[:8]}..." if len(value) > 8 else f"✅ {var}: SET")
        
#         if missing_vars:
#             logger.error(f"❌ Missing required environment variables:")
#             for var in missing_vars:
#                 logger.error(f"   - {var}")
#             raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
        
#         # Initialize database
#         create_tables()
#         logger.info("✅ Database initialized successfully")
#         logger.info("🎉 Application startup complete!")
        
#     except Exception as e:
#         logger.error(f"❌ Startup failed: {str(e)}")
#         raise

# #===============================================#
# # PYDANTIC MODELS: USER ACCOUNT RELATED CLASSES #
# #===============================================#

# class Token(BaseModel):
#     access_token: str
#     token_type: str

# class TokenData(BaseModel):
#     username: Optional[str] = None

# class UserCreate(BaseModel):
#     username: str
#     email: str
#     password: str

# class UserResponse(BaseModel):
#     id: int
#     username: str
#     email: str
#     created_at: datetime
    
#     class Config:
#         from_attributes = True

# class TranscriptRequest(BaseModel):
#     youtube_id: str
#     clean_transcript: bool = False

# class CreatePaymentIntentRequest(BaseModel):
#     price_id: str

# class ConfirmPaymentRequest(BaseModel):
#     payment_intent_id: str

# class SubscriptionRequest(BaseModel):
#     token: Optional[str] = None
#     subscription_tier: str

# class SubscriptionResponse(BaseModel):
#     tier: str
#     status: str
#     expiry_date: Optional[str] = None
#     limits: dict
#     usage: Optional[dict] = None
#     remaining: Optional[dict] = None
    
#     class Config:
#         from_attributes = True

# #==================================================#
# # HELPER FUNCTIONS: USER ACCOUNT RELATED FUNCTIONS #
# #==================================================#

# def verify_password(plain_password, hashed_password):
#     return pwd_context.verify(plain_password, hashed_password)

# def get_password_hash(password):
#     return pwd_context.hash(password)

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
#     to_encode = data.copy()
#     if expires_delta:
#         expire = datetime.utcnow() + expires_delta
#     else:
#         expire = datetime.utcnow() + timedelta(minutes=15)
#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
#     return encoded_jwt

# def get_user(db: Session, username: str):
#     return db.query(User).filter(User.username == username).first()

# def get_user_by_email(db: Session, email: str):
#     return db.query(User).filter(User.email == email).first()

# def authenticate_user(db: Session, username: str, password: str):
#     user = get_user(db, username)
#     if not user:
#         return False
#     if not verify_password(password, user.hashed_password):
#         return False
#     return user

# def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Invalid authentication credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
    
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")
#         if username is None:
#             raise credentials_exception
#     except jwt.PyJWTError:
#         raise credentials_exception
        
#     user = get_user(db, username)
#     if user is None:
#         raise credentials_exception
#     return user

# # Enhanced Stripe integration functions for main.py
# # Add these enhanced functions to replace the existing ones

# def get_or_create_stripe_customer(user, db: Session):
#     """
#     Enhanced Stripe customer creation with better tracking
#     This ensures customers appear in your Stripe dashboard automatically
#     """
#     try:
#         # Check if user already has a Stripe customer ID
#         if hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
#             try:
#                 # Verify the customer still exists in Stripe
#                 customer = stripe.Customer.retrieve(user.stripe_customer_id)
#                 logger.info(f"✅ Found existing Stripe customer: {customer.id} for user {user.username}")
#                 return customer
#             except stripe.error.InvalidRequestError:
#                 logger.info(f"📝 Stripe customer {user.stripe_customer_id} not found, creating new one")
#                 pass
        
#         # Create new Stripe customer
#         logger.info(f"🔄 Creating new Stripe customer for user: {user.username}")
#         customer = stripe.Customer.create(
#             email=user.email,
#             name=user.username,
#             description=f"YouTube Transcript Downloader user: {user.username}",
#             metadata={
#                 'user_id': str(user.id),
#                 'username': user.username,
#                 'signup_date': user.created_at.isoformat() if user.created_at else None,
#                 'app_name': 'YouTube Transcript Downloader'
#             }
#         )
        
#         # Save the Stripe customer ID to our database
#         if hasattr(user, 'stripe_customer_id'):
#             user.stripe_customer_id = customer.id
#             db.commit()
#             db.refresh(user)
#             logger.info(f"✅ Stripe customer created and saved: {customer.id}")
        
#         return customer
        
#     except Exception as e:
#         logger.error(f"❌ Error creating Stripe customer for user {user.username}: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to create payment customer. Please try again."
#         )


# #=========================================================== #
# # ENHANCED TRANSCRIPT FUNCTIONS: TRANSCRIPT RELATED FUNCTIONS #
# #===========================================================  #

# def check_subscription_limit(user_id: int, transcript_type: str, db: Session):
#     """Robust subscription limit check - handles missing columns gracefully"""
#     try:
#         # Get subscription info
#         subscription = db.query(Subscription).filter(Subscription.user_id == user_id).first()
        
#         if not subscription:
#             tier = "free"
#         else:
#             tier = subscription.tier
#             if subscription.expiry_date < datetime.now():
#                 tier = "free"
        
#         # Calculate current month start
#         month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
#         # Use a simple query that only uses basic columns that definitely exist
#         try:
#             # Try the enhanced query first
#             usage = db.query(TranscriptDownload).filter(
#                 TranscriptDownload.user_id == user_id,
#                 TranscriptDownload.transcript_type == transcript_type,
#                 TranscriptDownload.created_at >= month_start
#             ).count()
#         except Exception as e:
#             logger.warning(f"Enhanced query failed, using simple fallback: {e}")
            
#             # Fallback to basic SQL query that only uses guaranteed columns
#             result = db.execute(
#                 "SELECT COUNT(*) FROM transcript_downloads WHERE user_id = ? AND transcript_type = ? AND created_at >= ?",
#                 (user_id, transcript_type, month_start.isoformat())
#             )
#             usage = result.scalar() or 0
        
#         # Get limit for this tier and transcript type
#         limit = SUBSCRIPTION_LIMITS[tier].get(transcript_type, 0)
        
#         # Return True if user can download (hasn't reached limit)
#         if limit == float('inf'):
#             return True
        
#         return usage < limit
        
#     except Exception as e:
#         logger.error(f"Error checking subscription limit: {e}")
#         # If there's any error, default to allowing free tier limits
#         return True  # Allow download in case of errors to avoid blocking users

# def get_youtube_transcript_corrected(video_id: str, clean: bool = True) -> str:
#     """
#     ENHANCED YouTube transcript extraction with robust error handling
    
#     This function tries multiple methods in order of reliability:
#     1. youtube-transcript-api (fastest, but often blocked)
#     2. list_transcripts (better language detection)
#     3. yt-dlp (most robust, works when others fail)
#     4. Direct API (fallback for edge cases)
#     5. Demo content (last resort for testing)
    
#     Args:
#         video_id: 11-character YouTube video ID
#         clean: If True, returns clean text; if False, returns timestamped format
    
#     Returns:
#         Formatted transcript text
#     """
#     logger.info(f"🎯 ENHANCED transcript extraction for: {video_id}")
    
#     try:
#         from youtube_transcript_api import YouTubeTranscriptApi
        
#         # Method 1: Try youtube-transcript-api first (fastest when it works)
#         try:
#             logger.info("🔄 Trying youtube-transcript-api...")
            
#             # Try multiple English language variants for better success rate
#             language_codes = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
#             transcript_list = None
            
#             for lang in language_codes:
#                 try:
#                     transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
#                     if transcript_list and len(transcript_list) > 0:
#                         logger.info(f"✅ SUCCESS with language {lang}: {len(transcript_list)} segments")
#                         break
#                 except Exception as lang_error:
#                     logger.info(f"📝 Language {lang} failed: {str(lang_error)}")
#                     continue
            
#             # If we found a transcript, format and return it
#             if transcript_list and len(transcript_list) > 0:
#                 return format_transcript_enhanced(transcript_list, clean)
                
#         except Exception as e:
#             logger.info(f"📝 youtube-transcript-api failed: {str(e)}")
        
#         # Method 2: Try list_transcripts approach for better language detection
#         try:
#             logger.info("🔄 Trying list_transcripts method...")
#             transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
            
#             # Try to find any English transcript (manual or auto-generated)
#             english_transcript = None
#             try:
#                 # Prefer manual transcripts over auto-generated ones
#                 for transcript in transcript_list_obj:
#                     if transcript.language_code.startswith('en'):
#                         english_transcript = transcript
#                         logger.info(f"🎯 Found manual English transcript: {transcript.language_code}")
#                         break
                
#                 # If no manual English transcript, try auto-generated
#                 if not english_transcript:
#                     try:
#                         english_transcript = transcript_list_obj.find_generated_transcript(['en'])
#                         logger.info("🤖 Found auto-generated English transcript")
#                     except:
#                         pass
                
#                 # If we found any English transcript, fetch and format it
#                 if english_transcript:
#                     transcript_data = english_transcript.fetch()
#                     if transcript_data and len(transcript_data) > 0:
#                         logger.info(f"✅ LIST API SUCCESS: {len(transcript_data)} segments")
#                         return format_transcript_enhanced(transcript_data, clean)
                        
#             except Exception as inner_e:
#                 logger.info(f"📝 English transcript lookup failed: {str(inner_e)}")
                
#         except Exception as e:
#             logger.info(f"📝 List API failed: {str(e)}")
        
#         # Method 3: Enhanced yt-dlp method (most robust, works when APIs fail)
#         try:
#             logger.info("🔄 Trying enhanced yt-dlp method...")
#             import yt_dlp
            
#             # Configure yt-dlp for subtitle extraction only
#             ydl_opts = {
#                 'writesubtitles': True,        # Download manual subtitles
#                 'writeautomaticsub': True,     # Download auto-generated subtitles
#                 'subtitleslangs': ['en', 'en-US', 'en-GB'],  # English variants (MAYBE WE NEED TO CHANGE THIS LINE????)
#                 'skip_download': True,         # Don't download video, just metadata
#                 'quiet': True,                 # Suppress most output
#                 'no_warnings': True,           # Suppress warnings
#                 'extract_flat': False,         # Get full metadata
#             }
            
#             with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#                 url = f"https://www.youtube.com/watch?v={video_id}"
                
#                 try:
#                     # Extract video metadata including subtitle information
#                     info = ydl.extract_info(url, download=False)
                    
#                     # Special handling for live streams
#                     if info.get('is_live') or info.get('live_status') == 'is_live':
#                         logger.info("🔴 Live stream detected - trying live transcript extraction")
#                         return extract_live_transcript(video_id, clean)
                    
#                     # Try manual subtitles first (usually higher quality)
#                     subtitles_found = False
#                     if 'subtitles' in info and info['subtitles']:
#                         for lang in ['en', 'en-US', 'en-GB']:
#                             if lang in info['subtitles']:
#                                 for entry in info['subtitles'][lang]:
#                                     transcript_text = extract_subtitle_from_entry(ydl, entry, clean)
#                                     if transcript_text and not is_invalid_content(transcript_text):
#                                         logger.info(f"✅ YT-DLP MANUAL SUCCESS: {len(transcript_text)} chars")
#                                         return transcript_text
#                                     subtitles_found = True
                    
#                     # Fallback to automatic captions if manual subtitles don't work
#                     if 'automatic_captions' in info and info['automatic_captions']:
#                         for lang in ['en', 'en-US', 'en-GB']:
#                             if lang in info['automatic_captions']:
#                                 for entry in info['automatic_captions'][lang]:
#                                     transcript_text = extract_subtitle_from_entry(ydl, entry, clean)
#                                     if transcript_text and not is_invalid_content(transcript_text):
#                                         logger.info(f"✅ YT-DLP AUTO SUCCESS: {len(transcript_text)} chars")
#                                         return transcript_text
#                                     subtitles_found = True
                    
#                     # Log if subtitles were found but couldn't be processed
#                     if subtitles_found:
#                         logger.info("📝 Subtitles found but content was invalid/empty")
                    
#                 except Exception as yt_error:
#                     logger.info(f"📝 yt-dlp extraction failed: {str(yt_error)}")
                
#         except ImportError:
#             logger.info("📝 yt-dlp not installed, skipping...")
#         except Exception as e:
#             logger.info(f"📝 yt-dlp method failed: {str(e)}")
        
#         # Method 4: Direct API approach for edge cases (placeholder for future expansion)
#         try:
#             logger.info("🔄 Trying direct API approach...")
#             transcript_data = extract_with_direct_api(video_id)
#             if transcript_data:
#                 logger.info(f"✅ DIRECT API SUCCESS: {len(transcript_data)} segments")
#                 return format_transcript_enhanced(transcript_data, clean)
#         except Exception as e:
#             logger.info(f"📝 Direct API failed: {str(e)}")
        
#         # Final fallback: Demo content for testing (only when all methods fail)
#         logger.warning(f"⚠️ All extraction methods failed for {video_id}, falling back to demo content")
#         return get_demo_content(clean)
        
#     except Exception as e:
#         logger.error(f"💥 Critical error in transcript extraction for {video_id}: {str(e)}")
#         raise HTTPException(
#             status_code=404,
#             detail=f"Unable to extract transcript for video {video_id}. The video may not have captions enabled, be private, or be unavailable."
#         )

# def format_transcript_enhanced(transcript_list: list, clean: bool = True) -> str:
#     """
#     Enhanced transcript formatting with better text processing
    
#     This function takes raw transcript data and formats it into either:
#     - Clean format: Plain text with proper spacing and punctuation
#     - Timestamped format: Text with [MM:SS] timestamps for each segment
    
#     Args:
#         transcript_list: List of transcript segments with 'text' and 'start' keys
#         clean: If True, returns clean text; if False, returns timestamped format
    
#     Returns:
#         Formatted transcript string
#     """
#     if not transcript_list:
#         raise Exception("Empty transcript data")
    
#     if clean:
#         # Enhanced clean format - create readable paragraph text
#         texts = []
#         for item in transcript_list:
#             text = item.get('text', '').strip()
#             if text:
#                 # Clean and normalize the text
#                 text = clean_transcript_text(text)
#                 if text:  # Only add if text remains after cleaning
#                     texts.append(text)
        
#         # Join all text segments into one continuous string
#         result = ' '.join(texts)
        
#         # Final cleanup for better readability
#         result = ' '.join(result.split())  # Normalize whitespace
#         result = result.replace(' .', '.').replace(' ,', ',')  # Fix punctuation spacing
        
#         # Validate that we have meaningful content
#         if len(result) < 20:  # Too short, likely invalid
#             raise Exception("Transcript too short or invalid")
            
#         logger.info(f"✅ Enhanced clean transcript: {len(result)} characters")
#         return result
#     else:
#         # Enhanced timestamped format - each line has [MM:SS] timestamp
#         lines = []
#         for item in transcript_list:
#             start = item.get('start', 0)
#             text = item.get('text', '').strip()
#             if text:
#                 # Clean the text but preserve timestamps
#                 text = clean_transcript_text(text)
#                 if text:
#                     # Convert seconds to MM:SS format
#                     minutes = int(start // 60)
#                     seconds = int(start % 60)
#                     timestamp = f"[{minutes:02d}:{seconds:02d}]"
#                     lines.append(f"{timestamp} {text}")
        
#         # Validate that we have enough content
#         if len(lines) < 5:  # Too few lines, likely invalid
#             raise Exception("Transcript has too few valid segments")
            
#         result = '\n'.join(lines)
#         logger.info(f"✅ Enhanced timestamped transcript: {len(lines)} lines")
#         return result

# def clean_transcript_text(text: str) -> str:
#     """
#     Clean transcript text from common artifacts and formatting issues
    
#     This function removes:
#     - HTML entities (&amp;, &lt;, &gt;)
#     - HTML/XML tags
#     - Extra whitespace and line breaks
#     - Leading/trailing punctuation artifacts
    
#     Args:
#         text: Raw transcript text that may contain artifacts
    
#     Returns:
#         Clean, readable text
#     """
#     if not text:
#         return ""
    
#     # Fix common HTML entities
#     text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
#     text = text.replace('\n', ' ').replace('\r', ' ')
    
#     # Remove HTML/XML tags (like <c>, <i>, etc.)
#     import re
#     text = re.sub(r'<[^>]+>', '', text)
    
#     # Normalize whitespace (replace multiple spaces with single space)
#     text = ' '.join(text.split())
    
#     # Remove leading/trailing punctuation artifacts that don't belong
#     text = text.strip('.,!?;: ')
    
#     return text.strip()

# def is_invalid_content(content: str) -> bool:
#     """
#     Check if content is invalid (M3U8 playlist, metadata, etc.)
    
#     This function identifies content that looks like:
#     - M3U8 playlists (streaming metadata)
#     - API URLs with parameters
#     - Empty or too-short content
    
#     Args:
#         content: String content to validate
    
#     Returns:
#         True if content is invalid, False if it looks like real transcript text
#     """
#     if not content or len(content.strip()) < 10:
#         return True
    
#     # List of indicators that suggest this is not transcript content
#     invalid_indicators = [
#         '#EXTM3U',                           # M3U8 playlist header
#         '#EXT-X-VERSION',                    # M3U8 version
#         '#EXT-X-PLAYLIST-TYPE',              # M3U8 playlist type
#         '#EXT-X-TARGETDURATION',             # M3U8 duration
#         '#EXTINF:',                          # M3U8 segment info
#         'https://www.youtube.com/api/timedtext', # Direct API URL
#         'sparams=',                          # URL parameters
#         'signature=',                        # URL signature
#         'caps=asr'                           # Caption parameters
#     ]
    
#     content_lower = content.lower()
#     for indicator in invalid_indicators:
#         if indicator.lower() in content_lower:
#             logger.info(f"🚫 Invalid content detected: contains '{indicator}'")
#             return True
    
#     return False

# def extract_subtitle_from_entry(ydl, entry, clean: bool) -> str:
#     """
#     Extract and parse subtitle content from yt-dlp entry
    
#     This function handles the actual downloading and parsing of subtitle files
#     from yt-dlp subtitle entries. It validates URLs, downloads content,
#     and processes different subtitle formats.
    
#     Args:
#         ydl: yt-dlp YoutubeDL instance
#         entry: Subtitle entry from yt-dlp with URL and format info
#         clean: Whether to return clean or timestamped format
    
#     Returns:
#         Formatted transcript text, or empty string if extraction fails
#     """
#     try:
#         if 'url' not in entry:
#             logger.info("📝 No URL found in subtitle entry")
#             return ""
        
#         subtitle_url = entry['url']
#         ext = entry.get('ext', 'vtt')
        
#         # Skip if URL looks like M3U8 or contains suspicious patterns
#         suspicious_patterns = ['m3u8', 'playlist', 'range=', 'sparams=', 'signature=']
#         if any(pattern in subtitle_url.lower() for pattern in suspicious_patterns):
#             logger.info(f"🚫 Skipping suspicious URL: {subtitle_url[:100]}...")
#             return ""
        
#         try:
#             # Download subtitle content using yt-dlp's urlopen method
#             logger.info(f"📥 Downloading subtitle content: {ext} format")
#             subtitle_content = ydl.urlopen(subtitle_url).read().decode('utf-8', errors='ignore')
            
#             # Check if content is valid before parsing
#             if is_invalid_content(subtitle_content):
#                 logger.info(f"🚫 Invalid subtitle content detected for {ext} format")
#                 return ""
            
#             # Parse the content based on format
#             transcript_data = parse_subtitle_content_enhanced(subtitle_content, ext)
            
#             if transcript_data and len(transcript_data) > 0:
#                 logger.info(f"✅ Successfully parsed {len(transcript_data)} transcript segments")
#                 return format_transcript_enhanced(transcript_data, clean)
#             else:
#                 logger.info(f"📝 No valid transcript data found in {ext} content")
                
#         except Exception as url_error:
#             logger.info(f"📝 URL extraction failed: {str(url_error)}")
            
#     except Exception as e:
#         logger.info(f"📝 Entry extraction failed: {str(e)}")
    
#     return ""

# def parse_subtitle_content_enhanced(content: str, format_type: str) -> list:
#     """
#     Enhanced subtitle content parsing for multiple formats
    
#     Supports parsing of:
#     - VTT/WebVTT format (most common)
#     - SRV3/JSON format (YouTube's internal format)
#     - TTML/XML format (standard subtitle format)
    
#     Args:
#         content: Raw subtitle file content
#         format_type: Format indicator (vtt, srv3, json, ttml, xml)
    
#     Returns:
#         List of transcript segments with text, start time, and duration
#     """
#     transcript_data = []
    
#     try:
#         if format_type.lower() in ['vtt', 'webvtt']:
#             transcript_data = parse_vtt_content(content)
#         elif format_type.lower() in ['srv3', 'json']:
#             transcript_data = parse_srv3_content(content)
#         elif format_type.lower() in ['ttml', 'xml']:
#             transcript_data = parse_ttml_content(content)
#         else:
#             # Try to auto-detect format based on content
#             if content.strip().startswith('WEBVTT'):
#                 transcript_data = parse_vtt_content(content)
#             elif content.strip().startswith('{') or content.strip().startswith('['):
#                 transcript_data = parse_srv3_content(content)
#             elif '<' in content and '>' in content:
#                 transcript_data = parse_ttml_content(content)
    
#     except Exception as e:
#         logger.info(f"📝 Content parsing failed: {str(e)}")
    
#     return transcript_data

# def parse_vtt_content(content: str) -> list:
#     """
#     Parse WebVTT subtitle content
    
#     WebVTT format structure:
#     WEBVTT
    
#     00:00:01.000 --> 00:00:04.000
#     Hello, this is the first subtitle
    
#     Args:
#         content: WebVTT file content
    
#     Returns:
#         List of transcript segments
#     """
#     transcript_data = []
#     lines = content.split('\n')
#     current_start = 0
    
#     for i, line in enumerate(lines):
#         line = line.strip()
        
#         # Skip metadata and empty lines
#         if not line or line.startswith('WEBVTT') or line.startswith('NOTE') or line.isdigit():
#             continue
        
#         # Parse timestamp line (contains -->)
#         if '-->' in line:
#             try:
#                 start_time = line.split(' --> ')[0].strip()
#                 current_start = parse_vtt_timestamp(start_time)
#             except:
#                 current_start = 0
        
#         # Parse text line (actual subtitle content)
#         elif line and not line.startswith('<') and '-->' not in line:
#             clean_text = clean_vtt_text(line)
#             if clean_text and len(clean_text) > 1:
#                 transcript_data.append({
#                     'text': clean_text,
#                     'start': current_start,
#                     'duration': 3.0  # Default duration
#                 })
    
#     return transcript_data

# def clean_vtt_text(text: str) -> str:
#     """
#     Clean VTT text from formatting tags and artifacts
    
#     Removes VTT-specific formatting like:
#     - <c> color tags
#     - <i> italic tags
#     - {style} annotations
#     - HTML entities
    
#     Args:
#         text: Raw VTT text line
    
#     Returns:
#         Clean text content
#     """
#     # Remove VTT formatting tags
#     import re
#     text = re.sub(r'<[^>]+>', '', text)  # Remove HTML/VTT tags
#     text = re.sub(r'\{[^}]+\}', '', text)  # Remove style annotations
#     text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
#     return text.strip()

# def parse_vtt_timestamp(timestamp: str) -> float:
#     """
#     Parse VTT timestamp format to seconds
    
#     Supports formats:
#     - HH:MM:SS.mmm (hours:minutes:seconds.milliseconds)
#     - MM:SS.mmm (minutes:seconds.milliseconds)
    
#     Args:
#         timestamp: Timestamp string from VTT file
    
#     Returns:
#         Time in seconds as float
#     """
#     try:
#         # Handle both comma and dot as decimal separator
#         parts = timestamp.replace(',', '.').split(':')
#         if len(parts) == 3:
#             hours, minutes, seconds = parts
#             return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
#         elif len(parts) == 2:
#             minutes, seconds = parts
#             return int(minutes) * 60 + float(seconds)
#     except:
#         pass
#     return 0

# def parse_srv3_content(content: str) -> list:
#     """
#     Parse SRV3/JSON subtitle content (YouTube's internal format)
    
#     SRV3 format structure:
#     {
#       "events": [
#         {
#           "tStartMs": 1000,
#           "dDurationMs": 3000,
#           "segs": [{"utf8": "Hello world"}]
#         }
#       ]
#     }
    
#     Args:
#         content: SRV3/JSON file content
    
#     Returns:
#         List of transcript segments
#     """
#     transcript_data = []
#     try:
#         import json
#         data = json.loads(content)
        
#         if 'events' in data:
#             for event in data['events']:
#                 if 'segs' in event:
#                     text_segments = []
#                     for seg in event['segs']:
#                         if 'utf8' in seg:
#                             text_segments.append(seg['utf8'])
                    
#                     if text_segments:
#                         full_text = ''.join(text_segments).strip()
#                         if full_text and len(full_text) > 1:
#                             transcript_data.append({
#                                 'text': full_text,
#                                 'start': event.get('tStartMs', 0) / 1000.0,  # Convert ms to seconds
#                                 'duration': event.get('dDurationMs', 3000) / 1000.0
#                             })
#     except:
#         pass
    
#     return transcript_data

# def parse_ttml_content(content: str) -> list:
#     """
#     Parse TTML/XML subtitle content
    
#     TTML is a standard XML-based subtitle format used by many platforms.
    
#     Args:
#         content: TTML/XML file content
    
#     Returns:
#         List of transcript segments
#     """
#     transcript_data = []
#     try:
#         import xml.etree.ElementTree as ET
#         root = ET.fromstring(content)
        
#         # Find all text elements with timing information
#         for elem in root.iter():
#             if elem.text and elem.text.strip():
#                 start_time = 0
#                 if 'begin' in elem.attrib:
#                     start_time = parse_ttml_timestamp(elem.attrib['begin'])
                
#                 clean_text = elem.text.strip()
#                 if clean_text and len(clean_text) > 1:
#                     transcript_data.append({
#                         'text': clean_text,
#                         'start': start_time,
#                         'duration': 3.0  # Default duration
#                     })
#     except:
#         pass
    
#     return transcript_data

# def parse_ttml_timestamp(timestamp: str) -> float:
#     """
#     Parse TTML timestamp format to seconds
    
#     TTML supports various timestamp formats:
#     - "10s" (seconds)
#     - "01:30:45" (HH:MM:SS)
    
#     Args:
#         timestamp: TTML timestamp string
    
#     Returns:
#         Time in seconds as float
#     """
#     try:
#         if 's' in timestamp:
#             return float(timestamp.replace('s', ''))
#         elif ':' in timestamp:
#             parts = timestamp.split(':')
#             if len(parts) == 3:
#                 h, m, s = parts
#                 return int(h) * 3600 + int(m) * 60 + float(s)
#     except:
#         pass
#     return 0

# def extract_live_transcript(video_id: str, clean: bool) -> str:
#     """
#     Handle live stream transcript extraction
    
#     Live streams have special requirements:
#     - Transcripts may not be available during broadcast
#     - Content may be incomplete
#     - Should provide helpful feedback to users
    
#     Args:
#         video_id: YouTube video ID
#         clean: Whether to return clean or timestamped format
    
#     Returns:
#         Live transcript content or helpful message
#     """
#     logger.info(f"🔴 Attempting live transcript extraction for {video_id}")
    
#     # For live streams, try the standard API methods first
#     try:
#         from youtube_transcript_api import YouTubeTranscriptApi
#         transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
#         if transcript_list:
#             return format_transcript_enhanced(transcript_list, clean)
#     except:
#         pass
    
#     # Return informative message for live content
#     return "This appears to be a live stream. Live transcripts may not be available or may be incomplete. Please try again after the stream has ended."

# def extract_with_direct_api(video_id: str) -> list:
#     """
#     Direct API extraction method for edge cases
    
#     This is a placeholder for future implementation of additional
#     direct API methods that might become available.
    
#     Args:
#         video_id: YouTube video ID
    
#     Returns:
#         List of transcript segments (currently empty)
#     """
#     try:
#         # Placeholder for future direct API implementations
#         # Could include custom API endpoints, alternative services, etc.
#         return []
#     except:
#         return []

# def get_demo_content(clean: bool) -> str:
#     """
#     Fallback demo content for testing and when extraction fails
    
#     This provides sample content that demonstrates the expected format
#     while clearly indicating it's placeholder content.
    
#     Args:
#         clean: Whether to return clean or timestamped format
    
#     Returns:
#         Demo transcript content
#     """
#     demo_text = """Hello everyone, welcome to this video. Today we're going to be talking 
#     about some really interesting topics. We'll cover various aspects of the subject matter 
#     and provide you with valuable insights. This is sample transcript content for demonstration 
#     purposes. The actual transcript would contain the real audio content from the YouTube video. 
#     Thank you for watching, and don't forget to subscribe to our channel for more great content like this."""
    
#     if clean:
#         return demo_text
#     else:
#         # Add timestamps for unclean format demonstration
#         words = demo_text.split()
#         timestamped_lines = []
#         current_time = 0
        
#         for i in range(0, len(words), 8):  # Group words into 8-word chunks
#             chunk = ' '.join(words[i:i+8])
#             minutes = current_time // 60
#             seconds = current_time % 60
#             timestamp = f"[{minutes:02d}:{seconds:02d}]"
#             timestamped_lines.append(f"{timestamp} {chunk}")
#             current_time += 6  # Add 6 seconds per chunk
        
#         return '\n'.join(timestamped_lines)

# #=======================================================================
# # API ENDPOINTS: ALL THE ENDPOINTS OF THE YOUTUBE TRANS DOWNLOADER API #
# #=======================================================================

# @app.get("/")
# async def root():
#     return {"message": "YouTube Transcript Downloader API", "status": "running", "version": "1.0.0"}

# @app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
# def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
#     db_user = get_user(db, user_data.username)
#     if db_user:
#         raise HTTPException(status_code=400, detail="Username already registered")
    
#     email_exists = get_user_by_email(db, user_data.email)
#     if email_exists:
#         raise HTTPException(status_code=400, detail="Email already registered")
    
#     hashed_password = get_password_hash(user_data.password)
#     new_user = User(
#         username=user_data.username,
#         email=user_data.email,
#         hashed_password=hashed_password,
#         created_at=datetime.now()
#     )
    
#     try:
#         db.add(new_user)
#         db.commit()
#         db.refresh(new_user)
#         logger.info(f"User registered successfully: {user_data.username}")
#         return new_user
#     except Exception as e:
#         db.rollback()
#         logger.error(f"Error registering user: {str(e)}")
#         raise HTTPException(status_code=500, detail="Error registering user")

# @app.post("/token", response_model=Token)
# async def login_for_access_token(
#     form_data: OAuth2PasswordRequestForm = Depends(),
#     db: Session = Depends(get_db)
# ):
#     user = authenticate_user(db, form_data.username, form_data.password)
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect username or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
    
#     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token = create_access_token(
#         data={"sub": user.username}, expires_delta=access_token_expires
#     )
    
#     logger.info(f"User logged in successfully: {form_data.username}")
#     return {"access_token": access_token, "token_type": "bearer"}

# @app.get("/users/me", response_model=UserResponse)
# async def read_users_me(current_user: User = Depends(get_current_user)):
#     return current_user

# @app.post("/download_transcript/")
# async def download_transcript_corrected(
#     request: TranscriptRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Enhanced transcript downloader with robust extraction methods
    
#     This endpoint handles YouTube transcript extraction using multiple
#     fallback methods to ensure maximum success rate.
#     """
#     video_id = request.youtube_id.strip()
    
#     # Extract video ID from various YouTube URL formats
#     if 'youtube.com' in video_id or 'youtu.be' in video_id:
#         patterns = [
#             r'(?:youtube\.com\/watch\?v=)([^&\n?#]+)',      # Standard watch URL
#             r'(?:youtu\.be\/)([^&\n?#]+)',                  # Short URL
#             r'(?:youtube\.com\/embed\/)([^&\n?#]+)',        # Embed URL
#             r'(?:youtube\.com\/shorts\/)([^&\n?#]+)',       # Shorts URL
#             r'[?&]v=([^&\n?#]+)'                            # Video parameter
#         ]
        
#         for pattern in patterns:
#             match = re.search(pattern, video_id)
#             if match:
#                 video_id = match.group(1)[:11]  # YouTube IDs are 11 characters
#                 logger.info(f"✅ Extracted video ID: {video_id}")
#                 break
    
#     # Validate video ID format
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(
#             status_code=400, 
#             detail="Invalid YouTube video ID. Please provide a valid 11-character video ID or full YouTube URL."
#         )
    
#     logger.info(f"🎯 ENHANCED transcript request for: {video_id}")
    
#     # Check subscription limits based on transcript type
#     transcript_type = "clean" if request.clean_transcript else "unclean"
#     can_download = check_subscription_limit(user.id, transcript_type, db)
#     if not can_download:
#         raise HTTPException(
#             status_code=403, 
#             detail=f"You've reached your monthly limit for {transcript_type} transcripts. Please upgrade your plan."
#         )
    
#     # Extract transcript using enhanced method with multiple fallbacks
#     try:
#         transcript_text = get_youtube_transcript_corrected(video_id, clean=request.clean_transcript)
        
#         # Validate that we got meaningful content
#         if not transcript_text or len(transcript_text.strip()) < 10:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"No transcript content found for video {video_id}."
#             )
        
#         # Record successful download for usage tracking
#         new_download = TranscriptDownload(
#             user_id=user.id,
#             youtube_id=video_id,
#             transcript_type=transcript_type,
#             created_at=datetime.now()
#         )
        
#         db.add(new_download)
#         db.commit()
        
#         logger.info(f"🎉 ENHANCED SUCCESS: {user.username} downloaded {len(transcript_text)} chars for {video_id}")
        
#         return {
#             "transcript": transcript_text,
#             "youtube_id": video_id,
#             "message": "Transcript downloaded successfully"
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         db.rollback()
#         logger.error(f"💥 Enhanced extraction failed: {str(e)}")
        
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to extract transcript for video {video_id}. Error: {str(e)}"
#         )
# #==============================================

# @app.get("/subscription_status/")
# async def get_subscription_status_ultra_safe(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Ultra-safe subscription status - handles all database issues gracefully"""
#     try:
#         # Get basic subscription info
#         subscription = db.query(Subscription).filter(
#             Subscription.user_id == current_user.id
#         ).first()
        
#         # Determine current subscription tier
#         if not subscription:
#             tier = "free"
#             status = "inactive"
#             expiry_date = None
#         else:
#             # Check if subscription is expired
#             if hasattr(subscription, 'expiry_date') and subscription.expiry_date and subscription.expiry_date < datetime.now():
#                 tier = "free"
#                 status = "expired"
#                 expiry_date = subscription.expiry_date
#             else:
#                 tier = subscription.tier if subscription.tier else "free"
#                 status = "active" if tier != "free" else "inactive"
#                 expiry_date = subscription.expiry_date if hasattr(subscription, 'expiry_date') else None
        
#         # Calculate usage for current month using safe method
#         month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
#         # Safe usage calculation with multiple fallbacks
#         def get_safe_usage(transcript_type):
#             try:
#                 # Method 1: Try ORM query
#                 return db.query(TranscriptDownload).filter(
#                     TranscriptDownload.user_id == current_user.id,
#                     TranscriptDownload.transcript_type == transcript_type,
#                     TranscriptDownload.created_at >= month_start
#                 ).count()
#             except Exception:
#                 try:
#                     # Method 2: Raw SQL fallback
#                     result = db.execute(
#                         "SELECT COUNT(*) FROM transcript_downloads WHERE user_id = ? AND transcript_type = ? AND created_at >= ?",
#                         (current_user.id, transcript_type, month_start.isoformat())
#                     )
#                     return result.scalar() or 0
#                 except Exception:
#                     # Method 3: Ultimate fallback
#                     logger.warning(f"All usage queries failed for {transcript_type}, returning 0")
#                     return 0
        
#         # Get usage for all types safely
#         clean_usage = get_safe_usage("clean")
#         unclean_usage = get_safe_usage("unclean")
#         audio_usage = get_safe_usage("audio")
#         video_usage = get_safe_usage("video")
        
#         # Get limits based on current tier
#         limits = SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS["free"])
        
#         # Convert infinity to string for JSON serialization
#         json_limits = {}
#         for key, value in limits.items():
#             if value == float('inf'):
#                 json_limits[key] = 'unlimited'
#             else:
#                 json_limits[key] = value
        
#         logger.info(f"✅ Safe subscription status for {current_user.username}: tier={tier}, status={status}")
        
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
#         # Return safe defaults on any error
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

# @app.post("/confirm_payment/")
# async def confirm_payment_endpoint(
#     request: ConfirmPaymentRequest,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Enhanced payment confirmation with complete Stripe integration"""
#     try:
#         # Retrieve payment intent from Stripe
#         intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
        
#         if intent.status != 'succeeded':
#             raise HTTPException(status_code=400, detail=f"Payment not completed. Status: {intent.status}")

#         # Get plan details
#         plan_type = intent.metadata.get('plan_type', 'pro')
#         price_id = intent.metadata.get('price_id')
        
#         # Create or update subscription record
#         user_subscription = db.query(Subscription).filter(
#             Subscription.user_id == current_user.id
#         ).first()

#         if not user_subscription:
#             user_subscription = Subscription(
#                 user_id=current_user.id,
#                 tier=plan_type,
#                 start_date=datetime.utcnow(),
#                 expiry_date=datetime.utcnow() + timedelta(days=30),
#                 payment_id=request.payment_intent_id,
#                 auto_renew=True,
#                 stripe_price_id=price_id,
#                 status='active',
#                 current_period_start=datetime.utcnow(),
#                 current_period_end=datetime.utcnow() + timedelta(days=30)
#             )
#             db.add(user_subscription)
#         else:
#             user_subscription.tier = plan_type
#             user_subscription.start_date = datetime.utcnow()
#             user_subscription.expiry_date = datetime.utcnow() + timedelta(days=30)
#             user_subscription.payment_id = request.payment_intent_id
#             user_subscription.auto_renew = True
#             user_subscription.stripe_price_id = price_id
#             user_subscription.status = 'active'
#             user_subscription.current_period_start = datetime.utcnow()
#             user_subscription.current_period_end = datetime.utcnow() + timedelta(days=30)

#         # Record payment history
#         payment_record = PaymentHistory(
#             user_id=current_user.id,
#             stripe_payment_intent_id=request.payment_intent_id,
#             stripe_customer_id=intent.customer,
#             amount=intent.amount,
#             currency=intent.currency,
#             status='succeeded',
#             subscription_tier=plan_type,
#             created_at=datetime.utcnow(),
#             metadata=str(intent.metadata)
#         )
#         db.add(payment_record)

#         # Update user's Stripe customer ID if not already set
#         if hasattr(current_user, 'stripe_customer_id') and not current_user.stripe_customer_id:
#             current_user.stripe_customer_id = intent.customer

#         db.commit()
#         db.refresh(user_subscription)

#         logger.info(f"🎉 Payment confirmed: User {current_user.username} upgraded to {plan_type}")

#         return {
#             'success': True,
#             'subscription_tier': user_subscription.tier,
#             'expires_at': user_subscription.expiry_date.isoformat(),
#             'status': 'active',
#             'payment_id': request.payment_intent_id,
#             'customer_id': intent.customer
#         }

#     except Exception as e:
#         logger.error(f"❌ Payment confirmation error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to confirm payment: {str(e)}")

# #========================
# # Health check endpoints
# # =======================
# @app.get("/health/")
# async def health_check():
#     return {
#         "status": "healthy",
#         "stripe_configured": bool(os.getenv("STRIPE_SECRET_KEY")),
#         "timestamp": datetime.utcnow().isoformat()
#     }

# @app.get("/healthcheck")
# async def healthcheck():
#     return {"status": "ok", "version": "1.0.0"}

# #======================
# # Library info endpoint
# #======================
# @app.get("/library_info")
# async def library_info():
#     try:
#         from youtube_transcript_api import __version__
#         return {"youtube_transcript_api_version": __version__}
#     except:
#         return {"youtube_transcript_api_version": "unknown"}

# @app.get("/test_videos")
# async def get_test_videos():
#     """Get list of verified working video IDs for testing"""
#     working_videos = [
#         {
#             "id": "dQw4w9WgXcQ",
#             "title": "Rick Astley - Never Gonna Give You Up",
#             "description": "Classic music video with reliable captions"
#         },
#         {
#             "id": "jNQXAC9IVRw", 
#             "title": "Me at the zoo",
#             "description": "First YouTube video ever uploaded"
#         }
#     ]
    
#     return {
#         "message": "These video IDs have been verified to work for transcript extraction",
#         "videos": working_videos,
#         "usage": "Use any of these video IDs to test your transcript downloader",
#         "note": "These examples include demo content fallback for testing"
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

######################################################################
### IMPORTANT MESSAGE: DO NOT ALTER THIS MAIN.PY ANYMORE-- THANKS! ###
###################################################################### 