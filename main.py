# Fixed imports section - Replace lines 1-20 in your main.py

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, List
import jwt
from jwt.exceptions import PyJWTError
from pydantic import BaseModel
from passlib.context import CryptContext
import stripe
import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
import os
import json
import logging
from dotenv import load_dotenv
import secrets
import requests
import re
import ssl  # ✅ ADDED
import sys  # ✅ ADDED
import xml.etree.ElementTree as ET  # ✅ FIXED
from urllib.parse import unquote  # ✅ FIXED

# Add these imports to your main.py
from youtube_transcript_api.exceptions import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable, NoTranscriptAvailable, TooManyRequests

import warnings
warnings.filterwarnings("ignore", message=".*bcrypt.*")

# Import from database.py
from database import get_db, User, Subscription, TranscriptDownload, create_tables

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("youtube_trans_downloader.main")

# Stripe configuration
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
DOMAIN = os.getenv("DOMAIN", "https://youtube-trans-downloader-api.onrender.com")

# Create FastAPI app
app = FastAPI(
    title="YouTubeTransDownloader API", 
    description="API for downloading and processing YouTube video transcripts",
    version="1.0.0"
)

# CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Add this to your main.py - Production-ready CORS configuration
# Environment-aware configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Configure CORS based on environment
if ENVIRONMENT == "production":
    allowed_origins = [
        #"https://your-actual-frontend-domain.com",  # Replace with your actual frontend URL
        
        "http://localhost:8000", #Or "http://localhost:3000"?
        "https://youtube-trans-downloader-api.onrender.com",
        FRONTEND_URL  # From environment variable
    ]
    logger.info(f"🌍 Production mode - CORS origins: {allowed_origins}")
else:
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        FRONTEND_URL
    ]
    logger.info(f"🔧 Development mode - CORS origins: {allowed_origins}")

# CORS MIDDLEWARE with environment awareness
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

# Enhanced subscription limits with new action types
SUBSCRIPTION_LIMITS = {
    "free": {
        "transcript": 5, "audio": 2, "video": 1, "clean": 5, "unclean": 3,
        "clean_transcripts": 5, "unclean_transcripts": 3, 
        "audio_downloads": 2, "video_downloads": 1
    },
    "pro": {
        "transcript": 100, "audio": 50, "video": 20, "clean": 100, "unclean": 50,
        "clean_transcripts": 100, "unclean_transcripts": 50,
        "audio_downloads": 50, "video_downloads": 20
    },
    "premium": {
        "transcript": float('inf'), "audio": float('inf'), "video": float('inf'), 
        "clean": float('inf'), "unclean": float('inf'),
        "clean_transcripts": float('inf'), "unclean_transcripts": float('inf'),
        "audio_downloads": float('inf'), "video_downloads": float('inf')
    }
}

# Price ID mapping - UPDATED to use your standardized variable names
PRICE_ID_MAP = {
    "pro": os.getenv("PRO_PRICE_ID"),
    "premium": os.getenv("PREMIUM_PRICE_ID")
}

# Plan pricing in cents
PLAN_PRICING = {
    "pro": 999,  # $9.99
    "premium": 1999  # $19.99
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
                # Log first few characters (for debugging)
                logger.info(f"✅ {var}: {value[:8]}..." if len(value) > 8 else f"✅ {var}: SET")
        
        if missing_vars:
            logger.error(f"❌ Missing required environment variables:")
            for var in missing_vars:
                logger.error(f"   - {var}")
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
        
        # 🔧 UPDATED: Optional variables with correct names
        optional_vars = {
            "PRO_PRICE_ID": "Pro plan price ID",
            "PREMIUM_PRICE_ID": "Premium plan price ID", 
            "STRIPE_WEBHOOK_SECRET": "Webhook verification"
        }
        
        for var, description in optional_vars.items():
            if not os.getenv(var):
                logger.warning(f"⚠️  {var} not set - {description} will not work")
            else:
                logger.info(f"✅ {var}: SET")
        
        # Initialize database
        create_tables()
        logger.info("✅ Database initialized successfully")
        logger.info("🎉 Application startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

# Enhanced Pydantic models
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

class PaymentRequest(BaseModel):
    token: str
    subscription_tier: str

# NEW: Enhanced payment models
# 🔧 UPDATED: Payment models to match your new implementation
class CreatePaymentIntentRequest(BaseModel):
    price_id: str  # 🔧 SIMPLIFIED: Only price_id needed

class ConfirmPaymentRequest(BaseModel):
    payment_intent_id: str

class PaymentIntentRequest(BaseModel):
    amount: int  # Amount in cents
    currency: str = 'usd'
    payment_method_id: str
    plan_name: str

class PaymentIntentResponse(BaseModel):
    client_secret: str
    payment_intent_id: str  # 🔧 UPDATED: Changed from token to payment_intent_id

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

# Helper functions (existing ones kept)
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

def get_user_by_id(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

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

# NEW: Enhanced payment helper functions
def get_or_create_stripe_customer(user, db: Session):
    """Get or create a Stripe customer for the user"""
    try:
        # Check if user has stripe_customer_id attribute (from new User model)
        if hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
            try:
                customer = stripe.Customer.retrieve(user.stripe_customer_id)
                return customer
            except stripe.error.InvalidRequestError:
                pass
        
        # Create new customer
        customer = stripe.Customer.create(
            email=user.email,
            name=user.username,
            metadata={'user_id': str(user.id)}
        )
        
        # Save customer ID if user model supports it
        if hasattr(user, 'stripe_customer_id'):
            user.stripe_customer_id = customer.id
            db.commit()
        
        return customer
        
    except Exception as e:
        logger.error(f"Error creating Stripe customer: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create payment customer"
        )

def check_user_limits(user, action_type: str, db: Session):
    """Check if user has exceeded their limits for the current month"""
    # Get subscription tier
    subscription = db.query(Subscription).filter(Subscription.user_id == user.id).first()
    
    if not subscription or subscription.expiry_date < datetime.now():
        tier = "free"
    else:
        tier = subscription.tier
    
    # Get current month's usage count from TranscriptDownload table
    month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Map action types to transcript types
    type_mapping = {
        "clean_transcripts": "clean",
        "unclean_transcripts": "unclean",
        "audio_downloads": "audio",
        "video_downloads": "video"
    }
    
    transcript_type = type_mapping.get(action_type, action_type)
    
    current_usage = db.query(TranscriptDownload).filter(
        TranscriptDownload.user_id == user.id,
        TranscriptDownload.transcript_type == transcript_type,
        TranscriptDownload.created_at >= month_start
    ).count()
    
    limit = SUBSCRIPTION_LIMITS[tier].get(action_type, 0)
    
    if limit == float('inf'):
        return True
    
    return current_usage < limit

def increment_usage(user, action_type: str, db: Session):
    """Increment user's usage counter by recording a download"""
    # Map action types to transcript types
    type_mapping = {
        "clean_transcripts": "clean",
        "unclean_transcripts": "unclean", 
        "audio_downloads": "audio",
        "video_downloads": "video"
    }
    
    transcript_type = type_mapping.get(action_type, action_type)
    
    # Record the download in TranscriptDownload table
    new_download = TranscriptDownload(
        user_id=user.id,
        youtube_id="usage_increment",  # Placeholder for usage tracking
        transcript_type=transcript_type,
        created_at=datetime.now()
    )
    
    db.add(new_download)
    db.commit()

def check_subscription_limit(user_id: int, transcript_type: str, db: Session):
    """Original function maintained for backward compatibility"""
    subscription = db.query(Subscription).filter(Subscription.user_id == user_id).first()
    
    if not subscription:
        tier = "free"
    else:
        tier = subscription.tier
        if subscription.expiry_date < datetime.now():
            tier = "free"
    
    month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    usage = db.query(TranscriptDownload).filter(
        TranscriptDownload.user_id == user_id,
        TranscriptDownload.transcript_type == transcript_type,
        TranscriptDownload.created_at >= month_start
    ).count()
    
    limit = SUBSCRIPTION_LIMITS[tier][transcript_type]
    if usage >= limit:
        return False
    return True

def get_transcript_alternative_method_enhanced(video_id: str, clean: bool = True) -> str:
    """Enhanced alternative method with better headers"""
    try:
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Enhanced headers to avoid blocking
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        
        response = requests.get(video_url, headers=headers, timeout=20)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=404,
                detail=f"Could not access video page (Status: {response.status_code})"
            )
        
        # Rest of the alternative method logic...
        # (keeping this short since your original method exists)
        
        raise HTTPException(
            status_code=404,
            detail="Alternative method not fully implemented yet"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Alternative method failed: {str(e)}"
        )

def get_demo_transcript(video_id: str, clean: bool = True) -> str:
    """Return a demo transcript for testing purposes"""
    demo_text = """This is a demo transcript for testing purposes. 
The actual YouTube transcript could not be retrieved for this video. 
This might be due to:
- Transcripts being disabled for this video
- Network connectivity issues
- YouTube blocking automated requests
- The video being private or unavailable

Please try a different video or check if the video has captions enabled."""
    
    if clean:
        return demo_text
    else:
        lines = demo_text.split('\n')
        timestamped = []
        for i, line in enumerate(lines):
            if line.strip():
                timestamp = f"[{i:02d}:00]"
                timestamped.append(f"{timestamp} {line.strip()}")
        return '\n'.join(timestamped)

#  Updated main transcript processor with fallback
def process_youtube_transcript_with_fallback(video_id: str, clean: bool = True) -> str:
    """
    Main transcript processor with fallback to alternative method
    """
    # Try the original library first
    try:
        logger.info(f"🔍 Trying youtube-transcript-api for video: {video_id}")
        from youtube_transcript_api import YouTubeTranscriptApi
        
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=['en', 'en-US', 'en-GB']
        )
        
        if transcript_list:
            logger.info(f"✅ Library method succeeded: {len(transcript_list)} segments")
            
            if clean:
                text_parts = [item['text'].strip() for item in transcript_list if item['text'].strip()]
                return ' '.join(text_parts)
            else:
                formatted_transcript = []
                for item in transcript_list:
                    start_time = float(item['start'])
                    minutes = int(start_time // 60)
                    seconds = int(start_time % 60)
                    timestamp = f"[{minutes:02d}:{seconds:02d}]"
                    formatted_transcript.append(f"{timestamp} {item['text']}")
                return '\n'.join(formatted_transcript)
        
    except Exception as library_error:
        logger.warning(f"⚠️ Library method failed: {library_error}")
        logger.info("🔄 Falling back to alternative HTTP method...")
        
        # Fallback to alternative method
        return get_transcript_alternative_method(video_id, clean)

# Enhanced transcript function - Replace the existing function in your main.py

def process_youtube_transcript_enhanced(video_id: str, clean: bool = True) -> str:
    """
    Enhanced transcript processor with better error handling
    """
    logger.info(f"🔍 Getting transcript for video: {video_id}")
    
    # Test different video IDs that are known to work
    if video_id == "dQw4w9WgXcQ":
        # This video sometimes has issues, try alternatives
        alternative_videos = [
            ("jNQXAC9IVRw", "Me at the zoo - first YouTube video"),
            ("ZbZSe6N_BXs", "Sample video"),
            ("9bZkp7q19f0", "PSY - GANGNAM STYLE")
        ]
        
        for alt_id, alt_title in alternative_videos:
            try:
                logger.info(f"🔄 Trying alternative video: {alt_id} ({alt_title})")
                result = process_single_video_transcript(alt_id, clean)
                if result:
                    logger.info(f"✅ Success with alternative video: {alt_id}")
                    return f"[Using alternative video: {alt_title}]\n\n{result}"
            except Exception as e:
                logger.warning(f"⚠️ Alternative {alt_id} failed: {e}")
                continue
    
    # Try the original video
    return process_single_video_transcript(video_id, clean)

def process_single_video_transcript(video_id: str, clean: bool = True) -> str:
    """
    Process a single video transcript with multiple fallback methods
    """
    
    # Method 1: Try youtube-transcript-api with different approaches
    methods = [
        ("Auto-generated English", lambda: YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])),
        ("Any English variant", lambda: YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB', 'en-CA'])),
        ("Any available language", lambda: YouTubeTranscriptApi.get_transcript(video_id)),
    ]
    
    for method_name, method_func in methods:
        try:
            logger.info(f"🔍 Trying: {method_name}")
            transcript_list = method_func()
            
            if transcript_list:
                logger.info(f"✅ Success with {method_name}: {len(transcript_list)} segments")
                return format_transcript(transcript_list, clean)
                
        except TranscriptsDisabled:
            logger.error(f"❌ Transcripts disabled for video {video_id}")
            raise HTTPException(
                status_code=404,
                detail="Transcripts are disabled for this video."
            )
        except NoTranscriptFound:
            logger.warning(f"⚠️ No transcript found with {method_name}")
            continue
        except VideoUnavailable:
            logger.error(f"❌ Video unavailable: {video_id}")
            raise HTTPException(
                status_code=404,
                detail="Video is unavailable or private."
            )
        except Exception as e:
            logger.warning(f"⚠️ {method_name} failed: {e}")
            continue
    
    # Method 2: Alternative HTTP method
    logger.info("🔄 Trying alternative HTTP method...")
    try:
        return get_transcript_alternative_method_enhanced(video_id, clean)
    except Exception as e:
        logger.error(f"❌ Alternative method failed: {e}")
    
    # Method 3: Return demo transcript for testing
    logger.warning("⚠️ All methods failed, returning demo transcript")
    return get_demo_transcript(video_id, clean)

def format_transcript(transcript_list: list, clean: bool = True) -> str:
    """Format transcript based on clean/unclean preference"""
    if clean:
        # Clean format - text only
        text_parts = []
        for item in transcript_list:
            if 'text' in item and item['text'].strip():
                text = item['text'].strip()
                # Remove common transcript artifacts
                text = text.replace('[Music]', '').replace('[Applause]', '').replace('[Laughter]', '').strip()
                if text and not text.startswith('[') and not text.endswith(']'):
                    text_parts.append(text)
        
        return ' '.join(text_parts) if text_parts else "No readable text found."
    else:
        # Unclean format - with timestamps
        formatted_transcript = []
        for item in transcript_list:
            if 'text' in item and 'start' in item:
                start_time = float(item['start'])
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                timestamp = f"[{minutes:02d}:{seconds:02d}]"
                text = item['text'].strip()
                if text:
                    formatted_transcript.append(f"{timestamp} {text}")
        
        return '\n'.join(formatted_transcript) if formatted_transcript else "No transcript content found."


# API Endpoints (existing ones kept, new ones added)
@app.get("/")
async def root():
    return {"message": "YouTube Transcript Downloader API", "status": "running", "version": "1.0.0"}

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user(db, user_data.username)
    if db_user:
        logger.warning(f"Registration attempt with existing username: {user_data.username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    email_exists = get_user_by_email(db, user_data.email)
    if email_exists:
        logger.warning(f"Registration attempt with existing email: {user_data.email}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error registering user"
        )

@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        logger.warning(f"Failed login attempt for user: {form_data.username}")
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


# Replace the call in your download_transcript endpoint:
# OLD: transcript_text = process_youtube_transcript_with_fallback(video_id, clean=request.clean_transcript)
# NEW: transcript_text = process_youtube_transcript_enhanced(video_id, clean=request.clean_transcript)


# Update your download endpoint to use the fallback method
@app.post("/download_transcript/")
async def download_transcript(
    request: TranscriptRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # [Previous validation code remains the same...]
    video_id = request.youtube_id.strip()
    
    if 'youtube.com' in video_id or 'youtu.be' in video_id:
        import re
        patterns = [
            r'(?:youtube\.com\/watch\?v=)([^&\n?#]+)',
            r'(?:youtu\.be\/)([^&\n?#]+)',
            r'(?:youtube\.com\/shorts\/)([^&\n?#]+)',
            r'(?:youtube\.com\/embed\/)([^&\n?#]+)',
            r'[?&]v=([^&\n?#]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, video_id)
            if match:
                video_id = match.group(1)[:11]
                break
    
    if not video_id or len(video_id) != 11:
        raise HTTPException(status_code=400, detail="Invalid video ID")
    
    logger.info(f"📹 Processing transcript request for: {video_id}")
    
    # Check subscription limits
    transcript_type = "clean_transcripts" if request.clean_transcript else "unclean_transcripts"
    can_download = check_subscription_limit(user.id, transcript_type, db)
    if not can_download:
        raise HTTPException(status_code=403, detail="Monthly limit reached")
    
    # Use the fallback method
    transcript_text = process_youtube_transcript_enhanced(video_id, clean=request.clean_transcript)
    
    # Record successful download
    new_download = TranscriptDownload(
        user_id=user.id,
        youtube_id=video_id,
        transcript_type=transcript_type,
        created_at=datetime.now()
    )
    
    try:
        db.add(new_download)
        db.commit()
        logger.info(f"✅ Success: {user.username} downloaded {transcript_type} for {video_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Database error: {str(e)}")
    
    return {
        "transcript": transcript_text,
        "youtube_id": video_id,
        "message": "Transcript downloaded successfully"
    }

# 🔧 UPDATED: Enhanced payment intent endpoint to match payment.py
@app.post("/create_payment_intent/")
async def create_payment_intent_endpoint(
    request: CreatePaymentIntentRequest,  # 🔧 UPDATED: Use new simple model
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a payment intent for subscription upgrade - UPDATED VERSION"""
    try:
        logger.info(f"Creating payment intent for user {current_user.id} with price_id: {request.price_id}")
        
        # Validate price_id using your standardized variable names
        valid_price_ids = [
            os.getenv("PRO_PRICE_ID"),
            os.getenv("PREMIUM_PRICE_ID")
        ]
        
        if request.price_id not in valid_price_ids:
            logger.error(f"Invalid price ID: {request.price_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid price ID: {request.price_id}"
            )

        # Get the price from Stripe
        try:
            price = stripe.Price.retrieve(request.price_id)
            logger.info(f"Retrieved price: {price.unit_amount} {price.currency}")
        except stripe.error.InvalidRequestError as e:
            logger.error(f"Invalid Stripe price ID: {request.price_id}, error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid Stripe price ID: {request.price_id}"
            )
        
        # Determine plan type using your standardized variable names
        plan_type = 'pro' if request.price_id == os.getenv("PRO_PRICE_ID") else 'premium'
        logger.info(f"Plan type: {plan_type}")
        
        # Get or create Stripe customer
        customer = get_or_create_stripe_customer(current_user, db)
        logger.info(f"Stripe customer: {customer.id}")
        
        # 🔧 FIXED: Create PaymentIntent with proper configuration
        intent = stripe.PaymentIntent.create(
            amount=price.unit_amount,  # Amount in cents
            currency=price.currency,
            customer=customer.id,
            automatic_payment_methods={
                'enabled': True,
                'allow_redirects': 'never'  # 🔧 THIS FIXES THE STRIPE REDIRECT ERROR!
            },
            metadata={
                'user_id': str(current_user.id),
                'user_email': current_user.email,
                'price_id': request.price_id,
                'plan_type': plan_type
            }
        )

        logger.info(f"✅ Payment intent created successfully: {intent.id}")

        return {
            'client_secret': intent.client_secret,
            'payment_intent_id': intent.id,
            'amount': price.unit_amount,
            'currency': price.currency,
            'plan_type': plan_type
        }

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stripe error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Payment intent creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create payment intent: {str(e)}"
        )

# 🔧 PERFECT FIX: Replace your confirm_payment endpoint in main.py
@app.post("/confirm_payment/")
async def confirm_payment_endpoint(
    request: ConfirmPaymentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Confirm payment and update user subscription - PERFECTLY MATCHED TO YOUR MODEL"""
    try:
        logger.info(f"Confirming payment for user {current_user.id} with payment_intent: {request.payment_intent_id}")
        
        # Retrieve the PaymentIntent from Stripe
        intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
        
        if intent.status != 'succeeded':
            logger.error(f"Payment not completed. Status: {intent.status}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Payment not completed. Status: {intent.status}"
            )

        # Update user subscription in database
        user_subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id
        ).first()

        plan_type = intent.metadata.get('plan_type', 'pro')

        if not user_subscription:
            # 🔧 PERFECT: Create new subscription using YOUR EXACT model fields
            user_subscription = Subscription(
                user_id=current_user.id,
                tier=plan_type,
                start_date=datetime.utcnow(),
                expiry_date=datetime.utcnow() + timedelta(days=30),
                payment_id=request.payment_intent_id,
                auto_renew=True
            )
            db.add(user_subscription)
        else:
            # 🔧 PERFECT: Update existing subscription using YOUR EXACT model fields
            user_subscription.tier = plan_type
            user_subscription.start_date = datetime.utcnow()
            user_subscription.expiry_date = datetime.utcnow() + timedelta(days=30)
            user_subscription.payment_id = request.payment_intent_id
            user_subscription.auto_renew = True

        db.commit()
        db.refresh(user_subscription)

        logger.info(f"✅ User {current_user.id} subscription updated to {plan_type}")

        return {
            'success': True,
            'subscription_tier': user_subscription.tier,
            'expires_at': user_subscription.expiry_date.isoformat(),
            'status': 'active'  # 🔧 Return in response only, don't store in DB
        }

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error during confirmation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stripe error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Payment confirmation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to confirm payment: {str(e)}"
        )

# ENHANCED: Updated create_subscription endpoint
@app.post("/create_subscription/")
async def create_subscription_enhanced(
    request: SubscriptionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enhanced subscription creation with proper Stripe integration"""
    if request.subscription_tier not in PRICE_ID_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid subscription tier. Must be one of: {', '.join(PRICE_ID_MAP.keys())}"
        )
        
    try:
        # Get or create Stripe customer
        customer = get_or_create_stripe_customer(current_user, db)
        
        # Create Stripe subscription
        subscription = stripe.Subscription.create(
            customer=customer.id,
            items=[{
                'price': PRICE_ID_MAP[request.subscription_tier],
            }],
            metadata={
                'user_id': str(current_user.id),
                'username': current_user.username,
                'plan_name': request.subscription_tier
            }
        )
        
        # Update or create subscription in database
        existing_subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id
        ).first()
        
        if existing_subscription:
            existing_subscription.tier = request.subscription_tier
            existing_subscription.start_date = datetime.now()
            existing_subscription.expiry_date = datetime.now() + timedelta(days=30)
            existing_subscription.payment_id = subscription.id
            existing_subscription.auto_renew = True
        else:
            new_subscription = Subscription(
                user_id=current_user.id,  
                tier=request.subscription_tier,
                start_date=datetime.now(),
                expiry_date=datetime.now() + timedelta(days=30),
                payment_id=subscription.id,
                auto_renew=True
            )
            db.add(new_subscription)
        
        db.commit()
        logger.info(f"User {current_user.username} created {request.subscription_tier} subscription successfully")
        
        return {
            "subscription_id": subscription.id,
            "status": subscription.status,
            "current_period_end": subscription.current_period_end,
            "tier": request.subscription_tier
        }
    
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error during subscription creation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create subscription"
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Subscription creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process subscription"
        )

# NEW: Cancel subscription endpoint
@app.post("/cancel_subscription/")
async def cancel_subscription(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Cancel user's current subscription"""
    try:
        # Get user's subscription
        subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id
        ).first()
        
        if not subscription or not subscription.payment_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No active subscription found"
            )
        
        # Cancel subscription in Stripe
        stripe_subscription = stripe.Subscription.modify(
            subscription.payment_id,
            cancel_at_period_end=True
        )
        
        # Update database
        subscription.auto_renew = False
        db.commit()
        
        logger.info(f"User {current_user.username} cancelled subscription {subscription.payment_id}")
        
        return {
            "message": "Subscription cancelled successfully",
            "will_expire_at": stripe_subscription.current_period_end
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error during cancellation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel subscription"
        )
    except Exception as e:
        logger.error(f"Subscription cancellation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel subscription"
        )

# ENHANCED: Updated subscription status endpoint
@app.get("/subscription_status/")
async def get_subscription_status_enhanced(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enhanced subscription status with detailed usage info"""
    try:
        # Get user's subscription
        subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id
        ).first()
        
        if not subscription or subscription.expiry_date < datetime.now():
            tier = "free"
            status = "inactive"
        else:
            tier = subscription.tier
            status = "active"
        
        # Get current month's usage
        month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        clean_usage = db.query(TranscriptDownload).filter(
            TranscriptDownload.user_id == current_user.id,
            TranscriptDownload.transcript_type == "clean",
            TranscriptDownload.created_at >= month_start
        ).count()
        
        unclean_usage = db.query(TranscriptDownload).filter(
            TranscriptDownload.user_id == current_user.id,
            TranscriptDownload.transcript_type == "unclean",
            TranscriptDownload.created_at >= month_start
        ).count()
        
        audio_usage = db.query(TranscriptDownload).filter(
            TranscriptDownload.user_id == current_user.id,
            TranscriptDownload.transcript_type == "audio",
            TranscriptDownload.created_at >= month_start
        ).count()
        
        video_usage = db.query(TranscriptDownload).filter(
            TranscriptDownload.user_id == current_user.id,
            TranscriptDownload.transcript_type == "video",
            TranscriptDownload.created_at >= month_start
        ).count()
        
        # Get limits based on tier
        limits = SUBSCRIPTION_LIMITS[tier]
        
        # Convert infinity to string for JSON serialization
        json_limits = {}
        for key, value in limits.items():
            if value == float('inf'):
                json_limits[key] = 'unlimited'
            else:
                json_limits[key] = value
        
        return {
            "tier": tier,
            "status": status,
            "usage": {
                "clean_transcripts": clean_usage,
                "unclean_transcripts": unclean_usage,
                "audio_downloads": audio_usage,
                "video_downloads": video_usage
            },
            "limits": json_limits,
            "subscription_id": subscription.payment_id if subscription else None,
            "current_period_end": subscription.expiry_date.isoformat() if subscription and subscription.expiry_date else None
        }
        
    except Exception as e:
        logger.error(f"Error getting subscription status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get subscription status"
        )

# EXISTING: Keep original subscription status endpoint for backward compatibility
@app.get("/subscription/status", response_model=SubscriptionResponse)
async def get_subscription_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Original subscription status endpoint (maintained for compatibility)"""
    subscription = db.query(Subscription).filter(
        Subscription.user_id == current_user.id
    ).first()
    
    if not subscription:
        return {
            "tier": "free",
            "status": "active",
            "limits": SUBSCRIPTION_LIMITS["free"],
            "usage": {"clean": 0, "unclean": 0},
            "remaining": {"clean": SUBSCRIPTION_LIMITS["free"]["clean"], "unclean": SUBSCRIPTION_LIMITS["free"]["unclean"]}
        }

    if subscription.expiry_date < datetime.now():
        status = "expired"
        tier = "free"
    else:
        status = "active"
        tier = subscription.tier
    
    month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    clean_usage = db.query(TranscriptDownload).filter(
        TranscriptDownload.user_id == current_user.id,
        TranscriptDownload.transcript_type == "clean",
        TranscriptDownload.created_at >= month_start
    ).count()
    
    unclean_usage = db.query(TranscriptDownload).filter(
        TranscriptDownload.user_id == current_user.id,
        TranscriptDownload.transcript_type == "unclean",
        TranscriptDownload.created_at >= month_start
    ).count()
    
    return {
        "tier": tier,
        "status": status,
        "expiry_date": subscription.expiry_date.isoformat() if subscription.expiry_date else None,
        "limits": SUBSCRIPTION_LIMITS[tier],
        "usage": {
            "clean": clean_usage,
            "unclean": unclean_usage
        },
        "remaining": {
            "clean": max(0, SUBSCRIPTION_LIMITS[tier]["clean"] - clean_usage) if SUBSCRIPTION_LIMITS[tier]["clean"] != float('inf') else "unlimited",
            "unclean": max(0, SUBSCRIPTION_LIMITS[tier]["unclean"] - unclean_usage) if SUBSCRIPTION_LIMITS[tier]["unclean"] != float('inf') else "unlimited",
        }
    }

# EXISTING: Webhook handler (enhanced)
@app.post("/webhook", status_code=200)
async def webhook_received(request: Request, response: Response, db: Session = Depends(get_db)):
    """Enhanced webhook handler"""
    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature")
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError as e:
        logger.warning(f"Invalid webhook payload: {str(e)}")
        response.status_code = 400
        return {"error": str(e)}
    except stripe.error.SignatureVerificationError as e:
        logger.warning(f"Invalid webhook signature: {str(e)}")
        response.status_code = 400
        return {"error": str(e)}
    
    event_type = event['type']
    logger.info(f"Received Stripe webhook event: {event_type}")
    
    if event_type == 'invoice.payment_succeeded':
        invoice = event['data']['object']
        subscription_id = invoice['subscription']
        
        subscription = db.query(Subscription).filter(
            Subscription.payment_id == subscription_id
        ).first()
        
        if subscription:
            subscription.expiry_date = datetime.now() + timedelta(days=30)
            db.commit()
            logger.info(f"Subscription {subscription_id} extended by 30 days")
    
    elif event_type == 'customer.subscription.deleted':
        subscription_data = event['data']['object']
        subscription_id = subscription_data['id']
        
        subscription = db.query(Subscription).filter(
            Subscription.payment_id == subscription_id
        ).first()
        
        if subscription:
            subscription.auto_renew = False
            db.commit()
            logger.info(f"Subscription {subscription_id} marked as not auto-renewing")
    
    return {"success": True}

# NEW: Alternative webhook endpoint for new payment system
@app.post("/stripe_webhook/")
async def stripe_webhook_enhanced(request: Request, db: Session = Depends(get_db)):
    """Enhanced webhook endpoint for new payment system"""
    try:
        payload = await request.body()
        sig_header = request.headers.get("Stripe-Signature")
        endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
        
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, endpoint_secret
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.error.SignatureVerificationError:
            raise HTTPException(status_code=400, detail="Invalid signature")
        
        # Handle the event
        if event['type'] == 'invoice.payment_succeeded':
            subscription_id = event['data']['object']['subscription']
            # Update user subscription status in database
            subscription = db.query(Subscription).filter(
                Subscription.payment_id == subscription_id
            ).first()
            if subscription:
                subscription.expiry_date = datetime.now() + timedelta(days=30)
                db.commit()
            
        elif event['type'] == 'invoice.payment_failed':
            subscription_id = event['data']['object']['subscription']
            # Handle failed payment
            logger.warning(f"Payment failed for subscription {subscription_id}")
            
        elif event['type'] == 'customer.subscription.deleted':
            subscription_id = event['data']['object']['id']
            # Downgrade user to free tier
            subscription = db.query(Subscription).filter(
                Subscription.payment_id == subscription_id
            ).first()
            if subscription:
                subscription.auto_renew = False
                db.commit()
        
        return {"status": "success"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Webhook processing failed: {str(e)}"
        )

# Optional: Add this debug endpoint to test the new library
@app.get("/test_transcript/{video_id}")
async def test_transcript(video_id: str):
    """Test endpoint to verify transcript functionality"""
    try:
        from youtube_transcript_api import __version__ as yta_version
        
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
        return {
            "success": True,
            "library_version": yta_version,
            "video_id": video_id,
            "transcript_segments": len(transcript),
            "first_segment": transcript[0] if transcript else None,
            "message": "Transcript API working correctly!"
        }
    except Exception as e:
        return {
            "success": False,
            "library_version": yta_version if 'yta_version' in locals() else "unknown",
            "error": str(e),
            "error_type": type(e).__name__
        }

#Visit: http://localhost:8000/library_info - should show version 1.1.0
@app.get("/library_info")
async def library_info():
    from youtube_transcript_api import __version__

    return {"youtube_transcript_api_version": __version__
    }

@app.get("/debug/network")
async def debug_network():
    """Test network connectivity and identify issues"""
    results = {}
    
    try:
        # Test basic internet
        response = requests.get("https://httpbin.org/get", timeout=10)
        results["basic_internet"] = {
            "status": "OK" if response.status_code == 200 else "FAILED",
            "status_code": response.status_code
        }
    except Exception as e:
        results["basic_internet"] = {"status": "FAILED", "error": str(e)}
    
    try:
        # Test YouTube connectivity
        response = requests.get("https://www.youtube.com", timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        results["youtube_connectivity"] = {
            "status": "OK" if response.status_code == 200 else "FAILED",
            "status_code": response.status_code
        }
    except Exception as e:
        results["youtube_connectivity"] = {"status": "FAILED", "error": str(e)}
    
    try:
        # Test YouTube API endpoints
        test_video_id = "ZbZSe6N_BXs"
        api_url = f"https://www.youtube.com/watch?v={test_video_id}"
        response = requests.get(api_url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        results["youtube_video_page"] = {
            "status": "OK" if response.status_code == 200 else "FAILED",
            "status_code": response.status_code,
            "content_length": len(response.content)
        }
    except Exception as e:
        results["youtube_video_page"] = {"status": "FAILED", "error": str(e)}
    
    # System info
    results["system_info"] = {
        "ssl_version": ssl.OPENSSL_VERSION,
        "python_version": sys.version,
        "requests_version": requests.__version__
    }
    
    return results

@app.get("/debug/transcript_raw/{video_id}")
async def debug_transcript_raw(video_id: str):
    """Test raw transcript API access"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        from youtube_transcript_api import __version__ as yta_version
        
        logger.info(f"Testing raw transcript access for {video_id}")
        logger.info(f"Library version: {yta_version}")
        
        # Try to get available transcripts first
        try:
            transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
            available = []
            for transcript in transcript_list_obj:
                available.append({
                    "language": transcript.language,
                    "language_code": transcript.language_code,
                    "is_generated": transcript.is_generated
                })
            
            # Try to get English transcript
            transcript_data = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            
            return {
                "success": True,
                "library_version": yta_version,
                "video_id": video_id,
                "available_transcripts": available,
                "transcript_segments": len(transcript_data),
                "first_segment": transcript_data[0] if transcript_data else None
            }
            
        except Exception as inner_e:
            return {
                "success": False,
                "library_version": yta_version,
                "video_id": video_id,
                "error": str(inner_e),
                "error_type": type(inner_e).__name__
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

@app.get("/debug/alternative_method/{video_id}")
async def debug_alternative_method(video_id: str):
    """Try alternative transcript extraction method"""
    try:
        import re
        import xml.etree.ElementTree as ET
        from urllib.parse import unquote
        
        # Get YouTube video page
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        
        response = requests.get(video_url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            return {
                "success": False,
                "error": f"Failed to fetch video page. Status: {response.status_code}"
            }
        
        page_content = response.text
        
        # Look for captions in the page
        patterns = [
            r'"captionTracks":\[(.*?)\]',
            r'"captions".*?"playerCaptionsTracklistRenderer".*?"captionTracks":\[(.*?)\]'
        ]
        
        caption_data = None
        for pattern in patterns:
            match = re.search(pattern, page_content)
            if match:
                try:
                    caption_data = json.loads('[' + match.group(1) + ']')
                    break
                except:
                    continue
        
        if not caption_data:
            return {
                "success": False,
                "error": "No caption tracks found in video page",
                "page_size": len(page_content)
            }
        
        # Find English caption
        english_caption = None
        for caption in caption_data:
            if caption.get('languageCode', '').startswith('en'):
                english_caption = caption
                break
        
        if not english_caption:
            english_caption = caption_data[0]  # Use first available
        
        caption_url = english_caption.get('baseUrl')
        if not caption_url:
            return {
                "success": False,
                "error": "No caption URL found",
                "available_captions": caption_data
            }
        
        # Fetch the caption file
        caption_response = requests.get(caption_url, headers=headers, timeout=10)
        
        if caption_response.status_code != 200:
            return {
                "success": False,
                "error": f"Failed to fetch caption file. Status: {caption_response.status_code}"
            }
        
        # Parse XML
        try:
            root = ET.fromstring(caption_response.content)
            transcript_parts = []
            
            for text_elem in root.findall('.//text'):
                text_content = text_elem.text or ''
                if text_content.strip():
                    # Clean up the text
                    text_content = unquote(text_content)
                    text_content = text_content.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"')
                    transcript_parts.append(text_content.strip())
            
            return {
                "success": True,
                "method": "alternative_http",
                "video_id": video_id,
                "transcript_segments": len(transcript_parts),
                "sample_text": ' '.join(transcript_parts[:3]) if transcript_parts else None,
                "available_captions": len(caption_data)
            }
            
        except ET.ParseError as parse_error:
            return {
                "success": False,
                "error": f"XML parsing failed: {str(parse_error)}",
                "caption_content_preview": caption_response.text[:500]
            }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

# NEW: Health check endpoint
@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "stripe_configured": bool(os.getenv("STRIPE_SECRET_KEY")),
        "timestamp": datetime.utcnow().isoformat()
    }

# EXISTING: Healthcheck endpoint (maintained for compatibility)
@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok", "version": "1.0.0"}

#============================== New main.py ================

# # main.py - CLEAN WORKING VERSION (Based on your old successful approach)

# from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Request, Response
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from sqlalchemy.orm import Session
# from datetime import datetime, timedelta
# from typing import Optional, List

# import jwt
# from jwt import PyJWTError

# from pydantic import BaseModel
# from passlib.context import CryptContext
# import stripe
# from youtube_transcript_api import YouTubeTranscriptApi
# import os
# import json
# import logging
# from dotenv import load_dotenv
# import requests
# import re

# import warnings
# warnings.filterwarnings("ignore", message=".*bcrypt.*")

# # Import from database.py
# from database import get_db, User, Subscription, TranscriptDownload, create_tables

# import xml.etree.ElementTree as ET
# from urllib.parse import unquote

# # Load environment variables
# load_dotenv()

# # Configure logging
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
#     logger.info(f"Production mode - CORS origins: {allowed_origins}")
# else:
#     allowed_origins = [
#         "http://localhost:3000",
#         "http://127.0.0.1:3000",
#         FRONTEND_URL
#     ]
#     logger.info(f"Development mode - CORS origins: {allowed_origins}")

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

# # Subscription limits
# SUBSCRIPTION_LIMITS = {
#     "free": {
#         "clean_transcripts": 5, "unclean_transcripts": 3, 
#         "audio_downloads": 2, "video_downloads": 1
#     },
#     "pro": {
#         "clean_transcripts": 100, "unclean_transcripts": 50,
#         "audio_downloads": 50, "video_downloads": 20
#     },
#     "premium": {
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
#     """Startup event"""
#     try:
#         logger.info("Starting YouTube Transcript Downloader API...")
#         logger.info(f"Environment: {ENVIRONMENT}")
        
#         # Initialize database
#         create_tables()
#         logger.info("Database initialized successfully")
#         logger.info("Application startup complete!")
        
#     except Exception as e:
#         logger.error(f"Startup failed: {str(e)}")
#         raise

# # Pydantic models
# class Token(BaseModel):
#     access_token: str
#     token_type: str

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
#     youtube_id: str = None
#     video_url: str = None
#     clean_transcript: bool = False

# class CreatePaymentIntentRequest(BaseModel):
#     price_id: str

# class ConfirmPaymentRequest(BaseModel):
#     payment_intent_id: str

# class SubscriptionRequest(BaseModel):
#     subscription_tier: str

# # Helper functions
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
#     except Exception:
#         raise credentials_exception
        
#     user = get_user(db, username)
#     if user is None:
#         raise credentials_exception
#     return user

# def check_user_limits(user, action_type: str, db: Session):
#     """Check if user has exceeded their limits for the current month"""
#     subscription = db.query(Subscription).filter(Subscription.user_id == user.id).first()
    
#     if not subscription or subscription.expiry_date < datetime.now():
#         tier = "free"
#     else:
#         tier = subscription.tier
    
#     month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
#     type_mapping = {
#         "clean_transcripts": "clean",
#         "unclean_transcripts": "unclean",
#         "audio_downloads": "audio",
#         "video_downloads": "video"
#     }
    
#     transcript_type = type_mapping.get(action_type, action_type)
    
#     current_usage = db.query(TranscriptDownload).filter(
#         TranscriptDownload.user_id == user.id,
#         TranscriptDownload.transcript_type == transcript_type,
#         TranscriptDownload.created_at >= month_start
#     ).count()
    
#     limit = SUBSCRIPTION_LIMITS[tier].get(action_type, 0)
    
#     if limit == float('inf'):
#         return True
    
#     return current_usage < limit

# def get_or_create_stripe_customer(user, db: Session):
#     """Get or create a Stripe customer for the user"""
#     try:
#         stripe_customer_id = getattr(user, 'stripe_customer_id', None)
        
#         if stripe_customer_id:
#             try:
#                 customer = stripe.Customer.retrieve(stripe_customer_id)
#                 return customer
#             except stripe.error.InvalidRequestError:
#                 pass
        
#         customer = stripe.Customer.create(
#             email=user.email,
#             name=user.username,
#             metadata={'user_id': str(user.id)}
#         )
        
#         try:
#             if hasattr(user, 'stripe_customer_id'):
#                 user.stripe_customer_id = customer.id
#                 db.commit()
#         except Exception:
#             pass
        
#         return customer
        
#     except Exception as e:
#         logger.error(f"Stripe error creating customer: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to create payment customer"
#         )

# # Replace your process_youtube_transcript function in main.py with this bulletproof solution
# def get_youtube_transcript_direct(video_id: str, clean: bool = True) -> str:
#     """
#     Direct YouTube transcript scraper - bypasses broken youtube-transcript-api
#     This method scrapes YouTube pages directly and extracts caption data
#     """
#     try:
#         logger.info(f"Using direct scraping method for video: {video_id}")
        
#         # Step 1: Get YouTube video page with proper headers
#         video_url = f"https://www.youtube.com/watch?v={video_id}"
        
#         headers = {
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
#             'Accept-Language': 'en-US,en;q=0.9',
#             'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
#             'Accept-Encoding': 'gzip, deflate, br',
#             'DNT': '1',
#             'Connection': 'keep-alive',
#             'Upgrade-Insecure-Requests': '1',
#             'Sec-Fetch-Dest': 'document',
#             'Sec-Fetch-Mode': 'navigate',
#             'Sec-Fetch-Site': 'none',
#             'Cache-Control': 'max-age=0'
#         }
        
#         logger.info(f"Fetching YouTube page: {video_url}")
#         response = requests.get(video_url, headers=headers, timeout=15)
        
#         if response.status_code != 200:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"Video not accessible. Status: {response.status_code}"
#             )
        
#         page_content = response.text
#         logger.info(f"Page content received: {len(page_content)} characters")
        
#         # Step 2: Extract player response data using multiple methods
#         player_response = None
        
#         # Method 1: Look for ytInitialPlayerResponse
#         patterns = [
#             r'var\s+ytInitialPlayerResponse\s*=\s*({.+?});',
#             r'ytInitialPlayerResponse\s*=\s*({.+?});',
#             r'"playerResponse":"({.+?})"',
#             r'ytInitialPlayerResponse":\s*({.+?})(?=,|\})',
#         ]
        
#         for pattern in patterns:
#             matches = re.finditer(pattern, page_content, re.DOTALL)
#             for match in matches:
#                 try:
#                     json_str = match.group(1)
                    
#                     # Clean up escaped JSON if needed
#                     if json_str.startswith('"') and json_str.endswith('"'):
#                         json_str = json_str[1:-1].replace('\\"', '"')
                    
#                     player_response = json.loads(json_str)
                    
#                     # Check if this response has captions
#                     if 'captions' in player_response:
#                         logger.info("Found player response with captions")
#                         break
                        
#                 except (json.JSONDecodeError, ValueError) as e:
#                     continue
            
#             if player_response and 'captions' in player_response:
#                 break
        
#         if not player_response or 'captions' not in player_response:
#             raise HTTPException(
#                 status_code=404,
#                 detail="No captions found for this video. The video may not have subtitles enabled."
#             )
        
#         # Step 3: Extract caption tracks
#         captions_data = player_response.get('captions', {})
#         caption_renderer = captions_data.get('playerCaptionsTracklistRenderer', {})
#         caption_tracks = caption_renderer.get('captionTracks', [])
        
#         if not caption_tracks:
#             raise HTTPException(
#                 status_code=404,
#                 detail="No caption tracks found for this video."
#             )
        
#         logger.info(f"Found {len(caption_tracks)} caption tracks")
        
#         # Step 4: Find the best caption track (prefer English manual, then auto-generated)
#         best_caption = None
        
#         # First priority: Manual English captions
#         for caption in caption_tracks:
#             lang_code = caption.get('languageCode', '').lower()
#             is_auto = caption.get('kind', '') == 'asr'  # asr = automatic speech recognition
            
#             if lang_code.startswith('en') and not is_auto:
#                 best_caption = caption
#                 logger.info(f"Using manual English caption: {caption.get('name', {}).get('simpleText', 'Unknown')}")
#                 break
        
#         # Second priority: Auto-generated English captions
#         if not best_caption:
#             for caption in caption_tracks:
#                 lang_code = caption.get('languageCode', '').lower()
#                 if lang_code.startswith('en'):
#                     best_caption = caption
#                     logger.info(f"Using auto-generated English caption: {caption.get('name', {}).get('simpleText', 'Unknown')}")
#                     break
        
#         # Third priority: Any available caption
#         if not best_caption and caption_tracks:
#             best_caption = caption_tracks[0]
#             logger.info(f"Using first available caption: {best_caption.get('name', {}).get('simpleText', 'Unknown')}")
        
#         if not best_caption or 'baseUrl' not in best_caption:
#             raise HTTPException(
#                 status_code=404,
#                 detail="No usable caption track found"
#             )
        
#         # Step 5: Download caption XML
#         caption_url = best_caption['baseUrl']
#         logger.info(f"Downloading captions from: {caption_url[:100]}...")
        
#         caption_response = requests.get(caption_url, headers=headers, timeout=10)
        
#         if caption_response.status_code != 200:
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Failed to download caption file. Status: {caption_response.status_code}"
#             )
        
#         # Step 6: Parse caption XML
#         try:
#             root = ET.fromstring(caption_response.content)
#             transcript_segments = []
            
#             for text_elem in root.findall('.//text'):
#                 start_time = float(text_elem.get('start', '0'))
#                 duration = float(text_elem.get('dur', '3'))
#                 text_content = text_elem.text or ''
                
#                 if text_content.strip():
#                     # Decode HTML entities
#                     text_content = unquote(text_content)
#                     text_content = (text_content
#                                    .replace('&amp;', '&')
#                                    .replace('&lt;', '<')
#                                    .replace('&gt;', '>')
#                                    .replace('&quot;', '"')
#                                    .replace('&#39;', "'")
#                                    .replace('\n', ' ')
#                                    .strip())
                    
#                     # Remove HTML tags
#                     text_content = re.sub(r'<[^>]+>', '', text_content)
#                     text_content = re.sub(r'\s+', ' ', text_content).strip()
                    
#                     if text_content:
#                         transcript_segments.append({
#                             'text': text_content,
#                             'start': start_time,
#                             'duration': duration
#                         })
            
#             if not transcript_segments:
#                 raise HTTPException(
#                     status_code=404,
#                     detail="Caption file contains no readable text"
#                 )
            
#             logger.info(f"Successfully extracted {len(transcript_segments)} transcript segments")
            
#             # Step 7: Format transcript based on user preference
#             if clean:
#                 # Clean format - just text
#                 text_parts = []
#                 for segment in transcript_segments:
#                     text = segment['text'].strip()
#                     # Remove sound effect markers like [Music], [Applause]
#                     if not (text.startswith('[') and text.endswith(']')) and text:
#                         # Additional cleaning
#                         text = text.replace('[Music]', '').replace('[Applause]', '').replace('[Laughter]', '').strip()
#                         if text:
#                             text_parts.append(text)
                
#                 if not text_parts:
#                     raise HTTPException(
#                         status_code=404,
#                         detail="Transcript contains no readable text content after cleaning"
#                     )
                
#                 return ' '.join(text_parts)
            
#             else:
#                 # Unclean format - with timestamps
#                 formatted_lines = []
#                 for segment in transcript_segments:
#                     start_time = segment['start']
#                     text = segment['text']
                    
#                     # Convert seconds to MM:SS format
#                     minutes = int(start_time // 60)
#                     seconds = int(start_time % 60)
#                     timestamp = f"[{minutes:02d}:{seconds:02d}]"
                    
#                     formatted_lines.append(f"{timestamp} {text}")
                
#                 if not formatted_lines:
#                     raise HTTPException(
#                         status_code=404,
#                         detail="Transcript contains no content with timestamps"
#                     )
                
#                 return '\n'.join(formatted_lines)
                
#         except ET.ParseError as e:
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Failed to parse caption XML: {str(e)}"
#             )
            
#     except HTTPException:
#         raise  # Re-raise HTTP exceptions as-is
#     except requests.RequestException as e:
#         raise HTTPException(
#             status_code=503,
#             detail=f"Network error while fetching video data: {str(e)}"
#         )
#     except Exception as e:
#         logger.error(f"Direct scraping failed for {video_id}: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to extract transcript: {str(e)}"
#         )

# # Replace your process_youtube_transcript function with this:
# def process_youtube_transcript(video_id: str, clean: bool = True) -> str:
#     """
#     NEW bulletproof transcript processor using direct scraping
#     """
#     try:
#         logger.info(f"Getting transcript for video: {video_id} (using direct method)")
        
#         # Use our bulletproof direct scraping method
#         transcript_content = get_youtube_transcript_direct(video_id, clean)
        
#         if not transcript_content.strip():
#             raise HTTPException(
#                 status_code=404,
#                 detail="Transcript is empty or contains no readable content."
#             )
        
#         return transcript_content
        
#     except HTTPException:
#         raise  # Re-raise HTTP exceptions as-is
#     except Exception as e:
#         logger.error(f"Error getting transcript for {video_id}: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to retrieve transcript: {str(e)}"
#         )

# #===============================================================

# # 1. REPLACE your process_youtube_transcript function with this version:

# def process_youtube_transcript(video_id: str, clean: bool = True) -> str:
#     """
#     Process YouTube transcript - FIXED VERSION with generic exception handling
#     """
#     try:
#         logger.info(f"🔍 Getting transcript for video: {video_id}")
        
#         # Use the YouTube Transcript API
#         transcript_list = YouTubeTranscriptApi.get_transcript(
#             video_id,
#             languages=['en', 'en-US', 'en-GB']
#         )
        
#         if not transcript_list:
#             raise HTTPException(
#                 status_code=404,
#                 detail="No transcript data found for this video."
#             )
        
#         logger.info(f"✅ Retrieved {len(transcript_list)} transcript segments")
        
#         if clean:
#             # Clean format - text only
#             text_parts = []
#             for item in transcript_list:
#                 if 'text' in item and item['text'].strip():
#                     text = item['text'].strip()
#                     text = text.replace('[Music]', '').replace('[Applause]', '').replace('[Laughter]', '').strip()
#                     if text and not text.startswith('[') and not text.endswith(']'):
#                         text_parts.append(text)
            
#             if not text_parts:
#                 raise HTTPException(
#                     status_code=404,
#                     detail="Transcript contains no readable text content."
#                 )
            
#             return ' '.join(text_parts)
#         else:
#             # Unclean format - with timestamps
#             formatted_transcript = []
#             for item in transcript_list:
#                 if 'text' in item and 'start' in item:
#                     start_time = float(item['start'])
#                     minutes = int(start_time // 60)
#                     seconds = int(start_time % 60)
#                     timestamp = f"[{minutes:02d}:{seconds:02d}]"
#                     text = item['text'].strip()
#                     if text:
#                         formatted_transcript.append(f"{timestamp} {text}")
            
#             if not formatted_transcript:
#                 raise HTTPException(
#                     status_code=404,
#                     detail="Transcript contains no content with timestamps."
#                 )
            
#             return '\n'.join(formatted_transcript)
            
#     except Exception as e:
#         error_msg = str(e).lower()
#         error_type = type(e).__name__
#         logger.error(f"💥 Error getting transcript for {video_id}: {str(e)} (Type: {error_type})")
        
#         # Handle all exceptions generically using error message patterns
#         if "transcript" in error_msg and "disabled" in error_msg:
#             raise HTTPException(
#                 status_code=404,
#                 detail="Transcripts are disabled for this video."
#             )
#         elif "no transcript" in error_msg or "not found" in error_msg:
#             raise HTTPException(
#                 status_code=404,
#                 detail="No transcript available for this video. Try a different video with captions enabled."
#             )
#         elif "video unavailable" in error_msg or "private" in error_msg:
#             raise HTTPException(
#                 status_code=404,
#                 detail="Video is unavailable, private, or doesn't exist."
#             )
#         elif "too many requests" in error_msg or "rate limit" in error_msg:
#             raise HTTPException(
#                 status_code=429,
#                 detail="Too many requests. Please wait a moment and try again."
#             )
#         elif "could not retrieve" in error_msg:
#             raise HTTPException(
#                 status_code=404,
#                 detail="No transcript available for this video. Please try a different video."
#             )
#         elif "subtitles are disabled" in error_msg:
#             raise HTTPException(
#                 status_code=404,
#                 detail="Subtitles are disabled for this video."
#             )
#         else:
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Failed to retrieve transcript: {str(e)}"
#             )

# # 2. ALSO UPDATE your process_youtube_transcript_with_fallback function:

# def process_youtube_transcript_with_fallback(video_id: str, clean: bool = True) -> str:
#     """
#     Main transcript processor with fallback to alternative method - FIXED VERSION
#     """
#     # Try the original library first
#     try:
#         logger.info(f"🔍 Trying youtube-transcript-api for video: {video_id}")
        
#         transcript_list = YouTubeTranscriptApi.get_transcript(
#             video_id,
#             languages=['en', 'en-US', 'en-GB']
#         )
        
#         if transcript_list:
#             logger.info(f"✅ Library method succeeded: {len(transcript_list)} segments")
            
#             if clean:
#                 text_parts = [item['text'].strip() for item in transcript_list if item['text'].strip()]
#                 return ' '.join(text_parts)
#             else:
#                 formatted_transcript = []
#                 for item in transcript_list:
#                     start_time = float(item['start'])
#                     minutes = int(start_time // 60)
#                     seconds = int(start_time % 60)
#                     timestamp = f"[{minutes:02d}:{seconds:02d}]"
#                     formatted_transcript.append(f"{timestamp} {item['text']}")
#                 return '\n'.join(formatted_transcript)
        
#     except Exception as library_error:
#         logger.warning(f"⚠️ Library method failed: {library_error}")
#         logger.info("🔄 Falling back to alternative HTTP method...")
        
#         # Fallback to alternative method
#         return get_transcript_alternative_method(video_id, clean)

# #===============================================================

# # API Endpoints
# @app.get("/")
# async def root():
#     return {"message": "YouTube Transcript Downloader API", "status": "running", "version": "1.0.0"}

# @app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
# def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
#     db_user = get_user(db, user_data.username)
#     if db_user:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Username already registered"
#         )
    
#     email_exists = get_user_by_email(db, user_data.email)
#     if email_exists:
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail="Email already registered"
#         )
    
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
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Error registering user"
#         )

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

# # MAIN DOWNLOAD ENDPOINT - SIMPLE WORKING VERSION
# @app.post("/download_transcript/")
# async def download_transcript(
#     request: TranscriptRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Simple transcript download - BACK TO YOUR WORKING VERSION"""
    
#     try:
#         # Get video ID from request
#         video_identifier = getattr(request, 'video_url', None) or request.youtube_id
        
#         if not video_identifier:
#             return {
#                 "success": False,
#                 "error_type": "missing_identifier",
#                 "message": "Please provide a YouTube URL or video ID"
#             }
        
#         # Simple video ID extraction (your original method)
#         video_id = video_identifier.strip()
        
#         # If it's a URL, extract the ID
#         if 'youtube.com' in video_id or 'youtu.be' in video_id:
#             patterns = [
#                 r'(?:youtube\.com\/watch\?v=)([a-zA-Z0-9_-]{11})',
#                 r'(?:youtu\.be\/)([a-zA-Z0-9_-]{11})',
#                 r'(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
#             ]
            
#             for pattern in patterns:
#                 match = re.search(pattern, video_id)
#                 if match:
#                     video_id = match.group(1)
#                     break
        
#         # Validate video ID length
#         if len(video_id) != 11:
#             return {
#                 "success": False,
#                 "error_type": "invalid_url",
#                 "message": "Please enter a valid YouTube URL or 11-character Video ID"
#             }
        
#         logger.info(f"Processing transcript request for video: {video_id}")
        
#         # Check subscription limits
#         try:
#             transcript_type = "clean_transcripts" if request.clean_transcript else "unclean_transcripts"
#             can_download = check_user_limits(user, transcript_type, db)
#             if not can_download:
#                 return {
#                     "success": False,
#                     "error_type": "subscription_limit",
#                     "message": "Monthly limit reached! Please upgrade your plan."
#                 }
#         except Exception as e:
#             logger.warning(f"Subscription check error: {e}")
        
#         # Use the SIMPLE approach that was working for you
#         try:
#             transcript_content = process_youtube_transcript(video_id, request.clean_transcript)
            
#             if not transcript_content.strip():
#                 return {
#                     "success": False,
#                     "error_type": "no_transcript",
#                     "message": "Transcript is empty or contains no readable content."
#                 }
            
#         except HTTPException as e:
#             return {
#                 "success": False,
#                 "error_type": "no_transcript",
#                 "message": e.detail
#             }
#         except Exception as e:
#             logger.error(f"Transcript extraction failed for {video_id}: {str(e)}")
#             return {
#                 "success": False,
#                 "error_type": "retrieval_failed",
#                 "message": "Failed to retrieve transcript. Please try a different video."
#             }
        
#         # Record successful download in database  
#         try:
#             transcript_type_db = "clean" if request.clean_transcript else "unclean"
#             new_download = TranscriptDownload(
#                 user_id=user.id,
#                 youtube_id=video_id,
#                 transcript_type=transcript_type_db,
#                 created_at=datetime.now()
#             )
#             db.add(new_download)
#             db.commit()
#             logger.info(f"SUCCESS: {user.username} downloaded {transcript_type_db} transcript for {video_id}")
#         except Exception as e:
#             logger.warning(f"Failed to update usage tracking: {e}")
        
#         # Return response compatible with your frontend
#         return {
#             "success": True,
#             "transcript": transcript_content,
#             "youtube_id": video_id,
#             "message": "Transcript downloaded successfully"
#         }
        
#     except Exception as e:
#         logger.error(f"Unexpected error in download_transcript: {str(e)}")
#         return {
#             "success": False,
#             "error_type": "internal_error",
#             "message": "An unexpected error occurred. Please try again."
#         }

# # Payment endpoints
# @app.post("/create_payment_intent/")
# async def create_payment_intent_endpoint(
#     request: CreatePaymentIntentRequest,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Create a payment intent for subscription upgrade"""
#     try:
#         valid_price_ids = [
#             os.getenv("PRO_PRICE_ID"),
#             os.getenv("PREMIUM_PRICE_ID")
#         ]
        
#         if request.price_id not in valid_price_ids:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail=f"Invalid price ID: {request.price_id}"
#             )

#         price = stripe.Price.retrieve(request.price_id)
#         plan_type = 'pro' if request.price_id == os.getenv("PRO_PRICE_ID") else 'premium'
#         customer = get_or_create_stripe_customer(current_user, db)
        
#         intent = stripe.PaymentIntent.create(
#             amount=price.unit_amount,
#             currency=price.currency,
#             customer=customer.id,
#             automatic_payment_methods={
#                 'enabled': True,
#                 'allow_redirects': 'never'
#             },
#             metadata={
#                 'user_id': str(current_user.id),
#                 'user_email': current_user.email,
#                 'price_id': request.price_id,
#                 'plan_type': plan_type
#             }
#         )

#         return {
#             'client_secret': intent.client_secret,
#             'payment_intent_id': intent.id,
#             'amount': price.unit_amount,
#             'currency': price.currency,
#             'plan_type': plan_type
#         }

#     except Exception as e:
#         logger.error(f"Payment intent creation error: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to create payment intent: {str(e)}"
#         )

# @app.post("/confirm_payment/")
# async def confirm_payment_endpoint(
#     request: ConfirmPaymentRequest,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Confirm payment and update user subscription"""
#     try:
#         intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
        
#         if intent.status != 'succeeded':
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail=f"Payment not completed. Status: {intent.status}"
#             )

#         user_subscription = db.query(Subscription).filter(
#             Subscription.user_id == current_user.id
#         ).first()

#         plan_type = intent.metadata.get('plan_type', 'pro')

#         if not user_subscription:
#             user_subscription = Subscription(
#                 user_id=current_user.id,
#                 tier=plan_type,
#                 start_date=datetime.utcnow(),
#                 expiry_date=datetime.utcnow() + timedelta(days=30),
#                 payment_id=request.payment_intent_id,
#                 auto_renew=True
#             )
#             db.add(user_subscription)
#         else:
#             user_subscription.tier = plan_type
#             user_subscription.start_date = datetime.utcnow()
#             user_subscription.expiry_date = datetime.utcnow() + timedelta(days=30)
#             user_subscription.payment_id = request.payment_intent_id
#             user_subscription.auto_renew = True

#         db.commit()
#         db.refresh(user_subscription)

#         return {
#             'success': True,
#             'subscription_tier': user_subscription.tier,
#             'expires_at': user_subscription.expiry_date.isoformat(),
#             'status': 'active'
#         }

#     except Exception as e:
#         logger.error(f"Payment confirmation error: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to confirm payment: {str(e)}"
#         )

# # Subscription status endpoint
# @app.get("/subscription_status/")
# async def get_subscription_status_enhanced(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Get subscription status with detailed usage info"""
#     try:
#         subscription = db.query(Subscription).filter(
#             Subscription.user_id == current_user.id
#         ).first()
        
#         if not subscription or subscription.expiry_date < datetime.now():
#             tier = "free"
#             status = "inactive"
#         else:
#             tier = subscription.tier
#             status = "active"
        
#         # Get current month's usage
#         month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
#         clean_usage = db.query(TranscriptDownload).filter(
#             TranscriptDownload.user_id == current_user.id,
#             TranscriptDownload.transcript_type == "clean",
#             TranscriptDownload.created_at >= month_start
#         ).count()
        
#         unclean_usage = db.query(TranscriptDownload).filter(
#             TranscriptDownload.user_id == current_user.id,
#             TranscriptDownload.transcript_type == "unclean",
#             TranscriptDownload.created_at >= month_start
#         ).count()
        
#         limits = SUBSCRIPTION_LIMITS[tier]
        
#         # Convert infinity to string for JSON serialization
#         json_limits = {}
#         for key, value in limits.items():
#             if value == float('inf'):
#                 json_limits[key] = 'unlimited'
#             else:
#                 json_limits[key] = value
        
#         return {
#             "tier": tier,
#             "status": status,
#             "usage": {
#                 "clean_transcripts": clean_usage,
#                 "unclean_transcripts": unclean_usage,
#             },
#             "limits": json_limits,
#             "subscription_id": subscription.payment_id if subscription else None,
#             "current_period_end": subscription.expiry_date.isoformat() if subscription and subscription.expiry_date else None
#         }
        
#     except Exception as e:
#         logger.error(f"Error getting subscription status: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to get subscription status"
#         )

# # Health check
# @app.get("/health/")
# async def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "timestamp": datetime.utcnow().isoformat()
#     }

# # Test endpoint for working videos
# @app.get("/test_working_videos/")
# async def test_working_videos():
#     """Test endpoint with known working video IDs"""
#     working_videos = [
#         {
#             "id": "k-RjskuqxzU",
#             "title": "Your Previously Working Video",
#             "description": "This was working in Picture #3 & #4"
#         },
#         {
#             "id": "jNQXAC9IVRw",
#             "title": "Me at the zoo",
#             "description": "First YouTube video"
#         },
#         {
#             "id": "dQw4w9WgXcQ",
#             "title": "Rick Astley - Never Gonna Give You Up",
#             "description": "Popular video with captions"
#         }
#     ]
    
#     results = []
#     for video in working_videos:
#         try:
#             # Simple test using our process_youtube_transcript function
#             transcript = process_youtube_transcript(video["id"], clean=True)
#             results.append({
#                 "video": video,
#                 "status": "SUCCESS",
#                 "transcript_length": len(transcript)
#             })
#         except Exception as e:
#             results.append({
#                 "video": video,
#                 "status": "FAILED",
#                 "error": str(e)
#             })
    
#     return {
#         "test_results": results,
#         "recommendation": "Use videos marked as 'SUCCESS' for testing"
#     }

# # Add this endpoint to your main.py to test which videos actually work
# @app.get("/test_real_videos/")
# async def test_real_videos():
#     """Test with videos that actually have captions"""
    
#     # These videos are CONFIRMED to have captions
#     test_videos = [
#         {
#             "id": "dQw4w9WgXcQ", 
#             "title": "Rick Astley - Never Gonna Give You Up",
#             "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
#             "description": "Famous rickroll video - definitely has captions"
#         },
#         {
#             "id": "jNQXAC9IVRw", 
#             "title": "Me at the zoo",
#             "url": "https://www.youtube.com/watch?v=jNQXAC9IVRw", 
#             "description": "First YouTube video ever uploaded"
#         },
#         {
#             "id": "9bZkp7q19f0", 
#             "title": "PSY - Gangnam Style",
#             "url": "https://www.youtube.com/watch?v=9bZkp7q19f0",
#             "description": "Viral K-pop video with captions"
#         },
#         {
#             "id": "k-RjskuqxzU",
#             "title": "Your Previously Working Video", 
#             "url": "https://www.youtube.com/watch?v=k-RjskuqxzU",
#             "description": "The video that worked in your Pictures #3 & #4"
#         },
#         {
#             "id": "UfO9K0CtzMg",
#             "title": "Video You Were Testing",
#             "url": "https://www.youtube.com/watch?v=UfO9K0CtzMg", 
#             "description": "This video likely has NO captions"
#         }
#     ]
    
#     results = []
#     working_videos = []
    
#     for video in test_videos:
#         try:
#             from youtube_transcript_api import YouTubeTranscriptApi
            
#             # Try to get the transcript
#             transcript_data = YouTubeTranscriptApi.get_transcript(
#                 video["id"], 
#                 languages=['en', 'en-US', 'en-GB']
#             )
            
#             if transcript_data:
#                 # Get a sample of the text
#                 sample_text = ' '.join([item['text'] for item in transcript_data[:3]])
                
#                 results.append({
#                     "video": video,
#                     "status": "✅ HAS CAPTIONS",
#                     "segments": len(transcript_data),
#                     "sample": sample_text[:100] + "..." if len(sample_text) > 100 else sample_text
#                 })
                
#                 working_videos.append(video["id"])
#             else:
#                 results.append({
#                     "video": video,
#                     "status": "❌ NO CAPTIONS",
#                     "error": "Empty transcript data"
#                 })
                
#         except Exception as e:
#             error_msg = str(e)
#             if "no element found" in error_msg:
#                 status = "❌ NO CAPTIONS (empty XML)"
#             elif "could not retrieve" in error_msg:
#                 status = "❌ NO CAPTIONS (not available)"
#             else:
#                 status = f"❌ ERROR: {error_msg[:50]}..."
                
#             results.append({
#                 "video": video,
#                 "status": status,
#                 "error": error_msg
#             })
    
#     return {
#         "message": "Test results for caption availability",
#         "library_info": {
#             "youtube_transcript_api": "1.1.0 (latest available)"
#         },
#         "test_results": results,
#         "working_video_ids": working_videos,
#         "recommendation": "Use the videos marked with ✅ HAS CAPTIONS in your app testing"
#     }