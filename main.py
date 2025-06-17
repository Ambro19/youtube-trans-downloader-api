# main.py - Enhanced with Complete Stripe Payment Integration
# main.py - Completed all the required modifications including Testing the Debug Endpoints 

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, List

import jwt
from jwt import PyJWTError  # ✅ CORRECT: Import PyJWTError directly from jwt

# OR alternatively, you can use:
# import jwt
# from jwt.exceptions import InvalidTokenError as PyJWTError

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

import warnings
warnings.filterwarnings("ignore", message=".*bcrypt.*")

# Import from database.py
from database import get_db, User, Subscription, TranscriptDownload, create_tables

#from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled



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

# class TranscriptRequest(BaseModel):
#     youtube_id: str
#     clean_transcript: bool = False

# UPDATE YOUR TranscriptRequest MODEL to support the new format
class TranscriptRequest(BaseModel):
    youtube_id: str = None  # Keep for backward compatibility
    video_url: str = None   # New field for URLs
    clean_transcript: bool = False
    format: str = "clean"   # New format field
    advanced_cleaning: bool = False  # For future Pro features
    
    # Add validation to ensure at least one identifier is provided
    @classmethod
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
    
    def get_video_identifier(self):
        """Get video identifier from either youtube_id or video_url"""
        return self.video_url or self.youtube_id

class PaymentRequest(BaseModel):
    token: str
    subscription_tier: str

# NEW: Enhanced payment models
# 🔧 UPDATED: Payment models to match my new implementation
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
    except Exception:  # ✅ SAFEST: Catch all JWT-related exceptions generically
        raise credentials_exception
        raise credentials_exception
        
    user = get_user(db, username)
    if user is None:
        raise credentials_exception
    return user

def get_or_create_stripe_customer(user, db: Session):
    """Get or create a Stripe customer for the user - Enhanced version"""
    try:
        # Check if user has stripe_customer_id attribute and it's valid
        stripe_customer_id = getattr(user, 'stripe_customer_id', None)
        
        if stripe_customer_id:
            try:
                customer = stripe.Customer.retrieve(stripe_customer_id)
                logger.info(f"✅ Found existing Stripe customer: {customer.id}")
                return customer
            except stripe.error.InvalidRequestError:
                logger.warning(f"⚠️ Invalid Stripe customer ID: {stripe_customer_id}, creating new one")
        
        # Create new customer
        logger.info(f"🆕 Creating new Stripe customer for user: {user.username}")
        customer = stripe.Customer.create(
            email=user.email,
            name=user.username,
            metadata={'user_id': str(user.id)}
        )
        
        # Try to save customer ID if user model supports it
        try:
            if hasattr(user, 'stripe_customer_id'):
                user.stripe_customer_id = customer.id
                db.commit()
                logger.info(f"💾 Saved Stripe customer ID to database")
            else:
                logger.info(f"ℹ️ User model doesn't have stripe_customer_id field - customer created but not saved")
        except Exception as save_error:
            logger.warning(f"⚠️ Could not save Stripe customer ID: {save_error}")
            # Don't fail the whole process if we can't save the ID
        
        return customer
        
    except stripe.error.StripeError as e:
        logger.error(f"❌ Stripe error creating customer: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create payment customer"
        )
    except Exception as e:
        logger.error(f"❌ Unexpected error creating Stripe customer: {str(e)}")
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

# Alternative transcript processor that bypasses the youtube-transcript-api library
def get_transcript_alternative_method(video_id: str, clean: bool = True) -> str:
    """
    Alternative transcript method using direct HTTP requests
    This bypasses the youtube-transcript-api library entirely
    """
    try:
        logger.info(f"🔄 Using alternative method for video: {video_id}")
        
        # Step 1: Get the YouTube video page
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        
        logger.info(f"📡 Fetching video page: {video_url}")
        response = requests.get(video_url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=404,
                detail=f"Video page not accessible (Status: {response.status_code})"
            )
        
        page_content = response.text
        logger.info(f"📄 Got page content: {len(page_content)} characters")
        
        # Step 2: Extract caption track information
        caption_patterns = [
            r'"captionTracks":\[(.*?)\](?=,)',
            r'"captions".*?"playerCaptionsTracklistRenderer".*?"captionTracks":\[(.*?)\]',
            r'captionTracks":\[([^\]]+)\]'
        ]
        
        caption_data = None
        for pattern in caption_patterns:
            match = re.search(pattern, page_content, re.DOTALL)
            if match:
                try:
                    # Clean up the JSON string
                    json_str = '[' + match.group(1) + ']'
                    # Fix common JSON issues
                    json_str = re.sub(r'([{,]\s*)([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:', r'\1"\2":', json_str)
                    caption_data = json.loads(json_str)
                    logger.info(f"✅ Found {len(caption_data)} caption tracks")
                    break
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse caption JSON: {e}")
                    continue
        
        if not caption_data:
            raise HTTPException(
                status_code=404,
                detail="No captions found for this video. The video may not have subtitles enabled."
            )
        
        # Step 3: Find the best caption track (prefer English)
        best_caption = None
        for caption in caption_data:
            lang_code = caption.get('languageCode', '').lower()
            if lang_code.startswith('en'):
                best_caption = caption
                break
        
        if not best_caption and caption_data:
            best_caption = caption_data[0]  # Use first available
        
        if not best_caption or 'baseUrl' not in best_caption:
            raise HTTPException(
                status_code=404,
                detail="No usable caption track found"
            )
        
        caption_url = best_caption['baseUrl']
        logger.info(f"📥 Fetching captions from: {caption_url[:100]}...")
        
        # Step 4: Fetch the caption XML
        caption_response = requests.get(caption_url, headers=headers, timeout=10)
        
        if caption_response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download caption file (Status: {caption_response.status_code})"
            )
        
        # Step 5: Parse the XML caption file
        try:
            root = ET.fromstring(caption_response.content)
            transcript_data = []
            
            for text_elem in root.findall('.//text'):
                start_time = float(text_elem.get('start', '0'))
                duration = float(text_elem.get('dur', '0'))
                text_content = text_elem.text or ''
                
                if text_content.strip():
                    # Decode HTML entities and clean up
                    text_content = unquote(text_content)
                    text_content = (text_content
                                   .replace('&amp;', '&')
                                   .replace('&lt;', '<')
                                   .replace('&gt;', '>')
                                   .replace('&quot;', '"')
                                   .replace('&#39;', "'"))
                    
                    # Remove HTML tags
                    text_content = re.sub(r'<[^>]+>', '', text_content)
                    text_content = text_content.strip()
                    
                    if text_content:
                        transcript_data.append({
                            'text': text_content,
                            'start': start_time,
                            'duration': duration
                        })
            
            if not transcript_data:
                raise HTTPException(
                    status_code=404,
                    detail="Transcript file contains no readable text"
                )
            
            logger.info(f"✅ Extracted {len(transcript_data)} transcript segments")
            
            # Step 6: Format the transcript
            if clean:
                # Clean format - text only
                text_parts = []
                for item in transcript_data:
                    text = item['text'].strip()
                    # Remove music/sound effect markers
                    if not (text.startswith('[') and text.endswith(']')) and text:
                        text_parts.append(text)
                
                return ' '.join(text_parts)
            else:
                # Unclean format - with timestamps
                formatted_parts = []
                for item in transcript_data:
                    start_time = item['start']
                    minutes = int(start_time // 60)
                    seconds = int(start_time % 60)
                    timestamp = f"[{minutes:02d}:{seconds:02d}]"
                    formatted_parts.append(f"{timestamp} {item['text']}")
                
                return '\n'.join(formatted_parts)
                
        except ET.ParseError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse caption XML: {str(e)}"
            )
            
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except requests.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Network error while fetching transcript: {str(e)}"
        )
    except Exception as e:
        logger.error(f"💥 Alternative transcript method failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Transcript extraction failed: {str(e)}"
        )

# 3. ALSO UPDATE your process_youtube_transcript_with_fallback function:
def process_youtube_transcript_with_fallback(video_id: str, clean: bool = True) -> str:
    """
    Main transcript processor with fallback to alternative method - FIXED VERSION
    """
    # Try the original library first
    try:
        logger.info(f"🔍 Trying youtube-transcript-api for video: {video_id}")
        
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

def process_youtube_transcript(video_id: str, clean: bool = True) -> str:
    """
    Process YouTube transcript - FIXED VERSION with generic exception handling
    """
    try:
        logger.info(f"🔍 Getting transcript for video: {video_id}")
        
        # Use the YouTube Transcript API
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=['en', 'en-US', 'en-GB']
        )
        
        if not transcript_list:
            raise HTTPException(
                status_code=404,
                detail="No transcript data found for this video."
            )
        
        logger.info(f"✅ Retrieved {len(transcript_list)} transcript segments")
        
        if clean:
            # Clean format - text only
            text_parts = []
            for item in transcript_list:
                if 'text' in item and item['text'].strip():
                    text = item['text'].strip()
                    text = text.replace('[Music]', '').replace('[Applause]', '').replace('[Laughter]', '').strip()
                    if text and not text.startswith('[') and not text.endswith(']'):
                        text_parts.append(text)
            
            if not text_parts:
                raise HTTPException(
                    status_code=404,
                    detail="Transcript contains no readable text content."
                )
            
            return ' '.join(text_parts)
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
            
            if not formatted_transcript:
                raise HTTPException(
                    status_code=404,
                    detail="Transcript contains no content with timestamps."
                )
            
            return '\n'.join(formatted_transcript)
            
    except Exception as e:
        error_msg = str(e).lower()
        error_type = type(e).__name__
        logger.error(f"💥 Error getting transcript for {video_id}: {str(e)} (Type: {error_type})")
        
        # Handle all exceptions generically using error message patterns
        if "transcript" in error_msg and "disabled" in error_msg:
            raise HTTPException(
                status_code=404,
                detail="Transcripts are disabled for this video."
            )
        elif "no transcript" in error_msg or "not found" in error_msg:
            raise HTTPException(
                status_code=404,
                detail="No transcript available for this video. Try a different video with captions enabled."
            )
        elif "video unavailable" in error_msg or "private" in error_msg:
            raise HTTPException(
                status_code=404,
                detail="Video is unavailable, private, or doesn't exist."
            )
        elif "too many requests" in error_msg or "rate limit" in error_msg:
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please wait a moment and try again."
            )
        elif "could not retrieve" in error_msg:
            raise HTTPException(
                status_code=404,
                detail="No transcript available for this video. Please try a different video."
            )
        elif "subtitles are disabled" in error_msg:
            raise HTTPException(
                status_code=404,
                detail="Subtitles are disabled for this video."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve transcript: {str(e)}"
            )

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

#=====================================================
# Replace your download_transcript endpoint in main.py with this SIMPLE version
# This goes back to your original working approach with better error messages

@app.post("/download_transcript/")
async def download_transcript(
    request: TranscriptRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Simple transcript download - BACK TO WORKING VERSION"""
    
    try:
        # Get video ID from request
        video_identifier = getattr(request, 'video_url', None) or request.youtube_id
        
        if not video_identifier:
            return {
                "success": False,
                "error_type": "missing_identifier",
                "message": "Please provide a YouTube URL or video ID"
            }
        
        # Simple video ID extraction (your original method)
        video_id = video_identifier.strip()
        
        # If it's a URL, extract the ID
        if 'youtube.com' in video_id or 'youtu.be' in video_id:
            patterns = [
                r'(?:youtube\.com\/watch\?v=)([a-zA-Z0-9_-]{11})',
                r'(?:youtu\.be\/)([a-zA-Z0-9_-]{11})',
                r'(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, video_id)
                if match:
                    video_id = match.group(1)
                    break
        
        # Validate video ID length
        if len(video_id) != 11:
            return {
                "success": False,
                "error_type": "invalid_url",
                "message": "Please enter a valid YouTube URL or 11-character Video ID"
            }
        
        logger.info(f"Processing transcript request for video: {video_id}")
        
        # Check subscription limits
        try:
            transcript_type = "clean_transcripts" if request.clean_transcript else "unclean_transcripts"
            can_download = check_user_limits(user, transcript_type, db)
            if not can_download:
                return {
                    "success": False,
                    "error_type": "subscription_limit",
                    "message": "Monthly limit reached! Please upgrade your plan."
                }
        except Exception as e:
            logger.warning(f"Subscription check error: {e}")
        
        # Use the SIMPLE YouTube Transcript API approach (your original working method)
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            
            logger.info(f"Getting transcript for video: {video_id}")
            
            # Simple direct call - this was working for you before
            transcript_list = YouTubeTranscriptApi.get_transcript(
                video_id,
                languages=['en', 'en-US', 'en-GB']
            )
            
            if not transcript_list:
                return {
                    "success": False,
                    "error_type": "no_transcript",
                    "message": "No captions found for this video. Try a different video with captions enabled."
                }
            
            logger.info(f"Successfully retrieved {len(transcript_list)} transcript segments")
            
            # Format the transcript (your original working method)
            if request.clean_transcript:
                # Clean format - text only
                text_parts = []
                for item in transcript_list:
                    if 'text' in item and item['text'].strip():
                        text = item['text'].strip()
                        # Remove sound effect markers
                        if not (text.startswith('[') and text.endswith(']')):
                            text_parts.append(text)
                
                transcript_content = ' '.join(text_parts)
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
                
                transcript_content = '\n'.join(formatted_transcript)
            
            if not transcript_content.strip():
                return {
                    "success": False,
                    "error_type": "no_transcript",
                    "message": "Transcript is empty or contains no readable content."
                }
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"Transcript extraction failed for {video_id}: {str(e)}")
            
            # Better error messages based on the actual error
            if "no transcript" in error_msg or "could not retrieve" in error_msg:
                return {
                    "success": False,
                    "error_type": "no_transcript", 
                    "message": "No captions found for this video. The video may not have subtitles enabled.",
                    "suggestions": [
                        "Try a different video with captions enabled",
                        "Check if the video has closed captions (CC) available",
                        "Use one of the working example videos provided"
                    ]
                }
            elif "disabled" in error_msg:
                return {
                    "success": False,
                    "error_type": "transcripts_disabled",
                    "message": "Transcripts are disabled for this video."
                }
            elif "video unavailable" in error_msg or "private" in error_msg:
                return {
                    "success": False,
                    "error_type": "video_not_found",
                    "message": "Video is unavailable, private, or doesn't exist."
                }
            else:
                return {
                    "success": False,
                    "error_type": "retrieval_failed",
                    "message": "Failed to retrieve transcript. Please try a different video.",
                    "debug_info": str(e)
                }
        
        # Record successful download in database  
        try:
            transcript_type_db = "clean" if request.clean_transcript else "unclean"
            new_download = TranscriptDownload(
                user_id=user.id,
                youtube_id=video_id,
                transcript_type=transcript_type_db,
                created_at=datetime.now()
            )
            db.add(new_download)
            db.commit()
            logger.info(f"SUCCESS: {user.username} downloaded {transcript_type_db} transcript for {video_id}")
        except Exception as e:
            logger.warning(f"Failed to update usage tracking: {e}")
        
        # Return response compatible with your frontend
        return {
            "success": True,
            "transcript": transcript_content,
            "youtube_id": video_id,
            "message": "Transcript downloaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in download_transcript: {str(e)}")
        return {
            "success": False,
            "error_type": "internal_error",
            "message": "An unexpected error occurred. Please try again.",
            "debug_info": str(e)
        }
#================================================

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

# FIND: library_info endpoint at http://localhost:8000/library_info 
@app.get("/library_info")
async def library_info():
    """Get library version info - FIXED VERSION"""
    try:
        # Try different methods to get version
        import pkg_resources
        
        try:
            version = pkg_resources.get_distribution("youtube-transcript-api").version
        except:
            try:
                from importlib import metadata
                version = metadata.version("youtube-transcript-api")
            except:
                version = "1.1.0 (estimated)"
        
        return {
            "youtube_transcript_api_version": version,
            "pyjwt_version": "2.8.0",
            "status": "working"
        }
    except Exception as e:
        return {
            "youtube_transcript_api_version": "error",
            "error": str(e),
            "status": "error"
        }

# FIND: debug_network endpoint at http://localhost:8000/debug_network 
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


# FIND: debug_transcript_raw endpoint at http://localhost:8000/debug_transcript_raw 
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

#===========================

# ADD THESE NEW DEBUG ENDPOINTS (if not already present):
@app.get("/debug/video_check/{video_id}")
async def debug_video_check(video_id: str):
    """Debug endpoint to check video accessibility"""
    try:
        video_check = transcript_handler.check_video_exists(video_id)
        return {
            "video_id": video_id,
            "check_result": video_check,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e), "video_id": video_id}

@app.get("/debug/transcript_availability/{video_id}")
async def debug_transcript_availability(video_id: str):
    """Debug endpoint to check what transcripts are available"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        available_transcripts = []
        for transcript in transcript_list:
            transcript_info = {
                'language': transcript.language,
                'language_code': transcript.language_code,
                'is_generated': transcript.is_generated,
                'is_translatable': transcript.is_translatable
            }
            available_transcripts.append(transcript_info)
        
        return {
            "video_id": video_id,
            "available_transcripts": available_transcripts,
            "count": len(available_transcripts),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "video_id": video_id,
            "error": str(e),
            "available_transcripts": [],
            "count": 0
        }

@app.get("/debug/test_transcript/{video_id}")
async def debug_test_transcript(video_id: str):
    """Debug endpoint to test transcript retrieval"""
    try:
        transcript_data, method_used = transcript_handler.get_transcript_with_fallbacks(video_id)
        
        # Return just first few entries for testing
        sample_data = transcript_data[:3] if len(transcript_data) > 3 else transcript_data
        
        return {
            "video_id": video_id,
            "success": True,
            "method_used": method_used,
            "total_entries": len(transcript_data),
            "sample_entries": sample_data,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "video_id": video_id,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

#============================================
# ALSO ADD these test endpoints to help with debugging:

@app.get("/test_working_videos/")
async def test_working_videos():
    """Test endpoint with known working video IDs"""
    working_videos = [
        {
            "id": "ZbZSe6N_BXs",
            "title": "PHP in 100 Seconds",
            "url": "https://www.youtube.com/watch?v=ZbZSe6N_BXs",
            "description": "Short tech video with captions"
        },
        {
            "id": "kqtD5dpn9C8", 
            "title": "Python vs JavaScript",
            "url": "https://www.youtube.com/watch?v=kqtD5dpn9C8",
            "description": "Programming comparison video"
        },
        {
            "id": "UBP-ogtJLpg",
            "title": "How to Build a Simple Calculator",
            "url": "https://www.youtube.com/watch?v=UBP-ogtJLpg", 
            "description": "Tutorial video with captions"
        }
    ]
    
    results = []
    for video in working_videos:
        try:
            check_result = transcript_handler.check_video_exists(video["id"])
            results.append({
                "video": video,
                "exists": check_result.get("exists", False),
                "status": "OK" if check_result.get("exists", False) else "FAILED"
            })
        except Exception as e:
            results.append({
                "video": video,
                "exists": False,
                "status": "ERROR",
                "error": str(e)
            })
    
    return {
        "test_results": results,
        "recommendation": "Use videos marked as 'OK' for testing"
    }

@app.get("/debug_transcript/{video_id}")
async def debug_transcript_detailed(video_id: str):
    """Detailed debug endpoint for transcript issues"""
    debug_info = {
        "video_id": video_id,
        "timestamp": datetime.now().isoformat()
    }
    
    # Step 1: Check if video exists
    try:
        video_check = transcript_handler.check_video_exists(video_id)
        debug_info["video_check"] = video_check
    except Exception as e:
        debug_info["video_check"] = {"error": str(e)}
    
    # Step 2: Try each method individually
    methods_results = {}
    
    # Test YouTube Transcript API
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        methods_results["youtube_transcript_api"] = {
            "status": "SUCCESS",
            "segments": len(transcript),
            "sample": transcript[0] if transcript else None
        }
    except Exception as e:
        methods_results["youtube_transcript_api"] = {
            "status": "FAILED", 
            "error": str(e)
        }
    
    # Test our enhanced handler
    try:
        transcript_data, method_used = transcript_handler.get_transcript_with_fallbacks(video_id)
        methods_results["enhanced_handler"] = {
            "status": "SUCCESS",
            "method_used": method_used,
            "segments": len(transcript_data),
            "sample": transcript_data[0] if transcript_data else None
        }
    except Exception as e:
        methods_results["enhanced_handler"] = {
            "status": "FAILED",
            "error": str(e)
        }
    
    debug_info["methods_results"] = methods_results
    
    return debug_info

#=========================================
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
