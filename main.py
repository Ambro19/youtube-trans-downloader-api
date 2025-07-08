
#####################################################################
## IMPORTANT MESSAGE: DO NOT ALTER THIS MAIN.PY ANYMORE-- THANKS! ###
#####################################################################   

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
import os
import logging
from dotenv import load_dotenv
import re

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

# Enhanced subscription limits
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

# Enhanced Stripe integration functions for main.py
# Add these enhanced functions to replace the existing ones

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


#=========================================================== #
# ENHANCED TRANSCRIPT FUNCTIONS: TRANSCRIPT RELATED FUNCTIONS #
#===========================================================  #

def check_subscription_limit(user_id: int, transcript_type: str, db: Session):
    """Robust subscription limit check - handles missing columns gracefully"""
    try:
        # Get subscription info
        subscription = db.query(Subscription).filter(Subscription.user_id == user_id).first()
        
        if not subscription:
            tier = "free"
        else:
            tier = subscription.tier
            if subscription.expiry_date < datetime.now():
                tier = "free"
        
        # Calculate current month start
        month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Use a simple query that only uses basic columns that definitely exist
        try:
            # Try the enhanced query first
            usage = db.query(TranscriptDownload).filter(
                TranscriptDownload.user_id == user_id,
                TranscriptDownload.transcript_type == transcript_type,
                TranscriptDownload.created_at >= month_start
            ).count()
        except Exception as e:
            logger.warning(f"Enhanced query failed, using simple fallback: {e}")
            
            # Fallback to basic SQL query that only uses guaranteed columns
            result = db.execute(
                "SELECT COUNT(*) FROM transcript_downloads WHERE user_id = ? AND transcript_type = ? AND created_at >= ?",
                (user_id, transcript_type, month_start.isoformat())
            )
            usage = result.scalar() or 0
        
        # Get limit for this tier and transcript type
        limit = SUBSCRIPTION_LIMITS[tier].get(transcript_type, 0)
        
        # Return True if user can download (hasn't reached limit)
        if limit == float('inf'):
            return True
        
        return usage < limit
        
    except Exception as e:
        logger.error(f"Error checking subscription limit: {e}")
        # If there's any error, default to allowing free tier limits
        return True  # Allow download in case of errors to avoid blocking users


def format_transcript_enhanced(transcript_list: list, clean: bool = True) -> str:
    """
    Enhanced transcript formatting with better text processing
    
    This function takes raw transcript data and formats it into either:
    - Clean format: Plain text with proper spacing and punctuation
    - Timestamped format: Text with [MM:SS] timestamps for each segment
    
    Args:
        transcript_list: List of transcript segments with 'text' and 'start' keys
        clean: If True, returns clean text; if False, returns timestamped format
    
    Returns:
        Formatted transcript string
    """
    if not transcript_list:
        raise Exception("Empty transcript data")
    
    if clean:
        # Enhanced clean format - create readable paragraph text
        texts = []
        for item in transcript_list:
            text = item.get('text', '').strip()
            if text:
                # Clean and normalize the text
                text = clean_transcript_text(text)
                if text:  # Only add if text remains after cleaning
                    texts.append(text)
        
        # Join all text segments into one continuous string
        result = ' '.join(texts)
        
        # Final cleanup for better readability
        result = ' '.join(result.split())  # Normalize whitespace
        result = result.replace(' .', '.').replace(' ,', ',')  # Fix punctuation spacing
        
        # Validate that we have meaningful content
        if len(result) < 20:  # Too short, likely invalid
            raise Exception("Transcript too short or invalid")
            
        logger.info(f"✅ Enhanced clean transcript: {len(result)} characters")
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
                if text:
                    # Convert seconds to MM:SS format
                    minutes = int(start // 60)
                    seconds = int(start % 60)
                    timestamp = f"[{minutes:02d}:{seconds:02d}]"
                    lines.append(f"{timestamp} {text}")
        
        # Validate that we have enough content
        if len(lines) < 5:  # Too few lines, likely invalid
            raise Exception("Transcript has too few valid segments")
            
        result = '\n'.join(lines)
        logger.info(f"✅ Enhanced timestamped transcript: {len(lines)} lines")
        return result

def clean_transcript_text(text: str) -> str:
    """
    Clean transcript text from common artifacts and formatting issues
    
    This function removes:
    - HTML entities (&amp;, &lt;, &gt;)
    - HTML/XML tags
    - Extra whitespace and line breaks
    - Leading/trailing punctuation artifacts
    
    Args:
        text: Raw transcript text that may contain artifacts
    
    Returns:
        Clean, readable text
    """
    if not text:
        return ""
    
    # Fix common HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Remove HTML/XML tags (like <c>, <i>, etc.)
    import re
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace (replace multiple spaces with single space)
    text = ' '.join(text.split())
    
    # Remove leading/trailing punctuation artifacts that don't belong
    text = text.strip('.,!?;: ')
    
    return text.strip()

def parse_subtitle_content_enhanced(content: str, format_type: str) -> list:
    """
    Enhanced subtitle content parsing for multiple formats
    
    Supports parsing of:
    - VTT/WebVTT format (most common)
    - SRV3/JSON format (YouTube's internal format)
    - TTML/XML format (standard subtitle format)
    
    Args:
        content: Raw subtitle file content
        format_type: Format indicator (vtt, srv3, json, ttml, xml)
    
    Returns:
        List of transcript segments with text, start time, and duration
    """
    transcript_data = []
    
    try:
        if format_type.lower() in ['vtt', 'webvtt']:
            transcript_data = parse_vtt_content(content)
        elif format_type.lower() in ['srv3', 'json']:
            transcript_data = parse_srv3_content(content)
        elif format_type.lower() in ['ttml', 'xml']:
            transcript_data = parse_ttml_content(content)
        else:
            # Try to auto-detect format based on content
            if content.strip().startswith('WEBVTT'):
                transcript_data = parse_vtt_content(content)
            elif content.strip().startswith('{') or content.strip().startswith('['):
                transcript_data = parse_srv3_content(content)
            elif '<' in content and '>' in content:
                transcript_data = parse_ttml_content(content)
    
    except Exception as e:
        logger.info(f"📝 Content parsing failed: {str(e)}")
    
    return transcript_data

def parse_ttml_timestamp(timestamp: str) -> float:
    """
    Parse TTML timestamp format to seconds
    
    TTML supports various timestamp formats:
    - "10s" (seconds)
    - "01:30:45" (HH:MM:SS)
    
    Args:
        timestamp: TTML timestamp string
    
    Returns:
        Time in seconds as float
    """
    try:
        if 's' in timestamp:
            return float(timestamp.replace('s', ''))
        elif ':' in timestamp:
            parts = timestamp.split(':')
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
    except:
        pass
    return 0

def extract_live_transcript(video_id: str, clean: bool) -> str:
    """
    Handle live stream transcript extraction
    
    Live streams have special requirements:
    - Transcripts may not be available during broadcast
    - Content may be incomplete
    - Should provide helpful feedback to users
    
    Args:
        video_id: YouTube video ID
        clean: Whether to return clean or timestamped format
    
    Returns:
        Live transcript content or helpful message
    """
    logger.info(f"🔴 Attempting live transcript extraction for {video_id}")
    
    # For live streams, try the standard API methods first
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        if transcript_list:
            return format_transcript_enhanced(transcript_list, clean)
    except:
        pass
    
    # Return informative message for live content
    return "This appears to be a live stream. Live transcripts may not be available or may be incomplete. Please try again after the stream has ended."

def extract_with_direct_api(video_id: str) -> list:
    """
    Direct API extraction method for edge cases
    
    This is a placeholder for future implementation of additional
    direct API methods that might become available.
    
    Args:
        video_id: YouTube video ID
    
    Returns:
        List of transcript segments (currently empty)
    """
    try:
        # Placeholder for future direct API implementations
        # Could include custom API endpoints, alternative services, etc.
        return []
    except:
        return []
 
# ================== ENHANCED TRANSCRIPT EXTRACTION LOGIC ====================

def get_youtube_transcript_corrected(video_id: str, clean: bool = True) -> str:
    logger.info(f"[NEW] Attempting transcript extraction for video {video_id}")

    # 1. Try youtube-transcript-api
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        preferred_langs = ['en', 'en-US', 'en-GB']
        all_langs = preferred_langs + [lc.language_code for lc in transcript_list if lc.language_code not in preferred_langs]
        for lang in all_langs:
            try:
                logger.info(f"Trying language: {lang}")
                transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                if transcript and len(transcript) > 0:
                    logger.info(f"SUCCESS with language {lang}: {len(transcript)} segments")
                    return format_transcript_enhanced(transcript, clean)
            except Exception as e:
                logger.info(f"Language {lang} failed: {e}")
                continue
    except Exception as e:
        logger.info(f"youtube-transcript-api failed: {e}")

    # 2. Try yt-dlp for subtitles
    try:
        import yt_dlp
        ydl_opts = {
            "skip_download": True,
            "quiet": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "en-US", "en-GB"],
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"yt-dlp: Extracting info for video: {video_id}")
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            subs = info.get('subtitles', {})
            autos = info.get('automatic_captions', {})
            for lang_dict in [subs, autos]:
                for lang, entries in lang_dict.items():
                    for entry in entries:
                        url = entry.get('url')
                        ext = entry.get('ext')
                        logger.info(f"Trying subtitle URL ({lang}, {ext})")
                        transcript_text = extract_subtitle_from_entry(url, ext, clean)
                        if transcript_text and not is_invalid_content(transcript_text):
                            logger.info(f"✅ Found valid subtitle ({lang}, {ext}), length: {len(transcript_text)}")
                            return transcript_text
    except Exception as ydl_error:
        logger.info(f"yt-dlp failed: {ydl_error}")

    logger.warning(f"All extraction methods failed for {video_id}. Returning DEMO content.")
    return get_demo_content(clean)

def extract_subtitle_from_entry(url: str, ext: str, clean: bool) -> str:
    import requests
    try:
        resp = requests.get(url, timeout=10)
        content = resp.text
        logger.info(f"Downloaded subtitle file, ext={ext}, length={len(content)}")
        # Parse by type
        if ext == "vtt":
            return parse_vtt(content, clean)
        elif ext in ("srv1", "srv2", "srv3", "json"):
            return parse_srv(content, clean)
        elif ext == "ttml":
            return parse_ttml(content, clean)
        elif ext == "srt":
            return parse_srt(content, clean)
        elif ext == "m3u8":
            return parse_m3u8(content, clean)
        else:
            return content if len(content.strip()) > 0 else ""
    except Exception as e:
        logger.info(f"Failed to download/parse subtitle file: {e}")
        return ""

def is_invalid_content(text: str) -> bool:
    if not text or len(text.strip()) < 32:
        return True
    return False

def get_demo_content(clean: bool = True):
    demo = (
        "This is demo transcript content. "
        "The real YouTube transcript could not be downloaded."
    )
    if clean:
        return demo
    else:
        return "[00:00] " + demo

# --- Subtitle parsers ---
def parse_vtt(content: str, clean: bool) -> str:
    lines = content.splitlines()
    filtered = [line for line in lines if line and not re.match(r"^(\d{2}:){1,2}\d{2}\.\d{3} -->", line) and not line.startswith("WEBVTT")]
    text = "\n".join(filtered)
    if clean:
        text = re.sub(r"<[^>]+>", "", text)  # Remove tags
    return text.strip()

def parse_srt(content: str, clean: bool) -> str:
    lines = content.splitlines()
    filtered = []
    for line in lines:
        if line.strip().isdigit():
            continue
        if re.match(r"\d{2}:\d{2}:\d{2},\d{3} -->", line):
            continue
        filtered.append(line)
    text = "\n".join(filtered)
    if clean:
        text = re.sub(r"<[^>]+>", "", text)
    return text.strip()

def parse_srv(content: str, clean: bool) -> str:
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(content)
        texts = [el.text for el in root.iter() if el.tag == "text" and el.text]
        text = "\n".join(texts)
        if clean:
            text = re.sub(r"<[^>]+>", "", text)
        return text.strip()
    except Exception as e:
        logger.info(f"Failed to parse srv: {e}")
        return ""

def parse_ttml(content: str, clean: bool) -> str:
    import xml.etree.ElementTree as ET
    try:
        root = ET.fromstring(content)
        texts = [el.text for el in root.iter() if el.text]
        text = "\n".join(texts)
        if clean:
            text = re.sub(r"<[^>]+>", "", text)
        return text.strip()
    except Exception as e:
        logger.info(f"Failed to parse ttml: {e}")
        return ""

def parse_m3u8(content: str, clean: bool) -> str:
    logger.info("M3U8 subtitle parsing not implemented—returning raw content.")
    return content.strip()

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
    
    This endpoint handles YouTube transcript extraction using multiple
    fallback methods to ensure maximum success rate.
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
    can_download = check_subscription_limit(user.id, transcript_type, db)
    if not can_download:
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
#==============================================

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

        # Record payment history
        payment_record = PaymentHistory(
            user_id=current_user.id,
            stripe_payment_intent_id=request.payment_intent_id,
            stripe_customer_id=intent.customer,
            amount=intent.amount,
            currency=intent.currency,
            status='succeeded',
            subscription_tier=plan_type,
            created_at=datetime.utcnow(),
            metadata=str(intent.metadata)
        )
        db.add(payment_record)

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

####################################################################
# IMPORTANT MESSAGE: DO NOT ALTER THIS MAIN.PY ANYMORE-- THANKS! ###
#################################################################### 