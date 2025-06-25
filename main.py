# main.py - YouTube Transcript Downloader API
# CLEANED VERSION WITH INLINE COMMENTS - NO DEMO CONTENT

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
import ssl
import sys
import xml.etree.ElementTree as ET  # For parsing YouTube's XML transcript format

import warnings
warnings.filterwarnings("ignore", message=".*bcrypt.*")

# Import from database.py
from database import get_db, User, Subscription, TranscriptDownload, create_tables

# Load environment variables from .env file
load_dotenv()

# Configure application logging
logger = logging.getLogger("youtube_trans_downloader.main")

# Stripe payment configuration
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
DOMAIN = os.getenv("DOMAIN", "https://youtube-trans-downloader-api.onrender.com")

# Create FastAPI application instance
app = FastAPI(
    title="YouTubeTransDownloader API", 
    description="API for downloading and processing YouTube video transcripts",
    version="1.0.0"
)

# Environment-aware configuration for different deployment stages
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Configure CORS (Cross-Origin Resource Sharing) based on environment
if ENVIRONMENT == "production":
    # Production allowed origins - restrict to specific domains
    allowed_origins = [
        "http://localhost:8000",
        "https://youtube-trans-downloader-api.onrender.com",
        FRONTEND_URL
    ]
    logger.info(f"🌍 Production mode - CORS origins: {allowed_origins}")
else:
    # Development allowed origins - more permissive for local development
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        FRONTEND_URL
    ]
    logger.info(f"🔧 Development mode - CORS origins: {allowed_origins}")

# Add CORS middleware to allow frontend-backend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,  # Allow cookies and auth headers
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],  # Allow all headers
)

# Authentication setup using OAuth2 with password flow
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT (JSON Web Token) configuration for user authentication
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# Subscription tier limits - defines what each subscription level can access
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
        # Unlimited access for premium users
        "transcript": float('inf'), "audio": float('inf'), "video": float('inf'), 
        "clean": float('inf'), "unclean": float('inf'),
        "clean_transcripts": float('inf'), "unclean_transcripts": float('inf'),
        "audio_downloads": float('inf'), "video_downloads": float('inf')
    }
}

# Stripe price ID mapping for different subscription tiers
PRICE_ID_MAP = {
    "pro": os.getenv("PRO_PRICE_ID"),
    "premium": os.getenv("PREMIUM_PRICE_ID")
}

@app.on_event("startup")
async def startup_event():
    """
    Application startup event handler
    Validates environment variables and initializes the database
    """
    try:
        logger.info("🚀 Starting YouTube Transcript Downloader API...")
        logger.info(f"🌍 Environment: {ENVIRONMENT}")
        logger.info(f"🔗 Domain: {DOMAIN}")
        
        # Validate that critical environment variables are set
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
                # Log confirmation that variable is set (show first 8 chars for security)
                logger.info(f"✅ {var}: {value[:8]}..." if len(value) > 8 else f"✅ {var}: SET")
        
        if missing_vars:
            logger.error(f"❌ Missing required environment variables:")
            for var in missing_vars:
                logger.error(f"   - {var}")
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
        
        # Initialize database tables
        create_tables()
        logger.info("✅ Database initialized successfully")
        logger.info("🎉 Application startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

#====================================================
# PYDANTIC MODELS - Data validation and serialization
#====================================================

class Token(BaseModel):
    """JWT token response model"""
    access_token: str
    token_type: str

class TokenData(BaseModel):
    """JWT token payload data"""
    username: Optional[str] = None

class UserCreate(BaseModel):
    """User registration request model"""
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    """User information response model"""
    id: int
    username: str
    email: str
    created_at: datetime
    
    class Config:
        from_attributes = True  # Allow conversion from SQLAlchemy models

class TranscriptRequest(BaseModel):
    """Transcript download request model"""
    youtube_id: str  # YouTube video ID or URL
    clean_transcript: bool = False  # Whether to return clean (no timestamps) format

class CreatePaymentIntentRequest(BaseModel):
    """Payment intent creation request model"""
    price_id: str  # Stripe price ID for the subscription tier

class ConfirmPaymentRequest(BaseModel):
    """Payment confirmation request model"""
    payment_intent_id: str  # Stripe payment intent ID

class SubscriptionRequest(BaseModel):
    """Subscription creation request model"""
    token: Optional[str] = None
    subscription_tier: str  # "pro" or "premium"

class SubscriptionResponse(BaseModel):
    """Subscription status response model"""
    tier: str
    status: str
    expiry_date: Optional[str] = None
    limits: dict
    usage: Optional[dict] = None
    remaining: Optional[dict] = None
    
    class Config:
        from_attributes = True

#================================================
# HELPER FUNCTIONS - Authentication and utilities
#================================================

def verify_password(plain_password, hashed_password):
    """Verify a plain password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    """Generate a hash for a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create a JWT access token
    Args:
        data: Payload data to encode in the token
        expires_delta: Optional custom expiration time
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(db: Session, username: str):
    """Get user by username from database"""
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str):
    """Get user by email from database"""
    return db.query(User).filter(User.email == email).first()

def authenticate_user(db: Session, username: str, password: str):
    """
    Authenticate user with username and password
    Returns user object if authentication successful, False otherwise
    """
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """
    Dependency to get current authenticated user from JWT token
    Used in route dependencies to ensure user is authenticated
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode and validate JWT token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
        
    # Get user from database
    user = get_user(db, username)
    if user is None:
        raise credentials_exception
    return user

def get_or_create_stripe_customer(user, db: Session):
    """
    Get existing Stripe customer or create a new one for the user
    Used for payment processing
    """
    try:
        # Check if user already has a Stripe customer ID
        if hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
            try:
                customer = stripe.Customer.retrieve(user.stripe_customer_id)
                return customer
            except stripe.error.InvalidRequestError:
                pass  # Customer doesn't exist, create new one
        
        # Create new Stripe customer
        customer = stripe.Customer.create(
            email=user.email,
            name=user.username,
            metadata={'user_id': str(user.id)}
        )
        
        # Save customer ID to user record if supported
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

def check_subscription_limit(user_id: int, transcript_type: str, db: Session):
    """
    Check if user has exceeded their subscription limits for the current month
    Args:
        user_id: Database ID of the user
        transcript_type: Type of transcript ("clean" or "unclean")
        db: Database session
    Returns:
        Boolean indicating if user can download more transcripts
    """
    # Get user's current subscription
    subscription = db.query(Subscription).filter(Subscription.user_id == user_id).first()
    
    # Determine subscription tier
    if not subscription:
        tier = "free"
    else:
        tier = subscription.tier
        # Check if subscription has expired
        if subscription.expiry_date < datetime.now():
            tier = "free"
    
    # Calculate current month's usage
    month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    usage = db.query(TranscriptDownload).filter(
        TranscriptDownload.user_id == user_id,
        TranscriptDownload.transcript_type == transcript_type,
        TranscriptDownload.created_at >= month_start
    ).count()
    
    # Check against subscription limits
    limit = SUBSCRIPTION_LIMITS[tier][transcript_type]
    if usage >= limit:
        return False  # Limit exceeded
    return True  # Can still download

#=====================================   ===============
# REAL TRANSCRIPT EXTRACTION FUNCTIONS - NO DEMO CONTENT
#=====================================   =============== 

def get_real_youtube_transcript(video_id: str, clean: bool = True) -> str:
    """
    Extract REAL YouTube transcripts using multiple methods
    Args:
        video_id: 11-character YouTube video ID
        clean: Whether to return clean text (True) or timestamped format (False)
    Returns:
        Extracted transcript text
    Raises:
        HTTPException: If no transcript is available
    """
    logger.info(f"🎯 REAL transcript extraction for: {video_id}")
    
    # Method 1: YouTube Transcript API (most reliable method)
    try:
        logger.info("🚀 Trying YouTube Transcript API...")
        
        # Multiple API approaches with different language preferences
        approaches = [
            # Try English only first
            lambda: YouTubeTranscriptApi.get_transcript(video_id, languages=['en']),
            # Try English variants
            lambda: YouTubeTranscriptApi.get_transcript(video_id, languages=['en', 'en-US', 'en-GB']),
            # Try with formatting preservation
            lambda: YouTubeTranscriptApi.get_transcript(video_id, languages=['en'], preserve_formatting=True),
            # Try any available language as last resort
            lambda: YouTubeTranscriptApi.get_transcript(video_id),
        ]
        
        for i, approach in enumerate(approaches):
            try:
                logger.info(f"🔄 API Approach {i+1}...")
                transcript_list = approach()
                
                if transcript_list and len(transcript_list) > 0:
                    logger.info(f"✅ API SUCCESS: {len(transcript_list)} segments found")
                    
                    # Format the transcript based on requested format
                    if clean:
                        # Clean format - just the spoken text without timestamps
                        texts = []
                        for item in transcript_list:
                            text = item.get('text', '').strip()
                            # Filter out sound effects and music markers
                            if text and not (text.startswith('[') and text.endswith(']')):
                                texts.append(text)
                        
                        result = ' '.join(texts)
                        logger.info(f"✅ REAL CLEAN TRANSCRIPT: {len(result)} characters")
                        return result
                    else:
                        # Timestamped format - includes timing information
                        lines = []
                        for item in transcript_list:
                            start = item.get('start', 0)
                            text = item.get('text', '').strip()
                            if text:
                                # Convert seconds to MM:SS format
                                minutes = int(start // 60)
                                seconds = int(start % 60)
                                timestamp = f"[{minutes:02d}:{seconds:02d}]"
                                lines.append(f"{timestamp} {text}")
                        
                        result = '\n'.join(lines)
                        logger.info(f"✅ REAL TIMESTAMPED TRANSCRIPT: {len(lines)} lines")
                        return result
                        
            except Exception as e:
                logger.info(f"⚠️ API Approach {i+1} failed: {e}")
                continue
        
        logger.warning("⚠️ All API approaches failed")
        
    except Exception as e:
        logger.warning(f"⚠️ YouTube API completely failed: {e}")
    
    # Method 2: HTTP-based extraction (fallback when API fails)
    try:
        logger.info("🔄 Trying HTTP extraction...")
        return extract_transcript_http_robust(video_id, clean)
        
    except Exception as e:
        logger.warning(f"⚠️ HTTP extraction failed: {e}")
    
    # No more demo content - return proper error
    logger.error(f"❌ FAILED to extract real transcript for {video_id}")
    raise HTTPException(
        status_code=404, 
        detail=f"No transcript available for video {video_id}. This video may not have captions enabled."
    )

def extract_transcript_http_robust(video_id: str, clean: bool = True) -> str:
    """
    HTTP-based transcript extraction for cases where the API fails
    Scrapes YouTube page to find caption URLs and downloads transcript content
    """
    try:
        logger.info(f"🌐 HTTP extraction for: {video_id}")
        
        # Construct YouTube video URL
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        # Headers that mimic a real browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        }
        
        # Fetch the YouTube video page
        response = requests.get(video_url, headers=headers, timeout=20)
        if response.status_code != 200:
            raise Exception(f"Failed to load video page: {response.status_code}")
        
        page_content = response.text
        logger.info(f"📄 Page loaded: {len(page_content)} chars")
        
        # Extract caption URL from page content
        caption_url = None
        
        # Strategy 1: Look for captionTracks in the page JavaScript
        patterns = [
            r'"captionTracks":\s*(\[.*?\])',  # Standard pattern
            r'"captions".*?"captionTracks":\s*(\[.*?\])'  # Alternative pattern
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, page_content)
            for match in matches:
                try:
                    # Clean and parse the JSON data
                    json_str = match
                    # Fix unquoted JavaScript object keys
                    json_str = re.sub(r'([{,]\s*)([a-zA-Z_$][a-zA-Z0-9_$]*)\s*:', r'\1"\2":', json_str)
                    caption_tracks = json.loads(json_str)
                    
                    if caption_tracks and len(caption_tracks) > 0:
                        # Find English caption track
                        target_track = None
                        for track in caption_tracks:
                            if isinstance(track, dict) and 'baseUrl' in track:
                                lang = track.get('languageCode', '').lower()
                                if lang.startswith('en'):  # Prefer English
                                    target_track = track
                                    break
                        
                        # Use first available track if no English found
                        if not target_track and caption_tracks:
                            target_track = caption_tracks[0]
                        
                        if target_track and 'baseUrl' in target_track:
                            # Clean up the URL (decode HTML entities)
                            caption_url = target_track['baseUrl'].replace('\\u0026', '&').replace('\\/', '/')
                            logger.info("✅ Found caption URL from tracks")
                            break
                            
                except Exception as e:
                    continue  # Try next match
            
            if caption_url:
                break  # Found a working URL
        
        # Strategy 2: Direct URL search if track parsing failed
        if not caption_url:
            url_patterns = [
                r'"baseUrl":\s*"(https://www\.youtube\.com/api/timedtext[^"]*)"',
                r'(https://www\.youtube\.com/api/timedtext[^"&\s]*)'
            ]
            
            for pattern in url_patterns:
                matches = re.findall(pattern, page_content)
                if matches:
                    caption_url = matches[0].replace('\\u0026', '&').replace('\\/', '/')
                    logger.info("✅ Found caption URL directly")
                    break
        
        if not caption_url:
            raise Exception("No caption URL found in page")
        
        # Fetch the actual transcript content
        logger.info("📥 Fetching real transcript content...")
        caption_response = requests.get(caption_url, headers=headers, timeout=15)
        
        if caption_response.status_code != 200:
            raise Exception(f"Failed to fetch captions: {caption_response.status_code}")
        
        # Parse and return the transcript content
        return parse_real_transcript_content(caption_response.text, clean)
        
    except Exception as e:
        logger.error(f"❌ HTTP extraction failed: {e}")
        raise

def parse_real_transcript_content(content: str, clean: bool = True) -> str:
    """
    Parse transcript content from YouTube's caption format (typically XML)
    Args:
        content: Raw caption content from YouTube
        clean: Whether to return clean text or timestamped format
    Returns:
        Formatted transcript text
    """
    try:
        logger.info(f"📝 Parsing real transcript content...")
        
        # YouTube captions are typically in XML format
        transcript_segments = []
        
        # Method 1: XML parsing (YouTube's standard format)
        try:
            root = ET.fromstring(content)
            
            # Find all text elements in the XML
            for text_elem in root.findall('.//text'):
                start = float(text_elem.get('start', 0))  # Start time in seconds
                text = text_elem.text or ''
                
                if text.strip():
                    # Clean up HTML entities and formatting
                    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&#39;', "'")
                    text = re.sub(r'<[^>]+>', '', text)  # Remove any HTML tags
                    text = text.strip()
                    
                    if text:
                        transcript_segments.append({
                            'start': start,
                            'text': text
                        })
            
            if transcript_segments:
                logger.info(f"✅ XML parsing success: {len(transcript_segments)} segments")
                
        except Exception as xml_error:
            logger.info(f"XML parsing failed: {xml_error}")
            
            # Method 2: Regex fallback for plain text extraction
            text_patterns = [
                r'<text[^>]*start="[^"]*"[^>]*>([^<]+)</text>',  # Text with start attribute
                r'<text[^>]*>([^<]+)</text>',  # Any text element
                r'>([^<]+)</text>'  # Text before closing tag
            ]
            
            for pattern in text_patterns:
                matches = re.findall(pattern, content)
                if matches and len(matches) > 2:  # Ensure we have substantial content
                    logger.info(f"✅ Regex extraction: {len(matches)} segments")
                    
                    for i, text in enumerate(matches):
                        text = text.strip()
                        if text:
                            # Clean the text
                            text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&#39;', "'")
                            transcript_segments.append({
                                'start': i * 5,  # Approximate timing (5 seconds per segment)
                                'text': text
                            })
                    break  # Use first successful pattern
        
        if not transcript_segments:
            raise Exception("No transcript content found")
        
        # Format the output based on requested format
        if clean:
            # Clean format - just the text without timestamps
            texts = [seg['text'] for seg in transcript_segments if seg['text'].strip()]
            result = ' '.join(texts)
            logger.info(f"✅ REAL CLEAN TRANSCRIPT: {len(result)} characters")
            return result
        else:
            # Timestamped format - include timing information
            lines = []
            for seg in transcript_segments:
                start = seg['start']
                minutes = int(start // 60)
                seconds = int(start % 60)
                timestamp = f"[{minutes:02d}:{seconds:02d}]"
                lines.append(f"{timestamp} {seg['text']}")
            
            result = '\n'.join(lines)
            logger.info(f"✅ REAL TIMESTAMPED TRANSCRIPT: {len(lines)} lines")
            return result
        
    except Exception as e:
        logger.error(f"❌ Content parsing failed: {e}")
        raise

#===============  ================================
# API ENDPOINTS - Route handlers for HTTP requests
#===============  ================================

@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {"message": "YouTube Transcript Downloader API", "status": "running", "version": "1.0.0"}

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """
    User registration endpoint
    Creates a new user account with hashed password
    """
    # Check if username already exists
    db_user = get_user(db, user_data.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    # Check if email already exists
    email_exists = get_user_by_email(db, user_data.email)
    if email_exists:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    # Create new user with hashed password
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
    """
    User login endpoint
    Authenticates user and returns JWT access token
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create JWT token with expiration
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in successfully: {form_data.username}")
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """Get current authenticated user information"""
    return current_user

@app.post("/download_transcript/")
async def download_transcript_real(
    request: TranscriptRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    MAIN TRANSCRIPT DOWNLOAD ENDPOINT - NO DEMO CONTENT
    Downloads real YouTube transcripts in plain text format
    Supports YouTube videos, Shorts, and various URL formats
    """
    video_id = request.youtube_id.strip()
    
    # Extract video ID from various YouTube URL formats
    if 'youtube.com' in video_id or 'youtu.be' in video_id:
        patterns = [
            r'(?:youtube\.com\/shorts\/)([^&\n?#]+)',      # YouTube Shorts: /shorts/VIDEO_ID
            r'(?:youtube\.com\/watch\?v=)([^&\n?#]+)',     # Regular YouTube: /watch?v=VIDEO_ID
            r'(?:youtu\.be\/)([^&\n?#]+)',                 # Short URLs: youtu.be/VIDEO_ID
            r'(?:youtube\.com\/embed\/)([^&\n?#]+)',       # Embed URLs: /embed/VIDEO_ID
            r'[?&]v=([^&\n?#]+)'                           # Any URL with v= parameter
        ]
        
        for pattern in patterns:
            match = re.search(pattern, video_id)
            if match:
                video_id = match.group(1)[:11]  # YouTube video IDs are exactly 11 characters
                logger.info(f"✅ Extracted video ID: {video_id}")
                break
    
    # Validate video ID format
    if not video_id or len(video_id) != 11:
        raise HTTPException(status_code=400, detail="Invalid YouTube video ID")
    
    logger.info(f"🎯 REAL transcript request for: {video_id}")
    
    # Check user's subscription limits
    transcript_type = "clean" if request.clean_transcript else "unclean"
    can_download = check_subscription_limit(user.id, transcript_type, db)
    if not can_download:
        raise HTTPException(status_code=403, detail="Monthly limit reached")
    
    # Extract REAL transcript - NO demo content fallback
    try:
        logger.info(f"🎯 Extracting REAL transcript for: {video_id}")
        transcript_text = get_real_youtube_transcript(video_id, clean=request.clean_transcript)
        
        # Validate that we actually got meaningful content
        if not transcript_text or len(transcript_text.strip()) < 20:
            raise HTTPException(
                status_code=404,
                detail=f"No transcript content found for video {video_id}. This video may not have captions enabled."
            )
        
        # Record successful download in database for usage tracking
        new_download = TranscriptDownload(
            user_id=user.id,
            youtube_id=video_id,
            transcript_type=transcript_type,
            created_at=datetime.now()
        )
        
        db.add(new_download)
        db.commit()
        
        logger.info(f"✅ REAL TRANSCRIPT SUCCESS: {user.username} downloaded {len(transcript_text)} chars for {video_id}")
        
        return {
            "transcript": transcript_text,
            "youtube_id": video_id,
            "message": "Real transcript downloaded successfully"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (404, 403, etc.) without modification
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Real transcript extraction failed: {str(e)}")
        
        # Return error instead of demo content
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract transcript for video {video_id}. Error: {str(e)}"
        )

@app.get("/subscription_status/")
async def get_subscription_status_enhanced(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get user's subscription status and usage information
    Returns current tier, usage limits, and monthly usage statistics
    """
    try:
        # Get user's current subscription
        subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id
        ).first()
        
        # Determine current subscription tier and status
        if not subscription or subscription.expiry_date < datetime.now():
            tier = "free"
            status = "inactive"
        else:
            tier = subscription.tier
            status = "active"
        
        # Calculate current month's usage
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
        
        # Get subscription limits for current tier
        limits = SUBSCRIPTION_LIMITS[tier]
        
        # Convert infinity values to string for JSON serialization
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
            },
            "limits": json_limits,
            "subscription_id": subscription.payment_id if subscription else None,
            "current_period_end": subscription.expiry_date.isoformat() if subscription and subscription.expiry_date else None
        }
        
    except Exception as e:
        logger.error(f"Error getting subscription status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get subscription status")

#=============================================================
# PAYMENT ENDPOINTS - Stripe integration for subscriptions
#=============================================================

@app.post("/create_payment_intent/")
async def create_payment_intent_endpoint(
    request: CreatePaymentIntentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create Stripe payment intent for subscription upgrade
    Used by frontend to initiate payment process
    """
    try:
        # Validate price ID against configured values
        valid_price_ids = [os.getenv("PRO_PRICE_ID"), os.getenv("PREMIUM_PRICE_ID")]
        
        if request.price_id not in valid_price_ids:
            raise HTTPException(status_code=400, detail=f"Invalid price ID: {request.price_id}")

        # Get price details from Stripe
        price = stripe.Price.retrieve(request.price_id)
        plan_type = 'pro' if request.price_id == os.getenv("PRO_PRICE_ID") else 'premium'
        
        # Get or create Stripe customer for the user
        customer = get_or_create_stripe_customer(current_user, db)
        
        # Create payment intent
        intent = stripe.PaymentIntent.create(
            amount=price.unit_amount,  # Amount in cents
            currency=price.currency,
            customer=customer.id,
            automatic_payment_methods={'enabled': True, 'allow_redirects': 'never'},
            metadata={
                'user_id': str(current_user.id),
                'user_email': current_user.email,
                'price_id': request.price_id,
                'plan_type': plan_type
            }
        )

        return {
            'client_secret': intent.client_secret,  # Used by frontend to confirm payment
            'payment_intent_id': intent.id,
            'amount': price.unit_amount,
            'currency': price.currency,
            'plan_type': plan_type
        }

    except Exception as e:
        logger.error(f"Payment intent creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create payment intent: {str(e)}")

@app.post("/confirm_payment/")
async def confirm_payment_endpoint(
    request: ConfirmPaymentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Confirm payment and activate user subscription
    Called after successful payment to update user's subscription status
    """
    try:
        # Retrieve payment intent from Stripe to verify payment
        intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
        
        if intent.status != 'succeeded':
            raise HTTPException(status_code=400, detail=f"Payment not completed. Status: {intent.status}")

        # Get or create user subscription record
        user_subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id
        ).first()

        plan_type = intent.metadata.get('plan_type', 'pro')

        if not user_subscription:
            # Create new subscription record
            user_subscription = Subscription(
                user_id=current_user.id,
                tier=plan_type,
                start_date=datetime.utcnow(),
                expiry_date=datetime.utcnow() + timedelta(days=30),  # 30-day subscription
                payment_id=request.payment_intent_id,
                auto_renew=True
            )
            db.add(user_subscription)
        else:
            # Update existing subscription
            user_subscription.tier = plan_type
            user_subscription.start_date = datetime.utcnow()
            user_subscription.expiry_date = datetime.utcnow() + timedelta(days=30)
            user_subscription.payment_id = request.payment_intent_id
            user_subscription.auto_renew = True

        db.commit()
        db.refresh(user_subscription)

        return {
            'success': True,
            'subscription_tier': user_subscription.tier,
            'expires_at': user_subscription.expiry_date.isoformat(),
            'status': 'active'
        }

    except Exception as e:
        logger.error(f"Payment confirmation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to confirm payment: {str(e)}")

#========================
# HEALTH CHECK ENDPOINTS
# =======================

@app.get("/health/")
async def health_check():
    """Application health check endpoint"""
    return {
        "status": "healthy",
        "stripe_configured": bool(os.getenv("STRIPE_SECRET_KEY")),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/healthcheck")
async def healthcheck():
    """Simple health check endpoint"""
    return {"status": "ok", "version": "1.0.0"}

@app.get("/library_info")
async def library_info():
    """Get YouTube Transcript API library version information"""
    from youtube_transcript_api import __version__
    return {"youtube_transcript_api_version": __version__}

# Run the application if executed directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)