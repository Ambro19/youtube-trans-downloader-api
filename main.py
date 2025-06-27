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
import xml.etree.ElementTree as ET
import urllib.parse

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

#=============================
# PYDANTIC MODELS
#=============================

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

#=====================================
# HELPER FUNCTIONS
#======================================

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

def get_or_create_stripe_customer(user, db: Session):
    """Get or create a Stripe customer for the user"""
    try:
        if hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
            try:
                customer = stripe.Customer.retrieve(user.stripe_customer_id)
                return customer
            except stripe.error.InvalidRequestError:
                pass
        
        customer = stripe.Customer.create(
            email=user.email,
            name=user.username,
            metadata={'user_id': str(user.id)}
        )
        
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
    """Check subscription limits"""
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

#=====================================
# IMPROVED TRANSCRIPT FUNCTIONS
#=====================================

def get_youtube_transcript_robust(video_id: str, clean: bool = True) -> str:
    """
    Robust YouTube transcript extraction with multiple fallback methods
    """
    logger.info(f"🎯 Starting robust transcript extraction for: {video_id}")
    
    # Method 1: Official YouTube Transcript API with multiple language attempts
    try:
        logger.info("🔄 Trying YouTube Transcript API...")
        
        # Try different language configurations
        language_configs = [
            ['en'],  # English only
            ['en', 'en-US', 'en-GB'],  # English variants
            ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU'],  # More English variants
            None,  # Auto-detect any available language
        ]
        
        for i, languages in enumerate(language_configs):
            try:
                logger.info(f"📝 API attempt {i+1}: {languages if languages else 'auto-detect'}")
                
                if languages:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
                else:
                    # Get list of available transcripts and pick the first one
                    transcript_list_data = YouTubeTranscriptApi.list_transcripts(video_id)
                    # Try to get English first, then any manually created, then auto-generated
                    for transcript in transcript_list_data:
                        if 'en' in transcript.language_code.lower():
                            transcript_list = transcript.fetch()
                            break
                    else:
                        # If no English, get the first available
                        transcript_list = transcript_list_data._manually_created_transcripts[0].fetch() if transcript_list_data._manually_created_transcripts else transcript_list_data._generated_transcripts[0].fetch()
                
                if transcript_list and len(transcript_list) > 0:
                    logger.info(f"✅ API SUCCESS: {len(transcript_list)} segments found")
                    return format_transcript(transcript_list, clean)
                    
            except Exception as e:
                logger.info(f"⚠️ API attempt {i+1} failed: {str(e)}")
                continue
        
        logger.warning("⚠️ All YouTube API attempts failed")
        
    except Exception as e:
        logger.warning(f"⚠️ YouTube API completely failed: {str(e)}")
    
    # Method 2: Direct HTTP scraping
    try:
        logger.info("🌐 Trying HTTP scraping method...")
        return get_transcript_via_http(video_id, clean)
        
    except Exception as e:
        logger.warning(f"⚠️ HTTP scraping failed: {str(e)}")
    
    # Method 3: Alternative API approach
    try:
        logger.info("🔄 Trying alternative extraction...")
        return get_transcript_alternative_method(video_id, clean)
        
    except Exception as e:
        logger.warning(f"⚠️ Alternative method failed: {str(e)}")
    
    # If all methods fail, raise an exception
    logger.error(f"❌ All transcript extraction methods failed for video {video_id}")
    raise HTTPException(
        status_code=404,
        detail=f"No transcript available for video {video_id}. This video may not have captions enabled or may be private/restricted."
    )

def format_transcript(transcript_list: list, clean: bool = True) -> str:
    """Format transcript data into clean or timestamped format"""
    if clean:
        # Clean format - just text
        texts = []
        for item in transcript_list:
            text = item.get('text', '').strip()
            if text and not (text.startswith('[') and text.endswith(']')):
                # Clean up common artifacts
                text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                texts.append(text)
        
        result = ' '.join(texts)
        logger.info(f"✅ Clean transcript formatted: {len(result)} characters")
        return result
    else:
        # Timestamped format
        lines = []
        for item in transcript_list:
            start = item.get('start', 0)
            text = item.get('text', '').strip()
            if text:
                minutes = int(start // 60)
                seconds = int(start % 60)
                timestamp = f"[{minutes:02d}:{seconds:02d}]"
                lines.append(f"{timestamp} {text}")
        
        result = '\n'.join(lines)
        logger.info(f"✅ Timestamped transcript formatted: {len(lines)} lines")
        return result

def get_transcript_via_http(video_id: str, clean: bool = True) -> str:
    """HTTP-based transcript extraction"""
    try:
        logger.info(f"🌐 HTTP extraction for: {video_id}")
        
        # Build video URL
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Set up headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
        # Get the video page
        logger.info("📡 Fetching video page...")
        response = requests.get(video_url, headers=headers, timeout=30)
        if response.status_code != 200:
            raise Exception(f"Failed to load video page: {response.status_code}")
        
        page_content = response.text
        logger.info(f"📄 Page loaded: {len(page_content)} characters")
        
        # Extract player response data
        player_response = extract_player_response(page_content)
        if player_response:
            logger.info("✅ Player response found")
            caption_tracks = extract_caption_tracks(player_response)
            if caption_tracks:
                logger.info(f"✅ Found {len(caption_tracks)} caption tracks")
                return fetch_and_parse_captions(caption_tracks[0], clean)
        
        # Fallback: Direct caption URL search
        logger.info("🔄 Trying direct caption URL extraction...")
        caption_url = extract_caption_url_direct(page_content)
        if caption_url:
            logger.info("✅ Direct caption URL found")
            return fetch_caption_content(caption_url, clean)
        
        raise Exception("No caption data found in video page")
        
    except Exception as e:
        logger.error(f"❌ HTTP extraction failed: {str(e)}")
        raise

def extract_player_response(page_content: str) -> dict:
    """Extract player response JSON from page content"""
    try:
        # Look for player response in various formats
        patterns = [
            r'var ytInitialPlayerResponse = ({.+?});',
            r'"playerResponse":"({.+?})"',
            r'ytInitialPlayerResponse":\s*({.+?})(?:,|\s*})',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, page_content, re.DOTALL)
            for match in matches:
                try:
                    # Clean up the JSON string
                    json_str = match.strip()
                    if json_str.startswith('"') and json_str.endswith('"'):
                        json_str = json_str[1:-1].replace('\\"', '"')
                    
                    player_response = json.loads(json_str)
                    if 'captions' in player_response:
                        return player_response
                except json.JSONDecodeError:
                    continue
        
        return None
    except Exception as e:
        logger.warning(f"Failed to extract player response: {e}")
        return None

def extract_caption_tracks(player_response: dict) -> list:
    """Extract caption tracks from player response"""
    try:
        captions = player_response.get('captions', {})
        caption_tracks = captions.get('playerCaptionsTracklistRenderer', {}).get('captionTracks', [])
        
        if not caption_tracks:
            return []
        
        # Prefer English tracks
        english_tracks = [track for track in caption_tracks if 'en' in track.get('languageCode', '').lower()]
        return english_tracks if english_tracks else caption_tracks
        
    except Exception as e:
        logger.warning(f"Failed to extract caption tracks: {e}")
        return []

def extract_caption_url_direct(page_content: str) -> str:
    """Direct extraction of caption URL from page content"""
    try:
        # Search for caption URLs in the page
        patterns = [
            r'"captionTracks":\[{"baseUrl":"([^"]+)"',
            r'"baseUrl":"(https://www\.youtube\.com/api/timedtext[^"]*)"',
            r'(https://www\.youtube\.com/api/timedtext[^"&\s]*)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, page_content)
            for match in matches:
                # Clean up the URL
                url = match.replace('\\u0026', '&').replace('\\/', '/')
                if 'timedtext' in url:
                    return url
        
        return None
    except Exception as e:
        logger.warning(f"Failed direct caption URL extraction: {e}")
        return None

def fetch_and_parse_captions(caption_track: dict, clean: bool = True) -> str:
    """Fetch and parse caption content from a caption track"""
    try:
        base_url = caption_track.get('baseUrl', '')
        if not base_url:
            raise Exception("No base URL in caption track")
        
        return fetch_caption_content(base_url, clean)
        
    except Exception as e:
        logger.error(f"Failed to fetch captions: {e}")
        raise

def fetch_caption_content(caption_url: str, clean: bool = True) -> str:
    """Fetch and parse actual caption content"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        }
        
        logger.info("📥 Fetching caption content...")
        response = requests.get(caption_url, headers=headers, timeout=15)
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch captions: {response.status_code}")
        
        # Parse the XML content
        return parse_caption_xml(response.text, clean)
        
    except Exception as e:
        logger.error(f"Failed to fetch caption content: {e}")
        raise

def parse_caption_xml(xml_content: str, clean: bool = True) -> str:
    """Parse caption XML content"""
    try:
        logger.info("📝 Parsing caption XML...")
        
        # Clean up the XML content
        xml_content = xml_content.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&#39;', "'")
        
        # Parse XML
        root = ET.fromstring(xml_content)
        
        transcript_segments = []
        for text_elem in root.findall('.//text'):
            start = float(text_elem.get('start', 0))
            text = text_elem.text or ''
            
            if text.strip():
                # Clean up the text
                text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
                text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
                text = text.strip()
                
                if text:
                    transcript_segments.append({
                        'start': start,
                        'text': text
                    })
        
        if not transcript_segments:
            raise Exception("No transcript segments found in XML")
        
        # Format the output
        return format_transcript(transcript_segments, clean)
        
    except ET.ParseError as e:
        logger.error(f"XML parsing failed: {e}")
        # Try regex fallback
        return parse_caption_xml_regex(xml_content, clean)
    except Exception as e:
        logger.error(f"Caption XML parsing failed: {e}")
        raise

def parse_caption_xml_regex(xml_content: str, clean: bool = True) -> str:
    """Fallback regex-based XML parsing"""
    try:
        logger.info("🔄 Using regex fallback for XML parsing...")
        
        # Extract text content using regex
        text_pattern = r'<text[^>]*start="([^"]*)"[^>]*>([^<]+)</text>'
        matches = re.findall(text_pattern, xml_content)
        
        if not matches:
            # Try simpler pattern
            text_pattern = r'<text[^>]*>([^<]+)</text>'
            text_matches = re.findall(text_pattern, xml_content)
            matches = [(i * 5, text) for i, text in enumerate(text_matches)]  # Fake timestamps
        
        if not matches:
            raise Exception("No text content found with regex")
        
        transcript_segments = []
        for start_str, text in matches:
            try:
                start = float(start_str) if isinstance(start_str, str) else start_str
            except (ValueError, TypeError):
                start = len(transcript_segments) * 5  # Fallback timing
            
            text = text.strip()
            if text:
                # Clean up the text
                text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&#39;', "'")
                transcript_segments.append({
                    'start': start,
                    'text': text
                })
        
        if not transcript_segments:
            raise Exception("No transcript segments found")
        
        return format_transcript(transcript_segments, clean)
        
    except Exception as e:
        logger.error(f"Regex parsing failed: {e}")
        raise

def get_transcript_alternative_method(video_id: str, clean: bool = True) -> str:
    """Alternative method using different API approach"""
    try:
        logger.info(f"🔄 Alternative method for: {video_id}")
        
        # Try using the transcript API with different parameters
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # Get all available transcripts
        transcript_list_data = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try manually created transcripts first
        for transcript in transcript_list_data:
            try:
                if transcript.is_generated:
                    continue  # Skip auto-generated for now
                
                transcript_data = transcript.fetch()
                if transcript_data:
                    logger.info(f"✅ Manual transcript found: {transcript.language_code}")
                    return format_transcript(transcript_data, clean)
            except Exception as e:
                logger.info(f"Failed to fetch manual transcript {transcript.language_code}: {e}")
                continue
        
        # If no manual transcripts, try auto-generated
        for transcript in transcript_list_data:
            try:
                if not transcript.is_generated:
                    continue  # We already tried manual ones
                
                transcript_data = transcript.fetch()
                if transcript_data:
                    logger.info(f"✅ Auto-generated transcript found: {transcript.language_code}")
                    return format_transcript(transcript_data, clean)
            except Exception as e:
                logger.info(f"Failed to fetch auto transcript {transcript.language_code}: {e}")
                continue
        
        raise Exception("No transcripts available via alternative method")
        
    except Exception as e:
        logger.error(f"Alternative method failed: {e}")
        raise

#===============
# API ENDPOINTS
#===============

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
async def download_transcript(
    request: TranscriptRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Download YouTube transcript with robust extraction methods
    """
    video_id = request.youtube_id.strip()
    
    # Extract video ID from URLs (including Shorts)
    if 'youtube.com' in video_id or 'youtu.be' in video_id:
        patterns = [
            r'(?:youtube\.com\/shorts\/)([^&\n?#]+)',      # YouTube Shorts
            r'(?:youtube\.com\/watch\?v=)([^&\n?#]+)',     # Regular YouTube
            r'(?:youtu\.be\/)([^&\n?#]+)',                 # Short URLs
            r'(?:youtube\.com\/embed\/)([^&\n?#]+)',       # Embed URLs
            r'[?&]v=([^&\n?#]+)'                           # Any v= parameter
        ]
        
        for pattern in patterns:
            match = re.search(pattern, video_id)
            if match:
                video_id = match.group(1)[:11]
                logger.info(f"✅ Extracted video ID: {video_id}")
                break
    
    if not video_id or len(video_id) != 11:
        raise HTTPException(status_code=400, detail="Invalid YouTube video ID")
    
    logger.info(f"🎯 Transcript request for: {video_id}")
    
    # Check subscription limits
    transcript_type = "clean" if request.clean_transcript else "unclean"
    can_download = check_subscription_limit(user.id, transcript_type, db)
    if not can_download:
        raise HTTPException(status_code=403, detail="Monthly limit reached")
    
    # Extract transcript using robust method
    try:
        logger.info(f"🎯 Starting transcript extraction for: {video_id}")
        transcript_text = get_youtube_transcript_robust(video_id, clean=request.clean_transcript)
        
        # Validate we got real content
        if not transcript_text or len(transcript_text.strip()) < 10:
            raise HTTPException(
                status_code=404,
                detail=f"No transcript content found for video {video_id}. This video may not have captions enabled."
            )
        
        # Record successful download
        new_download = TranscriptDownload(
            user_id=user.id,
            youtube_id=video_id,
            transcript_type=transcript_type,
            created_at=datetime.now()
        )
        
        db.add(new_download)
        db.commit()
        
        logger.info(f"✅ TRANSCRIPT SUCCESS: {user.username} downloaded {len(transcript_text)} chars for {video_id}")
        
        return {
            "transcript": transcript_text,
            "youtube_id": video_id,
            "message": "Transcript downloaded successfully"
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (404, etc.)
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Transcript extraction failed: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract transcript for video {video_id}. Error: {str(e)}"
        )

@app.get("/subscription_status/")
async def get_subscription_status_enhanced(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enhanced subscription status"""
    try:
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
        
        # Get limits based on tier
        limits = SUBSCRIPTION_LIMITS[tier]
        
        # Convert infinity to string for JSON
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
# Payment endpoints (simplified - keeping only essential ones)
#=============================================================

@app.post("/create_payment_intent/")
async def create_payment_intent_endpoint(
    request: CreatePaymentIntentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create payment intent"""
    try:
        valid_price_ids = [os.getenv("PRO_PRICE_ID"), os.getenv("PREMIUM_PRICE_ID")]
        
        if request.price_id not in valid_price_ids:
            raise HTTPException(status_code=400, detail=f"Invalid price ID: {request.price_id}")

        price = stripe.Price.retrieve(request.price_id)
        plan_type = 'pro' if request.price_id == os.getenv("PRO_PRICE_ID") else 'premium'
        customer = get_or_create_stripe_customer(current_user, db)
        
        intent = stripe.PaymentIntent.create(
            amount=price.unit_amount,
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
            'client_secret': intent.client_secret,
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
    """Confirm payment and update subscription"""
    try:
        intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
        
        if intent.status != 'succeeded':
            raise HTTPException(status_code=400, detail=f"Payment not completed. Status: {intent.status}")

        user_subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id
        ).first()

        plan_type = intent.metadata.get('plan_type', 'pro')

        if not user_subscription:
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
    from youtube_transcript_api import __version__
    return {"youtube_transcript_api_version": __version__}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)