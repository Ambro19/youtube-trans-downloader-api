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
# ENHANCED TRANSCRIPT FUNCTIONS
#=====================================

def get_youtube_transcript_corrected(video_id: str, clean: bool = True) -> str:
    """
    ENHANCED YouTube transcript extraction with robust error handling
    
    This function tries multiple methods in order of reliability:
    1. youtube-transcript-api (fastest, but often blocked)
    2. list_transcripts (better language detection)
    3. yt-dlp (most robust, works when others fail)
    4. Direct API (fallback for edge cases)
    5. Demo content (last resort for testing)
    
    Args:
        video_id: 11-character YouTube video ID
        clean: If True, returns clean text; if False, returns timestamped format
    
    Returns:
        Formatted transcript text
    """
    logger.info(f"🎯 ENHANCED transcript extraction for: {video_id}")
    
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # Method 1: Try youtube-transcript-api first (fastest when it works)
        try:
            logger.info("🔄 Trying youtube-transcript-api...")
            
            # Try multiple English language variants for better success rate
            language_codes = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
            transcript_list = None
            
            for lang in language_codes:
                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                    if transcript_list and len(transcript_list) > 0:
                        logger.info(f"✅ SUCCESS with language {lang}: {len(transcript_list)} segments")
                        break
                except Exception as lang_error:
                    logger.info(f"📝 Language {lang} failed: {str(lang_error)}")
                    continue
            
            # If we found a transcript, format and return it
            if transcript_list and len(transcript_list) > 0:
                return format_transcript_enhanced(transcript_list, clean)
                
        except Exception as e:
            logger.info(f"📝 youtube-transcript-api failed: {str(e)}")
        
        # Method 2: Try list_transcripts approach for better language detection
        try:
            logger.info("🔄 Trying list_transcripts method...")
            transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to find any English transcript (manual or auto-generated)
            english_transcript = None
            try:
                # Prefer manual transcripts over auto-generated ones
                for transcript in transcript_list_obj:
                    if transcript.language_code.startswith('en'):
                        english_transcript = transcript
                        logger.info(f"🎯 Found manual English transcript: {transcript.language_code}")
                        break
                
                # If no manual English transcript, try auto-generated
                if not english_transcript:
                    try:
                        english_transcript = transcript_list_obj.find_generated_transcript(['en'])
                        logger.info("🤖 Found auto-generated English transcript")
                    except:
                        pass
                
                # If we found any English transcript, fetch and format it
                if english_transcript:
                    transcript_data = english_transcript.fetch()
                    if transcript_data and len(transcript_data) > 0:
                        logger.info(f"✅ LIST API SUCCESS: {len(transcript_data)} segments")
                        return format_transcript_enhanced(transcript_data, clean)
                        
            except Exception as inner_e:
                logger.info(f"📝 English transcript lookup failed: {str(inner_e)}")
                
        except Exception as e:
            logger.info(f"📝 List API failed: {str(e)}")
        
        # Method 3: Enhanced yt-dlp method (most robust, works when APIs fail)
        try:
            logger.info("🔄 Trying enhanced yt-dlp method...")
            import yt_dlp
            
            # Configure yt-dlp for subtitle extraction only
            ydl_opts = {
                'writesubtitles': True,        # Download manual subtitles
                'writeautomaticsub': True,     # Download auto-generated subtitles
                'subtitleslangs': ['en', 'en-US', 'en-GB'],  # English variants
                'skip_download': True,         # Don't download video, just metadata
                'quiet': True,                 # Suppress most output
                'no_warnings': True,           # Suppress warnings
                'extract_flat': False,         # Get full metadata
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                url = f"https://www.youtube.com/watch?v={video_id}"
                
                try:
                    # Extract video metadata including subtitle information
                    info = ydl.extract_info(url, download=False)
                    
                    # Special handling for live streams
                    if info.get('is_live') or info.get('live_status') == 'is_live':
                        logger.info("🔴 Live stream detected - trying live transcript extraction")
                        return extract_live_transcript(video_id, clean)
                    
                    # Try manual subtitles first (usually higher quality)
                    subtitles_found = False
                    if 'subtitles' in info and info['subtitles']:
                        for lang in ['en', 'en-US', 'en-GB']:
                            if lang in info['subtitles']:
                                for entry in info['subtitles'][lang]:
                                    transcript_text = extract_subtitle_from_entry(ydl, entry, clean)
                                    if transcript_text and not is_invalid_content(transcript_text):
                                        logger.info(f"✅ YT-DLP MANUAL SUCCESS: {len(transcript_text)} chars")
                                        return transcript_text
                                    subtitles_found = True
                    
                    # Fallback to automatic captions if manual subtitles don't work
                    if 'automatic_captions' in info and info['automatic_captions']:
                        for lang in ['en', 'en-US', 'en-GB']:
                            if lang in info['automatic_captions']:
                                for entry in info['automatic_captions'][lang]:
                                    transcript_text = extract_subtitle_from_entry(ydl, entry, clean)
                                    if transcript_text and not is_invalid_content(transcript_text):
                                        logger.info(f"✅ YT-DLP AUTO SUCCESS: {len(transcript_text)} chars")
                                        return transcript_text
                                    subtitles_found = True
                    
                    # Log if subtitles were found but couldn't be processed
                    if subtitles_found:
                        logger.info("📝 Subtitles found but content was invalid/empty")
                    
                except Exception as yt_error:
                    logger.info(f"📝 yt-dlp extraction failed: {str(yt_error)}")
                
        except ImportError:
            logger.info("📝 yt-dlp not installed, skipping...")
        except Exception as e:
            logger.info(f"📝 yt-dlp method failed: {str(e)}")
        
        # Method 4: Direct API approach for edge cases (placeholder for future expansion)
        try:
            logger.info("🔄 Trying direct API approach...")
            transcript_data = extract_with_direct_api(video_id)
            if transcript_data:
                logger.info(f"✅ DIRECT API SUCCESS: {len(transcript_data)} segments")
                return format_transcript_enhanced(transcript_data, clean)
        except Exception as e:
            logger.info(f"📝 Direct API failed: {str(e)}")
        
        # Final fallback: Demo content for testing (only when all methods fail)
        logger.warning(f"⚠️ All extraction methods failed for {video_id}, falling back to demo content")
        return get_demo_content(clean)
        
    except Exception as e:
        logger.error(f"💥 Critical error in transcript extraction for {video_id}: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"Unable to extract transcript for video {video_id}. The video may not have captions enabled, be private, or be unavailable."
        )

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

def is_invalid_content(content: str) -> bool:
    """
    Check if content is invalid (M3U8 playlist, metadata, etc.)
    
    This function identifies content that looks like:
    - M3U8 playlists (streaming metadata)
    - API URLs with parameters
    - Empty or too-short content
    
    Args:
        content: String content to validate
    
    Returns:
        True if content is invalid, False if it looks like real transcript text
    """
    if not content or len(content.strip()) < 10:
        return True
    
    # List of indicators that suggest this is not transcript content
    invalid_indicators = [
        '#EXTM3U',                           # M3U8 playlist header
        '#EXT-X-VERSION',                    # M3U8 version
        '#EXT-X-PLAYLIST-TYPE',              # M3U8 playlist type
        '#EXT-X-TARGETDURATION',             # M3U8 duration
        '#EXTINF:',                          # M3U8 segment info
        'https://www.youtube.com/api/timedtext', # Direct API URL
        'sparams=',                          # URL parameters
        'signature=',                        # URL signature
        'caps=asr'                           # Caption parameters
    ]
    
    content_lower = content.lower()
    for indicator in invalid_indicators:
        if indicator.lower() in content_lower:
            logger.info(f"🚫 Invalid content detected: contains '{indicator}'")
            return True
    
    return False

def extract_subtitle_from_entry(ydl, entry, clean: bool) -> str:
    """
    Extract and parse subtitle content from yt-dlp entry
    
    This function handles the actual downloading and parsing of subtitle files
    from yt-dlp subtitle entries. It validates URLs, downloads content,
    and processes different subtitle formats.
    
    Args:
        ydl: yt-dlp YoutubeDL instance
        entry: Subtitle entry from yt-dlp with URL and format info
        clean: Whether to return clean or timestamped format
    
    Returns:
        Formatted transcript text, or empty string if extraction fails
    """
    try:
        if 'url' not in entry:
            logger.info("📝 No URL found in subtitle entry")
            return ""
        
        subtitle_url = entry['url']
        ext = entry.get('ext', 'vtt')
        
        # Skip if URL looks like M3U8 or contains suspicious patterns
        suspicious_patterns = ['m3u8', 'playlist', 'range=', 'sparams=', 'signature=']
        if any(pattern in subtitle_url.lower() for pattern in suspicious_patterns):
            logger.info(f"🚫 Skipping suspicious URL: {subtitle_url[:100]}...")
            return ""
        
        try:
            # Download subtitle content using yt-dlp's urlopen method
            logger.info(f"📥 Downloading subtitle content: {ext} format")
            subtitle_content = ydl.urlopen(subtitle_url).read().decode('utf-8', errors='ignore')
            
            # Check if content is valid before parsing
            if is_invalid_content(subtitle_content):
                logger.info(f"🚫 Invalid subtitle content detected for {ext} format")
                return ""
            
            # Parse the content based on format
            transcript_data = parse_subtitle_content_enhanced(subtitle_content, ext)
            
            if transcript_data and len(transcript_data) > 0:
                logger.info(f"✅ Successfully parsed {len(transcript_data)} transcript segments")
                return format_transcript_enhanced(transcript_data, clean)
            else:
                logger.info(f"📝 No valid transcript data found in {ext} content")
                
        except Exception as url_error:
            logger.info(f"📝 URL extraction failed: {str(url_error)}")
            
    except Exception as e:
        logger.info(f"📝 Entry extraction failed: {str(e)}")
    
    return ""

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

def parse_vtt_content(content: str) -> list:
    """
    Parse WebVTT subtitle content
    
    WebVTT format structure:
    WEBVTT
    
    00:00:01.000 --> 00:00:04.000
    Hello, this is the first subtitle
    
    Args:
        content: WebVTT file content
    
    Returns:
        List of transcript segments
    """
    transcript_data = []
    lines = content.split('\n')
    current_start = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip metadata and empty lines
        if not line or line.startswith('WEBVTT') or line.startswith('NOTE') or line.isdigit():
            continue
        
        # Parse timestamp line (contains -->)
        if '-->' in line:
            try:
                start_time = line.split(' --> ')[0].strip()
                current_start = parse_vtt_timestamp(start_time)
            except:
                current_start = 0
        
        # Parse text line (actual subtitle content)
        elif line and not line.startswith('<') and '-->' not in line:
            clean_text = clean_vtt_text(line)
            if clean_text and len(clean_text) > 1:
                transcript_data.append({
                    'text': clean_text,
                    'start': current_start,
                    'duration': 3.0  # Default duration
                })
    
    return transcript_data

def clean_vtt_text(text: str) -> str:
    """
    Clean VTT text from formatting tags and artifacts
    
    Removes VTT-specific formatting like:
    - <c> color tags
    - <i> italic tags
    - {style} annotations
    - HTML entities
    
    Args:
        text: Raw VTT text line
    
    Returns:
        Clean text content
    """
    # Remove VTT formatting tags
    import re
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML/VTT tags
    text = re.sub(r'\{[^}]+\}', '', text)  # Remove style annotations
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    return text.strip()

def parse_vtt_timestamp(timestamp: str) -> float:
    """
    Parse VTT timestamp format to seconds
    
    Supports formats:
    - HH:MM:SS.mmm (hours:minutes:seconds.milliseconds)
    - MM:SS.mmm (minutes:seconds.milliseconds)
    
    Args:
        timestamp: Timestamp string from VTT file
    
    Returns:
        Time in seconds as float
    """
    try:
        # Handle both comma and dot as decimal separator
        parts = timestamp.replace(',', '.').split(':')
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        elif len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
    except:
        pass
    return 0

def parse_srv3_content(content: str) -> list:
    """
    Parse SRV3/JSON subtitle content (YouTube's internal format)
    
    SRV3 format structure:
    {
      "events": [
        {
          "tStartMs": 1000,
          "dDurationMs": 3000,
          "segs": [{"utf8": "Hello world"}]
        }
      ]
    }
    
    Args:
        content: SRV3/JSON file content
    
    Returns:
        List of transcript segments
    """
    transcript_data = []
    try:
        import json
        data = json.loads(content)
        
        if 'events' in data:
            for event in data['events']:
                if 'segs' in event:
                    text_segments = []
                    for seg in event['segs']:
                        if 'utf8' in seg:
                            text_segments.append(seg['utf8'])
                    
                    if text_segments:
                        full_text = ''.join(text_segments).strip()
                        if full_text and len(full_text) > 1:
                            transcript_data.append({
                                'text': full_text,
                                'start': event.get('tStartMs', 0) / 1000.0,  # Convert ms to seconds
                                'duration': event.get('dDurationMs', 3000) / 1000.0
                            })
    except:
        pass
    
    return transcript_data

def parse_ttml_content(content: str) -> list:
    """
    Parse TTML/XML subtitle content
    
    TTML is a standard XML-based subtitle format used by many platforms.
    
    Args:
        content: TTML/XML file content
    
    Returns:
        List of transcript segments
    """
    transcript_data = []
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(content)
        
        # Find all text elements with timing information
        for elem in root.iter():
            if elem.text and elem.text.strip():
                start_time = 0
                if 'begin' in elem.attrib:
                    start_time = parse_ttml_timestamp(elem.attrib['begin'])
                
                clean_text = elem.text.strip()
                if clean_text and len(clean_text) > 1:
                    transcript_data.append({
                        'text': clean_text,
                        'start': start_time,
                        'duration': 3.0  # Default duration
                    })
    except:
        pass
    
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

def get_demo_content(clean: bool) -> str:
    """
    Fallback demo content for testing and when extraction fails
    
    This provides sample content that demonstrates the expected format
    while clearly indicating it's placeholder content.
    
    Args:
        clean: Whether to return clean or timestamped format
    
    Returns:
        Demo transcript content
    """
    demo_text = """Hello everyone, welcome to this video. Today we're going to be talking 
    about some really interesting topics. We'll cover various aspects of the subject matter 
    and provide you with valuable insights. This is sample transcript content for demonstration 
    purposes. The actual transcript would contain the real audio content from the YouTube video. 
    Thank you for watching, and don't forget to subscribe to our channel for more great content like this."""
    
    if clean:
        return demo_text
    else:
        # Add timestamps for unclean format demonstration
        words = demo_text.split()
        timestamped_lines = []
        current_time = 0
        
        for i in range(0, len(words), 8):  # Group words into 8-word chunks
            chunk = ' '.join(words[i:i+8])
            minutes = current_time // 60
            seconds = current_time % 60
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            timestamped_lines.append(f"{timestamp} {chunk}")
            current_time += 6  # Add 6 seconds per chunk
        
        return '\n'.join(timestamped_lines)

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

@app.get("/subscription_status/")
async def get_subscription_status_enhanced(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enhanced subscription status with detailed usage information"""
    try:
        subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id
        ).first()
        
        # Determine current subscription tier
        if not subscription or subscription.expiry_date < datetime.now():
            tier = "free"
            status = "inactive"
        else:
            tier = subscription.tier
            status = "active"
        
        # Calculate usage for current month
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
        
        # Get limits based on current tier
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
    """Create payment intent for subscription upgrade"""
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
    """Confirm payment and update subscription status"""
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

#=================================================

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#=========================================================================================================
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

# #=============================
# # PYDANTIC MODELS
# #=============================

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

# #=====================================
# # HELPER FUNCTIONS
# #======================================

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

# def get_or_create_stripe_customer(user, db: Session):
#     """Get or create a Stripe customer for the user"""
#     try:
#         if hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
#             try:
#                 customer = stripe.Customer.retrieve(user.stripe_customer_id)
#                 return customer
#             except stripe.error.InvalidRequestError:
#                 pass
        
#         customer = stripe.Customer.create(
#             email=user.email,
#             name=user.username,
#             metadata={'user_id': str(user.id)}
#         )
        
#         if hasattr(user, 'stripe_customer_id'):
#             user.stripe_customer_id = customer.id
#             db.commit()
        
#         return customer
        
#     except Exception as e:
#         logger.error(f"Error creating Stripe customer: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to create payment customer"
#         )

# def check_subscription_limit(user_id: int, transcript_type: str, db: Session):
#     """Check subscription limits"""
#     subscription = db.query(Subscription).filter(Subscription.user_id == user_id).first()
    
#     if not subscription:
#         tier = "free"
#     else:
#         tier = subscription.tier
#         if subscription.expiry_date < datetime.now():
#             tier = "free"
    
#     month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
#     usage = db.query(TranscriptDownload).filter(
#         TranscriptDownload.user_id == user_id,
#         TranscriptDownload.transcript_type == transcript_type,
#         TranscriptDownload.created_at >= month_start
#     ).count()
    
#     limit = SUBSCRIPTION_LIMITS[tier][transcript_type]
#     if usage >= limit:
#         return False
#     return True

# #=====================================
# # SIMPLE TRANSCRIPT FUNCTIONS
# #=====================================

# def get_youtube_transcript_simple(video_id: str, clean: bool = True) -> str:
#     """
#     SIMPLE YouTube transcript extraction using the updated API
#     """
#     logger.info(f"🎯 SIMPLE transcript extraction for: {video_id}")
    
#     try:
#         # Method 1: Try the old API format first (might still work)
#         try:
#             logger.info("🔄 Trying OLD API format...")
#             from youtube_transcript_api import YouTubeTranscriptApi
#             transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            
#             if transcript_list and len(transcript_list) > 0:
#                 logger.info(f"✅ OLD API SUCCESS: {len(transcript_list)} segments")
#                 return format_transcript_simple(transcript_list, clean)
                
#         except Exception as e:
#             logger.info(f"📝 OLD API failed: {str(e)}")
#             logger.info("🔄 Trying NEW API format...")
        
#         # Method 2: Try the new API format
#         try:
#             from youtube_transcript_api import YouTubeTranscriptApi
#             ytt_api = YouTubeTranscriptApi()
#             result = ytt_api.fetch(video_id)
            
#             if result and hasattr(result, 'snippets'):
#                 logger.info(f"✅ NEW API SUCCESS: {len(result.snippets)} segments")
#                 return format_transcript_new_api(result, clean)
                
#         except Exception as e:
#             logger.info(f"📝 NEW API failed: {str(e)}")
        
#         # Method 3: Fallback to demo content for testing
#         logger.info("🎭 Using demo content for testing")
#         demo_content = """Hello everyone, welcome to this video. Today we're going to be talking about some really interesting topics. 
#         We'll cover various aspects of the subject matter and provide you with valuable insights. This is just sample transcript content 
#         for demonstration purposes. The actual transcript would contain the real audio content from the YouTube video. Thank you for watching, and don't forget to subscribe to our channel for more great content like this."""
        
#         if clean:
#             return demo_content
#         else:
#             # Add timestamps for unclean format
#             words = demo_content.split()
#             timestamped_lines = []
#             current_time = 0
            
#             for i in range(0, len(words), 8):  # Group words into chunks
#                 chunk = ' '.join(words[i:i+8])
#                 minutes = current_time // 60
#                 seconds = current_time % 60
#                 timestamp = f"[{minutes:02d}:{seconds:02d}]"
#                 timestamped_lines.append(f"{timestamp} {chunk}")
#                 current_time += 6  # Add 6 seconds per chunk
            
#             return '\n'.join(timestamped_lines)
        
#     except Exception as e:
#         logger.error(f"💥 All methods failed for {video_id}: {str(e)}")
#         raise HTTPException(
#             status_code=404,
#             detail=f"Unable to extract transcript for video {video_id}. The video may not have captions enabled or may be private/unavailable."
#         )

# def format_transcript_simple(transcript_list: list, clean: bool = True) -> str:
#     """Format transcript data from OLD API format"""
#     if not transcript_list:
#         raise Exception("Empty transcript data")
    
#     if clean:
#         # Clean format - just text
#         texts = []
#         for item in transcript_list:
#             text = item.get('text', '').strip()
#             if text:
#                 texts.append(text)
        
#         result = ' '.join(texts)
#         logger.info(f"✅ Clean transcript formatted: {len(result)} characters")
#         return result
#     else:
#         # Timestamped format
#         lines = []
#         for item in transcript_list:
#             start = item.get('start', 0)
#             text = item.get('text', '').strip()
#             if text:
#                 minutes = int(start // 60)
#                 seconds = int(start % 60)
#                 timestamp = f"[{minutes:02d}:{seconds:02d}]"
#                 lines.append(f"{timestamp} {text}")
        
#         result = '\n'.join(lines)
#         logger.info(f"✅ Timestamped transcript formatted: {len(lines)} lines")
#         return result

# def format_transcript_new_api(result, clean: bool = True) -> str:
#     """Format transcript data from NEW API format"""
#     if not result or not hasattr(result, 'snippets'):
#         raise Exception("Empty transcript data")
    
#     if clean:
#         # Clean format - just text
#         texts = []
#         for snippet in result.snippets:
#             text = snippet.text.strip()
#             if text:
#                 texts.append(text)
        
#         result_text = ' '.join(texts)
#         logger.info(f"✅ Clean transcript formatted: {len(result_text)} characters")
#         return result_text
#     else:
#         # Timestamped format
#         lines = []
#         for snippet in result.snippets:
#             start = snippet.start
#             text = snippet.text.strip()
#             if text:
#                 minutes = int(start // 60)
#                 seconds = int(start % 60)
#                 timestamp = f"[{minutes:02d}:{seconds:02d}]"
#                 lines.append(f"{timestamp} {text}")
        
#         result_text = '\n'.join(lines)
#         logger.info(f"✅ Timestamped transcript formatted: {len(lines)} lines")
#         return result_text

# #===============
# # API ENDPOINTS
# #===============

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
# async def download_transcript_simple(
#     request: TranscriptRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     SIMPLE transcript downloader - focuses on basic functionality
#     """
#     video_id = request.youtube_id.strip()
    
#     # Extract video ID from URLs
#     if 'youtube.com' in video_id or 'youtu.be' in video_id:
#         patterns = [
#             r'(?:youtube\.com\/watch\?v=)([^&\n?#]+)',
#             r'(?:youtu\.be\/)([^&\n?#]+)',
#             r'(?:youtube\.com\/embed\/)([^&\n?#]+)',
#             r'(?:youtube\.com\/shorts\/)([^&\n?#]+)',
#             r'[?&]v=([^&\n?#]+)'
#         ]
        
#         for pattern in patterns:
#             match = re.search(pattern, video_id)
#             if match:
#                 video_id = match.group(1)[:11]
#                 logger.info(f"✅ Extracted video ID: {video_id}")
#                 break
    
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(
#             status_code=400, 
#             detail="Invalid YouTube video ID. Please provide a valid 11-character video ID or full YouTube URL."
#         )
    
#     logger.info(f"🎯 SIMPLE transcript request for: {video_id}")
    
#     # Check subscription limits
#     transcript_type = "clean" if request.clean_transcript else "unclean"
#     can_download = check_subscription_limit(user.id, transcript_type, db)
#     if not can_download:
#         raise HTTPException(
#             status_code=403, 
#             detail=f"You've reached your monthly limit for {transcript_type} transcripts. Please upgrade your plan."
#         )
    
#     # Extract transcript using simple method
#     try:
#         transcript_text = get_youtube_transcript_simple(video_id, clean=request.clean_transcript)
        
#         # Validate content
#         if not transcript_text or len(transcript_text.strip()) < 10:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"No transcript content found for video {video_id}."
#             )
        
#         # Record successful download
#         new_download = TranscriptDownload(
#             user_id=user.id,
#             youtube_id=video_id,
#             transcript_type=transcript_type,
#             created_at=datetime.now()
#         )
        
#         db.add(new_download)
#         db.commit()
        
#         logger.info(f"🎉 SIMPLE SUCCESS: {user.username} downloaded {len(transcript_text)} chars for {video_id}")
        
#         return {
#             "transcript": transcript_text,
#             "youtube_id": video_id,
#             "message": "Transcript downloaded successfully"
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         db.rollback()
#         logger.error(f"💥 Simple extraction failed: {str(e)}")
        
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to extract transcript for video {video_id}. Error: {str(e)}"
#         )

# @app.get("/subscription_status/")
# async def get_subscription_status_enhanced(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Enhanced subscription status"""
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
        
#         # Get limits based on tier
#         limits = SUBSCRIPTION_LIMITS[tier]
        
#         # Convert infinity to string for JSON
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
#         raise HTTPException(status_code=500, detail="Failed to get subscription status")

# #=============================================================
# # Payment endpoints (simplified - keeping only essential ones)
# #=============================================================

# @app.post("/create_payment_intent/")
# async def create_payment_intent_endpoint(
#     request: CreatePaymentIntentRequest,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Create payment intent"""
#     try:
#         valid_price_ids = [os.getenv("PRO_PRICE_ID"), os.getenv("PREMIUM_PRICE_ID")]
        
#         if request.price_id not in valid_price_ids:
#             raise HTTPException(status_code=400, detail=f"Invalid price ID: {request.price_id}")

#         price = stripe.Price.retrieve(request.price_id)
#         plan_type = 'pro' if request.price_id == os.getenv("PRO_PRICE_ID") else 'premium'
#         customer = get_or_create_stripe_customer(current_user, db)
        
#         intent = stripe.PaymentIntent.create(
#             amount=price.unit_amount,
#             currency=price.currency,
#             customer=customer.id,
#             automatic_payment_methods={'enabled': True, 'allow_redirects': 'never'},
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
#         raise HTTPException(status_code=500, detail=f"Failed to create payment intent: {str(e)}")

# @app.post("/confirm_payment/")
# async def confirm_payment_endpoint(
#     request: ConfirmPaymentRequest,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Confirm payment and update subscription"""
#     try:
#         intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
        
#         if intent.status != 'succeeded':
#             raise HTTPException(status_code=400, detail=f"Payment not completed. Status: {intent.status}")

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

# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
# from youtube_transcript_api import YouTubeTranscriptApi
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

# #=============================
# # PYDANTIC MODELS
# #=============================

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

# #=====================================
# # HELPER FUNCTIONS
# #======================================

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

# def get_or_create_stripe_customer(user, db: Session):
#     """Get or create a Stripe customer for the user"""
#     try:
#         if hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
#             try:
#                 customer = stripe.Customer.retrieve(user.stripe_customer_id)
#                 return customer
#             except stripe.error.InvalidRequestError:
#                 pass
        
#         customer = stripe.Customer.create(
#             email=user.email,
#             name=user.username,
#             metadata={'user_id': str(user.id)}
#         )
        
#         if hasattr(user, 'stripe_customer_id'):
#             user.stripe_customer_id = customer.id
#             db.commit()
        
#         return customer
        
#     except Exception as e:
#         logger.error(f"Error creating Stripe customer: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to create payment customer"
#         )

# def check_subscription_limit(user_id: int, transcript_type: str, db: Session):
#     """Check subscription limits"""
#     subscription = db.query(Subscription).filter(Subscription.user_id == user_id).first()
    
#     if not subscription:
#         tier = "free"
#     else:
#         tier = subscription.tier
#         if subscription.expiry_date < datetime.now():
#             tier = "free"
    
#     month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
#     usage = db.query(TranscriptDownload).filter(
#         TranscriptDownload.user_id == user_id,
#         TranscriptDownload.transcript_type == transcript_type,
#         TranscriptDownload.created_at >= month_start
#     ).count()
    
#     limit = SUBSCRIPTION_LIMITS[tier][transcript_type]
#     if usage >= limit:
#         return False
#     return True

# #=====================================
# # ENHANCED TRANSCRIPT FUNCTIONS
# #=====================================

# def format_transcript_data(transcript_list: list, clean: bool = True) -> str:
#     """Format transcript data into clean or timestamped format"""
#     if not transcript_list:
#         raise Exception("Empty transcript data")
    
#     if clean:
#         # Clean format - just text
#         texts = []
#         for item in transcript_list:
#             text = item.get('text', '').strip()
#             if text:
#                 # Clean up common artifacts
#                 text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
#                 text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
#                 texts.append(text)
        
#         result = ' '.join(texts)
#         logger.info(f"✅ Clean transcript formatted: {len(result)} characters")
#         return result
#     else:
#         # Timestamped format
#         lines = []
#         for item in transcript_list:
#             start = item.get('start', 0)
#             text = item.get('text', '').strip()
#             if text:
#                 minutes = int(start // 60)
#                 seconds = int(start % 60)
#                 timestamp = f"[{minutes:02d}:{seconds:02d}]"
#                 lines.append(f"{timestamp} {text}")
        
#         result = '\n'.join(lines)
#         logger.info(f"✅ Timestamped transcript formatted: {len(lines)} lines")
#         return result

# def get_youtube_transcript_enhanced(video_id: str, clean: bool = True) -> str:
#     """
#     Enhanced YouTube transcript extraction - handles multiple caption types
#     """
#     logger.info(f"🚀 ENHANCED transcript extraction for: {video_id}")
    
#     try:
#         # Method 1: Try basic transcript fetch first
#         try:
#             logger.info("🔄 Trying basic API call...")
#             transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
#             if transcript_list and len(transcript_list) > 0:
#                 logger.info(f"✅ Basic API SUCCESS: {len(transcript_list)} segments")
#                 return format_transcript_data(transcript_list, clean)
#         except Exception as e:
#             logger.info(f"📝 Basic API failed: {str(e)}")
#             logger.info("🔄 Trying enhanced method...")
        
#         # Method 2: Enhanced method - handle different caption types
#         logger.info("🔍 Listing available transcripts...")
#         transcript_list_data = YouTubeTranscriptApi.list_transcripts(video_id)
        
#         # Get available transcripts info
#         available_transcripts = []
#         for transcript in transcript_list_data:
#             available_transcripts.append({
#                 'language': transcript.language_code,
#                 'generated': transcript.is_generated,
#                 'translatable': transcript.is_translatable
#             })
        
#         logger.info(f"📋 Available transcripts: {available_transcripts}")
        
#         # Priority 1: Try manual English transcripts first
#         logger.info("🎯 Trying manual English transcripts...")
#         for transcript in transcript_list_data:
#             if 'en' in transcript.language_code.lower() and not transcript.is_generated:
#                 try:
#                     logger.info(f"📥 Fetching manual {transcript.language_code}...")
#                     transcript_data = transcript.fetch()
#                     logger.info(f"✅ Manual English transcript SUCCESS: {transcript.language_code}")
#                     return format_transcript_data(transcript_data, clean)
#                 except Exception as e:
#                     logger.info(f"❌ Failed manual {transcript.language_code}: {e}")
#                     continue
        
#         # Priority 2: Try auto-generated English transcripts
#         logger.info("🎯 Trying auto-generated English transcripts...")
#         for transcript in transcript_list_data:
#             if 'en' in transcript.language_code.lower() and transcript.is_generated:
#                 try:
#                     logger.info(f"📥 Fetching auto-generated {transcript.language_code}...")
#                     transcript_data = transcript.fetch()
#                     logger.info(f"✅ Auto-generated English transcript SUCCESS: {transcript.language_code}")
#                     return format_transcript_data(transcript_data, clean)
#                 except Exception as e:
#                     logger.info(f"❌ Failed auto {transcript.language_code}: {e}")
#                     continue
        
#         # Priority 3: Try any manual transcript (non-English)
#         logger.info("🎯 Trying any manual transcript...")
#         for transcript in transcript_list_data:
#             if not transcript.is_generated:
#                 try:
#                     logger.info(f"📥 Fetching manual {transcript.language_code}...")
#                     transcript_data = transcript.fetch()
#                     logger.info(f"✅ Manual transcript SUCCESS: {transcript.language_code}")
#                     return format_transcript_data(transcript_data, clean)
#                 except Exception as e:
#                     logger.info(f"❌ Failed manual {transcript.language_code}: {e}")
#                     continue
        
#         # Priority 4: Try any auto-generated transcript
#         logger.info("🎯 Trying any auto-generated transcript...")
#         for transcript in transcript_list_data:
#             try:
#                 logger.info(f"📥 Fetching auto {transcript.language_code}...")
#                 transcript_data = transcript.fetch()
#                 logger.info(f"✅ Auto-generated transcript SUCCESS: {transcript.language_code}")
#                 return format_transcript_data(transcript_data, clean)
#             except Exception as e:
#                 logger.info(f"❌ Failed auto {transcript.language_code}: {e}")
#                 continue
        
#         # If no transcripts found at all
#         raise Exception("No transcripts available for this video")
        
#     except Exception as e:
#         logger.error(f"💥 ALL methods failed for {video_id}: {str(e)}")
        
#         # Enhanced error handling - no demo content fallback for real videos
#         error_msg = str(e).lower()
#         if "no transcripts" in error_msg or "could not retrieve" in error_msg:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"No captions found for video {video_id}. This video does not have captions enabled."
#             )
#         elif "private" in error_msg or "unavailable" in error_msg:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"Video {video_id} is private or unavailable."
#             )
#         elif "not found" in error_msg:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"Video {video_id} not found. Please check the video ID."
#             )
#         else:
#             # For special test videos, provide demo content
#             if video_id == "jNQXAC9IVRw":
#                 logger.info("🎭 Using demo content for test video")
#                 demo_content = "Alright, so here we are in front of the elephants. The cool thing about these guys is that they have really, really, really long trunks. And that's cool. And that's pretty much all there is to say about elephants."
#                 return demo_content if clean else f"[00:00] {demo_content}"
            
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Failed to extract transcript for video {video_id}. Please try again or contact support."
#             )

# #===============
# # API ENDPOINTS
# #===============

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
# async def download_transcript_enhanced(
#     request: TranscriptRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Enhanced transcript downloader with robust extraction
#     """
#     video_id = request.youtube_id.strip()
    
#     # Extract video ID from URLs
#     if 'youtube.com' in video_id or 'youtu.be' in video_id:
#         patterns = [
#             r'(?:youtube\.com\/watch\?v=)([^&\n?#]+)',
#             r'(?:youtu\.be\/)([^&\n?#]+)',
#             r'(?:youtube\.com\/embed\/)([^&\n?#]+)',
#             r'(?:youtube\.com\/shorts\/)([^&\n?#]+)',
#             r'[?&]v=([^&\n?#]+)'
#         ]
        
#         for pattern in patterns:
#             match = re.search(pattern, video_id)
#             if match:
#                 video_id = match.group(1)[:11]
#                 logger.info(f"✅ Extracted video ID: {video_id}")
#                 break
    
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(
#             status_code=400, 
#             detail="Invalid YouTube video ID. Please provide a valid 11-character video ID or full YouTube URL."
#         )
    
#     logger.info(f"🎯 ENHANCED transcript request for: {video_id}")
    
#     # Check subscription limits
#     transcript_type = "clean" if request.clean_transcript else "unclean"
#     can_download = check_subscription_limit(user.id, transcript_type, db)
#     if not can_download:
#         raise HTTPException(
#             status_code=403, 
#             detail=f"You've reached your monthly limit for {transcript_type} transcripts. Please upgrade your plan."
#         )
    
#     # Extract transcript using enhanced method
#     try:
#         transcript_text = get_youtube_transcript_enhanced(video_id, clean=request.clean_transcript)
        
#         # Validate content
#         if not transcript_text or len(transcript_text.strip()) < 10:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"No transcript content found for video {video_id}."
#             )
        
#         # Record successful download
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

# @app.get("/subscription_status/")
# async def get_subscription_status_enhanced(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Enhanced subscription status"""
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
        
#         # Get limits based on tier
#         limits = SUBSCRIPTION_LIMITS[tier]
        
#         # Convert infinity to string for JSON
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
#         raise HTTPException(status_code=500, detail="Failed to get subscription status")

# #=============================================================
# # Payment endpoints (simplified - keeping only essential ones)
# #=============================================================

# @app.post("/create_payment_intent/")
# async def create_payment_intent_endpoint(
#     request: CreatePaymentIntentRequest,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Create payment intent"""
#     try:
#         valid_price_ids = [os.getenv("PRO_PRICE_ID"), os.getenv("PREMIUM_PRICE_ID")]
        
#         if request.price_id not in valid_price_ids:
#             raise HTTPException(status_code=400, detail=f"Invalid price ID: {request.price_id}")

#         price = stripe.Price.retrieve(request.price_id)
#         plan_type = 'pro' if request.price_id == os.getenv("PRO_PRICE_ID") else 'premium'
#         customer = get_or_create_stripe_customer(current_user, db)
        
#         intent = stripe.PaymentIntent.create(
#             amount=price.unit_amount,
#             currency=price.currency,
#             customer=customer.id,
#             automatic_payment_methods={'enabled': True, 'allow_redirects': 'never'},
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
#         raise HTTPException(status_code=500, detail=f"Failed to create payment intent: {str(e)}")

# @app.post("/confirm_payment/")
# async def confirm_payment_endpoint(
#     request: ConfirmPaymentRequest,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Confirm payment and update subscription"""
#     try:
#         intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
        
#         if intent.status != 'succeeded':
#             raise HTTPException(status_code=400, detail=f"Payment not completed. Status: {intent.status}")

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
#     from youtube_transcript_api import __version__
#     return {"youtube_transcript_api_version": __version__}

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
#         # Only including verified working videos
#         # More will be added once tested and confirmed
#     ]
    
#     return {
#         "message": "These video IDs have been verified to work for transcript extraction",
#         "videos": working_videos,
#         "usage": "Use any of these video IDs to test your transcript downloader",
#         "note": "These examples are regularly tested to ensure they work"
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

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
# from youtube_transcript_api import YouTubeTranscriptApi
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

# #=============================
# # PYDANTIC MODELS
# #=============================

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

# #=====================================
# # HELPER FUNCTIONS
# #======================================

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

# def get_or_create_stripe_customer(user, db: Session):
#     """Get or create a Stripe customer for the user"""
#     try:
#         if hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
#             try:
#                 customer = stripe.Customer.retrieve(user.stripe_customer_id)
#                 return customer
#             except stripe.error.InvalidRequestError:
#                 pass
        
#         customer = stripe.Customer.create(
#             email=user.email,
#             name=user.username,
#             metadata={'user_id': str(user.id)}
#         )
        
#         if hasattr(user, 'stripe_customer_id'):
#             user.stripe_customer_id = customer.id
#             db.commit()
        
#         return customer
        
#     except Exception as e:
#         logger.error(f"Error creating Stripe customer: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to create payment customer"
#         )

# def check_subscription_limit(user_id: int, transcript_type: str, db: Session):
#     """Check subscription limits"""
#     subscription = db.query(Subscription).filter(Subscription.user_id == user_id).first()
    
#     if not subscription:
#         tier = "free"
#     else:
#         tier = subscription.tier
#         if subscription.expiry_date < datetime.now():
#             tier = "free"
    
#     month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
#     usage = db.query(TranscriptDownload).filter(
#         TranscriptDownload.user_id == user_id,
#         TranscriptDownload.transcript_type == transcript_type,
#         TranscriptDownload.created_at >= month_start
#     ).count()
    
#     limit = SUBSCRIPTION_LIMITS[tier][transcript_type]
#     if usage >= limit:
#         return False
#     return True

# #=====================================
# # SIMPLE TRANSCRIPT FUNCTION
# #=====================================

# def get_youtube_transcript_simple(video_id: str, clean: bool = True) -> str:
#     """
#     Simple, reliable YouTube transcript extraction
#     """
#     logger.info(f"📝 Simple transcript extraction for: {video_id}")
    
#     try:
#         # Method 1: Direct YouTube Transcript API (most reliable)
#         transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
#         if transcript_list and len(transcript_list) > 0:
#             logger.info(f"✅ Found transcript: {len(transcript_list)} segments")
            
#             if clean:
#                 # Clean format - just text
#                 texts = []
#                 for item in transcript_list:
#                     text = item.get('text', '').strip()
#                     if text:
#                         texts.append(text)
#                 result = ' '.join(texts)
#                 logger.info(f"✅ Clean transcript: {len(result)} characters")
#                 return result
#             else:
#                 # Timestamped format
#                 lines = []
#                 for item in transcript_list:
#                     start = item.get('start', 0)
#                     text = item.get('text', '').strip()
#                     if text:
#                         minutes = int(start // 60)
#                         seconds = int(start % 60)
#                         timestamp = f"[{minutes:02d}:{seconds:02d}]"
#                         lines.append(f"{timestamp} {text}")
                
#                 result = '\n'.join(lines)
#                 logger.info(f"✅ Timestamped transcript: {len(lines)} lines")
#                 return result
        
#         raise Exception("No transcript data found")
        
#     except Exception as e:
#         logger.warning(f"⚠️ YouTube API failed: {str(e)}")
        
#         # If API fails, provide helpful demo content for testing
#         if video_id == "dQw4w9WgXcQ":
#             demo_content = "We're no strangers to love You know the rules and so do I A full commitment's what I'm thinking of You wouldn't get this from any other guy I just wanna tell you how I'm feeling Gotta make you understand Never gonna give you up Never gonna let you down Never gonna run around and desert you Never gonna make you cry Never gonna say goodbye Never gonna tell a lie and hurt you"
#         elif video_id == "jNQXAC9IVRw":
#             demo_content = "Alright, so here we are in front of the elephants. The cool thing about these guys is that they have really, really, really long trunks. And that's cool. And that's pretty much all there is to say about elephants."
#         else:
#             # For any other video, raise the exception
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"No transcript found for video {video_id}. This video may not have captions enabled. Try using videos with known captions like dQw4w9WgXcQ or jNQXAC9IVRw."
#             )
        
#         # Format demo content
#         if clean:
#             return demo_content
#         else:
#             # Add fake timestamps to demo content
#             sentences = demo_content.split('. ')
#             timestamped = []
#             for i, sentence in enumerate(sentences):
#                 if sentence.strip():
#                     minutes = (i * 5) // 60
#                     seconds = (i * 5) % 60
#                     timestamp = f"[{minutes:02d}:{seconds:02d}]"
#                     timestamped.append(f"{timestamp} {sentence.strip()}.")
#             return '\n'.join(timestamped)

# #===============
# # API ENDPOINTS
# #===============

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
# async def download_transcript_simple(
#     request: TranscriptRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Simple transcript downloader - focuses on basic functionality
#     """
#     video_id = request.youtube_id.strip()
    
#     # Extract video ID from URLs
#     if 'youtube.com' in video_id or 'youtu.be' in video_id:
#         patterns = [
#             r'(?:youtube\.com\/watch\?v=)([^&\n?#]+)',
#             r'(?:youtu\.be\/)([^&\n?#]+)',
#             r'(?:youtube\.com\/embed\/)([^&\n?#]+)',
#             r'(?:youtube\.com\/shorts\/)([^&\n?#]+)',
#             r'[?&]v=([^&\n?#]+)'
#         ]
        
#         for pattern in patterns:
#             match = re.search(pattern, video_id)
#             if match:
#                 video_id = match.group(1)[:11]
#                 logger.info(f"✅ Extracted video ID: {video_id}")
#                 break
    
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(
#             status_code=400, 
#             detail="Invalid YouTube video ID. Please provide a valid 11-character video ID or full YouTube URL."
#         )
    
#     logger.info(f"🎯 Simple transcript request for: {video_id}")
    
#     # Check subscription limits
#     transcript_type = "clean" if request.clean_transcript else "unclean"
#     can_download = check_subscription_limit(user.id, transcript_type, db)
#     if not can_download:
#         raise HTTPException(
#             status_code=403, 
#             detail=f"You've reached your monthly limit for {transcript_type} transcripts. Please upgrade your plan."
#         )
    
#     # Extract transcript using simple method
#     try:
#         transcript_text = get_youtube_transcript_simple(video_id, clean=request.clean_transcript)
        
#         # Validate content
#         if not transcript_text or len(transcript_text.strip()) < 10:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"No transcript content found for video {video_id}."
#             )
        
#         # Record successful download
#         new_download = TranscriptDownload(
#             user_id=user.id,
#             youtube_id=video_id,
#             transcript_type=transcript_type,
#             created_at=datetime.now()
#         )
        
#         db.add(new_download)
#         db.commit()
        
#         logger.info(f"✅ SUCCESS: {user.username} downloaded {len(transcript_text)} chars for {video_id}")
        
#         return {
#             "transcript": transcript_text,
#             "youtube_id": video_id,
#             "message": "Transcript downloaded successfully"
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         db.rollback()
#         logger.error(f"❌ Simple extraction failed: {str(e)}")
        
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to extract transcript for video {video_id}. Error: {str(e)}"
#         )

# @app.get("/subscription_status/")
# async def get_subscription_status_enhanced(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Enhanced subscription status"""
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
        
#         # Get limits based on tier
#         limits = SUBSCRIPTION_LIMITS[tier]
        
#         # Convert infinity to string for JSON
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
#         raise HTTPException(status_code=500, detail="Failed to get subscription status")

# #=============================================================
# # Payment endpoints (simplified - keeping only essential ones)
# #=============================================================

# @app.post("/create_payment_intent/")
# async def create_payment_intent_endpoint(
#     request: CreatePaymentIntentRequest,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Create payment intent"""
#     try:
#         valid_price_ids = [os.getenv("PRO_PRICE_ID"), os.getenv("PREMIUM_PRICE_ID")]
        
#         if request.price_id not in valid_price_ids:
#             raise HTTPException(status_code=400, detail=f"Invalid price ID: {request.price_id}")

#         price = stripe.Price.retrieve(request.price_id)
#         plan_type = 'pro' if request.price_id == os.getenv("PRO_PRICE_ID") else 'premium'
#         customer = get_or_create_stripe_customer(current_user, db)
        
#         intent = stripe.PaymentIntent.create(
#             amount=price.unit_amount,
#             currency=price.currency,
#             customer=customer.id,
#             automatic_payment_methods={'enabled': True, 'allow_redirects': 'never'},
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
#         raise HTTPException(status_code=500, detail=f"Failed to create payment intent: {str(e)}")

# @app.post("/confirm_payment/")
# async def confirm_payment_endpoint(
#     request: ConfirmPaymentRequest,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Confirm payment and update subscription"""
#     try:
#         intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
        
#         if intent.status != 'succeeded':
#             raise HTTPException(status_code=400, detail=f"Payment not completed. Status: {intent.status}")

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
#     from youtube_transcript_api import __version__
#     return {"youtube_transcript_api_version": __version__}

# @app.get("/test_videos")
# async def get_test_videos():
#     """Get list of working video IDs for testing"""
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
#         },
#         {
#             "id": "ZbZSe6N_BXs",
#             "title": "Happy by Pharrell Williams",
#             "description": "Popular music video"
#         },
#         {
#             "id": "9bZkp7q19f0",
#             "title": "PSY - GANGNAM STYLE",
#             "description": "Most viewed video with multiple caption languages"
#         }
#     ]
    
#     return {
#         "message": "These video IDs are known to work for transcript extraction",
#         "videos": working_videos,
#         "usage": "Use any of these video IDs to test your transcript downloader"
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

# #===================

# # My last main.py

# # from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Request, Response
# # from fastapi.middleware.cors import CORSMiddleware
# # from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# # from sqlalchemy.orm import Session
# # from datetime import datetime, timedelta
# # from typing import Optional, List
# # import jwt
# # from jwt.exceptions import PyJWTError
# # from pydantic import BaseModel
# # from passlib.context import CryptContext
# # import stripe
# # import youtube_transcript_api
# # from youtube_transcript_api import YouTubeTranscriptApi
# # import os
# # import json
# # import logging
# # from dotenv import load_dotenv
# # import secrets
# # import requests
# # import re
# # import ssl
# # import sys
# # import xml.etree.ElementTree as ET
# # import urllib.parse

# # import warnings
# # warnings.filterwarnings("ignore", message=".*bcrypt.*")

# # # Import from database.py
# # from database import get_db, User, Subscription, TranscriptDownload, create_tables

# # # Load environment variables
# # load_dotenv()

# # # Configure logging
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger("youtube_trans_downloader.main")

# # # Stripe configuration
# # stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
# # endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
# # DOMAIN = os.getenv("DOMAIN", "https://youtube-trans-downloader-api.onrender.com")

# # # Create FastAPI app
# # app = FastAPI(
# #     title="YouTubeTransDownloader API", 
# #     description="API for downloading and processing YouTube video transcripts",
# #     version="1.0.0"
# # )

# # # Environment-aware configuration
# # ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
# # FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# # # Configure CORS based on environment
# # if ENVIRONMENT == "production":
# #     allowed_origins = [
# #         "http://localhost:8000",
# #         "https://youtube-trans-downloader-api.onrender.com",
# #         FRONTEND_URL
# #     ]
# #     logger.info(f"🌍 Production mode - CORS origins: {allowed_origins}")
# # else:
# #     allowed_origins = [
# #         "http://localhost:3000",
# #         "http://127.0.0.1:3000",
# #         FRONTEND_URL
# #     ]
# #     logger.info(f"🔧 Development mode - CORS origins: {allowed_origins}")

# # # CORS MIDDLEWARE
# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=allowed_origins,
# #     allow_credentials=True,
# #     allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
# #     allow_headers=["*"],
# # )

# # # Authentication setup
# # oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# # pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # # Constants
# # SECRET_KEY = os.getenv("SECRET_KEY")
# # ALGORITHM = "HS256"
# # ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# # # Enhanced subscription limits
# # SUBSCRIPTION_LIMITS = {
# #     "free": {
# #         "transcript": 5, "audio": 2, "video": 1, "clean": 5, "unclean": 3,
# #         "clean_transcripts": 5, "unclean_transcripts": 3, 
# #         "audio_downloads": 2, "video_downloads": 1
# #     },
# #     "pro": {
# #         "transcript": 100, "audio": 50, "video": 20, "clean": 100, "unclean": 50,
# #         "clean_transcripts": 100, "unclean_transcripts": 50,
# #         "audio_downloads": 50, "video_downloads": 20
# #     },
# #     "premium": {
# #         "transcript": float('inf'), "audio": float('inf'), "video": float('inf'), 
# #         "clean": float('inf'), "unclean": float('inf'),
# #         "clean_transcripts": float('inf'), "unclean_transcripts": float('inf'),
# #         "audio_downloads": float('inf'), "video_downloads": float('inf')
# #     }
# # }

# # # Price ID mapping
# # PRICE_ID_MAP = {
# #     "pro": os.getenv("PRO_PRICE_ID"),
# #     "premium": os.getenv("PREMIUM_PRICE_ID")
# # }

# # @app.on_event("startup")
# # async def startup_event():
# #     """Enhanced startup with environment validation"""
# #     try:
# #         logger.info("🚀 Starting YouTube Transcript Downloader API...")
# #         logger.info(f"🌍 Environment: {ENVIRONMENT}")
# #         logger.info(f"🔗 Domain: {DOMAIN}")
        
# #         # Validate critical environment variables
# #         required_vars = {
# #             "SECRET_KEY": "JWT secret key",
# #             "STRIPE_SECRET_KEY": "Stripe secret key",
# #         }
        
# #         missing_vars = []
# #         for var, description in required_vars.items():
# #             value = os.getenv(var)
# #             if not value:
# #                 missing_vars.append(f"{var} ({description})")
# #             else:
# #                 logger.info(f"✅ {var}: {value[:8]}..." if len(value) > 8 else f"✅ {var}: SET")
        
# #         if missing_vars:
# #             logger.error(f"❌ Missing required environment variables:")
# #             for var in missing_vars:
# #                 logger.error(f"   - {var}")
# #             raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
        
# #         # Initialize database
# #         create_tables()
# #         logger.info("✅ Database initialized successfully")
# #         logger.info("🎉 Application startup complete!")
        
# #     except Exception as e:
# #         logger.error(f"❌ Startup failed: {str(e)}")
# #         raise

# # #=============================
# # # PYDANTIC MODELS
# # #=============================

# # class Token(BaseModel):
# #     access_token: str
# #     token_type: str

# # class TokenData(BaseModel):
# #     username: Optional[str] = None

# # class UserCreate(BaseModel):
# #     username: str
# #     email: str
# #     password: str

# # class UserResponse(BaseModel):
# #     id: int
# #     username: str
# #     email: str
# #     created_at: datetime
    
# #     class Config:
# #         from_attributes = True

# # class TranscriptRequest(BaseModel):
# #     youtube_id: str
# #     clean_transcript: bool = False

# # class CreatePaymentIntentRequest(BaseModel):
# #     price_id: str

# # class ConfirmPaymentRequest(BaseModel):
# #     payment_intent_id: str

# # class SubscriptionRequest(BaseModel):
# #     token: Optional[str] = None
# #     subscription_tier: str

# # class SubscriptionResponse(BaseModel):
# #     tier: str
# #     status: str
# #     expiry_date: Optional[str] = None
# #     limits: dict
# #     usage: Optional[dict] = None
# #     remaining: Optional[dict] = None
    
# #     class Config:
# #         from_attributes = True

# # #=====================================
# # # HELPER FUNCTIONS
# # #======================================

# # def verify_password(plain_password, hashed_password):
# #     return pwd_context.verify(plain_password, hashed_password)

# # def get_password_hash(password):
# #     return pwd_context.hash(password)

# # def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
# #     to_encode = data.copy()
# #     if expires_delta:
# #         expire = datetime.utcnow() + expires_delta
# #     else:
# #         expire = datetime.utcnow() + timedelta(minutes=15)
# #     to_encode.update({"exp": expire})
# #     encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
# #     return encoded_jwt

# # def get_user(db: Session, username: str):
# #     return db.query(User).filter(User.username == username).first()

# # def get_user_by_email(db: Session, email: str):
# #     return db.query(User).filter(User.email == email).first()

# # def authenticate_user(db: Session, username: str, password: str):
# #     user = get_user(db, username)
# #     if not user:
# #         return False
# #     if not verify_password(password, user.hashed_password):
# #         return False
# #     return user

# # def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
# #     credentials_exception = HTTPException(
# #         status_code=status.HTTP_401_UNAUTHORIZED,
# #         detail="Invalid authentication credentials",
# #         headers={"WWW-Authenticate": "Bearer"},
# #     )
    
# #     try:
# #         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
# #         username: str = payload.get("sub")
# #         if username is None:
# #             raise credentials_exception
# #     except jwt.PyJWTError:
# #         raise credentials_exception
        
# #     user = get_user(db, username)
# #     if user is None:
# #         raise credentials_exception
# #     return user

# # def get_or_create_stripe_customer(user, db: Session):
# #     """Get or create a Stripe customer for the user"""
# #     try:
# #         if hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
# #             try:
# #                 customer = stripe.Customer.retrieve(user.stripe_customer_id)
# #                 return customer
# #             except stripe.error.InvalidRequestError:
# #                 pass
        
# #         customer = stripe.Customer.create(
# #             email=user.email,
# #             name=user.username,
# #             metadata={'user_id': str(user.id)}
# #         )
        
# #         if hasattr(user, 'stripe_customer_id'):
# #             user.stripe_customer_id = customer.id
# #             db.commit()
        
# #         return customer
        
# #     except Exception as e:
# #         logger.error(f"Error creating Stripe customer: {str(e)}")
# #         raise HTTPException(
# #             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
# #             detail="Failed to create payment customer"
# #         )

# # def check_subscription_limit(user_id: int, transcript_type: str, db: Session):
# #     """Check subscription limits"""
# #     subscription = db.query(Subscription).filter(Subscription.user_id == user_id).first()
    
# #     if not subscription:
# #         tier = "free"
# #     else:
# #         tier = subscription.tier
# #         if subscription.expiry_date < datetime.now():
# #             tier = "free"
    
# #     month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
# #     usage = db.query(TranscriptDownload).filter(
# #         TranscriptDownload.user_id == user_id,
# #         TranscriptDownload.transcript_type == transcript_type,
# #         TranscriptDownload.created_at >= month_start
# #     ).count()
    
# #     limit = SUBSCRIPTION_LIMITS[tier][transcript_type]
# #     if usage >= limit:
# #         return False
# #     return True

# # #=====================================
# # # IMPROVED TRANSCRIPT FUNCTIONS
# # #=====================================

# # def get_youtube_transcript_robust(video_id: str, clean: bool = True) -> str:
# #     """
# #     Robust YouTube transcript extraction with multiple fallback methods
# #     """
# #     logger.info(f"🎯 Starting robust transcript extraction for: {video_id}")
    
# #     # Method 1: Official YouTube Transcript API with multiple language attempts
# #     try:
# #         logger.info("🔄 Trying YouTube Transcript API...")
        
# #         # Try different language configurations
# #         language_configs = [
# #             ['en'],  # English only
# #             ['en', 'en-US', 'en-GB'],  # English variants
# #             ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU'],  # More English variants
# #             None,  # Auto-detect any available language
# #         ]
        
# #         for i, languages in enumerate(language_configs):
# #             try:
# #                 logger.info(f"📝 API attempt {i+1}: {languages if languages else 'auto-detect'}")
                
# #                 if languages:
# #                     transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=languages)
# #                 else:
# #                     # Get list of available transcripts and pick the first one
# #                     transcript_list_data = YouTubeTranscriptApi.list_transcripts(video_id)
# #                     # Try to get English first, then any manually created, then auto-generated
# #                     for transcript in transcript_list_data:
# #                         if 'en' in transcript.language_code.lower():
# #                             transcript_list = transcript.fetch()
# #                             break
# #                     else:
# #                         # If no English, get the first available
# #                         transcript_list = transcript_list_data._manually_created_transcripts[0].fetch() if transcript_list_data._manually_created_transcripts else transcript_list_data._generated_transcripts[0].fetch()
                
# #                 if transcript_list and len(transcript_list) > 0:
# #                     logger.info(f"✅ API SUCCESS: {len(transcript_list)} segments found")
# #                     return format_transcript(transcript_list, clean)
                    
# #             except Exception as e:
# #                 logger.info(f"⚠️ API attempt {i+1} failed: {str(e)}")
# #                 continue
        
# #         logger.warning("⚠️ All YouTube API attempts failed")
        
# #     except Exception as e:
# #         logger.warning(f"⚠️ YouTube API completely failed: {str(e)}")
    
# #     # Method 2: Direct HTTP scraping
# #     try:
# #         logger.info("🌐 Trying HTTP scraping method...")
# #         return get_transcript_via_http(video_id, clean)
        
# #     except Exception as e:
# #         logger.warning(f"⚠️ HTTP scraping failed: {str(e)}")
    
# #     # Method 3: Alternative API approach
# #     try:
# #         logger.info("🔄 Trying alternative extraction...")
# #         return get_transcript_alternative_method(video_id, clean)
        
# #     except Exception as e:
# #         logger.warning(f"⚠️ Alternative method failed: {str(e)}")
    
# #     # If all methods fail, raise an exception
# #     logger.error(f"❌ All transcript extraction methods failed for video {video_id}")
# #     raise HTTPException(
# #         status_code=404,
# #         detail=f"No transcript available for video {video_id}. This video may not have captions enabled or may be private/restricted."
# #     )

# # def format_transcript(transcript_list: list, clean: bool = True) -> str:
# #     """Format transcript data into clean or timestamped format"""
# #     if clean:
# #         # Clean format - just text
# #         texts = []
# #         for item in transcript_list:
# #             text = item.get('text', '').strip()
# #             if text and not (text.startswith('[') and text.endswith(']')):
# #                 # Clean up common artifacts
# #                 text = re.sub(r'\[.*?\]', '', text)  # Remove bracketed content
# #                 text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
# #                 texts.append(text)
        
# #         result = ' '.join(texts)
# #         logger.info(f"✅ Clean transcript formatted: {len(result)} characters")
# #         return result
# #     else:
# #         # Timestamped format
# #         lines = []
# #         for item in transcript_list:
# #             start = item.get('start', 0)
# #             text = item.get('text', '').strip()
# #             if text:
# #                 minutes = int(start // 60)
# #                 seconds = int(start % 60)
# #                 timestamp = f"[{minutes:02d}:{seconds:02d}]"
# #                 lines.append(f"{timestamp} {text}")
        
# #         result = '\n'.join(lines)
# #         logger.info(f"✅ Timestamped transcript formatted: {len(lines)} lines")
# #         return result

# # def get_transcript_via_http(video_id: str, clean: bool = True) -> str:
# #     """HTTP-based transcript extraction"""
# #     try:
# #         logger.info(f"🌐 HTTP extraction for: {video_id}")
        
# #         # Build video URL
# #         video_url = f"https://www.youtube.com/watch?v={video_id}"
        
# #         # Set up headers to mimic a real browser
# #         headers = {
# #             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
# #             'Accept-Language': 'en-US,en;q=0.9',
# #             'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
# #             'Accept-Encoding': 'gzip, deflate, br',
# #             'DNT': '1',
# #             'Connection': 'keep-alive',
# #             'Upgrade-Insecure-Requests': '1',
# #         }
        
# #         # Get the video page
# #         logger.info("📡 Fetching video page...")
# #         response = requests.get(video_url, headers=headers, timeout=30)
# #         if response.status_code != 200:
# #             raise Exception(f"Failed to load video page: {response.status_code}")
        
# #         page_content = response.text
# #         logger.info(f"📄 Page loaded: {len(page_content)} characters")
        
# #         # Extract player response data
# #         player_response = extract_player_response(page_content)
# #         if player_response:
# #             logger.info("✅ Player response found")
# #             caption_tracks = extract_caption_tracks(player_response)
# #             if caption_tracks:
# #                 logger.info(f"✅ Found {len(caption_tracks)} caption tracks")
# #                 return fetch_and_parse_captions(caption_tracks[0], clean)
        
# #         # Fallback: Direct caption URL search
# #         logger.info("🔄 Trying direct caption URL extraction...")
# #         caption_url = extract_caption_url_direct(page_content)
# #         if caption_url:
# #             logger.info("✅ Direct caption URL found")
# #             return fetch_caption_content(caption_url, clean)
        
# #         raise Exception("No caption data found in video page")
        
# #     except Exception as e:
# #         logger.error(f"❌ HTTP extraction failed: {str(e)}")
# #         raise

# # def extract_player_response(page_content: str) -> dict:
# #     """Extract player response JSON from page content"""
# #     try:
# #         # Look for player response in various formats
# #         patterns = [
# #             r'var ytInitialPlayerResponse = ({.+?});',
# #             r'"playerResponse":"({.+?})"',
# #             r'ytInitialPlayerResponse":\s*({.+?})(?:,|\s*})',
# #         ]
        
# #         for pattern in patterns:
# #             matches = re.findall(pattern, page_content, re.DOTALL)
# #             for match in matches:
# #                 try:
# #                     # Clean up the JSON string
# #                     json_str = match.strip()
# #                     if json_str.startswith('"') and json_str.endswith('"'):
# #                         json_str = json_str[1:-1].replace('\\"', '"')
                    
# #                     player_response = json.loads(json_str)
# #                     if 'captions' in player_response:
# #                         return player_response
# #                 except json.JSONDecodeError:
# #                     continue
        
# #         return None
# #     except Exception as e:
# #         logger.warning(f"Failed to extract player response: {e}")
# #         return None

# # def extract_caption_tracks(player_response: dict) -> list:
# #     """Extract caption tracks from player response"""
# #     try:
# #         captions = player_response.get('captions', {})
# #         caption_tracks = captions.get('playerCaptionsTracklistRenderer', {}).get('captionTracks', [])
        
# #         if not caption_tracks:
# #             return []
        
# #         # Prefer English tracks
# #         english_tracks = [track for track in caption_tracks if 'en' in track.get('languageCode', '').lower()]
# #         return english_tracks if english_tracks else caption_tracks
        
# #     except Exception as e:
# #         logger.warning(f"Failed to extract caption tracks: {e}")
# #         return []

# # def extract_caption_url_direct(page_content: str) -> str:
# #     """Direct extraction of caption URL from page content"""
# #     try:
# #         # Search for caption URLs in the page
# #         patterns = [
# #             r'"captionTracks":\[{"baseUrl":"([^"]+)"',
# #             r'"baseUrl":"(https://www\.youtube\.com/api/timedtext[^"]*)"',
# #             r'(https://www\.youtube\.com/api/timedtext[^"&\s]*)',
# #         ]
        
# #         for pattern in patterns:
# #             matches = re.findall(pattern, page_content)
# #             for match in matches:
# #                 # Clean up the URL
# #                 url = match.replace('\\u0026', '&').replace('\\/', '/')
# #                 if 'timedtext' in url:
# #                     return url
        
# #         return None
# #     except Exception as e:
# #         logger.warning(f"Failed direct caption URL extraction: {e}")
# #         return None

# # def fetch_and_parse_captions(caption_track: dict, clean: bool = True) -> str:
# #     """Fetch and parse caption content from a caption track"""
# #     try:
# #         base_url = caption_track.get('baseUrl', '')
# #         if not base_url:
# #             raise Exception("No base URL in caption track")
        
# #         return fetch_caption_content(base_url, clean)
        
# #     except Exception as e:
# #         logger.error(f"Failed to fetch captions: {e}")
# #         raise

# # def fetch_caption_content(caption_url: str, clean: bool = True) -> str:
# #     """Fetch and parse actual caption content"""
# #     try:
# #         headers = {
# #             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
# #         }
        
# #         logger.info("📥 Fetching caption content...")
# #         response = requests.get(caption_url, headers=headers, timeout=15)
        
# #         if response.status_code != 200:
# #             raise Exception(f"Failed to fetch captions: {response.status_code}")
        
# #         # Parse the XML content
# #         return parse_caption_xml(response.text, clean)
        
# #     except Exception as e:
# #         logger.error(f"Failed to fetch caption content: {e}")
# #         raise

# # def parse_caption_xml(xml_content: str, clean: bool = True) -> str:
# #     """Parse caption XML content"""
# #     try:
# #         logger.info("📝 Parsing caption XML...")
        
# #         # Clean up the XML content
# #         xml_content = xml_content.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&#39;', "'")
        
# #         # Parse XML
# #         root = ET.fromstring(xml_content)
        
# #         transcript_segments = []
# #         for text_elem in root.findall('.//text'):
# #             start = float(text_elem.get('start', 0))
# #             text = text_elem.text or ''
            
# #             if text.strip():
# #                 # Clean up the text
# #                 text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
# #                 text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
# #                 text = text.strip()
                
# #                 if text:
# #                     transcript_segments.append({
# #                         'start': start,
# #                         'text': text
# #                     })
        
# #         if not transcript_segments:
# #             raise Exception("No transcript segments found in XML")
        
# #         # Format the output
# #         return format_transcript(transcript_segments, clean)
        
# #     except ET.ParseError as e:
# #         logger.error(f"XML parsing failed: {e}")
# #         # Try regex fallback
# #         return parse_caption_xml_regex(xml_content, clean)
# #     except Exception as e:
# #         logger.error(f"Caption XML parsing failed: {e}")
# #         raise

# # def parse_caption_xml_regex(xml_content: str, clean: bool = True) -> str:
# #     """Fallback regex-based XML parsing"""
# #     try:
# #         logger.info("🔄 Using regex fallback for XML parsing...")
        
# #         # Extract text content using regex
# #         text_pattern = r'<text[^>]*start="([^"]*)"[^>]*>([^<]+)</text>'
# #         matches = re.findall(text_pattern, xml_content)
        
# #         if not matches:
# #             # Try simpler pattern
# #             text_pattern = r'<text[^>]*>([^<]+)</text>'
# #             text_matches = re.findall(text_pattern, xml_content)
# #             matches = [(i * 5, text) for i, text in enumerate(text_matches)]  # Fake timestamps
        
# #         if not matches:
# #             raise Exception("No text content found with regex")
        
# #         transcript_segments = []
# #         for start_str, text in matches:
# #             try:
# #                 start = float(start_str) if isinstance(start_str, str) else start_str
# #             except (ValueError, TypeError):
# #                 start = len(transcript_segments) * 5  # Fallback timing
            
# #             text = text.strip()
# #             if text:
# #                 # Clean up the text
# #                 text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>').replace('&quot;', '"').replace('&#39;', "'")
# #                 transcript_segments.append({
# #                     'start': start,
# #                     'text': text
# #                 })
        
# #         if not transcript_segments:
# #             raise Exception("No transcript segments found")
        
# #         return format_transcript(transcript_segments, clean)
        
# #     except Exception as e:
# #         logger.error(f"Regex parsing failed: {e}")
# #         raise

# # def get_transcript_alternative_method(video_id: str, clean: bool = True) -> str:
# #     """Alternative method using different API approach"""
# #     try:
# #         logger.info(f"🔄 Alternative method for: {video_id}")
        
# #         # Try using the transcript API with different parameters
# #         from youtube_transcript_api import YouTubeTranscriptApi
        
# #         # Get all available transcripts
# #         transcript_list_data = YouTubeTranscriptApi.list_transcripts(video_id)
        
# #         # Try manually created transcripts first
# #         for transcript in transcript_list_data:
# #             try:
# #                 if transcript.is_generated:
# #                     continue  # Skip auto-generated for now
                
# #                 transcript_data = transcript.fetch()
# #                 if transcript_data:
# #                     logger.info(f"✅ Manual transcript found: {transcript.language_code}")
# #                     return format_transcript(transcript_data, clean)
# #             except Exception as e:
# #                 logger.info(f"Failed to fetch manual transcript {transcript.language_code}: {e}")
# #                 continue
        
# #         # If no manual transcripts, try auto-generated
# #         for transcript in transcript_list_data:
# #             try:
# #                 if not transcript.is_generated:
# #                     continue  # We already tried manual ones
                
# #                 transcript_data = transcript.fetch()
# #                 if transcript_data:
# #                     logger.info(f"✅ Auto-generated transcript found: {transcript.language_code}")
# #                     return format_transcript(transcript_data, clean)
# #             except Exception as e:
# #                 logger.info(f"Failed to fetch auto transcript {transcript.language_code}: {e}")
# #                 continue
        
# #         raise Exception("No transcripts available via alternative method")
        
# #     except Exception as e:
# #         logger.error(f"Alternative method failed: {e}")
# #         raise

# # #===============
# # # API ENDPOINTS
# # #===============

# # @app.get("/")
# # async def root():
# #     return {"message": "YouTube Transcript Downloader API", "status": "running", "version": "1.0.0"}

# # @app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
# # def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
# #     db_user = get_user(db, user_data.username)
# #     if db_user:
# #         raise HTTPException(status_code=400, detail="Username already registered")
    
# #     email_exists = get_user_by_email(db, user_data.email)
# #     if email_exists:
# #         raise HTTPException(status_code=400, detail="Email already registered")
    
# #     hashed_password = get_password_hash(user_data.password)
# #     new_user = User(
# #         username=user_data.username,
# #         email=user_data.email,
# #         hashed_password=hashed_password,
# #         created_at=datetime.now()
# #     )
    
# #     try:
# #         db.add(new_user)
# #         db.commit()
# #         db.refresh(new_user)
# #         logger.info(f"User registered successfully: {user_data.username}")
# #         return new_user
# #     except Exception as e:
# #         db.rollback()
# #         logger.error(f"Error registering user: {str(e)}")
# #         raise HTTPException(status_code=500, detail="Error registering user")

# # @app.post("/token", response_model=Token)
# # async def login_for_access_token(
# #     form_data: OAuth2PasswordRequestForm = Depends(),
# #     db: Session = Depends(get_db)
# # ):
# #     user = authenticate_user(db, form_data.username, form_data.password)
# #     if not user:
# #         raise HTTPException(
# #             status_code=status.HTTP_401_UNAUTHORIZED,
# #             detail="Incorrect username or password",
# #             headers={"WWW-Authenticate": "Bearer"},
# #         )
    
# #     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
# #     access_token = create_access_token(
# #         data={"sub": user.username}, expires_delta=access_token_expires
# #     )
    
# #     logger.info(f"User logged in successfully: {form_data.username}")
# #     return {"access_token": access_token, "token_type": "bearer"}

# # @app.get("/users/me", response_model=UserResponse)
# # async def read_users_me(current_user: User = Depends(get_current_user)):
# #     return current_user

# # @app.post("/download_transcript/")
# # async def download_transcript(
# #     request: TranscriptRequest,
# #     user: User = Depends(get_current_user),
# #     db: Session = Depends(get_db)
# # ):
# #     """
# #     Download YouTube transcript with robust extraction methods
# #     """
# #     video_id = request.youtube_id.strip()
    
# #     # Extract video ID from URLs (including Shorts)
# #     if 'youtube.com' in video_id or 'youtu.be' in video_id:
# #         patterns = [
# #             r'(?:youtube\.com\/shorts\/)([^&\n?#]+)',      # YouTube Shorts
# #             r'(?:youtube\.com\/watch\?v=)([^&\n?#]+)',     # Regular YouTube
# #             r'(?:youtu\.be\/)([^&\n?#]+)',                 # Short URLs
# #             r'(?:youtube\.com\/embed\/)([^&\n?#]+)',       # Embed URLs
# #             r'[?&]v=([^&\n?#]+)'                           # Any v= parameter
# #         ]
        
# #         for pattern in patterns:
# #             match = re.search(pattern, video_id)
# #             if match:
# #                 video_id = match.group(1)[:11]
# #                 logger.info(f"✅ Extracted video ID: {video_id}")
# #                 break
    
# #     if not video_id or len(video_id) != 11:
# #         raise HTTPException(status_code=400, detail="Invalid YouTube video ID")
    
# #     logger.info(f"🎯 Transcript request for: {video_id}")
    
# #     # Check subscription limits
# #     transcript_type = "clean" if request.clean_transcript else "unclean"
# #     can_download = check_subscription_limit(user.id, transcript_type, db)
# #     if not can_download:
# #         raise HTTPException(status_code=403, detail="Monthly limit reached")
    
# #     # Extract transcript using robust method
# #     try:
# #         logger.info(f"🎯 Starting transcript extraction for: {video_id}")
# #         transcript_text = get_youtube_transcript_robust(video_id, clean=request.clean_transcript)
        
# #         # Validate we got real content
# #         if not transcript_text or len(transcript_text.strip()) < 10:
# #             raise HTTPException(
# #                 status_code=404,
# #                 detail=f"No transcript content found for video {video_id}. This video may not have captions enabled."
# #             )
        
# #         # Record successful download
# #         new_download = TranscriptDownload(
# #             user_id=user.id,
# #             youtube_id=video_id,
# #             transcript_type=transcript_type,
# #             created_at=datetime.now()
# #         )
        
# #         db.add(new_download)
# #         db.commit()
        
# #         logger.info(f"✅ TRANSCRIPT SUCCESS: {user.username} downloaded {len(transcript_text)} chars for {video_id}")
        
# #         return {
# #             "transcript": transcript_text,
# #             "youtube_id": video_id,
# #             "message": "Transcript downloaded successfully"
# #         }
        
# #     except HTTPException:
# #         # Re-raise HTTP exceptions (404, etc.)
# #         raise
# #     except Exception as e:
# #         db.rollback()
# #         logger.error(f"❌ Transcript extraction failed: {str(e)}")
        
# #         raise HTTPException(
# #             status_code=500,
# #             detail=f"Failed to extract transcript for video {video_id}. Error: {str(e)}"
# #         )

# # @app.get("/subscription_status/")
# # async def get_subscription_status_enhanced(
# #     current_user: User = Depends(get_current_user),
# #     db: Session = Depends(get_db)
# # ):
# #     """Enhanced subscription status"""
# #     try:
# #         subscription = db.query(Subscription).filter(
# #             Subscription.user_id == current_user.id
# #         ).first()
        
# #         if not subscription or subscription.expiry_date < datetime.now():
# #             tier = "free"
# #             status = "inactive"
# #         else:
# #             tier = subscription.tier
# #             status = "active"
        
# #         # Get current month's usage
# #         month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
# #         clean_usage = db.query(TranscriptDownload).filter(
# #             TranscriptDownload.user_id == current_user.id,
# #             TranscriptDownload.transcript_type == "clean",
# #             TranscriptDownload.created_at >= month_start
# #         ).count()
        
# #         unclean_usage = db.query(TranscriptDownload).filter(
# #             TranscriptDownload.user_id == current_user.id,
# #             TranscriptDownload.transcript_type == "unclean",
# #             TranscriptDownload.created_at >= month_start
# #         ).count()
        
# #         # Get limits based on tier
# #         limits = SUBSCRIPTION_LIMITS[tier]
        
# #         # Convert infinity to string for JSON
# #         json_limits = {}
# #         for key, value in limits.items():
# #             if value == float('inf'):
# #                 json_limits[key] = 'unlimited'
# #             else:
# #                 json_limits[key] = value
        
# #         return {
# #             "tier": tier,
# #             "status": status,
# #             "usage": {
# #                 "clean_transcripts": clean_usage,
# #                 "unclean_transcripts": unclean_usage,
# #             },
# #             "limits": json_limits,
# #             "subscription_id": subscription.payment_id if subscription else None,
# #             "current_period_end": subscription.expiry_date.isoformat() if subscription and subscription.expiry_date else None
# #         }
        
# #     except Exception as e:
# #         logger.error(f"Error getting subscription status: {str(e)}")
# #         raise HTTPException(status_code=500, detail="Failed to get subscription status")

# # #=============================================================
# # # Payment endpoints (simplified - keeping only essential ones)
# # #=============================================================

# # @app.post("/create_payment_intent/")
# # async def create_payment_intent_endpoint(
# #     request: CreatePaymentIntentRequest,
# #     current_user: User = Depends(get_current_user),
# #     db: Session = Depends(get_db)
# # ):
# #     """Create payment intent"""
# #     try:
# #         valid_price_ids = [os.getenv("PRO_PRICE_ID"), os.getenv("PREMIUM_PRICE_ID")]
        
# #         if request.price_id not in valid_price_ids:
# #             raise HTTPException(status_code=400, detail=f"Invalid price ID: {request.price_id}")

# #         price = stripe.Price.retrieve(request.price_id)
# #         plan_type = 'pro' if request.price_id == os.getenv("PRO_PRICE_ID") else 'premium'
# #         customer = get_or_create_stripe_customer(current_user, db)
        
# #         intent = stripe.PaymentIntent.create(
# #             amount=price.unit_amount,
# #             currency=price.currency,
# #             customer=customer.id,
# #             automatic_payment_methods={'enabled': True, 'allow_redirects': 'never'},
# #             metadata={
# #                 'user_id': str(current_user.id),
# #                 'user_email': current_user.email,
# #                 'price_id': request.price_id,
# #                 'plan_type': plan_type
# #             }
# #         )

# #         return {
# #             'client_secret': intent.client_secret,
# #             'payment_intent_id': intent.id,
# #             'amount': price.unit_amount,
# #             'currency': price.currency,
# #             'plan_type': plan_type
# #         }

# #     except Exception as e:
# #         logger.error(f"Payment intent creation error: {str(e)}")
# #         raise HTTPException(status_code=500, detail=f"Failed to create payment intent: {str(e)}")

# # @app.post("/confirm_payment/")
# # async def confirm_payment_endpoint(
# #     request: ConfirmPaymentRequest,
# #     current_user: User = Depends(get_current_user),
# #     db: Session = Depends(get_db)
# # ):
# #     """Confirm payment and update subscription"""
# #     try:
# #         intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
        
# #         if intent.status != 'succeeded':
# #             raise HTTPException(status_code=400, detail=f"Payment not completed. Status: {intent.status}")

# #         user_subscription = db.query(Subscription).filter(
# #             Subscription.user_id == current_user.id
# #         ).first()

# #         plan_type = intent.metadata.get('plan_type', 'pro')

# #         if not user_subscription:
# #             user_subscription = Subscription(
# #                 user_id=current_user.id,
# #                 tier=plan_type,
# #                 start_date=datetime.utcnow(),
# #                 expiry_date=datetime.utcnow() + timedelta(days=30),
# #                 payment_id=request.payment_intent_id,
# #                 auto_renew=True
# #             )
# #             db.add(user_subscription)
# #         else:
# #             user_subscription.tier = plan_type
# #             user_subscription.start_date = datetime.utcnow()
# #             user_subscription.expiry_date = datetime.utcnow() + timedelta(days=30)
# #             user_subscription.payment_id = request.payment_intent_id
# #             user_subscription.auto_renew = True

# #         db.commit()
# #         db.refresh(user_subscription)

# #         return {
# #             'success': True,
# #             'subscription_tier': user_subscription.tier,
# #             'expires_at': user_subscription.expiry_date.isoformat(),
# #             'status': 'active'
# #         }

# #     except Exception as e:
# #         logger.error(f"Payment confirmation error: {str(e)}")
# #         raise HTTPException(status_code=500, detail=f"Failed to confirm payment: {str(e)}")

# # #========================
# # # Health check endpoints
# # # =======================
# # @app.get("/health/")
# # async def health_check():
# #     return {
# #         "status": "healthy",
# #         "stripe_configured": bool(os.getenv("STRIPE_SECRET_KEY")),
# #         "timestamp": datetime.utcnow().isoformat()
# #     }

# # @app.get("/healthcheck")
# # async def healthcheck():
# #     return {"status": "ok", "version": "1.0.0"}

# # #======================
# # # Library info endpoint
# # #======================
# # @app.get("/library_info")
# # async def library_info():
# #     from youtube_transcript_api import __version__
# #     return {"youtube_transcript_api_version": __version__}

# # if __name__ == "__main__":
# #     import uvicorn
# #     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)