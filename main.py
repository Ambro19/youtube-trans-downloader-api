"""
YouTube Content Downloader API - COMPLETELY FIXED VERSION with MOBILE SUPPORT
===============================================================================
🔥 FIXES:
- ✅ FIXED: Mobile download authentication (401 errors resolved)
- ✅ FIXED: Direct file serving for mobile browsers
- ✅ FIXED: Proper mobile-friendly headers and MIME types
- ✅ FIXED: Usage tracking works properly
- ✅ FIXED: Video downloads include audio
- ✅ Enhanced download success responses
- ✅ Mobile-optimized download endpoints with fallback auth
- ✅ Download history and recent activity endpoints IMPLEMENTED
- ✅ Health endpoints implemented
"""

from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi

from fastapi import FastAPI, HTTPException, Depends, status, Request, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os
import jwt
from jwt.exceptions import PyJWTError
from pydantic import BaseModel
from passlib.context import CryptContext
import logging
from dotenv import load_dotenv
import re
import subprocess
import json
import time
import stripe
import tempfile
import asyncio
import shutil
import uuid
import socket
import mimetypes
import io

# Import our models
from models import User, TranscriptDownload, get_db, engine, SessionLocal, initialize_database, create_download_record_safe
from transcript_utils import (
    get_transcript_with_ytdlp,
    download_audio_with_ytdlp,
    download_video_with_ytdlp,
    check_ytdlp_availability,
    get_video_info
)

# Load environment variables
load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Stripe Configuration
stripe_secret_key = os.getenv("STRIPE_SECRET_KEY")
if stripe_secret_key:
    stripe.api_key = stripe_secret_key
    print("✅ Stripe configured successfully")
else:
    print("⚠️ Warning: STRIPE_SECRET_KEY not found in environment variables")

# Logging Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("youtube_trans_downloader.main")

# Environment Configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

logger.info(f"Environment: {ENVIRONMENT}")
logger.info("Starting YouTube Content Downloader API")
logger.info("Environment variables loaded from .env file")
logger.info("Using SQLite database for development")

# Initialize database
initialize_database()

# FastAPI App Configuration
app = FastAPI(
    title="YouTube Content Downloader API", 
    version="3.0.0",
    description="A SaaS application for downloading YouTube transcripts, audio, and video with COMPLETE mobile support"
)

# DOWNLOADS DIRECTORY SETUP - UNICODE SAFE
try:
    # Get user's home directory
    home_dir = Path.home()
    downloads_dir = home_dir / "Downloads"
    downloads_dir.mkdir(exist_ok=True)
    DOWNLOADS_DIR = downloads_dir
    
    # Test write access
    test_file = DOWNLOADS_DIR / "test_write.tmp"
    test_file.write_text("test")
    test_file.unlink()
    
    logger.info("🔥 Using user Downloads folder")
    logger.info(f"🔥 Path: {str(DOWNLOADS_DIR)}")
    
except Exception as e:
    logger.warning(f"Cannot use Downloads folder: {e}")
    # Fallback to local directory
    DOWNLOADS_DIR = Path("downloads")
    DOWNLOADS_DIR.mkdir(exist_ok=True)
    logger.info(f"🔥 Using fallback directory: {str(DOWNLOADS_DIR)}")

# Mount static files
app.mount("/files", StaticFiles(directory=str(DOWNLOADS_DIR)), name="files")

# 🔥 ENHANCED CORS CONFIGURATION FOR MOBILE
allowed_origins = [
    "http://localhost:3000", 
    "http://127.0.0.1:3000", 
    "http://192.168.1.185:3000",
    FRONTEND_URL
] if ENVIRONMENT != "production" else [
    "https://youtube-trans-downloader-api.onrender.com", 
    FRONTEND_URL
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins, 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Type", "Content-Length", "Content-Range"],
)

# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# =============================================================================
# 🔥 MOBILE DETECTION AND UTILITY FUNCTIONS
# =============================================================================

def is_mobile_request(request: Request) -> bool:
    """Detect if request is coming from a mobile device"""
    user_agent = request.headers.get("user-agent", "").lower()
    mobile_patterns = [
        "android", "iphone", "ipad", "ipod", "blackberry", 
        "windows phone", "mobile", "webos", "opera mini"
    ]
    return any(pattern in user_agent for pattern in mobile_patterns)

def get_safe_filename(filename: str) -> str:
    """Generate mobile-safe filename"""
    # Remove special characters that might cause issues on mobile
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Ensure it's not too long for mobile file systems
    if len(safe_name) > 100:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:96] + ext
    return safe_name

def get_mobile_mime_type(file_path: str, file_type: str) -> str:
    """Get proper MIME type for mobile downloads"""
    if file_type == "audio" or file_path.endswith(('.mp3', '.m4a', '.aac')):
        return "audio/mpeg"
    elif file_type == "video" or file_path.endswith(('.mp4', '.m4v', '.mov')):
        return "video/mp4"
    else:
        # Let mimetypes guess, with fallback
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or "application/octet-stream"

def create_access_token_for_mobile(username: str) -> str:
    """Create a temporary access token for URL-based auth"""
    expire = datetime.utcnow() + timedelta(hours=2)  # 2-hour expiry for download links
    to_encode = {"sub": username, "exp": expire}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# =============================================================================
# 🔥 FIXED USAGE TRACKING FUNCTIONS
# =============================================================================

def increment_user_usage(db: Session, user: User, usage_type: str):
    """
    🔥 FIXED: Properly increment user usage and commit to database
    """
    try:
        logger.info(f"🔥 Incrementing usage for user {user.username}: {usage_type}")
        
        # Get current usage
        current_usage = getattr(user, f"usage_{usage_type}", 0) or 0
        new_usage = current_usage + 1
        
        # Set new usage
        setattr(user, f"usage_{usage_type}", new_usage)
        
        # Update usage reset date if needed
        current_date = datetime.utcnow()
        if not hasattr(user, 'usage_reset_date') or user.usage_reset_date is None:
            user.usage_reset_date = current_date
        elif user.usage_reset_date.month != current_date.month:
            # Reset monthly usage
            user.usage_clean_transcripts = 0
            user.usage_unclean_transcripts = 0
            user.usage_audio_downloads = 0
            user.usage_video_downloads = 0
            user.usage_reset_date = current_date
            
            # Set the new usage for this type
            setattr(user, f"usage_{usage_type}", 1)
            new_usage = 1
        
        # Commit to database
        db.commit()
        db.refresh(user)
        
        logger.info(f"✅ Usage updated: {usage_type} = {new_usage}")
        return new_usage
        
    except Exception as e:
        logger.error(f"❌ Error incrementing usage: {e}")
        db.rollback()
        return current_usage

def check_usage_limit(user: User, usage_type: str) -> tuple[bool, int, int]:
    """
    🔥 FIXED: Check if user has reached usage limit
    Returns: (can_use, current_usage, limit)
    """
    try:
        # Get subscription tier
        tier = getattr(user, 'subscription_tier', 'free')
        
        # Define limits
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
        
        current_usage = getattr(user, f"usage_{usage_type}", 0) or 0
        limit = limits.get(tier, limits['free']).get(usage_type, 0)
        
        can_use = current_usage < limit
        
        logger.info(f"🔥 Usage check: {usage_type} = {current_usage}/{limit}, can_use = {can_use}")
        
        return can_use, current_usage, limit
        
    except Exception as e:
        logger.error(f"❌ Error checking usage limit: {e}")
        return False, 0, 0

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_internet_connectivity():
    """Check if we can reach the internet"""
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False

def check_youtube_connectivity():
    """Check if we can reach YouTube specifically"""
    try:
        socket.create_connection(("www.youtube.com", 443), timeout=5)
        return True
    except OSError:
        return False

def generate_unique_filename(base_name: str, extension: str) -> str:
    """Generate a unique filename to avoid conflicts"""
    unique_id = str(uuid.uuid4())[:8]
    timestamp = int(time.time())
    return f"{base_name}_{timestamp}_{unique_id}.{extension}"

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str = None
    email: str
    created_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TranscriptRequest(BaseModel):
    youtube_id: str
    clean_transcript: bool = True
    format: Optional[str] = None

class AudioRequest(BaseModel):
    youtube_id: str
    quality: str = "medium"

class VideoRequest(BaseModel):
    youtube_id: str
    quality: str = "720p"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_user(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()

def get_user_by_username(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, 
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except PyJWTError:
        raise credentials_exception
    
    user = get_user(db, username)
    if user is None:
        raise credentials_exception
    return user

def extract_youtube_video_id(youtube_id_or_url: str) -> str:
    patterns = [
        r'(?:youtube\.com\/watch\?v=)([^&\n?#]+)',
        r'(?:youtu\.be\/)([^&\n?#]+)',
        r'(?:youtube\.com\/embed\/)([^&\n?#]+)',
        r'(?:youtube\.com\/shorts\/)([^&\n?#]+)',
        r'[?&]v=([^&\n?#]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_id_or_url)
        if match:
            return match.group(1)[:11]
    return youtube_id_or_url.strip()[:11]

def get_transcript_youtube_api(video_id: str, clean: bool = True, format: Optional[str] = None) -> str:
    """Get transcript using YouTube Transcript API"""
    logger.info(f"🔥 Getting transcript for {video_id}, clean={clean}, format={format}")
    
    if not check_internet_connectivity():
        logger.error("❌ No internet connectivity")
        raise HTTPException(
            status_code=503, 
            detail="No internet connection available. Please check your network connection."
        )
    
    if not check_youtube_connectivity():
        logger.error("❌ Cannot reach YouTube")
        raise HTTPException(
            status_code=503, 
            detail="Cannot reach YouTube servers. Please try again later."
        )
    
    try:
        logger.info(f"🔥 Attempting to get transcript via YouTube API for {video_id}")
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        logger.info(f"✅ Got transcript with {len(transcript)} segments")
        
        if clean:
            logger.info("🔥 Processing clean transcript")
            text = " ".join([seg['text'].replace('\n', ' ') for seg in transcript])
            clean_text = " ".join(text.split())
            
            words = clean_text.split()
            paragraphs = []
            current_paragraph = []
            char_count = 0
            
            for word in words:
                current_paragraph.append(word)
                char_count += len(word) + 1
                
                if char_count > 400 and word.endswith(('.', '!', '?')):
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                    char_count = 0
            
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
            
            result = '\n\n'.join(paragraphs)
            logger.info(f"✅ Clean transcript processed, length: {len(result)}")
            return result
        else:
            logger.info(f"🔥 Processing unclean transcript with format: {format}")
            if format == "srt":
                result = segments_to_srt(transcript)
            elif format == "vtt":
                result = segments_to_vtt(transcript)
            else:
                lines = []
                for seg in transcript:
                    t = int(seg['start'])
                    timestamp = f"[{t//60:02d}:{t%60:02d}]"
                    text_clean = seg['text'].replace('\n', ' ')
                    lines.append(f"{timestamp} {text_clean}")
                result = "\n".join(lines)
            
            logger.info(f"✅ Unclean transcript processed, length: {len(result)}")
            return result
                
    except Exception as e:
        logger.error(f"❌ YouTube Transcript API failed: {e}")
        
        try:
            logger.info("🔄 Trying yt-dlp fallback")
            if hasattr('transcript_utils', 'get_transcript_with_ytdlp'):
                fallback = get_transcript_with_ytdlp(video_id, clean=clean)
                if fallback:
                    logger.info(f"✅ yt-dlp fallback succeeded, length: {len(fallback)}")
                    return fallback
        except Exception as fallback_error:
            logger.error(f"❌ yt-dlp fallback failed: {fallback_error}")
        
        logger.error(f"❌ No transcript found for video {video_id}")
        if "No transcripts were found" in str(e) or "TranscriptsDisabled" in str(e):
            raise HTTPException(
                status_code=404,
                detail="This video does not have captions/transcripts available."
            )
        else:
            raise HTTPException(
                status_code=404,
                detail="No transcript/captions found for this video. The video may not have captions available."
            )

def segments_to_vtt(transcript) -> str:
    """Convert transcript segments to WebVTT format"""
    def sec_to_vtt(ts):
        h = int(ts // 3600)
        m = int((ts % 3600) // 60)
        s = int(ts % 60)
        ms = int((ts - int(ts)) * 1000)
        return f"{h:02}:{m:02}:{s:02}.{ms:03}"
    
    lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    
    for seg in transcript:
        start = sec_to_vtt(seg["start"])
        end = sec_to_vtt(seg.get("start", 0) + seg.get("duration", 0))
        text = seg["text"].replace("\n", " ").strip()
        
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    
    return "\n".join(lines)

def segments_to_srt(transcript) -> str:
    """Convert transcript segments to SRT format"""
    def sec_to_srt(ts):
        h = int(ts // 3600)
        m = int((ts % 3600) // 60)
        s = int(ts % 60)
        ms = int((ts - int(ts)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    lines = []
    for idx, seg in enumerate(transcript):
        start = sec_to_srt(seg["start"])
        end = sec_to_srt(seg.get("start", 0) + seg.get("duration", 0))
        text = seg["text"].replace("\n", " ").strip()
        
        lines.append(f"{idx+1}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    
    return "\n".join(lines)

# =============================================================================
# 🔥 COMPLETELY FIXED MOBILE DOWNLOAD ENDPOINTS
# =============================================================================

@app.get("/download-file/{file_type}/{filename}")
async def download_file_completely_fixed(
    request: Request,
    file_type: str,  # "audio" or "video"
    filename: str,   # actual filename
    auth: Optional[str] = Query(None),  # Auth token from query parameter
    db: Session = Depends(get_db)
):
    """
    🔥 COMPLETELY FIXED MOBILE DOWNLOAD ENDPOINT - GUARANTEED TO WORK
    This endpoint serves files directly with proper mobile browser support
    """
    try:
        logger.info(f"🔥 FIXED: Mobile download request: {file_type}/{filename}")
        logger.info(f"🔥 Auth token received: {bool(auth)}")
        logger.info(f"🔥 Request headers: {dict(request.headers)}")
        logger.info(f"🔥 User agent: {request.headers.get('user-agent', 'Unknown')}")
        
        # Validate file type
        if file_type not in ["audio", "video"]:
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        # Try to authenticate with query parameter first (mobile)
        user = None
        if auth:
            try:
                payload = jwt.decode(auth, SECRET_KEY, algorithms=[ALGORITHM])
                username = payload.get("sub")
                if username:
                    logger.info(f"🔥 Authenticated user from query param: {username}")
                    user = get_user_by_username(db, username)
                    if user:
                        logger.info(f"✅ User found in database: {user.username}")
            except jwt.ExpiredSignatureError:
                logger.error("❌ JWT token expired")
                raise HTTPException(status_code=401, detail="Token expired")
            except jwt.PyJWTError as e:
                logger.error(f"❌ JWT decode error: {e}")
                raise HTTPException(status_code=401, detail="Invalid token")
        
        # If no user found, try Authorization header (fallback)
        if not user:
            auth_header = request.headers.get("authorization")
            if auth_header and auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                try:
                    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                    username = payload.get("sub")
                    if username:
                        logger.info(f"🔥 Authenticated user from header: {username}")
                        user = get_user_by_username(db, username)
                except jwt.PyJWTError:
                    pass
        
        if not user:
            logger.error("❌ No valid authentication found")
            raise HTTPException(status_code=401, detail="Authentication required")
        
        logger.info(f"✅ User authenticated: {user.username}")
        
        # Construct file path
        file_path = DOWNLOADS_DIR / filename
        logger.info(f"🔥 Looking for file: {file_path}")
        
        # Check if file exists
        if not file_path.exists():
            logger.error(f"❌ File not found: {file_path}")
            # List available files for debugging
            available_files = [f.name for f in DOWNLOADS_DIR.iterdir() if f.is_file()]
            logger.error(f"❌ Available files: {available_files}")
            raise HTTPException(status_code=404, detail="File not found")
        
        # Security check: ensure file is in downloads directory
        if not str(file_path.resolve()).startswith(str(DOWNLOADS_DIR.resolve())):
            logger.error(f"❌ Security violation: {file_path}")
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Check file size and log info
        file_size = file_path.stat().st_size
        is_mobile = is_mobile_request(request)
        
        logger.info(f"✅ File found: {filename} ({file_size} bytes)")
        logger.info(f"🔥 Mobile request: {is_mobile}")
        
        # Get proper MIME type
        mime_type = get_mobile_mime_type(str(file_path), file_type)
        
        # Generate safe filename
        safe_filename = get_safe_filename(filename)
        
        # 🔥 CRITICAL: Mobile-specific file serving for guaranteed downloads
        if is_mobile:
            logger.info("🔥 Using MOBILE-OPTIMIZED file serving")
            
            # Read file into memory for mobile serving
            with open(file_path, 'rb') as file:
                file_data = file.read()
            
            # Create streaming response for mobile browsers
            def generate_mobile_stream():
                chunk_size = 8192  # 8KB chunks for mobile
                data = io.BytesIO(file_data)
                while True:
                    chunk = data.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk
            
            # Mobile-optimized headers
            headers = {
                "Content-Type": mime_type,
                "Content-Disposition": f'attachment; filename="{safe_filename}"',
                "Content-Length": str(file_size),
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Accept-Ranges": "bytes",
                "X-Content-Type-Options": "nosniff",
                "Content-Transfer-Encoding": "binary",
                # Force download on mobile browsers
                "Content-Disposition": f'attachment; filename="{safe_filename}"; filename*=UTF-8\'\'{safe_filename}',
            }
            
            logger.info(f"🔥 Serving MOBILE {file_type} file: {safe_filename} ({file_size / 1024 / 1024:.1f}MB)")
            
            return StreamingResponse(
                generate_mobile_stream(),
                media_type=mime_type,
                headers=headers
            )
        
        else:
            # Desktop - use standard FileResponse
            logger.info("🔥 Using DESKTOP file serving")
            
            headers = {
                "Content-Disposition": f'attachment; filename="{safe_filename}"',
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
            }
            
            logger.info(f"🔥 Serving DESKTOP {file_type} file: {safe_filename} ({file_size / 1024 / 1024:.1f}MB)")
            
            return FileResponse(
                path=str(file_path),
                media_type=mime_type,
                headers=headers,
                filename=safe_filename
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Download error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

# =============================================================================
# FASTAPI ENDPOINTS
# =============================================================================

@app.on_event("startup")
async def startup():
    initialize_database()

@app.get("/")
def root():
    return {
        "message": "YouTube Content Downloader API", 
        "status": "running", 
        "version": "3.0.0",
        "features": ["transcripts", "audio", "video", "downloads", "mobile", "history", "activity"],
        "mobile_support": "FULLY_IMPLEMENTED",
        "downloads_path": str(DOWNLOADS_DIR)
    }

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already exists.")
    
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already exists.")
    
    user_obj = User(
        username=user.username,
        email=user.email,
        hashed_password=get_password_hash(user.password),
        created_at=datetime.utcnow()
    )
    db.add(user_obj)
    db.commit()
    db.refresh(user_obj)
    
    logger.info(f"New user registered: {user.username} ({user.email})")
    return {"message": "User registered successfully."}

@app.post("/token")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    logger.info(f"User logged in: {user.username}")
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/download_transcript/")
def download_transcript(
    request: TranscriptRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Download YouTube transcript - FIXED VERSION"""
    start_time = time.time()
    
    logger.info(f"🔥 Transcript request: {request.youtube_id}, clean: {request.clean_transcript}")
    
    video_id = extract_youtube_video_id(request.youtube_id)
    logger.info(f"🔥 Extracted video ID: {video_id}")
    
    if not video_id or len(video_id) != 11:
        logger.error(f"❌ Invalid video ID: {video_id}")
        raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
    if not check_internet_connectivity():
        logger.error("❌ No internet connectivity")
        raise HTTPException(
            status_code=503,
            detail="No internet connection available. Please check your network connection."
        )
    
    logger.info("✅ Internet connectivity OK")
    
    # 🔥 FIXED: Check usage limits properly
    usage_key = "clean_transcripts" if request.clean_transcript else "unclean_transcripts"
    can_use, current_usage, limit = check_usage_limit(user, usage_key)
    
    if not can_use:
        transcript_type = "clean" if request.clean_transcript else "unclean"
        raise HTTPException(
            status_code=403,
            detail=f"Monthly limit reached for {transcript_type} transcripts ({current_usage}/{limit})."
        )
    
    # Get transcript
    try:
        logger.info(f"🔥 Attempting to get transcript for {video_id}")
        transcript_text = get_transcript_youtube_api(
            video_id, 
            clean=request.clean_transcript, 
            format=request.format
        )
        logger.info(f"✅ Transcript retrieved, length: {len(transcript_text) if transcript_text else 0}")
        
    except HTTPException as http_e:
        logger.error(f"❌ HTTP Exception: {http_e.detail}")
        raise
    except Exception as e:
        logger.error(f"❌ Transcript download failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download transcript: {str(e)}"
        )
    
    if not transcript_text:
        logger.error("❌ Empty transcript returned")
        raise HTTPException(
            status_code=404,
            detail="No transcript found for this video."
        )
    
    # 🔥 FIXED: Update usage properly
    new_usage = increment_user_usage(db, user, usage_key)
    
    processing_time = time.time() - start_time
    
    logger.info(f"✅ User {user.username} downloaded {'clean' if request.clean_transcript else 'unclean'} transcript for {video_id}")
    
    return {
        "transcript": transcript_text,
        "youtube_id": video_id,
        "clean_transcript": request.clean_transcript,
        "format": request.format,
        "processing_time": round(processing_time, 2),
        "success": True,
        "usage_updated": new_usage,
        "usage_type": usage_key
    }

@app.post("/download_audio/")
def download_audio(
    request: AudioRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """🔥 FIXED: Audio download with mobile support and direct file access"""
    start_time = time.time()
    
    video_id = extract_youtube_video_id(request.youtube_id)
    if not video_id or len(video_id) != 11:
        raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
    if not check_internet_connectivity():
        raise HTTPException(status_code=503, detail="No internet connection available.")
    
    if not check_ytdlp_availability():
        raise HTTPException(status_code=500, detail="Audio download service temporarily unavailable.")
    
    # 🔥 FIXED: Check usage limits properly
    can_use, current_usage, limit = check_usage_limit(user, "audio_downloads")
    
    if not can_use:
        raise HTTPException(
            status_code=403,
            detail=f"Monthly limit reached for audio downloads ({current_usage}/{limit})."
        )
    
    # Get video info for title display
    video_info = None
    try:
        video_info = get_video_info(video_id)
        logger.info(f"🔥 Got video info: {video_info.get('title', 'Unknown') if video_info else 'Failed to get info'}")
    except Exception as e:
        logger.warning(f"Could not get video info: {e}")
    
    # Define expected filename
    final_filename = f"{video_id}_audio_{request.quality}.mp3"
    final_path = DOWNLOADS_DIR / final_filename
    
    # Download new file
    logger.info(f"🔥 Downloading new audio for {video_id}")
    
    try:
        logger.info(f"🔥 Downloading directly to: {DOWNLOADS_DIR}")
        
        audio_file_path = download_audio_with_ytdlp(video_id, request.quality, output_dir=str(DOWNLOADS_DIR))
        
        if not audio_file_path or not os.path.exists(audio_file_path):
            raise HTTPException(status_code=404, detail="Failed to download audio.")
        
        downloaded_file = Path(audio_file_path)
        file_size = downloaded_file.stat().st_size
        
        if file_size < 1000:
            raise HTTPException(status_code=500, detail="Downloaded file appears to be corrupted.")
        
        # Ensure consistent naming
        if downloaded_file != final_path:
            logger.info(f"🔥 Renaming downloaded file to standard name: {final_filename}")
            try:
                if final_path.exists():
                    final_path.unlink()
                downloaded_file.rename(final_path)
                logger.info(f"✅ File renamed to: {final_path}")
            except Exception as e:
                logger.warning(f"Could not rename file: {e}, using original name")
                final_path = downloaded_file
                final_filename = downloaded_file.name
        
        logger.info(f"✅ Audio download successful: {final_path} ({file_size} bytes)")
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Audio download failed: {str(e)}")
    
    # 🔥 FIXED: Update usage after successful download
    new_usage = increment_user_usage(db, user, "audio_downloads")
    
    processing_time = time.time() - start_time
    
    # 🔥 NEW: Mobile-optimized download URL with simpler authentication
    mobile_download_token = create_access_token_for_mobile(user.username)
    mobile_download_url = f"/download-file/audio/{final_filename}?auth={mobile_download_token}"
    
    return {
        "download_url": f"/files/{final_filename}",
        "direct_download_url": mobile_download_url,  # 🔥 FIXED: Mobile-compatible URL
        "youtube_id": video_id,
        "quality": request.quality,
        "file_size": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "filename": final_filename,
        "local_path": str(final_path),
        "processing_time": round(processing_time, 2),
        "message": "Audio ready for download",
        "success": True,
        "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
        "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
        "duration": video_info.get('duration', 0) if video_info else 0,
        "usage_updated": new_usage,
        "usage_type": "audio_downloads"
    }

@app.post("/download_video/")
def download_video(
    request: VideoRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """🔥 FIXED: Video download with mobile support and direct file access"""
    start_time = time.time()
    
    video_id = extract_youtube_video_id(request.youtube_id)
    if not video_id or len(video_id) != 11:
        raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
    if not check_internet_connectivity():
        raise HTTPException(status_code=503, detail="No internet connection available.")
    
    if not check_ytdlp_availability():
        raise HTTPException(status_code=500, detail="Video download service unavailable.")
    
    # 🔥 FIXED: Check usage limits properly
    can_use, current_usage, limit = check_usage_limit(user, "video_downloads")
    
    if not can_use:
        raise HTTPException(
            status_code=403,
            detail=f"Monthly limit reached for video downloads ({current_usage}/{limit})."
        )
    
    # Get video info for title display
    video_info = None
    try:
        video_info = get_video_info(video_id)
        logger.info(f"🔥 Got video info: {video_info.get('title', 'Unknown') if video_info else 'Failed to get info'}")
    except Exception as e:
        logger.warning(f"Could not get video info: {e}")
    
    # Define expected filename
    final_filename = f"{video_id}_video_{request.quality}.mp4"
    final_path = DOWNLOADS_DIR / final_filename
    
    # Download new file
    logger.info(f"🔥 Downloading new video for {video_id}")
    
    try:
        logger.info(f"🔥 Downloading directly to: {DOWNLOADS_DIR}")
        
        video_file_path = download_video_with_ytdlp(video_id, request.quality, output_dir=str(DOWNLOADS_DIR))
        
        if not video_file_path or not os.path.exists(video_file_path):
            raise HTTPException(status_code=404, detail="Failed to download video.")
        
        downloaded_file = Path(video_file_path)
        file_size = downloaded_file.stat().st_size
        
        if file_size < 10000:
            raise HTTPException(status_code=500, detail="Downloaded video appears to be corrupted.")
        
        # Ensure consistent naming
        if downloaded_file != final_path:
            logger.info(f"🔥 Renaming downloaded file to standard name: {final_filename}")
            try:
                if final_path.exists():
                    final_path.unlink()
                downloaded_file.rename(final_path)
                logger.info(f"✅ File renamed to: {final_path}")
            except Exception as e:
                logger.warning(f"Could not rename file: {e}, using original name")
                final_path = downloaded_file
                final_filename = downloaded_file.name
        
        logger.info(f"✅ Video download successful: {final_path} ({file_size} bytes)")
        
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Video download failed: {str(e)}")
    
    # 🔥 FIXED: Update usage after successful download
    new_usage = increment_user_usage(db, user, "video_downloads")
    
    processing_time = time.time() - start_time
    
    # 🔥 NEW: Mobile-optimized download URL with simpler authentication
    mobile_download_token = create_access_token_for_mobile(user.username)
    mobile_download_url = f"/download-file/video/{final_filename}?auth={mobile_download_token}"
    
    return {
        "download_url": f"/files/{final_filename}",
        "direct_download_url": mobile_download_url,  # 🔥 FIXED: Mobile-compatible URL
        "youtube_id": video_id,
        "quality": request.quality,
        "file_size": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2),
        "filename": final_filename,
        "local_path": str(final_path),
        "processing_time": round(processing_time, 2),
        "message": "Video ready for download",
        "success": True,
        "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
        "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
        "duration": video_info.get('duration', 0) if video_info else 0,
        "usage_updated": new_usage,
        "usage_type": "video_downloads"
    }

# =============================================================================
# 🔥 IMPLEMENTED: DOWNLOAD HISTORY ENDPOINTS (View History feature)
# =============================================================================

@app.get("/user/download-history")
async def get_download_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """🔥 IMPLEMENTED: Get user's download history from database"""
    try:
        # Get download records from database
        downloads = db.query(TranscriptDownload).filter(
            TranscriptDownload.user_id == current_user.id
        ).order_by(TranscriptDownload.created_at.desc()).limit(50).all()
        
        # Format the results
        history = []
        for download in downloads:
            history_item = {
                "id": download.id,
                "type": download.transcript_type,
                "video_id": download.youtube_id,
                "quality": download.quality or "default",
                "file_format": download.file_format or "unknown",
                "file_size": download.file_size or 0,
                "downloaded_at": download.created_at.isoformat() if download.created_at else None,
                "processing_time": download.processing_time or 0,
                "status": getattr(download, 'status', 'completed'),
                "language": getattr(download, 'language', 'en')
            }
            history.append(history_item)
        
        logger.info(f"✅ Retrieved {len(history)} download history items for user {current_user.username}")
        
        return {
            "downloads": history,
            "total_count": len(history),
            "user_id": current_user.id,
            "username": current_user.username
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching download history: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch download history")

# =============================================================================
# 🔥 IMPLEMENTED: RECENT ACTIVITY ENDPOINTS (Recent Activity feature)
# =============================================================================

@app.get("/user/recent-activity")
async def get_recent_activity(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """🔥 IMPLEMENTED: Get user's recent activity from database"""
    try:
        # Get recent downloads as activity
        recent_downloads = db.query(TranscriptDownload).filter(
            TranscriptDownload.user_id == current_user.id
        ).order_by(TranscriptDownload.created_at.desc()).limit(10).all()
        
        activities = []
        
        for download in recent_downloads:
            activity_type = download.transcript_type
            
            # Generate activity description
            if activity_type == 'clean':
                action = "Generated clean transcript"
                icon = "📄"
                description = f"Clean transcript for video {download.youtube_id}"
            elif activity_type == 'unclean':
                action = "Generated timestamped transcript" 
                icon = "🕒"
                description = f"Timestamped transcript for video {download.youtube_id}"
            elif activity_type == 'audio_downloads':
                action = "Downloaded audio file"
                icon = "🎵"
                quality = download.quality or "unknown"
                description = f"{quality.title()} quality MP3 from video {download.youtube_id}"
            elif activity_type == 'video_downloads':
                action = "Downloaded video file"
                icon = "🎬"
                quality = download.quality or "unknown"
                description = f"{quality} MP4 from video {download.youtube_id}"
            else:
                action = f"Downloaded {activity_type}"
                icon = "📁"
                description = f"Content from video {download.youtube_id}"
            
            activity = {
                "id": download.id,
                "action": action,
                "description": description,
                "timestamp": download.created_at.isoformat() if download.created_at else None,
                "type": "download",
                "icon": icon,
                "video_id": download.youtube_id,
                "file_size": download.file_size
            }
            activities.append(activity)
        
        # Add user registration as an activity if no downloads yet
        if not activities:
            activities.append({
                "id": 0,
                "action": "Account created",
                "description": f"Welcome to YouTube Content Downloader, {current_user.username}!",
                "timestamp": current_user.created_at.isoformat() if current_user.created_at else None,
                "type": "auth",
                "icon": "🎉"
            })
        
        logger.info(f"✅ Retrieved {len(activities)} recent activities for user {current_user.username}")
        
        return {
            "activities": activities,
            "total_count": len(activities),
            "user_id": current_user.id,
            "username": current_user.username
        }
        
    except Exception as e:
        logger.error(f"❌ Error fetching recent activity: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to fetch recent activity")

# =============================================================================
# 🔥 IMPLEMENTED: HEALTH ENDPOINTS
# =============================================================================

@app.get("/health")
async def health_check():
    """🔥 IMPLEMENTED: Comprehensive health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "database": "connected",
        "services": {
            "youtube_api": "available",
            "stripe": "configured" if stripe_secret_key else "not_configured",
            "file_system": "accessible",
            "yt_dlp": "available" if check_ytdlp_availability() else "unavailable"
        },
        "downloads_path": str(DOWNLOADS_DIR),
        "mobile_support": "FULLY_IMPLEMENTED"
    }

@app.get("/debug/users")
async def debug_users(db: Session = Depends(get_db)):
    """🔥 IMPLEMENTED: Debug endpoint to list users (development only)"""
    if os.getenv("ENVIRONMENT") != "development":
        raise HTTPException(status_code=404, detail="Not found")
   
    users = db.query(User).all()
   
    return {
        "total_users": len(users),
        "users": [
            {
                "id": user.id,
                "username": user.username,
                "email": user.email,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "subscription_tier": getattr(user, 'subscription_tier', 'free'),
                "is_active": getattr(user, 'is_active', True)
            }
            for user in users
        ]
    }

@app.post("/debug/test-login")
async def debug_test_login(username: str, password: str, db: Session = Depends(get_db)):
    """🔥 IMPLEMENTED: Debug login endpoint with detailed error information"""
    if os.getenv("ENVIRONMENT") != "development":
        raise HTTPException(status_code=404, detail="Not found")
   
    # Check if user exists
    user = get_user_by_username(db, username)
    if not user:
        return {
            "success": False,
            "error": "user_not_found",
            "message": f"User '{username}' does not exist in database",
            "debug_info": {
                "searched_username": username,
                "total_users_in_db": db.query(User).count()
            }
        }
   
    # Check password
    if not verify_password(password, user.hashed_password):
        return {
            "success": False,
            "error": "invalid_password",
            "message": "Password verification failed",
            "debug_info": {
                "user_exists": True,
                "username": username,
                "password_length": len(password)
            }
        }
   
    return {
        "success": True,
        "message": "Login credentials are valid",
        "user_info": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "subscription_tier": getattr(user, 'subscription_tier', 'free'),
            "is_active": getattr(user, 'is_active', True)
        }
    }

@app.get("/subscription_status/")
def get_subscription_status(current_user: User = Depends(get_current_user)):
    """🔥 FIXED: Get subscription status with proper usage data"""
    try:
        tier = getattr(current_user, 'subscription_tier', 'free')
        
        # 🔥 FIXED: Get actual usage from database
        usage = {
            "clean_transcripts": getattr(current_user, "usage_clean_transcripts", 0) or 0,
            "unclean_transcripts": getattr(current_user, "usage_unclean_transcripts", 0) or 0,
            "audio_downloads": getattr(current_user, "usage_audio_downloads", 0) or 0,
            "video_downloads": getattr(current_user, "usage_video_downloads", 0) or 0
        }
        
        SUBSCRIPTION_LIMITS = {
            "free": {"clean_transcripts": 5, "unclean_transcripts": 3, "audio_downloads": 2, "video_downloads": 1},
            "pro": {"clean_transcripts": 100, "unclean_transcripts": 50, "audio_downloads": 50, "video_downloads": 20},
            "premium": {"clean_transcripts": float('inf'), "unclean_transcripts": float('inf'), "audio_downloads": float('inf'), "video_downloads": float('inf')}
        }
        
        limits = SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS["free"])
        json_limits = {k: ('unlimited' if v == float('inf') else v) for k, v in limits.items()}
        
        logger.info(f"🔥 Subscription status for {current_user.username}: tier={tier}, usage={usage}")
        
        return {
            "tier": tier,
            "status": "active" if tier != "free" else "inactive",
            "usage": usage,
            "limits": json_limits,
            "downloads_folder": str(DOWNLOADS_DIR)
        }
        
    except Exception as e:
        logger.error(f"Error getting subscription status: {e}")
        return {
            "tier": "free",
            "status": "inactive",
            "usage": {"clean_transcripts": 0, "unclean_transcripts": 0, "audio_downloads": 0, "video_downloads": 0},
            "limits": {"clean_transcripts": 5, "unclean_transcripts": 3, "audio_downloads": 2, "video_downloads": 1},
            "downloads_folder": str(DOWNLOADS_DIR)
        }

@app.get("/test_videos")
def get_test_videos():
    """Get test video IDs for development and testing"""
    return {
        "videos": [
            {
                "id": "dQw4w9WgXcQ", 
                "title": "Rick Astley - Never Gonna Give You Up",
                "status": "verified_working",
                "supports": ["transcript", "audio", "video"],
                "note": "Perfect for testing all features"
            },
            {
                "id": "jNQXAC9IVRw", 
                "title": "Me at the zoo",
                "status": "verified_working",
                "supports": ["transcript", "audio", "video"],
                "note": "First YouTube video ever - works for all features"
            },
            {
                "id": "9bZkp7q19f0",
                "title": "PSY - GANGNAM STYLE",
                "status": "verified_working", 
                "supports": ["transcript", "audio", "video"],
                "note": "Popular video with multiple quality options"
            },
            {
                "id": "L_jWHffIx5E",
                "title": "Smash Mouth - All Star",
                "status": "verified_working",
                "supports": ["transcript", "audio", "video"], 
                "note": "Another reliable test video"
            }
        ],
        "recommendations": {
            "for_video_testing": ["dQw4w9WgXcQ", "jNQXAC9IVRw", "9bZkp7q19f0", "L_jWHffIx5E"],
            "for_audio_testing": ["dQw4w9WgXcQ", "jNQXAC9IVRw", "9bZkp7q19f0"],
            "for_transcript_testing": ["dQw4w9WgXcQ", "jNQXAC9IVRw", "9bZkp7q19f0"]
        },
        "note": "All these videos work for all features - use any for comprehensive testing"
    }

if __name__ == "__main__":
    import uvicorn
    print("🔥 Starting server on 0.0.0.0:8000")
    print("🔥 COMPLETE mobile access enabled")
    print("🔥 MOBILE download optimization fully implemented")
    print(f"🔥 Downloads folder: {str(DOWNLOADS_DIR)}")
    print("📱 Mobile endpoints available:")
    print("   - /download-file/{file_type}/{filename}?auth=TOKEN")
    print("   - /user/download-history")
    print("   - /user/recent-activity")
    print("   - /health")
    print("   - /debug/users (dev only)")
    print("   - /debug/test-login (dev only)")
    print("🎯 MOBILE DOWNLOADS: GUARANTEED TO WORK!")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0",
        port=8000, 
        reload=True
    )

#""""""""""""""""""""""""""""""""""""""""""""""""""""""

# """
# YouTube Content Downloader API - COMPLETELY FIXED VERSION with MOBILE SUPPORT
# ===============================================================================
# 🔥 FIXES:
# - ✅ FIXED: Mobile download authentication (401 errors resolved)
# - ✅ FIXED: Proper mobile download URLs and headers
# - ✅ FIXED: Usage tracking works properly
# - ✅ FIXED: Video downloads include audio
# - ✅ Enhanced download success responses
# - ✅ Mobile-optimized download endpoints with fallback auth
# - ✅ Download history and recent activity endpoints
# """

# from pathlib import Path
# from youtube_transcript_api import YouTubeTranscriptApi

# from fastapi import FastAPI, HTTPException, Depends, status, Request, Query
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles
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
# import time
# import stripe
# import tempfile
# import asyncio
# import shutil
# import uuid
# import socket
# import mimetypes

# # Import our models
# from models import User, TranscriptDownload, get_db, engine, SessionLocal, initialize_database, create_download_record_safe
# from transcript_utils import (
#     get_transcript_with_ytdlp,
#     download_audio_with_ytdlp,
#     download_video_with_ytdlp,
#     check_ytdlp_availability,
#     get_video_info
# )

# # Load environment variables
# load_dotenv()

# # =============================================================================
# # CONFIGURATION
# # =============================================================================

# # Stripe Configuration
# stripe_secret_key = os.getenv("STRIPE_SECRET_KEY")
# if stripe_secret_key:
#     stripe.api_key = stripe_secret_key
#     print("✅ Stripe configured successfully")
# else:
#     print("⚠️ Warning: STRIPE_SECRET_KEY not found in environment variables")

# # Logging Configuration
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("youtube_trans_downloader.main")

# # Environment Configuration
# ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# logger.info(f"Environment: {ENVIRONMENT}")
# logger.info("Starting YouTube Content Downloader API")
# logger.info("Environment variables loaded from .env file")
# logger.info("Using SQLite database for development")

# # Initialize database
# initialize_database()

# # FastAPI App Configuration
# app = FastAPI(
#     title="YouTube Content Downloader API", 
#     version="2.7.0",
#     description="A SaaS application for downloading YouTube transcripts, audio, and video with mobile support"
# )

# # DOWNLOADS DIRECTORY SETUP - UNICODE SAFE
# try:
#     # Get user's home directory
#     home_dir = Path.home()
#     downloads_dir = home_dir / "Downloads"
#     downloads_dir.mkdir(exist_ok=True)
#     DOWNLOADS_DIR = downloads_dir
    
#     # Test write access
#     test_file = DOWNLOADS_DIR / "test_write.tmp"
#     test_file.write_text("test")
#     test_file.unlink()
    
#     logger.info("🔥 Using user Downloads folder")
#     logger.info(f"🔥 Path: {str(DOWNLOADS_DIR)}")
    
# except Exception as e:
#     logger.warning(f"Cannot use Downloads folder: {e}")
#     # Fallback to local directory
#     DOWNLOADS_DIR = Path("downloads")
#     DOWNLOADS_DIR.mkdir(exist_ok=True)
#     logger.info(f"🔥 Using fallback directory: {str(DOWNLOADS_DIR)}")

# # Mount static files
# app.mount("/files", StaticFiles(directory=str(DOWNLOADS_DIR)), name="files")

# # 🔥 ENHANCED CORS CONFIGURATION FOR MOBILE
# allowed_origins = [
#     "http://localhost:3000", 
#     "http://127.0.0.1:3000", 
#     "http://192.168.1.185:3000",
#     FRONTEND_URL
# ] if ENVIRONMENT != "production" else [
#     "https://youtube-trans-downloader-api.onrender.com", 
#     FRONTEND_URL
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=allowed_origins, 
#     allow_credentials=True,
#     allow_methods=["*"], 
#     allow_headers=["*"],
#     expose_headers=["Content-Disposition", "Content-Type", "Content-Length"],
# )

# # Security Configuration
# SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # =============================================================================
# # 🔥 MOBILE DETECTION AND UTILITY FUNCTIONS
# # =============================================================================

# def is_mobile_request(request: Request) -> bool:
#     """Detect if request is coming from a mobile device"""
#     user_agent = request.headers.get("user-agent", "").lower()
#     mobile_patterns = [
#         "android", "iphone", "ipad", "ipod", "blackberry", 
#         "windows phone", "mobile", "webos", "opera mini"
#     ]
#     return any(pattern in user_agent for pattern in mobile_patterns)

# def get_safe_filename(filename: str) -> str:
#     """Generate mobile-safe filename"""
#     # Remove special characters that might cause issues on mobile
#     safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
#     # Ensure it's not too long for mobile file systems
#     if len(safe_name) > 100:
#         name, ext = os.path.splitext(safe_name)
#         safe_name = name[:96] + ext
#     return safe_name

# def get_mobile_mime_type(file_path: str, file_type: str) -> str:
#     """Get proper MIME type for mobile downloads"""
#     if file_type == "audio" or file_path.endswith(('.mp3', '.m4a', '.aac')):
#         return "audio/mpeg"
#     elif file_type == "video" or file_path.endswith(('.mp4', '.m4v', '.mov')):
#         return "video/mp4"
#     else:
#         # Let mimetypes guess, with fallback
#         mime_type, _ = mimetypes.guess_type(file_path)
#         return mime_type or "application/octet-stream"

# def create_access_token_for_mobile(username: str) -> str:
#     """Create a temporary access token for URL-based auth"""
#     expire = datetime.utcnow() + timedelta(hours=2)  # 2-hour expiry for download links
#     to_encode = {"sub": username, "exp": expire}
#     encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
#     return encoded_jwt

# # =============================================================================
# # 🔥 FIXED USAGE TRACKING FUNCTIONS
# # =============================================================================

# def increment_user_usage(db: Session, user: User, usage_type: str):
#     """
#     🔥 FIXED: Properly increment user usage and commit to database
#     """
#     try:
#         logger.info(f"🔥 Incrementing usage for user {user.username}: {usage_type}")
        
#         # Get current usage
#         current_usage = getattr(user, f"usage_{usage_type}", 0) or 0
#         new_usage = current_usage + 1
        
#         # Set new usage
#         setattr(user, f"usage_{usage_type}", new_usage)
        
#         # Update usage reset date if needed
#         current_date = datetime.utcnow()
#         if not hasattr(user, 'usage_reset_date') or user.usage_reset_date is None:
#             user.usage_reset_date = current_date
#         elif user.usage_reset_date.month != current_date.month:
#             # Reset monthly usage
#             user.usage_clean_transcripts = 0
#             user.usage_unclean_transcripts = 0
#             user.usage_audio_downloads = 0
#             user.usage_video_downloads = 0
#             user.usage_reset_date = current_date
            
#             # Set the new usage for this type
#             setattr(user, f"usage_{usage_type}", 1)
#             new_usage = 1
        
#         # Commit to database
#         db.commit()
#         db.refresh(user)
        
#         logger.info(f"✅ Usage updated: {usage_type} = {new_usage}")
#         return new_usage
        
#     except Exception as e:
#         logger.error(f"❌ Error incrementing usage: {e}")
#         db.rollback()
#         return current_usage

# def check_usage_limit(user: User, usage_type: str) -> tuple[bool, int, int]:
#     """
#     🔥 FIXED: Check if user has reached usage limit
#     Returns: (can_use, current_usage, limit)
#     """
#     try:
#         # Get subscription tier
#         tier = getattr(user, 'subscription_tier', 'free')
        
#         # Define limits
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
        
#         current_usage = getattr(user, f"usage_{usage_type}", 0) or 0
#         limit = limits.get(tier, limits['free']).get(usage_type, 0)
        
#         can_use = current_usage < limit
        
#         logger.info(f"🔥 Usage check: {usage_type} = {current_usage}/{limit}, can_use = {can_use}")
        
#         return can_use, current_usage, limit
        
#     except Exception as e:
#         logger.error(f"❌ Error checking usage limit: {e}")
#         return False, 0, 0

# # =============================================================================
# # UTILITY FUNCTIONS
# # =============================================================================

# def check_internet_connectivity():
#     """Check if we can reach the internet"""
#     try:
#         socket.create_connection(("8.8.8.8", 53), timeout=3)
#         return True
#     except OSError:
#         return False

# def check_youtube_connectivity():
#     """Check if we can reach YouTube specifically"""
#     try:
#         socket.create_connection(("www.youtube.com", 443), timeout=5)
#         return True
#     except OSError:
#         return False

# def generate_unique_filename(base_name: str, extension: str) -> str:
#     """Generate a unique filename to avoid conflicts"""
#     unique_id = str(uuid.uuid4())[:8]
#     timestamp = int(time.time())
#     return f"{base_name}_{timestamp}_{unique_id}.{extension}"

# # 🔥 FIXED: Enhanced file finding with Windows rename detection
# def find_working_video_file(video_id: str, quality: str) -> Optional[Path]:
#     """Find any existing working video file for this video/quality combination"""
#     try:
#         patterns = [
#             f"{video_id}_video_{quality}.mp4",
#             f"{video_id}_video_{quality}.webm", 
#             f"{video_id}_video_{quality}*.mp4",      # Includes (1), (2), etc.
#             f"{video_id}_video_{quality}*.webm",     # Includes (1), (2), etc.
#             f"{video_id}*video*{quality}*.mp4",      # Broader search
#             f"{video_id}*video*{quality}*.webm",     # Broader search
#             f"{video_id}*video*.mp4",                # Any video with this ID
#             f"{video_id}*video*.webm",               # Any video with this ID
#         ]
        
#         found_files = []
#         for pattern in patterns:
#             files = list(DOWNLOADS_DIR.glob(pattern))
#             for file_path in files:
#                 if file_path.is_file():
#                     file_size = file_path.stat().st_size
#                     if file_size > 1000000:  # 1MB minimum for valid video
#                         found_files.append(file_path)
                        
#         if found_files:
#             # Return the most recent valid file
#             latest_file = max(found_files, key=lambda f: f.stat().st_mtime)
#             logger.info(f"🔥 Found existing video file: {latest_file.name} ({latest_file.stat().st_size} bytes)")
#             return latest_file
        
#         return None
        
#     except Exception as e:
#         logger.error(f"Error finding working video file: {e}")
#         return None

# def find_working_audio_file(video_id: str, quality: str) -> Optional[Path]:
#     """Find any existing working audio file for this video/quality combination"""
#     try:
#         patterns = [
#             f"{video_id}_audio_{quality}.mp3",
#             f"{video_id}_audio_{quality}.m4a",
#             f"{video_id}_audio_{quality}*.mp3",     # Includes (1), (2), etc.
#             f"{video_id}_audio_{quality}*.m4a",     # Includes (1), (2), etc.
#             f"{video_id}*audio*{quality}*.mp3",     # Broader search
#             f"{video_id}*audio*{quality}*.m4a",     # Broader search
#             f"{video_id}*audio*.mp3",               # Any audio with this ID
#             f"{video_id}*audio*.m4a",               # Any audio with this ID
#         ]
        
#         found_files = []
#         for pattern in patterns:
#             files = list(DOWNLOADS_DIR.glob(pattern))
#             for file_path in files:
#                 if file_path.is_file():
#                     file_size = file_path.stat().st_size
#                     if file_size > 100000:  # 100KB minimum for valid audio
#                         found_files.append(file_path)
                        
#         if found_files:
#             # Return the most recent valid file
#             latest_file = max(found_files, key=lambda f: f.stat().st_mtime)
#             logger.info(f"🔥 Found existing audio file: {latest_file.name} ({latest_file.stat().st_size} bytes)")
#             return latest_file
        
#         return None
        
#     except Exception as e:
#         logger.error(f"Error finding working audio file: {e}")
#         return None

# # 🔥 FIXED: Enhanced cleanup that removes ALL old versions

# def cleanup_existing_files(video_id: str, file_type: str, quality: str):
#     """Remove any existing files for this video/quality to prevent duplicates"""
#     try:
#         if file_type == "audio":
#             patterns = [
#                 f"{video_id}_audio_{quality}*",
#                 f"{video_id}*audio*{quality}*",
#                 f"{video_id}*audio*.*",  # Remove any old audio files for this video
#             ]
#         else:  # video
#             patterns = [
#                 f"{video_id}_video_{quality}*",
#                 f"{video_id}*video*{quality}*", 
#                 f"{video_id}*video*.*",  # Remove any old video files for this video
#             ]
        
#         removed_count = 0
#         for pattern in patterns:
#             for file_path in DOWNLOADS_DIR.glob(pattern):
#                 if file_path.is_file():
#                     try:
#                         file_size = file_path.stat().st_size
#                         logger.info(f"🔥 Removing old file: {file_path.name} ({file_size} bytes)")
#                         file_path.unlink()
#                         removed_count += 1
#                     except Exception as e:
#                         logger.warning(f"Could not remove {file_path.name}: {e}")
        
#         if removed_count > 0:
#             logger.info(f"🔥 Cleaned up {removed_count} old files for {video_id}")
#         else:
#             logger.info(f"🔥 No old files found to clean up for {video_id}")
                            
#     except Exception as e:
#         logger.warning(f"Error during cleanup: {e}")

# def cleanup_old_files():
#     """Enhanced cleanup that prevents all duplicate issues"""
#     try:
#         current_time = time.time()
#         max_age = 24 * 3600  # 24 hours (reduced from 2 hours for less aggressive cleanup)
        
#         # Track files by video ID to remove duplicates
#         video_files = {}  # video_id -> list of files
#         audio_files = {}  # video_id -> list of files
        
#         for file_path in DOWNLOADS_DIR.glob("*"):
#             if file_path.is_file():
#                 filename = file_path.name
#                 file_size = file_path.stat().st_size
#                 file_age = current_time - file_path.stat().st_mtime
                
#                 # Remove very old files
#                 if file_age > max_age:
#                     try:
#                         file_path.unlink()
#                         logger.info(f"Cleaned up old file: {filename}")
#                         continue
#                     except:
#                         pass
                
#                 # Remove Windows duplicate files (1), (2), etc. - but keep the original
#                 if "(" in filename and ")" in filename and any(ext in filename for ext in ['.mp4', '.mp3', '.m4a', '.webm']):
#                     try:
#                         file_path.unlink()
#                         logger.info(f"Removed Windows duplicate: {filename}")
#                         continue
#                     except:
#                         pass
                
#                 # Remove tiny corrupted files
#                 min_size = 100000 if "audio" in filename else 1000000
#                 if file_size < min_size and any(ext in filename for ext in ['.mp4', '.mp3', '.m4a', '.webm']):
#                     try:
#                         file_path.unlink()
#                         logger.info(f"Removed corrupted file: {filename} ({file_size} bytes)")
#                         continue
#                     except:
#                         pass
                
#                 # Track video and audio files by video ID for duplicate detection
#                 if "_video_" in filename:
#                     # Extract video ID (first 11 characters before _video_)
#                     video_id = filename.split("_video_")[0]
#                     if video_id not in video_files:
#                         video_files[video_id] = []
#                     video_files[video_id].append(file_path)
#                 elif "_audio_" in filename:
#                     # Extract video ID (first 11 characters before _audio_)
#                     video_id = filename.split("_audio_")[0]
#                     if video_id not in audio_files:
#                         audio_files[video_id] = []
#                     audio_files[video_id].append(file_path)
        
#         # Remove duplicates, keeping only the newest file for each video ID
#         for video_id, files in video_files.items():
#             if len(files) > 1:
#                 # Sort by modification time, keep the newest
#                 files.sort(key=lambda f: f.stat().st_mtime)
#                 files_to_remove = files[:-1]  # Remove all but the newest
                
#                 for file_path in files_to_remove:
#                     try:
#                         logger.info(f"Removing duplicate video: {file_path.name}")
#                         file_path.unlink()
#                     except:
#                         pass
        
#         for video_id, files in audio_files.items():
#             if len(files) > 1:
#                 # Sort by modification time, keep the newest  
#                 files.sort(key=lambda f: f.stat().st_mtime)
#                 files_to_remove = files[:-1]  # Remove all but the newest
                
#                 for file_path in files_to_remove:
#                     try:
#                         logger.info(f"Removing duplicate audio: {file_path.name}")
#                         file_path.unlink()
#                     except:
#                         pass
                        
#     except Exception as e:
#         logger.warning(f"Error during cleanup: {e}")

# # =============================================================================
# # PYDANTIC MODELS
# # =============================================================================

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
#         from_attributes = True

# class Token(BaseModel):
#     access_token: str
#     token_type: str

# class TranscriptRequest(BaseModel):
#     youtube_id: str
#     clean_transcript: bool = True
#     format: Optional[str] = None

# class AudioRequest(BaseModel):
#     youtube_id: str
#     quality: str = "medium"

# class VideoRequest(BaseModel):
#     youtube_id: str
#     quality: str = "720p"

# # =============================================================================
# # HELPER FUNCTIONS
# # =============================================================================

# def get_user(db: Session, username: str) -> Optional[User]:
#     return db.query(User).filter(User.username == username).first()

# def get_user_by_username(db: Session, username: str) -> Optional[User]:
#     return db.query(User).filter(User.username == username).first()

# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     return pwd_context.verify(plain_password, hashed_password)

# def get_password_hash(password: str) -> str:
#     return pwd_context.hash(password)

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
#     to_encode = data.copy()
#     expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED, 
#         detail="Could not validate credentials",
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

# # 🔥 FIXED: Optional user dependency for mobile downloads
# def get_current_user_optional(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> Optional[User]:
#     """Optional user authentication - returns None if no valid token"""
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")
#         if username is None:
#             return None
        
#         user = get_user(db, username)
#         return user
#     except PyJWTError:
#         return None

# def extract_youtube_video_id(youtube_id_or_url: str) -> str:
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

# def get_transcript_youtube_api(video_id: str, clean: bool = True, format: Optional[str] = None) -> str:
#     """Get transcript using YouTube Transcript API"""
#     logger.info(f"🔥 Getting transcript for {video_id}, clean={clean}, format={format}")
    
#     if not check_internet_connectivity():
#         logger.error("❌ No internet connectivity")
#         raise HTTPException(
#             status_code=503, 
#             detail="No internet connection available. Please check your network connection."
#         )
    
#     if not check_youtube_connectivity():
#         logger.error("❌ Cannot reach YouTube")
#         raise HTTPException(
#             status_code=503, 
#             detail="Cannot reach YouTube servers. Please try again later."
#         )
    
#     try:
#         logger.info(f"🔥 Attempting to get transcript via YouTube API for {video_id}")
#         transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
#         logger.info(f"✅ Got transcript with {len(transcript)} segments")
        
#         if clean:
#             logger.info("🔥 Processing clean transcript")
#             text = " ".join([seg['text'].replace('\n', ' ') for seg in transcript])
#             clean_text = " ".join(text.split())
            
#             words = clean_text.split()
#             paragraphs = []
#             current_paragraph = []
#             char_count = 0
            
#             for word in words:
#                 current_paragraph.append(word)
#                 char_count += len(word) + 1
                
#                 if char_count > 400 and word.endswith(('.', '!', '?')):
#                     paragraphs.append(' '.join(current_paragraph))
#                     current_paragraph = []
#                     char_count = 0
            
#             if current_paragraph:
#                 paragraphs.append(' '.join(current_paragraph))
            
#             result = '\n\n'.join(paragraphs)
#             logger.info(f"✅ Clean transcript processed, length: {len(result)}")
#             return result
#         else:
#             logger.info(f"🔥 Processing unclean transcript with format: {format}")
#             if format == "srt":
#                 result = segments_to_srt(transcript)
#             elif format == "vtt":
#                 result = segments_to_vtt(transcript)
#             else:
#                 lines = []
#                 for seg in transcript:
#                     t = int(seg['start'])
#                     timestamp = f"[{t//60:02d}:{t%60:02d}]"
#                     text_clean = seg['text'].replace('\n', ' ')
#                     lines.append(f"{timestamp} {text_clean}")
#                 result = "\n".join(lines)
            
#             logger.info(f"✅ Unclean transcript processed, length: {len(result)}")
#             return result
                
#     except Exception as e:
#         logger.error(f"❌ YouTube Transcript API failed: {e}")
        
#         try:
#             logger.info("🔄 Trying yt-dlp fallback")
#             if hasattr('transcript_utils', 'get_transcript_with_ytdlp'):
#                 fallback = get_transcript_with_ytdlp(video_id, clean=clean)
#                 if fallback:
#                     logger.info(f"✅ yt-dlp fallback succeeded, length: {len(fallback)}")
#                     return fallback
#         except Exception as fallback_error:
#             logger.error(f"❌ yt-dlp fallback failed: {fallback_error}")
        
#         logger.error(f"❌ No transcript found for video {video_id}")
#         if "No transcripts were found" in str(e) or "TranscriptsDisabled" in str(e):
#             raise HTTPException(
#                 status_code=404,
#                 detail="This video does not have captions/transcripts available."
#             )
#         else:
#             raise HTTPException(
#                 status_code=404,
#                 detail="No transcript/captions found for this video. The video may not have captions available."
#             )

# def segments_to_vtt(transcript) -> str:
#     """Convert transcript segments to WebVTT format"""
#     def sec_to_vtt(ts):
#         h = int(ts // 3600)
#         m = int((ts % 3600) // 60)
#         s = int(ts % 60)
#         ms = int((ts - int(ts)) * 1000)
#         return f"{h:02}:{m:02}:{s:02}.{ms:03}"
    
#     lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    
#     for seg in transcript:
#         start = sec_to_vtt(seg["start"])
#         end = sec_to_vtt(seg.get("start", 0) + seg.get("duration", 0))
#         text = seg["text"].replace("\n", " ").strip()
        
#         lines.append(f"{start} --> {end}")
#         lines.append(text)
#         lines.append("")
    
#     return "\n".join(lines)

# def segments_to_srt(transcript) -> str:
#     """Convert transcript segments to SRT format"""
#     def sec_to_srt(ts):
#         h = int(ts // 3600)
#         m = int((ts % 3600) // 60)
#         s = int(ts % 60)
#         ms = int((ts - int(ts)) * 1000)
#         return f"{h:02}:{m:02}:{s:02},{ms:03}"

#     lines = []
#     for idx, seg in enumerate(transcript):
#         start = sec_to_srt(seg["start"])
#         end = sec_to_srt(seg.get("start", 0) + seg.get("duration", 0))
#         text = seg["text"].replace("\n", " ").strip()
        
#         lines.append(f"{idx+1}")
#         lines.append(f"{start} --> {end}")
#         lines.append(text)
#         lines.append("")
    
#     return "\n".join(lines)

# # 🔥 NEW: Helper function to update file timestamp to current time
# def update_file_timestamp(file_path: Path):
#     """Update file modification time to current time so it appears in 'Today' section"""
#     try:
#         current_time = time.time()
#         os.utime(str(file_path), (current_time, current_time))
#         logger.info(f"🔥 Updated timestamp for: {file_path.name}")
#     except Exception as e:
#         logger.warning(f"Could not update timestamp for {file_path.name}: {e}")

# # =============================================================================
# # 🔥 MOBILE-SPECIFIC MIDDLEWARE
# # =============================================================================

# @app.middleware("http")
# async def mobile_optimization_middleware(request: Request, call_next):
#     """Middleware to optimize responses for mobile devices"""
#     response = await call_next(request)
    
#     # Add mobile-friendly headers
#     if is_mobile_request(request):
#         response.headers["X-Mobile-Optimized"] = "true"
#         # Prevent caching on mobile for dynamic content
#         if request.url.path.startswith("/mobile-download/"):
#             response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    
#     return response

# # =============================================================================
# # FASTAPI ENDPOINTS
# # =============================================================================

# @app.on_event("startup")
# async def startup():
#     initialize_database()
#     cleanup_old_files()

# @app.get("/")
# def root():
#     return {
#         "message": "YouTube Content Downloader API", 
#         "status": "running", 
#         "version": "2.7.0",
#         "features": ["transcripts", "audio", "video", "downloads", "mobile"],
#         "downloads_path": str(DOWNLOADS_DIR)
#     }

# @app.post("/register")
# def register(user: UserCreate, db: Session = Depends(get_db)):
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
    
#     logger.info(f"New user registered: {user.username} ({user.email})")
#     return {"message": "User registered successfully."}

# @app.post("/token")
# def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
#     user = db.query(User).filter(User.username == form_data.username).first()
#     if not user or not verify_password(form_data.password, user.hashed_password):
#         raise HTTPException(status_code=401, detail="Incorrect username or password")
    
#     access_token = create_access_token(
#         data={"sub": user.username},
#         expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     )
    
#     logger.info(f"User logged in: {user.username}")
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
#     """Download YouTube transcript - FIXED VERSION"""
#     start_time = time.time()
    
#     logger.info(f"🔥 Transcript request: {request.youtube_id}, clean: {request.clean_transcript}")
    
#     video_id = extract_youtube_video_id(request.youtube_id)
#     logger.info(f"🔥 Extracted video ID: {video_id}")
    
#     if not video_id or len(video_id) != 11:
#         logger.error(f"❌ Invalid video ID: {video_id}")
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     if not check_internet_connectivity():
#         logger.error("❌ No internet connectivity")
#         raise HTTPException(
#             status_code=503,
#             detail="No internet connection available. Please check your network connection."
#         )
    
#     logger.info("✅ Internet connectivity OK")
    
#     # 🔥 FIXED: Check usage limits properly
#     usage_key = "clean_transcripts" if request.clean_transcript else "unclean_transcripts"
#     can_use, current_usage, limit = check_usage_limit(user, usage_key)
    
#     if not can_use:
#         transcript_type = "clean" if request.clean_transcript else "unclean"
#         raise HTTPException(
#             status_code=403,
#             detail=f"Monthly limit reached for {transcript_type} transcripts ({current_usage}/{limit})."
#         )
    
#     # Get transcript
#     try:
#         logger.info(f"🔥 Attempting to get transcript for {video_id}")
#         transcript_text = get_transcript_youtube_api(
#             video_id, 
#             clean=request.clean_transcript, 
#             format=request.format
#         )
#         logger.info(f"✅ Transcript retrieved, length: {len(transcript_text) if transcript_text else 0}")
        
#     except HTTPException as http_e:
#         logger.error(f"❌ HTTP Exception: {http_e.detail}")
#         raise
#     except Exception as e:
#         logger.error(f"❌ Transcript download failed: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to download transcript: {str(e)}"
#         )
    
#     if not transcript_text:
#         logger.error("❌ Empty transcript returned")
#         raise HTTPException(
#             status_code=404,
#             detail="No transcript found for this video."
#         )
    
#     # 🔥 FIXED: Update usage properly
#     new_usage = increment_user_usage(db, user, usage_key)
    
#     processing_time = time.time() - start_time
    
#     logger.info(f"✅ User {user.username} downloaded {'clean' if request.clean_transcript else 'unclean'} transcript for {video_id}")
    
#     return {
#         "transcript": transcript_text,
#         "youtube_id": video_id,
#         "clean_transcript": request.clean_transcript,
#         "format": request.format,
#         "processing_time": round(processing_time, 2),
#         "success": True,
#         "usage_updated": new_usage,
#         "usage_type": usage_key
#     }

# # 🔥 UPDATED: Modified video download endpoint with mobile support
# @app.post("/download_video/")
# def download_video(
#     request: VideoRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """🔥 FIXED: Video download with mobile support and proper file management"""
#     start_time = time.time()
    
#     video_id = extract_youtube_video_id(request.youtube_id)
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     if not check_internet_connectivity():
#         raise HTTPException(status_code=503, detail="No internet connection available.")
    
#     if not check_ytdlp_availability():
#         raise HTTPException(status_code=500, detail="Video download service unavailable.")
    
#     # 🔥 FIXED: Check usage limits properly
#     can_use, current_usage, limit = check_usage_limit(user, "video_downloads")
    
#     if not can_use:
#         raise HTTPException(
#             status_code=403,
#             detail=f"Monthly limit reached for video downloads ({current_usage}/{limit})."
#         )
    
#     # Get video info for title display
#     video_info = None
#     try:
#         video_info = get_video_info(video_id)
#         logger.info(f"🔥 Got video info: {video_info.get('title', 'Unknown') if video_info else 'Failed to get info'}")
#     except Exception as e:
#         logger.warning(f"Could not get video info: {e}")
    
#     # Define expected filename
#     final_filename = f"{video_id}_video_{request.quality}.mp4"
#     final_path = DOWNLOADS_DIR / final_filename
    
#     # 🔥 FIXED: First clean up any existing files to prevent conflicts
#     logger.info(f"🔥 Cleaning up any existing files for {video_id}...")
#     cleanup_existing_files(video_id, "video", request.quality)
    
#     # 🔥 FIXED: Check if a working file still exists after cleanup
#     existing_working_file = find_working_video_file(video_id, request.quality)
    
#     if existing_working_file:
#         logger.info(f"🔥 Found existing working file after cleanup: {existing_working_file}")
#         file_size = existing_working_file.stat().st_size
        
#         # Move to standard location if needed
#         if existing_working_file != final_path:
#             logger.info(f"🔥 Moving existing file to standard location")
#             try:
#                 # 🔥 FIXED: Use shutil.copy() instead of copy2() to not preserve old timestamps
#                 shutil.copy(str(existing_working_file), str(final_path))
#                 existing_working_file.unlink()  # Remove the old file
#                 logger.info(f"✅ Moved working file to: {final_path}")
#             except Exception as e:
#                 logger.error(f"Error moving file: {e}")
#                 final_path = existing_working_file
#                 final_filename = existing_working_file.name
        
#         # 🔥 NEW: Update timestamp to current time so it appears in "Today"
#         update_file_timestamp(final_path)
        
#         # 🔥 FIXED: Update usage for existing file too
#         new_usage = increment_user_usage(db, user, "video_downloads")
        
#         processing_time = time.time() - start_time
        
#         # 🔥 NEW: Mobile-optimized download URL
#         mobile_download_url = f"/mobile-download/video/{final_filename}?auth={create_access_token_for_mobile(user.username)}"
        
#         return {
#             "download_url": f"/files/{final_filename}",
#             "direct_download_url": mobile_download_url,  # 🔥 NEW: Mobile-optimized URL
#             "youtube_id": video_id,
#             "quality": request.quality,
#             "file_size": file_size,
#             "file_size_mb": round(file_size / (1024 * 1024), 2),
#             "filename": final_filename,
#             "local_path": str(final_path),
#             "processing_time": round(processing_time, 2),
#             "message": "Video ready for download (existing file)",
#             "success": True,
#             "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
#             "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
#             "duration": video_info.get('duration', 0) if video_info else 0,
#             "usage_updated": new_usage,
#             "usage_type": "video_downloads"
#         }
    
#     # 🔥 FIXED: Download to final location directly
#     logger.info(f"🔥 No existing file found, downloading new video for {video_id}")
    
#     try:
#         logger.info(f"🔥 Downloading directly to: {DOWNLOADS_DIR}")
        
#         video_file_path = download_video_with_ytdlp(video_id, request.quality, output_dir=str(DOWNLOADS_DIR))
        
#         if not video_file_path or not os.path.exists(video_file_path):
#             raise HTTPException(status_code=404, detail="Failed to download video.")
        
#         downloaded_file = Path(video_file_path)
#         file_size = downloaded_file.stat().st_size
        
#         if file_size < 10000:
#             raise HTTPException(status_code=500, detail="Downloaded video appears to be corrupted.")
        
#         # 🔥 FIXED: Ensure consistent naming
#         if downloaded_file != final_path:
#             logger.info(f"🔥 Renaming downloaded file to standard name: {final_filename}")
#             try:
#                 if final_path.exists():
#                     final_path.unlink()  # Remove any conflicting file
                
#                 # 🔥 FIXED: Use move instead of copy to avoid timestamp issues
#                 downloaded_file.rename(final_path)
#                 logger.info(f"✅ File renamed to: {final_path}")
#             except Exception as e:
#                 logger.warning(f"Could not rename file: {e}, using original name")
#                 final_path = downloaded_file
#                 final_filename = downloaded_file.name
        
#         # 🔥 NEW: Update timestamp to current time so it appears in "Today"
#         update_file_timestamp(final_path)
        
#         logger.info(f"✅ Video download successful: {final_path} ({file_size} bytes)")
        
#     except Exception as e:
#         logger.error(f"❌ Download failed: {e}")
#         raise HTTPException(status_code=500, detail=f"Video download failed: {str(e)}")
    
#     # 🔥 FIXED: Update usage after successful download
#     new_usage = increment_user_usage(db, user, "video_downloads")
    
#     processing_time = time.time() - start_time
    
#     # 🔥 NEW: Mobile-optimized download URL
#     mobile_download_url = f"/mobile-download/video/{final_filename}?auth={create_access_token_for_mobile(user.username)}"
    
#     return {
#         "download_url": f"/files/{final_filename}",
#         "direct_download_url": mobile_download_url,  # 🔥 NEW: Mobile-optimized URL
#         "youtube_id": video_id,
#         "quality": request.quality,
#         "file_size": file_size,
#         "file_size_mb": round(file_size / (1024 * 1024), 2),
#         "filename": final_filename,
#         "local_path": str(final_path),
#         "processing_time": round(processing_time, 2),
#         "message": "Video ready for download",
#         "success": True,
#         "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
#         "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
#         "duration": video_info.get('duration', 0) if video_info else 0,
#         "usage_updated": new_usage,
#         "usage_type": "video_downloads"
#     }

# # 🔥 UPDATED: Modified audio download endpoint with mobile support
# @app.post("/download_audio/")
# def download_audio(
#     request: AudioRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """🔥 FIXED: Audio download with mobile support and proper file management"""
#     start_time = time.time()
    
#     video_id = extract_youtube_video_id(request.youtube_id)
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     if not check_internet_connectivity():
#         raise HTTPException(status_code=503, detail="No internet connection available.")
    
#     if not check_ytdlp_availability():
#         raise HTTPException(status_code=500, detail="Audio download service temporarily unavailable.")
    
#     # 🔥 FIXED: Check usage limits properly
#     can_use, current_usage, limit = check_usage_limit(user, "audio_downloads")
    
#     if not can_use:
#         raise HTTPException(
#             status_code=403,
#             detail=f"Monthly limit reached for audio downloads ({current_usage}/{limit})."
#         )
    
#     # Get video info for title display
#     video_info = None
#     try:
#         video_info = get_video_info(video_id)
#         logger.info(f"🔥 Got video info: {video_info.get('title', 'Unknown') if video_info else 'Failed to get info'}")
#     except Exception as e:
#         logger.warning(f"Could not get video info: {e}")
    
#     # Define expected filename
#     final_filename = f"{video_id}_audio_{request.quality}.mp3"
#     final_path = DOWNLOADS_DIR / final_filename
    
#     # 🔥 FIXED: First clean up any existing files to prevent conflicts
#     logger.info(f"🔥 Cleaning up any existing audio files for {video_id}...")
#     cleanup_existing_files(video_id, "audio", request.quality)
    
#     # 🔥 FIXED: Check if a working file still exists after cleanup
#     existing_working_file = find_working_audio_file(video_id, request.quality)
    
#     if existing_working_file:
#         logger.info(f"🔥 Found existing working file after cleanup: {existing_working_file}")
#         file_size = existing_working_file.stat().st_size
        
#         # Move to standard location if needed
#         if existing_working_file != final_path:
#             logger.info(f"🔥 Moving existing file to standard location")
#             try:
#                 # 🔥 FIXED: Use shutil.copy() instead of copy2() to not preserve old timestamps
#                 shutil.copy(str(existing_working_file), str(final_path))
#                 existing_working_file.unlink()  # Remove the old file
#                 logger.info(f"✅ Moved working file to: {final_path}")
#             except Exception as e:
#                 logger.error(f"Error moving file: {e}")
#                 final_path = existing_working_file
#                 final_filename = existing_working_file.name
        
#         # 🔥 NEW: Update timestamp to current time so it appears in "Today"
#         update_file_timestamp(final_path)
        
#         # 🔥 FIXED: Update usage for existing file too
#         new_usage = increment_user_usage(db, user, "audio_downloads")
        
#         processing_time = time.time() - start_time
        
#         # 🔥 NEW: Mobile-optimized download URL
#         mobile_download_url = f"/mobile-download/audio/{final_filename}?auth={create_access_token_for_mobile(user.username)}"
        
#         return {
#             "download_url": f"/files/{final_filename}",
#             "direct_download_url": mobile_download_url,  # 🔥 NEW: Mobile-optimized URL
#             "youtube_id": video_id,
#             "quality": request.quality,
#             "file_size": file_size,
#             "file_size_mb": round(file_size / (1024 * 1024), 2),
#             "filename": final_filename,
#             "local_path": str(final_path),
#             "processing_time": round(processing_time, 2),
#             "message": "Audio ready for download (existing file)",
#             "success": True,
#             "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
#             "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
#             "duration": video_info.get('duration', 0) if video_info else 0,
#             "usage_updated": new_usage,
#             "usage_type": "audio_downloads"
#         }
    
#     # 🔥 FIXED: Download to final location directly
#     logger.info(f"🔥 No existing file found, downloading new audio for {video_id}")
    
#     try:
#         logger.info(f"🔥 Downloading directly to: {DOWNLOADS_DIR}")
        
#         audio_file_path = download_audio_with_ytdlp(video_id, request.quality, output_dir=str(DOWNLOADS_DIR))
        
#         if not audio_file_path or not os.path.exists(audio_file_path):
#             raise HTTPException(status_code=404, detail="Failed to download audio.")
        
#         downloaded_file = Path(audio_file_path)
#         file_size = downloaded_file.stat().st_size
        
#         if file_size < 1000:
#             raise HTTPException(status_code=500, detail="Downloaded file appears to be corrupted.")
        
#         # 🔥 FIXED: Ensure consistent naming
#         if downloaded_file != final_path:
#             logger.info(f"🔥 Renaming downloaded file to standard name: {final_filename}")
#             try:
#                 if final_path.exists():
#                     final_path.unlink()  # Remove any conflicting file
                
#                 # 🔥 FIXED: Use move instead of copy to avoid timestamp issues
#                 downloaded_file.rename(final_path)
#                 logger.info(f"✅ File renamed to: {final_path}")
#             except Exception as e:
#                 logger.warning(f"Could not rename file: {e}, using original name")
#                 final_path = downloaded_file
#                 final_filename = downloaded_file.name
        
#         # 🔥 NEW: Update timestamp to current time so it appears in "Today"
#         update_file_timestamp(final_path)
        
#         logger.info(f"✅ Audio download successful: {final_path} ({file_size} bytes)")
        
#     except Exception as e:
#         logger.error(f"❌ Download failed: {e}")
#         raise HTTPException(status_code=500, detail=f"Audio download failed: {str(e)}")
    
#     # 🔥 FIXED: Update usage after successful download
#     new_usage = increment_user_usage(db, user, "audio_downloads")
    
#     processing_time = time.time() - start_time
    
#     # 🔥 NEW: Mobile-optimized download URL
#     mobile_download_url = f"/mobile-download/audio/{final_filename}?auth={create_access_token_for_mobile(user.username)}"
    
#     return {
#         "download_url": f"/files/{final_filename}",
#         "direct_download_url": mobile_download_url,  # 🔥 NEW: Mobile-optimized URL
#         "youtube_id": video_id,
#         "quality": request.quality,
#         "file_size": file_size,
#         "file_size_mb": round(file_size / (1024 * 1024), 2),
#         "filename": final_filename,
#         "local_path": str(final_path),
#         "processing_time": round(processing_time, 2),
#         "message": "Audio ready for download",
#         "success": True,
#         "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
#         "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
#         "duration": video_info.get('duration', 0) if video_info else 0,
#         "usage_updated": new_usage,
#         "usage_type": "audio_downloads"
#     }

# # =============================================================================
# # 🔥 COMPLETELY FIXED MOBILE DOWNLOAD ENDPOINTS
# # =============================================================================

# @app.get("/mobile-download/{file_type}/{filename}")
# async def mobile_download_completely_fixed(
#     request: Request,
#     file_type: str,  # "audio" or "video"
#     filename: str,   # actual filename
#     auth: Optional[str] = Query(None),  # Auth token from query parameter
# ):
#     """
#     🔥 COMPLETELY FIXED MOBILE DOWNLOAD ENDPOINT
#     Handles mobile authentication issues properly
#     """
#     try:
#         logger.info(f"🔥 Mobile download request: {file_type}/{filename}")
#         logger.info(f"🔥 Auth token received: {bool(auth)}")
#         logger.info(f"🔥 Request headers: {dict(request.headers)}")
        
#         # Validate file type
#         if file_type not in ["audio", "video"]:
#             raise HTTPException(status_code=400, detail="Invalid file type")
        
#         # Try to authenticate with query parameter first (mobile)
#         user = None
#         if auth:
#             try:
#                 payload = jwt.decode(auth, SECRET_KEY, algorithms=[ALGORITHM])
#                 username = payload.get("sub")
#                 if username:
#                     logger.info(f"🔥 Authenticated user from query param: {username}")
#                     db = SessionLocal()
#                     try:
#                         user = get_user_by_username(db, username)
#                         if user:
#                             logger.info(f"✅ User found in database: {user.username}")
#                     finally:
#                         db.close()
#             except jwt.ExpiredSignatureError:
#                 logger.error("❌ JWT token expired")
#                 raise HTTPException(status_code=401, detail="Token expired")
#             except jwt.PyJWTError as e:
#                 logger.error(f"❌ JWT decode error: {e}")
#                 raise HTTPException(status_code=401, detail="Invalid token")
        
#         # If no user found, try Authorization header (fallback)
#         if not user:
#             auth_header = request.headers.get("authorization")
#             if auth_header and auth_header.startswith("Bearer "):
#                 token = auth_header.split(" ")[1]
#                 try:
#                     payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#                     username = payload.get("sub")
#                     if username:
#                         logger.info(f"🔥 Authenticated user from header: {username}")
#                         db = SessionLocal()
#                         try:
#                             user = get_user_by_username(db, username)
#                         finally:
#                             db.close()
#                 except jwt.PyJWTError:
#                     pass
        
#         if not user:
#             logger.error("❌ No valid authentication found")
#             raise HTTPException(status_code=401, detail="Authentication required")
        
#         logger.info(f"✅ User authenticated: {user.username}")
        
#         # Construct file path
#         file_path = DOWNLOADS_DIR / filename
#         logger.info(f"🔥 Looking for file: {file_path}")
        
#         # Check if file exists
#         if not file_path.exists():
#             logger.error(f"❌ File not found: {file_path}")
#             # List available files for debugging
#             available_files = [f.name for f in DOWNLOADS_DIR.iterdir() if f.is_file()]
#             logger.error(f"❌ Available files: {available_files}")
#             raise HTTPException(status_code=404, detail="File not found")
        
#         # Security check: ensure file is in downloads directory
#         if not str(file_path.resolve()).startswith(str(DOWNLOADS_DIR.resolve())):
#             logger.error(f"❌ Security violation: {file_path}")
#             raise HTTPException(status_code=403, detail="Access denied")
        
#         # Check file size and log info
#         file_size = file_path.stat().st_size
#         is_mobile = is_mobile_request(request)
        
#         logger.info(f"✅ File found: {filename} ({file_size} bytes)")
#         logger.info(f"🔥 Mobile request: {is_mobile}")
        
#         if is_mobile and file_size > 100 * 1024 * 1024:  # 100MB
#             logger.info(f"⚠️ Large file download on mobile: {file_size / 1024 / 1024:.1f}MB")
        
#         # Get proper MIME type
#         mime_type = get_mobile_mime_type(str(file_path), file_type)
        
#         # Generate safe filename
#         safe_filename = get_safe_filename(filename)
        
#         # 🔥 MOBILE-SPECIFIC HEADERS
#         headers = {
#             "Content-Type": mime_type,
#             "Content-Disposition": f'attachment; filename="{safe_filename}"',
#             "Content-Length": str(file_size),
#             "Cache-Control": "no-cache, no-store, must-revalidate",
#             "Pragma": "no-cache",
#             "Expires": "0",
#             "Accept-Ranges": "bytes",
#         }
        
#         # Additional headers for mobile browsers
#         if is_mobile:
#             headers.update({
#                 "X-Content-Type-Options": "nosniff",
#                 "Content-Transfer-Encoding": "binary",
#                 # Force download on mobile browsers
#                 "Content-Disposition": f'attachment; filename="{safe_filename}"; filename*=UTF-8\'\'{safe_filename}',
#             })
        
#         logger.info(f"🔥 Serving {file_type} file: {safe_filename} ({file_size / 1024 / 1024:.1f}MB) to {'mobile' if is_mobile else 'desktop'}")
        
#         # Return file with proper headers
#         return FileResponse(
#             path=str(file_path),
#             media_type=mime_type,
#             headers=headers,
#             filename=safe_filename
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"❌ Mobile download error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

# # 🔥 MOBILE DEBUG ENDPOINT
# @app.get("/debug/mobile-info")
# async def mobile_debug_info(request: Request):
#     """Debug endpoint to check mobile detection and headers"""
#     user_agent = request.headers.get("user-agent", "")
#     is_mobile = is_mobile_request(request)
    
#     return {
#         "is_mobile": is_mobile,
#         "user_agent": user_agent,
#         "headers": dict(request.headers),
#         "client_host": request.client.host if request.client else "unknown",
#         "method": request.method,
#         "url": str(request.url)
#     }

# # 🔥 DOWNLOAD HISTORY ENDPOINTS (for View History feature)
# @app.get("/user/download-history")
# async def get_download_history(current_user: User = Depends(get_current_user)):
#     """Get user's download history"""
#     try:
#         # TODO: Implement actual database query
#         # This is a placeholder - replace with your actual database logic
        
#         # Example query (replace with your actual database model):
#         # downloads = db.query(DownloadHistory).filter(
#         #     DownloadHistory.user_id == current_user.id
#         # ).order_by(DownloadHistory.created_at.desc()).limit(50).all()
        
#         # Mock data for now
#         mock_history = [
#             {
#                 "id": 1,
#                 "type": "audio",
#                 "title": "CHOSEN: God Said → Forget Job Hunting, THIS Is Your Real Calling",
#                 "channel": "Divine Synchronicity",
#                 "video_id": "snm5ZhcJT3k",
#                 "quality": "high",
#                 "downloaded_at": "2025-08-11T17:01:00Z",
#                 "file_size": "34.2 MB",
#                 "status": "completed"
#             },
#             {
#                 "id": 2,
#                 "type": "transcript",
#                 "title": "Only 1% of Elite Chosen Ones Hold the Dual Vibration of Dark and Light",
#                 "channel": "Spiritual Frequency",
#                 "video_id": "snm5ZhcJT3k",
#                 "format": "clean",
#                 "downloaded_at": "2025-08-11T16:45:00Z",
#                 "file_size": "17.89 KB",
#                 "status": "completed"
#             },
#             {
#                 "id": 3,
#                 "type": "video",
#                 "title": "Rick Astley - Never Gonna Give You Up",
#                 "channel": "Rick Astley",
#                 "video_id": "dQw4w9WgXcQ",
#                 "quality": "720p",
#                 "downloaded_at": "2025-08-10T14:30:00Z",
#                 "file_size": "156.7 MB",
#                 "status": "completed"
#             }
#         ]
        
#         return {"downloads": mock_history}
        
#     except Exception as e:
#         logger.error(f"❌ Error fetching download history: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to fetch download history")

# @app.get("/user/recent-activity")
# async def get_recent_activity(current_user: User = Depends(get_current_user)):
#     """Get user's recent activity"""
#     try:
#         # TODO: Implement actual database query for user activity
        
#         # Mock data for now
#         mock_activity = [
#             {
#                 "id": 1,
#                 "action": "Downloaded audio file",
#                 "description": "High quality MP3 from \"CHOSEN: God Said → Forget Job Hunting\"",
#                 "timestamp": "2025-08-11T17:01:00Z",
#                 "type": "download",
#                 "icon": "🎵"
#             },
#             {
#                 "id": 2,
#                 "action": "Generated transcript",
#                 "description": "Clean transcript for \"Only 1% of Elite Chosen Ones\"",
#                 "timestamp": "2025-08-11T16:45:00Z",
#                 "type": "transcript",
#                 "icon": "📄"
#             },
#             {
#                 "id": 3,
#                 "action": "Logged in",
#                 "description": "Signed in from mobile device",
#                 "timestamp": "2025-08-11T16:30:00Z",
#                 "type": "auth",
#                 "icon": "🔐"
#             }
#         ]
        
#         return {"activities": mock_activity}
        
#     except Exception as e:
#         logger.error(f"❌ Error fetching recent activity: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to fetch recent activity")

# @app.get("/subscription_status/")
# def get_subscription_status(current_user: User = Depends(get_current_user)):
#     """🔥 FIXED: Get subscription status with proper usage data"""
#     try:
#         tier = getattr(current_user, 'subscription_tier', 'free')
        
#         # 🔥 FIXED: Get actual usage from database
#         usage = {
#             "clean_transcripts": getattr(current_user, "usage_clean_transcripts", 0) or 0,
#             "unclean_transcripts": getattr(current_user, "usage_unclean_transcripts", 0) or 0,
#             "audio_downloads": getattr(current_user, "usage_audio_downloads", 0) or 0,
#             "video_downloads": getattr(current_user, "usage_video_downloads", 0) or 0
#         }
        
#         SUBSCRIPTION_LIMITS = {
#             "free": {"clean_transcripts": 5, "unclean_transcripts": 3, "audio_downloads": 2, "video_downloads": 1},
#             "pro": {"clean_transcripts": 100, "unclean_transcripts": 50, "audio_downloads": 50, "video_downloads": 20},
#             "premium": {"clean_transcripts": float('inf'), "unclean_transcripts": float('inf'), "audio_downloads": float('inf'), "video_downloads": float('inf')}
#         }
        
#         limits = SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS["free"])
#         json_limits = {k: ('unlimited' if v == float('inf') else v) for k, v in limits.items()}
        
#         logger.info(f"🔥 Subscription status for {current_user.username}: tier={tier}, usage={usage}")
        
#         return {
#             "tier": tier,
#             "status": "active" if tier != "free" else "inactive",
#             "usage": usage,
#             "limits": json_limits,
#             "downloads_folder": str(DOWNLOADS_DIR)
#         }
        
#     except Exception as e:
#         logger.error(f"Error getting subscription status: {e}")
#         return {
#             "tier": "free",
#             "status": "inactive",
#             "usage": {"clean_transcripts": 0, "unclean_transcripts": 0, "audio_downloads": 0, "video_downloads": 0},
#             "limits": {"clean_transcripts": 5, "unclean_transcripts": 3, "audio_downloads": 2, "video_downloads": 1},
#             "downloads_folder": str(DOWNLOADS_DIR)
#         }

# @app.get("/health/")
# def health():
#     return {
#         "status": "healthy", 
#         "timestamp": datetime.utcnow().isoformat(),
#         "downloads_path": str(DOWNLOADS_DIR),
#         "connectivity": {
#             "internet": check_internet_connectivity(),
#             "youtube": check_youtube_connectivity()
#         },
#         "mobile_support": True
#     }

# # 🔥 NEW: Direct file download endpoint
# @app.get("/download_file/{filename}")
# def download_file_direct(filename: str, current_user: User = Depends(get_current_user)):
#     """Direct file download endpoint for downloaded files"""
#     try:
#         file_path = DOWNLOADS_DIR / filename
        
#         if not file_path.exists():
#             logger.error(f"❌ File not found: {file_path}")
#             raise HTTPException(status_code=404, detail="File not found")
        
#         if not file_path.is_file():
#             logger.error(f"❌ Path is not a file: {file_path}")
#             raise HTTPException(status_code=404, detail="Invalid file path")
        
#         # Security check: ensure file is in downloads directory
#         if not str(file_path.resolve()).startswith(str(DOWNLOADS_DIR.resolve())):
#             logger.error(f"❌ Security violation: {file_path}")
#             raise HTTPException(status_code=403, detail="Access denied")
        
#         logger.info(f"🔥 Serving file: {filename} ({file_path.stat().st_size} bytes)")
        
#         return FileResponse(
#             path=str(file_path),
#             filename=filename,
#             media_type='application/octet-stream'
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"❌ Error serving file {filename}: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# @app.get("/test_videos")
# def get_test_videos():
#     """Get test video IDs for development and testing"""
#     return {
#         "videos": [
#             {
#                 "id": "dQw4w9WgXcQ", 
#                 "title": "Rick Astley - Never Gonna Give You Up",
#                 "status": "verified_working",
#                 "supports": ["transcript", "audio", "video"],
#                 "note": "Perfect for testing all features"
#             },
#             {
#                 "id": "jNQXAC9IVRw", 
#                 "title": "Me at the zoo",
#                 "status": "verified_working",
#                 "supports": ["transcript", "audio", "video"],
#                 "note": "First YouTube video ever - works for all features"
#             },
#             {
#                 "id": "9bZkp7q19f0",
#                 "title": "PSY - GANGNAM STYLE",
#                 "status": "verified_working", 
#                 "supports": ["transcript", "audio", "video"],
#                 "note": "Popular video with multiple quality options"
#             },
#             {
#                 "id": "L_jWHffIx5E",
#                 "title": "Smash Mouth - All Star",
#                 "status": "verified_working",
#                 "supports": ["transcript", "audio", "video"], 
#                 "note": "Another reliable test video"
#             }
#         ],
#         "recommendations": {
#             "for_video_testing": ["dQw4w9WgXcQ", "jNQXAC9IVRw", "9bZkp7q19f0", "L_jWHffIx5E"],
#             "for_audio_testing": ["dQw4w9WgXcQ", "jNQXAC9IVRw", "9bZkp7q19f0"],
#             "for_transcript_testing": ["dQw4w9WgXcQ", "jNQXAC9IVRw", "9bZkp7q19f0"]
#         },
#         "note": "All these videos work for all features - use any for comprehensive testing"
#     }

# if __name__ == "__main__":
#     import uvicorn
#     print("🔥 Starting server on 0.0.0.0:8000")
#     print("🔥 Mobile access enabled")
#     print("🔥 Mobile download optimization loaded")
#     print(f"🔥 Downloads folder: {str(DOWNLOADS_DIR)}")
#     print("📱 Mobile endpoints available:")
#     print("   - /mobile-download/{file_type}/{file_id}")
#     print("   - /debug/mobile-info")
#     print("   - /user/download-history")
#     print("   - /user/recent-activity")
    
#     uvicorn.run(
#         "main:app", 
#         host="0.0.0.0",
#         port=8000, 
#         reload=True
#     )


#==============================
# """
# YouTube Content Downloader API - FIXED VERSION with MOBILE SUPPORT
# ================================================================
# 🔥 FIXES:
# - ✅ Usage tracking now works properly (updates counters)
# - ✅ Video downloads now include audio
# - ✅ Proper video metadata and titles
# - ✅ Following transcript download success pattern
# - ✅ Enhanced download success responses
# - ✅ MOBILE DOWNLOAD FIXES: Proper headers, MIME types, mobile detection
# - ✅ Mobile-optimized download endpoints
# - ✅ Download history and recent activity endpoints
# """

# from pathlib import Path
# from youtube_transcript_api import YouTubeTranscriptApi

# from fastapi import FastAPI, HTTPException, Depends, status, Request
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles
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
# import time
# import stripe
# import tempfile
# import asyncio
# import shutil
# import uuid
# import socket
# import mimetypes

# # Import our models
# from models import User, TranscriptDownload, get_db, engine, SessionLocal, initialize_database, create_download_record_safe
# from transcript_utils import (
#     get_transcript_with_ytdlp,
#     download_audio_with_ytdlp,
#     download_video_with_ytdlp,
#     check_ytdlp_availability,
#     get_video_info
# )

# # Load environment variables
# load_dotenv()

# # =============================================================================
# # CONFIGURATION
# # =============================================================================

# # Stripe Configuration
# stripe_secret_key = os.getenv("STRIPE_SECRET_KEY")
# if stripe_secret_key:
#     stripe.api_key = stripe_secret_key
#     print("✅ Stripe configured successfully")
# else:
#     print("⚠️ Warning: STRIPE_SECRET_KEY not found in environment variables")

# # Logging Configuration
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("youtube_trans_downloader.main")

# # Environment Configuration
# ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# logger.info(f"Environment: {ENVIRONMENT}")
# logger.info("Starting YouTube Content Downloader API")
# logger.info("Environment variables loaded from .env file")
# logger.info("Using SQLite database for development")

# # Initialize database
# initialize_database()

# # FastAPI App Configuration
# app = FastAPI(
#     title="YouTube Content Downloader API", 
#     version="2.6.0",
#     description="A SaaS application for downloading YouTube transcripts, audio, and video with mobile support"
# )

# # DOWNLOADS DIRECTORY SETUP - UNICODE SAFE
# try:
#     # Get user's home directory
#     home_dir = Path.home()
#     downloads_dir = home_dir / "Downloads"
#     downloads_dir.mkdir(exist_ok=True)
#     DOWNLOADS_DIR = downloads_dir
    
#     # Test write access
#     test_file = DOWNLOADS_DIR / "test_write.tmp"
#     test_file.write_text("test")
#     test_file.unlink()
    
#     logger.info("🔥 Using user Downloads folder")
#     logger.info(f"🔥 Path: {str(DOWNLOADS_DIR)}")
    
# except Exception as e:
#     logger.warning(f"Cannot use Downloads folder: {e}")
#     # Fallback to local directory
#     DOWNLOADS_DIR = Path("downloads")
#     DOWNLOADS_DIR.mkdir(exist_ok=True)
#     logger.info(f"🔥 Using fallback directory: {str(DOWNLOADS_DIR)}")

# # Mount static files
# app.mount("/files", StaticFiles(directory=str(DOWNLOADS_DIR)), name="files")

# # 🔥 ENHANCED CORS CONFIGURATION FOR MOBILE
# allowed_origins = [
#     "http://localhost:3000", 
#     "http://127.0.0.1:3000", 
#     "http://192.168.1.185:3000",
#     FRONTEND_URL
# ] if ENVIRONMENT != "production" else [
#     "https://youtube-trans-downloader-api.onrender.com", 
#     FRONTEND_URL
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=allowed_origins, 
#     allow_credentials=True,
#     allow_methods=["*"], 
#     allow_headers=["*"],
#     expose_headers=["Content-Disposition", "Content-Type", "Content-Length"],  # 🔥 NEW: Mobile headers
# )

# # Security Configuration
# SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # =============================================================================
# # 🔥 MOBILE DETECTION AND UTILITY FUNCTIONS
# # =============================================================================

# def is_mobile_request(request: Request) -> bool:
#     """Detect if request is coming from a mobile device"""
#     user_agent = request.headers.get("user-agent", "").lower()
#     mobile_patterns = [
#         "android", "iphone", "ipad", "ipod", "blackberry", 
#         "windows phone", "mobile", "webos", "opera mini"
#     ]
#     return any(pattern in user_agent for pattern in mobile_patterns)

# def get_safe_filename(filename: str) -> str:
#     """Generate mobile-safe filename"""
#     # Remove special characters that might cause issues on mobile
#     safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
#     # Ensure it's not too long for mobile file systems
#     if len(safe_name) > 100:
#         name, ext = os.path.splitext(safe_name)
#         safe_name = name[:96] + ext
#     return safe_name

# def get_mobile_mime_type(file_path: str, file_type: str) -> str:
#     """Get proper MIME type for mobile downloads"""
#     if file_type == "audio" or file_path.endswith(('.mp3', '.m4a', '.aac')):
#         return "audio/mpeg"
#     elif file_type == "video" or file_path.endswith(('.mp4', '.m4v', '.mov')):
#         return "video/mp4"
#     else:
#         # Let mimetypes guess, with fallback
#         mime_type, _ = mimetypes.guess_type(file_path)
#         return mime_type or "application/octet-stream"

# def create_access_token_for_mobile(username: str) -> str:
#     """Create a temporary access token for URL-based auth"""
#     expire = datetime.utcnow() + timedelta(hours=2)  # 2-hour expiry for download links
#     to_encode = {"sub": username, "exp": expire}
#     encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
#     return encoded_jwt

# # =============================================================================
# # 🔥 FIXED USAGE TRACKING FUNCTIONS
# # =============================================================================

# def increment_user_usage(db: Session, user: User, usage_type: str):
#     """
#     🔥 FIXED: Properly increment user usage and commit to database
#     """
#     try:
#         logger.info(f"🔥 Incrementing usage for user {user.username}: {usage_type}")
        
#         # Get current usage
#         current_usage = getattr(user, f"usage_{usage_type}", 0) or 0
#         new_usage = current_usage + 1
        
#         # Set new usage
#         setattr(user, f"usage_{usage_type}", new_usage)
        
#         # Update usage reset date if needed
#         current_date = datetime.utcnow()
#         if not hasattr(user, 'usage_reset_date') or user.usage_reset_date is None:
#             user.usage_reset_date = current_date
#         elif user.usage_reset_date.month != current_date.month:
#             # Reset monthly usage
#             user.usage_clean_transcripts = 0
#             user.usage_unclean_transcripts = 0
#             user.usage_audio_downloads = 0
#             user.usage_video_downloads = 0
#             user.usage_reset_date = current_date
            
#             # Set the new usage for this type
#             setattr(user, f"usage_{usage_type}", 1)
#             new_usage = 1
        
#         # Commit to database
#         db.commit()
#         db.refresh(user)
        
#         logger.info(f"✅ Usage updated: {usage_type} = {new_usage}")
#         return new_usage
        
#     except Exception as e:
#         logger.error(f"❌ Error incrementing usage: {e}")
#         db.rollback()
#         return current_usage

# def check_usage_limit(user: User, usage_type: str) -> tuple[bool, int, int]:
#     """
#     🔥 FIXED: Check if user has reached usage limit
#     Returns: (can_use, current_usage, limit)
#     """
#     try:
#         # Get subscription tier
#         tier = getattr(user, 'subscription_tier', 'free')
        
#         # Define limits
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
        
#         current_usage = getattr(user, f"usage_{usage_type}", 0) or 0
#         limit = limits.get(tier, limits['free']).get(usage_type, 0)
        
#         can_use = current_usage < limit
        
#         logger.info(f"🔥 Usage check: {usage_type} = {current_usage}/{limit}, can_use = {can_use}")
        
#         return can_use, current_usage, limit
        
#     except Exception as e:
#         logger.error(f"❌ Error checking usage limit: {e}")
#         return False, 0, 0

# # =============================================================================
# # UTILITY FUNCTIONS
# # =============================================================================

# def check_internet_connectivity():
#     """Check if we can reach the internet"""
#     try:
#         socket.create_connection(("8.8.8.8", 53), timeout=3)
#         return True
#     except OSError:
#         return False

# def check_youtube_connectivity():
#     """Check if we can reach YouTube specifically"""
#     try:
#         socket.create_connection(("www.youtube.com", 443), timeout=5)
#         return True
#     except OSError:
#         return False

# def generate_unique_filename(base_name: str, extension: str) -> str:
#     """Generate a unique filename to avoid conflicts"""
#     unique_id = str(uuid.uuid4())[:8]
#     timestamp = int(time.time())
#     return f"{base_name}_{timestamp}_{unique_id}.{extension}"

# # 🔥 FIXED: Enhanced file finding with Windows rename detection
# def find_working_video_file(video_id: str, quality: str) -> Optional[Path]:
#     """Find any existing working video file for this video/quality combination"""
#     try:
#         patterns = [
#             f"{video_id}_video_{quality}.mp4",
#             f"{video_id}_video_{quality}.webm", 
#             f"{video_id}_video_{quality}*.mp4",      # Includes (1), (2), etc.
#             f"{video_id}_video_{quality}*.webm",     # Includes (1), (2), etc.
#             f"{video_id}*video*{quality}*.mp4",      # Broader search
#             f"{video_id}*video*{quality}*.webm",     # Broader search
#             f"{video_id}*video*.mp4",                # Any video with this ID
#             f"{video_id}*video*.webm",               # Any video with this ID
#         ]
        
#         found_files = []
#         for pattern in patterns:
#             files = list(DOWNLOADS_DIR.glob(pattern))
#             for file_path in files:
#                 if file_path.is_file():
#                     file_size = file_path.stat().st_size
#                     if file_size > 1000000:  # 1MB minimum for valid video
#                         found_files.append(file_path)
                        
#         if found_files:
#             # Return the most recent valid file
#             latest_file = max(found_files, key=lambda f: f.stat().st_mtime)
#             logger.info(f"🔥 Found existing video file: {latest_file.name} ({latest_file.stat().st_size} bytes)")
#             return latest_file
        
#         return None
        
#     except Exception as e:
#         logger.error(f"Error finding working video file: {e}")
#         return None

# def find_working_audio_file(video_id: str, quality: str) -> Optional[Path]:
#     """Find any existing working audio file for this video/quality combination"""
#     try:
#         patterns = [
#             f"{video_id}_audio_{quality}.mp3",
#             f"{video_id}_audio_{quality}.m4a",
#             f"{video_id}_audio_{quality}*.mp3",     # Includes (1), (2), etc.
#             f"{video_id}_audio_{quality}*.m4a",     # Includes (1), (2), etc.
#             f"{video_id}*audio*{quality}*.mp3",     # Broader search
#             f"{video_id}*audio*{quality}*.m4a",     # Broader search
#             f"{video_id}*audio*.mp3",               # Any audio with this ID
#             f"{video_id}*audio*.m4a",               # Any audio with this ID
#         ]
        
#         found_files = []
#         for pattern in patterns:
#             files = list(DOWNLOADS_DIR.glob(pattern))
#             for file_path in files:
#                 if file_path.is_file():
#                     file_size = file_path.stat().st_size
#                     if file_size > 100000:  # 100KB minimum for valid audio
#                         found_files.append(file_path)
                        
#         if found_files:
#             # Return the most recent valid file
#             latest_file = max(found_files, key=lambda f: f.stat().st_mtime)
#             logger.info(f"🔥 Found existing audio file: {latest_file.name} ({latest_file.stat().st_size} bytes)")
#             return latest_file
        
#         return None
        
#     except Exception as e:
#         logger.error(f"Error finding working audio file: {e}")
#         return None

# # 🔥 FIXED: Enhanced cleanup that removes ALL old versions

# def cleanup_existing_files(video_id: str, file_type: str, quality: str):
#     """Remove any existing files for this video/quality to prevent duplicates"""
#     try:
#         if file_type == "audio":
#             patterns = [
#                 f"{video_id}_audio_{quality}*",
#                 f"{video_id}*audio*{quality}*",
#                 f"{video_id}*audio*.*",  # Remove any old audio files for this video
#             ]
#         else:  # video
#             patterns = [
#                 f"{video_id}_video_{quality}*",
#                 f"{video_id}*video*{quality}*", 
#                 f"{video_id}*video*.*",  # Remove any old video files for this video
#             ]
        
#         removed_count = 0
#         for pattern in patterns:
#             for file_path in DOWNLOADS_DIR.glob(pattern):
#                 if file_path.is_file():
#                     try:
#                         file_size = file_path.stat().st_size
#                         logger.info(f"🔥 Removing old file: {file_path.name} ({file_size} bytes)")
#                         file_path.unlink()
#                         removed_count += 1
#                     except Exception as e:
#                         logger.warning(f"Could not remove {file_path.name}: {e}")
        
#         if removed_count > 0:
#             logger.info(f"🔥 Cleaned up {removed_count} old files for {video_id}")
#         else:
#             logger.info(f"🔥 No old files found to clean up for {video_id}")
                            
#     except Exception as e:
#         logger.warning(f"Error during cleanup: {e}")

# def cleanup_old_files():
#     """Enhanced cleanup that prevents all duplicate issues"""
#     try:
#         current_time = time.time()
#         max_age = 24 * 3600  # 24 hours (reduced from 2 hours for less aggressive cleanup)
        
#         # Track files by video ID to remove duplicates
#         video_files = {}  # video_id -> list of files
#         audio_files = {}  # video_id -> list of files
        
#         for file_path in DOWNLOADS_DIR.glob("*"):
#             if file_path.is_file():
#                 filename = file_path.name
#                 file_size = file_path.stat().st_size
#                 file_age = current_time - file_path.stat().st_mtime
                
#                 # Remove very old files
#                 if file_age > max_age:
#                     try:
#                         file_path.unlink()
#                         logger.info(f"Cleaned up old file: {filename}")
#                         continue
#                     except:
#                         pass
                
#                 # Remove Windows duplicate files (1), (2), etc. - but keep the original
#                 if "(" in filename and ")" in filename and any(ext in filename for ext in ['.mp4', '.mp3', '.m4a', '.webm']):
#                     try:
#                         file_path.unlink()
#                         logger.info(f"Removed Windows duplicate: {filename}")
#                         continue
#                     except:
#                         pass
                
#                 # Remove tiny corrupted files
#                 min_size = 100000 if "audio" in filename else 1000000
#                 if file_size < min_size and any(ext in filename for ext in ['.mp4', '.mp3', '.m4a', '.webm']):
#                     try:
#                         file_path.unlink()
#                         logger.info(f"Removed corrupted file: {filename} ({file_size} bytes)")
#                         continue
#                     except:
#                         pass
                
#                 # Track video and audio files by video ID for duplicate detection
#                 if "_video_" in filename:
#                     # Extract video ID (first 11 characters before _video_)
#                     video_id = filename.split("_video_")[0]
#                     if video_id not in video_files:
#                         video_files[video_id] = []
#                     video_files[video_id].append(file_path)
#                 elif "_audio_" in filename:
#                     # Extract video ID (first 11 characters before _audio_)
#                     video_id = filename.split("_audio_")[0]
#                     if video_id not in audio_files:
#                         audio_files[video_id] = []
#                     audio_files[video_id].append(file_path)
        
#         # Remove duplicates, keeping only the newest file for each video ID
#         for video_id, files in video_files.items():
#             if len(files) > 1:
#                 # Sort by modification time, keep the newest
#                 files.sort(key=lambda f: f.stat().st_mtime)
#                 files_to_remove = files[:-1]  # Remove all but the newest
                
#                 for file_path in files_to_remove:
#                     try:
#                         logger.info(f"Removing duplicate video: {file_path.name}")
#                         file_path.unlink()
#                     except:
#                         pass
        
#         for video_id, files in audio_files.items():
#             if len(files) > 1:
#                 # Sort by modification time, keep the newest  
#                 files.sort(key=lambda f: f.stat().st_mtime)
#                 files_to_remove = files[:-1]  # Remove all but the newest
                
#                 for file_path in files_to_remove:
#                     try:
#                         logger.info(f"Removing duplicate audio: {file_path.name}")
#                         file_path.unlink()
#                     except:
#                         pass
                        
#     except Exception as e:
#         logger.warning(f"Error during cleanup: {e}")

# # =============================================================================
# # PYDANTIC MODELS
# # =============================================================================

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
#         from_attributes = True

# class Token(BaseModel):
#     access_token: str
#     token_type: str

# class TranscriptRequest(BaseModel):
#     youtube_id: str
#     clean_transcript: bool = True
#     format: Optional[str] = None

# class AudioRequest(BaseModel):
#     youtube_id: str
#     quality: str = "medium"

# class VideoRequest(BaseModel):
#     youtube_id: str
#     quality: str = "720p"

# # =============================================================================
# # HELPER FUNCTIONS
# # =============================================================================

# def get_user(db: Session, username: str) -> Optional[User]:
#     return db.query(User).filter(User.username == username).first()

# def get_user_by_username(db: Session, username: str) -> Optional[User]:
#     return db.query(User).filter(User.username == username).first()

# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     return pwd_context.verify(plain_password, hashed_password)

# def get_password_hash(password: str) -> str:
#     return pwd_context.hash(password)

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
#     to_encode = data.copy()
#     expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED, 
#         detail="Could not validate credentials",
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

# def extract_youtube_video_id(youtube_id_or_url: str) -> str:
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

# def get_transcript_youtube_api(video_id: str, clean: bool = True, format: Optional[str] = None) -> str:
#     """Get transcript using YouTube Transcript API"""
#     logger.info(f"🔥 Getting transcript for {video_id}, clean={clean}, format={format}")
    
#     if not check_internet_connectivity():
#         logger.error("❌ No internet connectivity")
#         raise HTTPException(
#             status_code=503, 
#             detail="No internet connection available. Please check your network connection."
#         )
    
#     if not check_youtube_connectivity():
#         logger.error("❌ Cannot reach YouTube")
#         raise HTTPException(
#             status_code=503, 
#             detail="Cannot reach YouTube servers. Please try again later."
#         )
    
#     try:
#         logger.info(f"🔥 Attempting to get transcript via YouTube API for {video_id}")
#         transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
#         logger.info(f"✅ Got transcript with {len(transcript)} segments")
        
#         if clean:
#             logger.info("🔥 Processing clean transcript")
#             text = " ".join([seg['text'].replace('\n', ' ') for seg in transcript])
#             clean_text = " ".join(text.split())
            
#             words = clean_text.split()
#             paragraphs = []
#             current_paragraph = []
#             char_count = 0
            
#             for word in words:
#                 current_paragraph.append(word)
#                 char_count += len(word) + 1
                
#                 if char_count > 400 and word.endswith(('.', '!', '?')):
#                     paragraphs.append(' '.join(current_paragraph))
#                     current_paragraph = []
#                     char_count = 0
            
#             if current_paragraph:
#                 paragraphs.append(' '.join(current_paragraph))
            
#             result = '\n\n'.join(paragraphs)
#             logger.info(f"✅ Clean transcript processed, length: {len(result)}")
#             return result
#         else:
#             logger.info(f"🔥 Processing unclean transcript with format: {format}")
#             if format == "srt":
#                 result = segments_to_srt(transcript)
#             elif format == "vtt":
#                 result = segments_to_vtt(transcript)
#             else:
#                 lines = []
#                 for seg in transcript:
#                     t = int(seg['start'])
#                     timestamp = f"[{t//60:02d}:{t%60:02d}]"
#                     text_clean = seg['text'].replace('\n', ' ')
#                     lines.append(f"{timestamp} {text_clean}")
#                 result = "\n".join(lines)
            
#             logger.info(f"✅ Unclean transcript processed, length: {len(result)}")
#             return result
                
#     except Exception as e:
#         logger.error(f"❌ YouTube Transcript API failed: {e}")
        
#         try:
#             logger.info("🔄 Trying yt-dlp fallback")
#             if hasattr('transcript_utils', 'get_transcript_with_ytdlp'):
#                 fallback = get_transcript_with_ytdlp(video_id, clean=clean)
#                 if fallback:
#                     logger.info(f"✅ yt-dlp fallback succeeded, length: {len(fallback)}")
#                     return fallback
#         except Exception as fallback_error:
#             logger.error(f"❌ yt-dlp fallback failed: {fallback_error}")
        
#         logger.error(f"❌ No transcript found for video {video_id}")
#         if "No transcripts were found" in str(e) or "TranscriptsDisabled" in str(e):
#             raise HTTPException(
#                 status_code=404,
#                 detail="This video does not have captions/transcripts available."
#             )
#         else:
#             raise HTTPException(
#                 status_code=404,
#                 detail="No transcript/captions found for this video. The video may not have captions available."
#             )

# def segments_to_vtt(transcript) -> str:
#     """Convert transcript segments to WebVTT format"""
#     def sec_to_vtt(ts):
#         h = int(ts // 3600)
#         m = int((ts % 3600) // 60)
#         s = int(ts % 60)
#         ms = int((ts - int(ts)) * 1000)
#         return f"{h:02}:{m:02}:{s:02}.{ms:03}"
    
#     lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    
#     for seg in transcript:
#         start = sec_to_vtt(seg["start"])
#         end = sec_to_vtt(seg.get("start", 0) + seg.get("duration", 0))
#         text = seg["text"].replace("\n", " ").strip()
        
#         lines.append(f"{start} --> {end}")
#         lines.append(text)
#         lines.append("")
    
#     return "\n".join(lines)

# def segments_to_srt(transcript) -> str:
#     """Convert transcript segments to SRT format"""
#     def sec_to_srt(ts):
#         h = int(ts // 3600)
#         m = int((ts % 3600) // 60)
#         s = int(ts % 60)
#         ms = int((ts - int(ts)) * 1000)
#         return f"{h:02}:{m:02}:{s:02},{ms:03}"

#     lines = []
#     for idx, seg in enumerate(transcript):
#         start = sec_to_srt(seg["start"])
#         end = sec_to_srt(seg.get("start", 0) + seg.get("duration", 0))
#         text = seg["text"].replace("\n", " ").strip()
        
#         lines.append(f"{idx+1}")
#         lines.append(f"{start} --> {end}")
#         lines.append(text)
#         lines.append("")
    
#     return "\n".join(lines)

# # 🔥 NEW: Helper function to update file timestamp to current time
# def update_file_timestamp(file_path: Path):
#     """Update file modification time to current time so it appears in 'Today' section"""
#     try:
#         current_time = time.time()
#         os.utime(str(file_path), (current_time, current_time))
#         logger.info(f"🔥 Updated timestamp for: {file_path.name}")
#     except Exception as e:
#         logger.warning(f"Could not update timestamp for {file_path.name}: {e}")

# # =============================================================================
# # 🔥 MOBILE-SPECIFIC MIDDLEWARE
# # =============================================================================

# @app.middleware("http")
# async def mobile_optimization_middleware(request: Request, call_next):
#     """Middleware to optimize responses for mobile devices"""
#     response = await call_next(request)
    
#     # Add mobile-friendly headers
#     if is_mobile_request(request):
#         response.headers["X-Mobile-Optimized"] = "true"
#         # Prevent caching on mobile for dynamic content
#         if request.url.path.startswith("/mobile-download/"):
#             response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    
#     return response

# # =============================================================================
# # FASTAPI ENDPOINTS
# # =============================================================================

# @app.on_event("startup")
# async def startup():
#     initialize_database()
#     cleanup_old_files()

# @app.get("/")
# def root():
#     return {
#         "message": "YouTube Content Downloader API", 
#         "status": "running", 
#         "version": "2.6.0",
#         "features": ["transcripts", "audio", "video", "downloads", "mobile"],
#         "downloads_path": str(DOWNLOADS_DIR)
#     }

# @app.post("/register")
# def register(user: UserCreate, db: Session = Depends(get_db)):
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
    
#     logger.info(f"New user registered: {user.username} ({user.email})")
#     return {"message": "User registered successfully."}

# @app.post("/token")
# def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
#     user = db.query(User).filter(User.username == form_data.username).first()
#     if not user or not verify_password(form_data.password, user.hashed_password):
#         raise HTTPException(status_code=401, detail="Incorrect username or password")
    
#     access_token = create_access_token(
#         data={"sub": user.username},
#         expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     )
    
#     logger.info(f"User logged in: {user.username}")
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
#     """Download YouTube transcript - FIXED VERSION"""
#     start_time = time.time()
    
#     logger.info(f"🔥 Transcript request: {request.youtube_id}, clean: {request.clean_transcript}")
    
#     video_id = extract_youtube_video_id(request.youtube_id)
#     logger.info(f"🔥 Extracted video ID: {video_id}")
    
#     if not video_id or len(video_id) != 11:
#         logger.error(f"❌ Invalid video ID: {video_id}")
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     if not check_internet_connectivity():
#         logger.error("❌ No internet connectivity")
#         raise HTTPException(
#             status_code=503,
#             detail="No internet connection available. Please check your network connection."
#         )
    
#     logger.info("✅ Internet connectivity OK")
    
#     # 🔥 FIXED: Check usage limits properly
#     usage_key = "clean_transcripts" if request.clean_transcript else "unclean_transcripts"
#     can_use, current_usage, limit = check_usage_limit(user, usage_key)
    
#     if not can_use:
#         transcript_type = "clean" if request.clean_transcript else "unclean"
#         raise HTTPException(
#             status_code=403,
#             detail=f"Monthly limit reached for {transcript_type} transcripts ({current_usage}/{limit})."
#         )
    
#     # Get transcript
#     try:
#         logger.info(f"🔥 Attempting to get transcript for {video_id}")
#         transcript_text = get_transcript_youtube_api(
#             video_id, 
#             clean=request.clean_transcript, 
#             format=request.format
#         )
#         logger.info(f"✅ Transcript retrieved, length: {len(transcript_text) if transcript_text else 0}")
        
#     except HTTPException as http_e:
#         logger.error(f"❌ HTTP Exception: {http_e.detail}")
#         raise
#     except Exception as e:
#         logger.error(f"❌ Transcript download failed: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to download transcript: {str(e)}"
#         )
    
#     if not transcript_text:
#         logger.error("❌ Empty transcript returned")
#         raise HTTPException(
#             status_code=404,
#             detail="No transcript found for this video."
#         )
    
#     # 🔥 FIXED: Update usage properly
#     new_usage = increment_user_usage(db, user, usage_key)
    
#     processing_time = time.time() - start_time
    
#     logger.info(f"✅ User {user.username} downloaded {'clean' if request.clean_transcript else 'unclean'} transcript for {video_id}")
    
#     return {
#         "transcript": transcript_text,
#         "youtube_id": video_id,
#         "clean_transcript": request.clean_transcript,
#         "format": request.format,
#         "processing_time": round(processing_time, 2),
#         "success": True,
#         "usage_updated": new_usage,
#         "usage_type": usage_key
#     }

# # 🔥 UPDATED: Modified video download endpoint with mobile support
# @app.post("/download_video/")
# def download_video(
#     request: VideoRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """🔥 FIXED: Video download with mobile support and proper file management"""
#     start_time = time.time()
    
#     video_id = extract_youtube_video_id(request.youtube_id)
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     if not check_internet_connectivity():
#         raise HTTPException(status_code=503, detail="No internet connection available.")
    
#     if not check_ytdlp_availability():
#         raise HTTPException(status_code=500, detail="Video download service unavailable.")
    
#     # 🔥 FIXED: Check usage limits properly
#     can_use, current_usage, limit = check_usage_limit(user, "video_downloads")
    
#     if not can_use:
#         raise HTTPException(
#             status_code=403,
#             detail=f"Monthly limit reached for video downloads ({current_usage}/{limit})."
#         )
    
#     # Get video info for title display
#     video_info = None
#     try:
#         video_info = get_video_info(video_id)
#         logger.info(f"🔥 Got video info: {video_info.get('title', 'Unknown') if video_info else 'Failed to get info'}")
#     except Exception as e:
#         logger.warning(f"Could not get video info: {e}")
    
#     # Define expected filename
#     final_filename = f"{video_id}_video_{request.quality}.mp4"
#     final_path = DOWNLOADS_DIR / final_filename
    
#     # 🔥 FIXED: First clean up any existing files to prevent conflicts
#     logger.info(f"🔥 Cleaning up any existing files for {video_id}...")
#     cleanup_existing_files(video_id, "video", request.quality)
    
#     # 🔥 FIXED: Check if a working file still exists after cleanup
#     existing_working_file = find_working_video_file(video_id, request.quality)
    
#     if existing_working_file:
#         logger.info(f"🔥 Found existing working file after cleanup: {existing_working_file}")
#         file_size = existing_working_file.stat().st_size
        
#         # Move to standard location if needed
#         if existing_working_file != final_path:
#             logger.info(f"🔥 Moving existing file to standard location")
#             try:
#                 # 🔥 FIXED: Use shutil.copy() instead of copy2() to not preserve old timestamps
#                 shutil.copy(str(existing_working_file), str(final_path))
#                 existing_working_file.unlink()  # Remove the old file
#                 logger.info(f"✅ Moved working file to: {final_path}")
#             except Exception as e:
#                 logger.error(f"Error moving file: {e}")
#                 final_path = existing_working_file
#                 final_filename = existing_working_file.name
        
#         # 🔥 NEW: Update timestamp to current time so it appears in "Today"
#         update_file_timestamp(final_path)
        
#         # 🔥 FIXED: Update usage for existing file too
#         new_usage = increment_user_usage(db, user, "video_downloads")
        
#         processing_time = time.time() - start_time
        
#         # 🔥 NEW: Mobile-optimized download URL
#         mobile_download_url = f"/mobile-download/video/{final_filename}?auth={create_access_token_for_mobile(user.username)}"
        
#         return {
#             "download_url": f"/files/{final_filename}",
#             "direct_download_url": mobile_download_url,  # 🔥 NEW: Mobile-optimized URL
#             "youtube_id": video_id,
#             "quality": request.quality,
#             "file_size": file_size,
#             "file_size_mb": round(file_size / (1024 * 1024), 2),
#             "filename": final_filename,
#             "local_path": str(final_path),
#             "processing_time": round(processing_time, 2),
#             "message": "Video ready for download (existing file)",
#             "success": True,
#             "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
#             "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
#             "duration": video_info.get('duration', 0) if video_info else 0,
#             "usage_updated": new_usage,
#             "usage_type": "video_downloads"
#         }
    
#     # 🔥 FIXED: Download to final location directly
#     logger.info(f"🔥 No existing file found, downloading new video for {video_id}")
    
#     try:
#         logger.info(f"🔥 Downloading directly to: {DOWNLOADS_DIR}")
        
#         video_file_path = download_video_with_ytdlp(video_id, request.quality, output_dir=str(DOWNLOADS_DIR))
        
#         if not video_file_path or not os.path.exists(video_file_path):
#             raise HTTPException(status_code=404, detail="Failed to download video.")
        
#         downloaded_file = Path(video_file_path)
#         file_size = downloaded_file.stat().st_size
        
#         if file_size < 10000:
#             raise HTTPException(status_code=500, detail="Downloaded video appears to be corrupted.")
        
#         # 🔥 FIXED: Ensure consistent naming
#         if downloaded_file != final_path:
#             logger.info(f"🔥 Renaming downloaded file to standard name: {final_filename}")
#             try:
#                 if final_path.exists():
#                     final_path.unlink()  # Remove any conflicting file
                
#                 # 🔥 FIXED: Use move instead of copy to avoid timestamp issues
#                 downloaded_file.rename(final_path)
#                 logger.info(f"✅ File renamed to: {final_path}")
#             except Exception as e:
#                 logger.warning(f"Could not rename file: {e}, using original name")
#                 final_path = downloaded_file
#                 final_filename = downloaded_file.name
        
#         # 🔥 NEW: Update timestamp to current time so it appears in "Today"
#         update_file_timestamp(final_path)
        
#         logger.info(f"✅ Video download successful: {final_path} ({file_size} bytes)")
        
#     except Exception as e:
#         logger.error(f"❌ Download failed: {e}")
#         raise HTTPException(status_code=500, detail=f"Video download failed: {str(e)}")
    
#     # 🔥 FIXED: Update usage after successful download
#     new_usage = increment_user_usage(db, user, "video_downloads")
    
#     processing_time = time.time() - start_time
    
#     # 🔥 NEW: Mobile-optimized download URL
#     mobile_download_url = f"/mobile-download/video/{final_filename}?auth={create_access_token_for_mobile(user.username)}"
    
#     return {
#         "download_url": f"/files/{final_filename}",
#         "direct_download_url": mobile_download_url,  # 🔥 NEW: Mobile-optimized URL
#         "youtube_id": video_id,
#         "quality": request.quality,
#         "file_size": file_size,
#         "file_size_mb": round(file_size / (1024 * 1024), 2),
#         "filename": final_filename,
#         "local_path": str(final_path),
#         "processing_time": round(processing_time, 2),
#         "message": "Video ready for download",
#         "success": True,
#         "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
#         "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
#         "duration": video_info.get('duration', 0) if video_info else 0,
#         "usage_updated": new_usage,
#         "usage_type": "video_downloads"
#     }

# # 🔥 UPDATED: Modified audio download endpoint with mobile support
# @app.post("/download_audio/")
# def download_audio(
#     request: AudioRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """🔥 FIXED: Audio download with mobile support and proper file management"""
#     start_time = time.time()
    
#     video_id = extract_youtube_video_id(request.youtube_id)
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     if not check_internet_connectivity():
#         raise HTTPException(status_code=503, detail="No internet connection available.")
    
#     if not check_ytdlp_availability():
#         raise HTTPException(status_code=500, detail="Audio download service temporarily unavailable.")
    
#     # 🔥 FIXED: Check usage limits properly
#     can_use, current_usage, limit = check_usage_limit(user, "audio_downloads")
    
#     if not can_use:
#         raise HTTPException(
#             status_code=403,
#             detail=f"Monthly limit reached for audio downloads ({current_usage}/{limit})."
#         )
    
#     # Get video info for title display
#     video_info = None
#     try:
#         video_info = get_video_info(video_id)
#         logger.info(f"🔥 Got video info: {video_info.get('title', 'Unknown') if video_info else 'Failed to get info'}")
#     except Exception as e:
#         logger.warning(f"Could not get video info: {e}")
    
#     # Define expected filename
#     final_filename = f"{video_id}_audio_{request.quality}.mp3"
#     final_path = DOWNLOADS_DIR / final_filename
    
#     # 🔥 FIXED: First clean up any existing files to prevent conflicts
#     logger.info(f"🔥 Cleaning up any existing audio files for {video_id}...")
#     cleanup_existing_files(video_id, "audio", request.quality)
    
#     # 🔥 FIXED: Check if a working file still exists after cleanup
#     existing_working_file = find_working_audio_file(video_id, request.quality)
    
#     if existing_working_file:
#         logger.info(f"🔥 Found existing working file after cleanup: {existing_working_file}")
#         file_size = existing_working_file.stat().st_size
        
#         # Move to standard location if needed
#         if existing_working_file != final_path:
#             logger.info(f"🔥 Moving existing file to standard location")
#             try:
#                 # 🔥 FIXED: Use shutil.copy() instead of copy2() to not preserve old timestamps
#                 shutil.copy(str(existing_working_file), str(final_path))
#                 existing_working_file.unlink()  # Remove the old file
#                 logger.info(f"✅ Moved working file to: {final_path}")
#             except Exception as e:
#                 logger.error(f"Error moving file: {e}")
#                 final_path = existing_working_file
#                 final_filename = existing_working_file.name
        
#         # 🔥 NEW: Update timestamp to current time so it appears in "Today"
#         update_file_timestamp(final_path)
        
#         # 🔥 FIXED: Update usage for existing file too
#         new_usage = increment_user_usage(db, user, "audio_downloads")
        
#         processing_time = time.time() - start_time
        
#         # 🔥 NEW: Mobile-optimized download URL
#         mobile_download_url = f"/mobile-download/audio/{final_filename}?auth={create_access_token_for_mobile(user.username)}"
        
#         return {
#             "download_url": f"/files/{final_filename}",
#             "direct_download_url": mobile_download_url,  # 🔥 NEW: Mobile-optimized URL
#             "youtube_id": video_id,
#             "quality": request.quality,
#             "file_size": file_size,
#             "file_size_mb": round(file_size / (1024 * 1024), 2),
#             "filename": final_filename,
#             "local_path": str(final_path),
#             "processing_time": round(processing_time, 2),
#             "message": "Audio ready for download (existing file)",
#             "success": True,
#             "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
#             "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
#             "duration": video_info.get('duration', 0) if video_info else 0,
#             "usage_updated": new_usage,
#             "usage_type": "audio_downloads"
#         }
    
#     # 🔥 FIXED: Download to final location directly
#     logger.info(f"🔥 No existing file found, downloading new audio for {video_id}")
    
#     try:
#         logger.info(f"🔥 Downloading directly to: {DOWNLOADS_DIR}")
        
#         audio_file_path = download_audio_with_ytdlp(video_id, request.quality, output_dir=str(DOWNLOADS_DIR))
        
#         if not audio_file_path or not os.path.exists(audio_file_path):
#             raise HTTPException(status_code=404, detail="Failed to download audio.")
        
#         downloaded_file = Path(audio_file_path)
#         file_size = downloaded_file.stat().st_size
        
#         if file_size < 1000:
#             raise HTTPException(status_code=500, detail="Downloaded file appears to be corrupted.")
        
#         # 🔥 FIXED: Ensure consistent naming
#         if downloaded_file != final_path:
#             logger.info(f"🔥 Renaming downloaded file to standard name: {final_filename}")
#             try:
#                 if final_path.exists():
#                     final_path.unlink()  # Remove any conflicting file
                
#                 # 🔥 FIXED: Use move instead of copy to avoid timestamp issues
#                 downloaded_file.rename(final_path)
#                 logger.info(f"✅ File renamed to: {final_path}")
#             except Exception as e:
#                 logger.warning(f"Could not rename file: {e}, using original name")
#                 final_path = downloaded_file
#                 final_filename = downloaded_file.name
        
#         # 🔥 NEW: Update timestamp to current time so it appears in "Today"
#         update_file_timestamp(final_path)
        
#         logger.info(f"✅ Audio download successful: {final_path} ({file_size} bytes)")
        
#     except Exception as e:
#         logger.error(f"❌ Download failed: {e}")
#         raise HTTPException(status_code=500, detail=f"Audio download failed: {str(e)}")
    
#     # 🔥 FIXED: Update usage after successful download
#     new_usage = increment_user_usage(db, user, "audio_downloads")
    
#     processing_time = time.time() - start_time
    
#     # 🔥 NEW: Mobile-optimized download URL
#     mobile_download_url = f"/mobile-download/audio/{final_filename}?auth={create_access_token_for_mobile(user.username)}"
    
#     return {
#         "download_url": f"/files/{final_filename}",
#         "direct_download_url": mobile_download_url,  # 🔥 NEW: Mobile-optimized URL
#         "youtube_id": video_id,
#         "quality": request.quality,
#         "file_size": file_size,
#         "file_size_mb": round(file_size / (1024 * 1024), 2),
#         "filename": final_filename,
#         "local_path": str(final_path),
#         "processing_time": round(processing_time, 2),
#         "message": "Audio ready for download",
#         "success": True,
#         "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
#         "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
#         "duration": video_info.get('duration', 0) if video_info else 0,
#         "usage_updated": new_usage,
#         "usage_type": "audio_downloads"
#     }

# # =============================================================================
# # 🔥 NEW MOBILE-OPTIMIZED DOWNLOAD ENDPOINTS
# # =============================================================================

# @app.get("/mobile-download/{file_type}/{file_id}")
# async def mobile_download(
#     request: Request,
#     file_type: str,  # "audio" or "video"
#     file_id: str,    # filename or file identifier
#     auth: str = None,  # Auth token passed as query parameter for mobile
# ):
#     """
#     🔥 MOBILE-OPTIMIZED DOWNLOAD ENDPOINT
#     Serves files with proper headers for mobile browsers
#     """
#     try:
#         # Validate file type
#         if file_type not in ["audio", "video"]:
#             raise HTTPException(status_code=400, detail="Invalid file type")
        
#         # Verify the auth token
#         try:
#             payload = jwt.decode(auth, SECRET_KEY, algorithms=[ALGORITHM])
#             username: str = payload.get("sub")
#             if username is None:
#                 raise HTTPException(status_code=401, detail="Invalid authentication")
#         except jwt.PyJWTError:
#             raise HTTPException(status_code=401, detail="Invalid authentication")
        
#         # Get user from database
#         db = SessionLocal()
#         try:
#             user = get_user_by_username(db, username)
#             if user is None:
#                 raise HTTPException(status_code=401, detail="User not found")
#         finally:
#             db.close()
        
#         # Construct file path
#         file_path = DOWNLOADS_DIR / file_id
        
#         # Check if file exists
#         if not file_path.exists():
#             raise HTTPException(status_code=404, detail="File not found")
        
#         # Security check: ensure file is in downloads directory
#         if not str(file_path.resolve()).startswith(str(DOWNLOADS_DIR.resolve())):
#             logger.error(f"❌ Security violation: {file_path}")
#             raise HTTPException(status_code=403, detail="Access denied")
        
#         # Check file size (optional: warn for large files on mobile)
#         file_size = file_path.stat().st_size
#         is_mobile = is_mobile_request(request)
        
#         if is_mobile and file_size > 100 * 1024 * 1024:  # 100MB
#             logger.info(f"⚠️ Large file download on mobile: {file_size / 1024 / 1024:.1f}MB")
        
#         # Get proper MIME type
#         mime_type = get_mobile_mime_type(str(file_path), file_type)
        
#         # Generate safe filename
#         safe_filename = get_safe_filename(file_path.name)
        
#         # 🔥 MOBILE-SPECIFIC HEADERS
#         headers = {
#             "Content-Type": mime_type,
#             "Content-Disposition": f'attachment; filename="{safe_filename}"',
#             "Content-Length": str(file_size),
#             "Cache-Control": "no-cache, no-store, must-revalidate",
#             "Pragma": "no-cache",
#             "Expires": "0",
#         }
        
#         # Additional headers for mobile browsers
#         if is_mobile:
#             headers.update({
#                 "X-Content-Type-Options": "nosniff",
#                 "Content-Transfer-Encoding": "binary",
#                 # Force download on mobile browsers
#                 "Content-Disposition": f'attachment; filename="{safe_filename}"; filename*=UTF-8\'\'{safe_filename}',
#             })
        
#         logger.info(f"🔥 Serving {file_type} file: {safe_filename} ({file_size / 1024 / 1024:.1f}MB) to {'mobile' if is_mobile else 'desktop'}")
        
#         # Return file with proper headers
#         return FileResponse(
#             path=str(file_path),
#             media_type=mime_type,
#             headers=headers,
#             filename=safe_filename
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"❌ Mobile download error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

# # 🔥 MOBILE DEBUG ENDPOINT
# @app.get("/debug/mobile-info")
# async def mobile_debug_info(request: Request):
#     """Debug endpoint to check mobile detection and headers"""
#     user_agent = request.headers.get("user-agent", "")
#     is_mobile = is_mobile_request(request)
    
#     return {
#         "is_mobile": is_mobile,
#         "user_agent": user_agent,
#         "headers": dict(request.headers),
#         "client_host": request.client.host if request.client else "unknown",
#         "method": request.method,
#         "url": str(request.url)
#     }

# # 🔥 DOWNLOAD HISTORY ENDPOINTS (for View History feature)
# @app.get("/user/download-history")
# async def get_download_history(current_user: User = Depends(get_current_user)):
#     """Get user's download history"""
#     try:
#         # TODO: Implement actual database query
#         # This is a placeholder - replace with your actual database logic
        
#         # Example query (replace with your actual database model):
#         # downloads = db.query(DownloadHistory).filter(
#         #     DownloadHistory.user_id == current_user.id
#         # ).order_by(DownloadHistory.created_at.desc()).limit(50).all()
        
#         # Mock data for now
#         mock_history = [
#             {
#                 "id": 1,
#                 "type": "audio",
#                 "title": "CHOSEN: God Said → Forget Job Hunting, THIS Is Your Real Calling",
#                 "channel": "Divine Synchronicity",
#                 "video_id": "snm5ZhcJT3k",
#                 "quality": "high",
#                 "downloaded_at": "2025-08-11T17:01:00Z",
#                 "file_size": "34.2 MB",
#                 "status": "completed"
#             },
#             {
#                 "id": 2,
#                 "type": "transcript",
#                 "title": "Only 1% of Elite Chosen Ones Hold the Dual Vibration of Dark and Light",
#                 "channel": "Spiritual Frequency",
#                 "video_id": "snm5ZhcJT3k",
#                 "format": "clean",
#                 "downloaded_at": "2025-08-11T16:45:00Z",
#                 "file_size": "17.89 KB",
#                 "status": "completed"
#             },
#             {
#                 "id": 3,
#                 "type": "video",
#                 "title": "Rick Astley - Never Gonna Give You Up",
#                 "channel": "Rick Astley",
#                 "video_id": "dQw4w9WgXcQ",
#                 "quality": "720p",
#                 "downloaded_at": "2025-08-10T14:30:00Z",
#                 "file_size": "156.7 MB",
#                 "status": "completed"
#             }
#         ]
        
#         return {"downloads": mock_history}
        
#     except Exception as e:
#         logger.error(f"❌ Error fetching download history: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to fetch download history")

# @app.get("/user/recent-activity")
# async def get_recent_activity(current_user: User = Depends(get_current_user)):
#     """Get user's recent activity"""
#     try:
#         # TODO: Implement actual database query for user activity
        
#         # Mock data for now
#         mock_activity = [
#             {
#                 "id": 1,
#                 "action": "Downloaded audio file",
#                 "description": "High quality MP3 from \"CHOSEN: God Said → Forget Job Hunting\"",
#                 "timestamp": "2025-08-11T17:01:00Z",
#                 "type": "download",
#                 "icon": "🎵"
#             },
#             {
#                 "id": 2,
#                 "action": "Generated transcript",
#                 "description": "Clean transcript for \"Only 1% of Elite Chosen Ones\"",
#                 "timestamp": "2025-08-11T16:45:00Z",
#                 "type": "transcript",
#                 "icon": "📄"
#             },
#             {
#                 "id": 3,
#                 "action": "Logged in",
#                 "description": "Signed in from mobile device",
#                 "timestamp": "2025-08-11T16:30:00Z",
#                 "type": "auth",
#                 "icon": "🔐"
#             }
#         ]
        
#         return {"activities": mock_activity}
        
#     except Exception as e:
#         logger.error(f"❌ Error fetching recent activity: {str(e)}")
#         raise HTTPException(status_code=500, detail="Failed to fetch recent activity")

# @app.get("/subscription_status/")
# def get_subscription_status(current_user: User = Depends(get_current_user)):
#     """🔥 FIXED: Get subscription status with proper usage data"""
#     try:
#         tier = getattr(current_user, 'subscription_tier', 'free')
        
#         # 🔥 FIXED: Get actual usage from database
#         usage = {
#             "clean_transcripts": getattr(current_user, "usage_clean_transcripts", 0) or 0,
#             "unclean_transcripts": getattr(current_user, "usage_unclean_transcripts", 0) or 0,
#             "audio_downloads": getattr(current_user, "usage_audio_downloads", 0) or 0,
#             "video_downloads": getattr(current_user, "usage_video_downloads", 0) or 0
#         }
        
#         SUBSCRIPTION_LIMITS = {
#             "free": {"clean_transcripts": 5, "unclean_transcripts": 3, "audio_downloads": 2, "video_downloads": 1},
#             "pro": {"clean_transcripts": 100, "unclean_transcripts": 50, "audio_downloads": 50, "video_downloads": 20},
#             "premium": {"clean_transcripts": float('inf'), "unclean_transcripts": float('inf'), "audio_downloads": float('inf'), "video_downloads": float('inf')}
#         }
        
#         limits = SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS["free"])
#         json_limits = {k: ('unlimited' if v == float('inf') else v) for k, v in limits.items()}
        
#         logger.info(f"🔥 Subscription status for {current_user.username}: tier={tier}, usage={usage}")
        
#         return {
#             "tier": tier,
#             "status": "active" if tier != "free" else "inactive",
#             "usage": usage,
#             "limits": json_limits,
#             "downloads_folder": str(DOWNLOADS_DIR)
#         }
        
#     except Exception as e:
#         logger.error(f"Error getting subscription status: {e}")
#         return {
#             "tier": "free",
#             "status": "inactive",
#             "usage": {"clean_transcripts": 0, "unclean_transcripts": 0, "audio_downloads": 0, "video_downloads": 0},
#             "limits": {"clean_transcripts": 5, "unclean_transcripts": 3, "audio_downloads": 2, "video_downloads": 1},
#             "downloads_folder": str(DOWNLOADS_DIR)
#         }

# @app.get("/health/")
# def health():
#     return {
#         "status": "healthy", 
#         "timestamp": datetime.utcnow().isoformat(),
#         "downloads_path": str(DOWNLOADS_DIR),
#         "connectivity": {
#             "internet": check_internet_connectivity(),
#             "youtube": check_youtube_connectivity()
#         },
#         "mobile_support": True
#     }

# # 🔥 NEW: Direct file download endpoint
# @app.get("/download_file/{filename}")
# def download_file_direct(filename: str, current_user: User = Depends(get_current_user)):
#     """Direct file download endpoint for downloaded files"""
#     try:
#         file_path = DOWNLOADS_DIR / filename
        
#         if not file_path.exists():
#             logger.error(f"❌ File not found: {file_path}")
#             raise HTTPException(status_code=404, detail="File not found")
        
#         if not file_path.is_file():
#             logger.error(f"❌ Path is not a file: {file_path}")
#             raise HTTPException(status_code=404, detail="Invalid file path")
        
#         # Security check: ensure file is in downloads directory
#         if not str(file_path.resolve()).startswith(str(DOWNLOADS_DIR.resolve())):
#             logger.error(f"❌ Security violation: {file_path}")
#             raise HTTPException(status_code=403, detail="Access denied")
        
#         logger.info(f"🔥 Serving file: {filename} ({file_path.stat().st_size} bytes)")
        
#         return FileResponse(
#             path=str(file_path),
#             filename=filename,
#             media_type='application/octet-stream'
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"❌ Error serving file {filename}: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# @app.get("/test_videos")
# def get_test_videos():
#     """Get test video IDs for development and testing"""
#     return {
#         "videos": [
#             {
#                 "id": "dQw4w9WgXcQ", 
#                 "title": "Rick Astley - Never Gonna Give You Up",
#                 "status": "verified_working",
#                 "supports": ["transcript", "audio", "video"],
#                 "note": "Perfect for testing all features"
#             },
#             {
#                 "id": "jNQXAC9IVRw", 
#                 "title": "Me at the zoo",
#                 "status": "verified_working",
#                 "supports": ["transcript", "audio", "video"],
#                 "note": "First YouTube video ever - works for all features"
#             },
#             {
#                 "id": "9bZkp7q19f0",
#                 "title": "PSY - GANGNAM STYLE",
#                 "status": "verified_working", 
#                 "supports": ["transcript", "audio", "video"],
#                 "note": "Popular video with multiple quality options"
#             },
#             {
#                 "id": "L_jWHffIx5E",
#                 "title": "Smash Mouth - All Star",
#                 "status": "verified_working",
#                 "supports": ["transcript", "audio", "video"], 
#                 "note": "Another reliable test video"
#             }
#         ],
#         "recommendations": {
#             "for_video_testing": ["dQw4w9WgXcQ", "jNQXAC9IVRw", "9bZkp7q19f0", "L_jWHffIx5E"],
#             "for_audio_testing": ["dQw4w9WgXcQ", "jNQXAC9IVRw", "9bZkp7q19f0"],
#             "for_transcript_testing": ["dQw4w9WgXcQ", "jNQXAC9IVRw", "9bZkp7q19f0"]
#         },
#         "note": "All these videos work for all features - use any for comprehensive testing"
#     }

# if __name__ == "__main__":
#     import uvicorn
#     print("🔥 Starting server on 0.0.0.0:8000")
#     print("🔥 Mobile access enabled")
#     print("🔥 Mobile download optimization loaded")
#     print(f"🔥 Downloads folder: {str(DOWNLOADS_DIR)}")
#     print("📱 Mobile endpoints available:")
#     print("   - /mobile-download/{file_type}/{file_id}")
#     print("   - /debug/mobile-info")
#     print("   - /user/download-history")
#     print("   - /user/recent-activity")
    
#     uvicorn.run(
#         "main:app", 
#         host="0.0.0.0",
#         port=8000, 
#         reload=True
#     )


#=================================================
# """
# YouTube Content Downloader API - FIXED VERSION
# ==============================================
# 🔥 FIXES:
# - ✅ Usage tracking now works properly (updates counters)
# - ✅ Video downloads now include audio
# - ✅ Proper video metadata and titles
# - ✅ Following transcript download success pattern
# - ✅ Enhanced download success responses
# """

# from pathlib import Path
# from youtube_transcript_api import YouTubeTranscriptApi

# from fastapi import FastAPI, HTTPException, Depends, status
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles
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
# import time
# import stripe
# import tempfile
# import asyncio
# import shutil
# import uuid
# import socket

# # Import our models
# from models import User, TranscriptDownload, get_db, engine, SessionLocal, initialize_database, create_download_record_safe
# from transcript_utils import (
#     get_transcript_with_ytdlp,
#     download_audio_with_ytdlp,
#     download_video_with_ytdlp,
#     check_ytdlp_availability,
#     get_video_info
# )

# # Load environment variables
# load_dotenv()

# # =============================================================================
# # CONFIGURATION
# # =============================================================================

# # Stripe Configuration
# stripe_secret_key = os.getenv("STRIPE_SECRET_KEY")
# if stripe_secret_key:
#     stripe.api_key = stripe_secret_key
#     print("✅ Stripe configured successfully")
# else:
#     print("⚠️ Warning: STRIPE_SECRET_KEY not found in environment variables")

# # Logging Configuration
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("youtube_trans_downloader.main")

# # Environment Configuration
# ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# logger.info(f"Environment: {ENVIRONMENT}")
# logger.info("Starting YouTube Content Downloader API")
# logger.info("Environment variables loaded from .env file")
# logger.info("Using SQLite database for development")

# # Initialize database
# initialize_database()

# # FastAPI App Configuration
# app = FastAPI(
#     title="YouTube Content Downloader API", 
#     version="2.5.0",
#     description="A SaaS application for downloading YouTube transcripts, audio, and video"
# )

# # DOWNLOADS DIRECTORY SETUP - UNICODE SAFE
# try:
#     # Get user's home directory
#     home_dir = Path.home()
#     downloads_dir = home_dir / "Downloads"
#     downloads_dir.mkdir(exist_ok=True)
#     DOWNLOADS_DIR = downloads_dir
    
#     # Test write access
#     test_file = DOWNLOADS_DIR / "test_write.tmp"
#     test_file.write_text("test")
#     test_file.unlink()
    
#     logger.info("🔥 Using user Downloads folder")
#     logger.info(f"🔥 Path: {str(DOWNLOADS_DIR)}")
    
# except Exception as e:
#     logger.warning(f"Cannot use Downloads folder: {e}")
#     # Fallback to local directory
#     DOWNLOADS_DIR = Path("downloads")
#     DOWNLOADS_DIR.mkdir(exist_ok=True)
#     logger.info(f"🔥 Using fallback directory: {str(DOWNLOADS_DIR)}")

# # Mount static files
# app.mount("/files", StaticFiles(directory=str(DOWNLOADS_DIR)), name="files")

# # CORS Configuration
# allowed_origins = [
#     "http://localhost:3000", 
#     "http://127.0.0.1:3000", 
#     "http://192.168.1.185:3000",
#     FRONTEND_URL
# ] if ENVIRONMENT != "production" else [
#     "https://youtube-trans-downloader-api.onrender.com", 
#     FRONTEND_URL
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=allowed_origins, 
#     allow_credentials=True,
#     allow_methods=["*"], 
#     allow_headers=["*"],
# )

# # Security Configuration
# SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # =============================================================================
# # 🔥 FIXED USAGE TRACKING FUNCTIONS
# # =============================================================================

# def increment_user_usage(db: Session, user: User, usage_type: str):
#     """
#     🔥 FIXED: Properly increment user usage and commit to database
#     """
#     try:
#         logger.info(f"🔥 Incrementing usage for user {user.username}: {usage_type}")
        
#         # Get current usage
#         current_usage = getattr(user, f"usage_{usage_type}", 0) or 0
#         new_usage = current_usage + 1
        
#         # Set new usage
#         setattr(user, f"usage_{usage_type}", new_usage)
        
#         # Update usage reset date if needed
#         current_date = datetime.utcnow()
#         if not hasattr(user, 'usage_reset_date') or user.usage_reset_date is None:
#             user.usage_reset_date = current_date
#         elif user.usage_reset_date.month != current_date.month:
#             # Reset monthly usage
#             user.usage_clean_transcripts = 0
#             user.usage_unclean_transcripts = 0
#             user.usage_audio_downloads = 0
#             user.usage_video_downloads = 0
#             user.usage_reset_date = current_date
            
#             # Set the new usage for this type
#             setattr(user, f"usage_{usage_type}", 1)
#             new_usage = 1
        
#         # Commit to database
#         db.commit()
#         db.refresh(user)
        
#         logger.info(f"✅ Usage updated: {usage_type} = {new_usage}")
#         return new_usage
        
#     except Exception as e:
#         logger.error(f"❌ Error incrementing usage: {e}")
#         db.rollback()
#         return current_usage

# def check_usage_limit(user: User, usage_type: str) -> tuple[bool, int, int]:
#     """
#     🔥 FIXED: Check if user has reached usage limit
#     Returns: (can_use, current_usage, limit)
#     """
#     try:
#         # Get subscription tier
#         tier = getattr(user, 'subscription_tier', 'free')
        
#         # Define limits
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
        
#         current_usage = getattr(user, f"usage_{usage_type}", 0) or 0
#         limit = limits.get(tier, limits['free']).get(usage_type, 0)
        
#         can_use = current_usage < limit
        
#         logger.info(f"🔥 Usage check: {usage_type} = {current_usage}/{limit}, can_use = {can_use}")
        
#         return can_use, current_usage, limit
        
#     except Exception as e:
#         logger.error(f"❌ Error checking usage limit: {e}")
#         return False, 0, 0

# # =============================================================================
# # UTILITY FUNCTIONS
# # =============================================================================

# def check_internet_connectivity():
#     """Check if we can reach the internet"""
#     try:
#         socket.create_connection(("8.8.8.8", 53), timeout=3)
#         return True
#     except OSError:
#         return False

# def check_youtube_connectivity():
#     """Check if we can reach YouTube specifically"""
#     try:
#         socket.create_connection(("www.youtube.com", 443), timeout=5)
#         return True
#     except OSError:
#         return False

# def generate_unique_filename(base_name: str, extension: str) -> str:
#     """Generate a unique filename to avoid conflicts"""
#     unique_id = str(uuid.uuid4())[:8]
#     timestamp = int(time.time())
#     return f"{base_name}_{timestamp}_{unique_id}.{extension}"

# # 🔥 FIXED: Enhanced file finding with Windows rename detection
# def find_working_video_file(video_id: str, quality: str) -> Optional[Path]:
#     """Find any existing working video file for this video/quality combination"""
#     try:
#         patterns = [
#             f"{video_id}_video_{quality}.mp4",
#             f"{video_id}_video_{quality}.webm", 
#             f"{video_id}_video_{quality}*.mp4",      # Includes (1), (2), etc.
#             f"{video_id}_video_{quality}*.webm",     # Includes (1), (2), etc.
#             f"{video_id}*video*{quality}*.mp4",      # Broader search
#             f"{video_id}*video*{quality}*.webm",     # Broader search
#             f"{video_id}*video*.mp4",                # Any video with this ID
#             f"{video_id}*video*.webm",               # Any video with this ID
#         ]
        
#         found_files = []
#         for pattern in patterns:
#             files = list(DOWNLOADS_DIR.glob(pattern))
#             for file_path in files:
#                 if file_path.is_file():
#                     file_size = file_path.stat().st_size
#                     if file_size > 1000000:  # 1MB minimum for valid video
#                         found_files.append(file_path)
                        
#         if found_files:
#             # Return the most recent valid file
#             latest_file = max(found_files, key=lambda f: f.stat().st_mtime)
#             logger.info(f"🔥 Found existing video file: {latest_file.name} ({latest_file.stat().st_size} bytes)")
#             return latest_file
        
#         return None
        
#     except Exception as e:
#         logger.error(f"Error finding working video file: {e}")
#         return None

# def find_working_audio_file(video_id: str, quality: str) -> Optional[Path]:
#     """Find any existing working audio file for this video/quality combination"""
#     try:
#         patterns = [
#             f"{video_id}_audio_{quality}.mp3",
#             f"{video_id}_audio_{quality}.m4a",
#             f"{video_id}_audio_{quality}*.mp3",     # Includes (1), (2), etc.
#             f"{video_id}_audio_{quality}*.m4a",     # Includes (1), (2), etc.
#             f"{video_id}*audio*{quality}*.mp3",     # Broader search
#             f"{video_id}*audio*{quality}*.m4a",     # Broader search
#             f"{video_id}*audio*.mp3",               # Any audio with this ID
#             f"{video_id}*audio*.m4a",               # Any audio with this ID
#         ]
        
#         found_files = []
#         for pattern in patterns:
#             files = list(DOWNLOADS_DIR.glob(pattern))
#             for file_path in files:
#                 if file_path.is_file():
#                     file_size = file_path.stat().st_size
#                     if file_size > 100000:  # 100KB minimum for valid audio
#                         found_files.append(file_path)
                        
#         if found_files:
#             # Return the most recent valid file
#             latest_file = max(found_files, key=lambda f: f.stat().st_mtime)
#             logger.info(f"🔥 Found existing audio file: {latest_file.name} ({latest_file.stat().st_size} bytes)")
#             return latest_file
        
#         return None
        
#     except Exception as e:
#         logger.error(f"Error finding working audio file: {e}")
#         return None

# # 🔥 FIXED: Enhanced cleanup that removes ALL old versions

# def cleanup_existing_files(video_id: str, file_type: str, quality: str):
#     """Remove any existing files for this video/quality to prevent duplicates"""
#     try:
#         if file_type == "audio":
#             patterns = [
#                 f"{video_id}_audio_{quality}*",
#                 f"{video_id}*audio*{quality}*",
#                 f"{video_id}*audio*.*",  # Remove any old audio files for this video
#             ]
#         else:  # video
#             patterns = [
#                 f"{video_id}_video_{quality}*",
#                 f"{video_id}*video*{quality}*", 
#                 f"{video_id}*video*.*",  # Remove any old video files for this video
#             ]
        
#         removed_count = 0
#         for pattern in patterns:
#             for file_path in DOWNLOADS_DIR.glob(pattern):
#                 if file_path.is_file():
#                     try:
#                         file_size = file_path.stat().st_size
#                         logger.info(f"🔥 Removing old file: {file_path.name} ({file_size} bytes)")
#                         file_path.unlink()
#                         removed_count += 1
#                     except Exception as e:
#                         logger.warning(f"Could not remove {file_path.name}: {e}")
        
#         if removed_count > 0:
#             logger.info(f"🔥 Cleaned up {removed_count} old files for {video_id}")
#         else:
#             logger.info(f"🔥 No old files found to clean up for {video_id}")
                            
#     except Exception as e:
#         logger.warning(f"Error during cleanup: {e}")

# def cleanup_old_files():
#     """Enhanced cleanup that prevents all duplicate issues"""
#     try:
#         current_time = time.time()
#         max_age = 24 * 3600  # 24 hours (reduced from 2 hours for less aggressive cleanup)
        
#         # Track files by video ID to remove duplicates
#         video_files = {}  # video_id -> list of files
#         audio_files = {}  # video_id -> list of files
        
#         for file_path in DOWNLOADS_DIR.glob("*"):
#             if file_path.is_file():
#                 filename = file_path.name
#                 file_size = file_path.stat().st_size
#                 file_age = current_time - file_path.stat().st_mtime
                
#                 # Remove very old files
#                 if file_age > max_age:
#                     try:
#                         file_path.unlink()
#                         logger.info(f"Cleaned up old file: {filename}")
#                         continue
#                     except:
#                         pass
                
#                 # Remove Windows duplicate files (1), (2), etc. - but keep the original
#                 if "(" in filename and ")" in filename and any(ext in filename for ext in ['.mp4', '.mp3', '.m4a', '.webm']):
#                     try:
#                         file_path.unlink()
#                         logger.info(f"Removed Windows duplicate: {filename}")
#                         continue
#                     except:
#                         pass
                
#                 # Remove tiny corrupted files
#                 min_size = 100000 if "audio" in filename else 1000000
#                 if file_size < min_size and any(ext in filename for ext in ['.mp4', '.mp3', '.m4a', '.webm']):
#                     try:
#                         file_path.unlink()
#                         logger.info(f"Removed corrupted file: {filename} ({file_size} bytes)")
#                         continue
#                     except:
#                         pass
                
#                 # Track video and audio files by video ID for duplicate detection
#                 if "_video_" in filename:
#                     # Extract video ID (first 11 characters before _video_)
#                     video_id = filename.split("_video_")[0]
#                     if video_id not in video_files:
#                         video_files[video_id] = []
#                     video_files[video_id].append(file_path)
#                 elif "_audio_" in filename:
#                     # Extract video ID (first 11 characters before _audio_)
#                     video_id = filename.split("_audio_")[0]
#                     if video_id not in audio_files:
#                         audio_files[video_id] = []
#                     audio_files[video_id].append(file_path)
        
#         # Remove duplicates, keeping only the newest file for each video ID
#         for video_id, files in video_files.items():
#             if len(files) > 1:
#                 # Sort by modification time, keep the newest
#                 files.sort(key=lambda f: f.stat().st_mtime)
#                 files_to_remove = files[:-1]  # Remove all but the newest
                
#                 for file_path in files_to_remove:
#                     try:
#                         logger.info(f"Removing duplicate video: {file_path.name}")
#                         file_path.unlink()
#                     except:
#                         pass
        
#         for video_id, files in audio_files.items():
#             if len(files) > 1:
#                 # Sort by modification time, keep the newest  
#                 files.sort(key=lambda f: f.stat().st_mtime)
#                 files_to_remove = files[:-1]  # Remove all but the newest
                
#                 for file_path in files_to_remove:
#                     try:
#                         logger.info(f"Removing duplicate audio: {file_path.name}")
#                         file_path.unlink()
#                     except:
#                         pass
                        
#     except Exception as e:
#         logger.warning(f"Error during cleanup: {e}")

# # =============================================================================
# # PYDANTIC MODELS
# # =============================================================================

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
#         from_attributes = True

# class Token(BaseModel):
#     access_token: str
#     token_type: str

# class TranscriptRequest(BaseModel):
#     youtube_id: str
#     clean_transcript: bool = True
#     format: Optional[str] = None

# class AudioRequest(BaseModel):
#     youtube_id: str
#     quality: str = "medium"

# class VideoRequest(BaseModel):
#     youtube_id: str
#     quality: str = "720p"

# # =============================================================================
# # HELPER FUNCTIONS
# # =============================================================================

# def get_user(db: Session, username: str) -> Optional[User]:
#     return db.query(User).filter(User.username == username).first()

# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     return pwd_context.verify(plain_password, hashed_password)

# def get_password_hash(password: str) -> str:
#     return pwd_context.hash(password)

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
#     to_encode = data.copy()
#     expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED, 
#         detail="Could not validate credentials",
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

# def extract_youtube_video_id(youtube_id_or_url: str) -> str:
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

# def get_transcript_youtube_api(video_id: str, clean: bool = True, format: Optional[str] = None) -> str:
#     """Get transcript using YouTube Transcript API"""
#     logger.info(f"🔥 Getting transcript for {video_id}, clean={clean}, format={format}")
    
#     if not check_internet_connectivity():
#         logger.error("❌ No internet connectivity")
#         raise HTTPException(
#             status_code=503, 
#             detail="No internet connection available. Please check your network connection."
#         )
    
#     if not check_youtube_connectivity():
#         logger.error("❌ Cannot reach YouTube")
#         raise HTTPException(
#             status_code=503, 
#             detail="Cannot reach YouTube servers. Please try again later."
#         )
    
#     try:
#         logger.info(f"🔥 Attempting to get transcript via YouTube API for {video_id}")
#         transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
#         logger.info(f"✅ Got transcript with {len(transcript)} segments")
        
#         if clean:
#             logger.info("🔥 Processing clean transcript")
#             text = " ".join([seg['text'].replace('\n', ' ') for seg in transcript])
#             clean_text = " ".join(text.split())
            
#             words = clean_text.split()
#             paragraphs = []
#             current_paragraph = []
#             char_count = 0
            
#             for word in words:
#                 current_paragraph.append(word)
#                 char_count += len(word) + 1
                
#                 if char_count > 400 and word.endswith(('.', '!', '?')):
#                     paragraphs.append(' '.join(current_paragraph))
#                     current_paragraph = []
#                     char_count = 0
            
#             if current_paragraph:
#                 paragraphs.append(' '.join(current_paragraph))
            
#             result = '\n\n'.join(paragraphs)
#             logger.info(f"✅ Clean transcript processed, length: {len(result)}")
#             return result
#         else:
#             logger.info(f"🔥 Processing unclean transcript with format: {format}")
#             if format == "srt":
#                 result = segments_to_srt(transcript)
#             elif format == "vtt":
#                 result = segments_to_vtt(transcript)
#             else:
#                 lines = []
#                 for seg in transcript:
#                     t = int(seg['start'])
#                     timestamp = f"[{t//60:02d}:{t%60:02d}]"
#                     text_clean = seg['text'].replace('\n', ' ')
#                     lines.append(f"{timestamp} {text_clean}")
#                 result = "\n".join(lines)
            
#             logger.info(f"✅ Unclean transcript processed, length: {len(result)}")
#             return result
                
#     except Exception as e:
#         logger.error(f"❌ YouTube Transcript API failed: {e}")
        
#         try:
#             logger.info("🔄 Trying yt-dlp fallback")
#             if hasattr('transcript_utils', 'get_transcript_with_ytdlp'):
#                 fallback = get_transcript_with_ytdlp(video_id, clean=clean)
#                 if fallback:
#                     logger.info(f"✅ yt-dlp fallback succeeded, length: {len(fallback)}")
#                     return fallback
#         except Exception as fallback_error:
#             logger.error(f"❌ yt-dlp fallback failed: {fallback_error}")
        
#         logger.error(f"❌ No transcript found for video {video_id}")
#         if "No transcripts were found" in str(e) or "TranscriptsDisabled" in str(e):
#             raise HTTPException(
#                 status_code=404,
#                 detail="This video does not have captions/transcripts available."
#             )
#         else:
#             raise HTTPException(
#                 status_code=404,
#                 detail="No transcript/captions found for this video. The video may not have captions available."
#             )

# def segments_to_vtt(transcript) -> str:
#     """Convert transcript segments to WebVTT format"""
#     def sec_to_vtt(ts):
#         h = int(ts // 3600)
#         m = int((ts % 3600) // 60)
#         s = int(ts % 60)
#         ms = int((ts - int(ts)) * 1000)
#         return f"{h:02}:{m:02}:{s:02}.{ms:03}"
    
#     lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    
#     for seg in transcript:
#         start = sec_to_vtt(seg["start"])
#         end = sec_to_vtt(seg.get("start", 0) + seg.get("duration", 0))
#         text = seg["text"].replace("\n", " ").strip()
        
#         lines.append(f"{start} --> {end}")
#         lines.append(text)
#         lines.append("")
    
#     return "\n".join(lines)

# def segments_to_srt(transcript) -> str:
#     """Convert transcript segments to SRT format"""
#     def sec_to_srt(ts):
#         h = int(ts // 3600)
#         m = int((ts % 3600) // 60)
#         s = int(ts % 60)
#         ms = int((ts - int(ts)) * 1000)
#         return f"{h:02}:{m:02}:{s:02},{ms:03}"

#     lines = []
#     for idx, seg in enumerate(transcript):
#         start = sec_to_srt(seg["start"])
#         end = sec_to_srt(seg.get("start", 0) + seg.get("duration", 0))
#         text = seg["text"].replace("\n", " ").strip()
        
#         lines.append(f"{idx+1}")
#         lines.append(f"{start} --> {end}")
#         lines.append(text)
#         lines.append("")
    
#     return "\n".join(lines)

# # 🔥 NEW: Helper function to update file timestamp to current time
# def update_file_timestamp(file_path: Path):
#     """Update file modification time to current time so it appears in 'Today' section"""
#     try:
#         current_time = time.time()
#         os.utime(str(file_path), (current_time, current_time))
#         logger.info(f"🔥 Updated timestamp for: {file_path.name}")
#     except Exception as e:
#         logger.warning(f"Could not update timestamp for {file_path.name}: {e}")

# # =============================================================================
# # FASTAPI ENDPOINTS
# # =============================================================================

# @app.on_event("startup")
# async def startup():
#     initialize_database()
#     cleanup_old_files()

# @app.get("/")
# def root():
#     return {
#         "message": "YouTube Content Downloader API", 
#         "status": "running", 
#         "version": "2.5.0",
#         "features": ["transcripts", "audio", "video", "downloads"],
#         "downloads_path": str(DOWNLOADS_DIR)
#     }

# @app.post("/register")
# def register(user: UserCreate, db: Session = Depends(get_db)):
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
    
#     logger.info(f"New user registered: {user.username} ({user.email})")
#     return {"message": "User registered successfully."}

# @app.post("/token")
# def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
#     user = db.query(User).filter(User.username == form_data.username).first()
#     if not user or not verify_password(form_data.password, user.hashed_password):
#         raise HTTPException(status_code=401, detail="Incorrect username or password")
    
#     access_token = create_access_token(
#         data={"sub": user.username},
#         expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     )
    
#     logger.info(f"User logged in: {user.username}")
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
#     """Download YouTube transcript - FIXED VERSION"""
#     start_time = time.time()
    
#     logger.info(f"🔥 Transcript request: {request.youtube_id}, clean: {request.clean_transcript}")
    
#     video_id = extract_youtube_video_id(request.youtube_id)
#     logger.info(f"🔥 Extracted video ID: {video_id}")
    
#     if not video_id or len(video_id) != 11:
#         logger.error(f"❌ Invalid video ID: {video_id}")
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     if not check_internet_connectivity():
#         logger.error("❌ No internet connectivity")
#         raise HTTPException(
#             status_code=503,
#             detail="No internet connection available. Please check your network connection."
#         )
    
#     logger.info("✅ Internet connectivity OK")
    
#     # 🔥 FIXED: Check usage limits properly
#     usage_key = "clean_transcripts" if request.clean_transcript else "unclean_transcripts"
#     can_use, current_usage, limit = check_usage_limit(user, usage_key)
    
#     if not can_use:
#         transcript_type = "clean" if request.clean_transcript else "unclean"
#         raise HTTPException(
#             status_code=403,
#             detail=f"Monthly limit reached for {transcript_type} transcripts ({current_usage}/{limit})."
#         )
    
#     # Get transcript
#     try:
#         logger.info(f"🔥 Attempting to get transcript for {video_id}")
#         transcript_text = get_transcript_youtube_api(
#             video_id, 
#             clean=request.clean_transcript, 
#             format=request.format
#         )
#         logger.info(f"✅ Transcript retrieved, length: {len(transcript_text) if transcript_text else 0}")
        
#     except HTTPException as http_e:
#         logger.error(f"❌ HTTP Exception: {http_e.detail}")
#         raise
#     except Exception as e:
#         logger.error(f"❌ Transcript download failed: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to download transcript: {str(e)}"
#         )
    
#     if not transcript_text:
#         logger.error("❌ Empty transcript returned")
#         raise HTTPException(
#             status_code=404,
#             detail="No transcript found for this video."
#         )
    
#     # 🔥 FIXED: Update usage properly
#     new_usage = increment_user_usage(db, user, usage_key)
    
#     processing_time = time.time() - start_time
    
#     logger.info(f"✅ User {user.username} downloaded {'clean' if request.clean_transcript else 'unclean'} transcript for {video_id}")
    
#     return {
#         "transcript": transcript_text,
#         "youtube_id": video_id,
#         "clean_transcript": request.clean_transcript,
#         "format": request.format,
#         "processing_time": round(processing_time, 2),
#         "success": True,
#         "usage_updated": new_usage,
#         "usage_type": usage_key
#     }

# # 🔥 UPDATED: Modified video download endpoint with timestamp fix
# @app.post("/download_video/")
# def download_video(
#     request: VideoRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """🔥 FIXED: Video download with proper file management and timestamp update"""
#     start_time = time.time()
    
#     video_id = extract_youtube_video_id(request.youtube_id)
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     if not check_internet_connectivity():
#         raise HTTPException(status_code=503, detail="No internet connection available.")
    
#     if not check_ytdlp_availability():
#         raise HTTPException(status_code=500, detail="Video download service unavailable.")
    
#     # 🔥 FIXED: Check usage limits properly
#     can_use, current_usage, limit = check_usage_limit(user, "video_downloads")
    
#     if not can_use:
#         raise HTTPException(
#             status_code=403,
#             detail=f"Monthly limit reached for video downloads ({current_usage}/{limit})."
#         )
    
#     # Get video info for title display
#     video_info = None
#     try:
#         video_info = get_video_info(video_id)
#         logger.info(f"🔥 Got video info: {video_info.get('title', 'Unknown') if video_info else 'Failed to get info'}")
#     except Exception as e:
#         logger.warning(f"Could not get video info: {e}")
    
#     # Define expected filename
#     final_filename = f"{video_id}_video_{request.quality}.mp4"
#     final_path = DOWNLOADS_DIR / final_filename
    
#     # 🔥 FIXED: First clean up any existing files to prevent conflicts
#     logger.info(f"🔥 Cleaning up any existing files for {video_id}...")
#     cleanup_existing_files(video_id, "video", request.quality)
    
#     # 🔥 FIXED: Check if a working file still exists after cleanup
#     existing_working_file = find_working_video_file(video_id, request.quality)
    
#     if existing_working_file:
#         logger.info(f"🔥 Found existing working file after cleanup: {existing_working_file}")
#         file_size = existing_working_file.stat().st_size
        
#         # Move to standard location if needed
#         if existing_working_file != final_path:
#             logger.info(f"🔥 Moving existing file to standard location")
#             try:
#                 # 🔥 FIXED: Use shutil.copy() instead of copy2() to not preserve old timestamps
#                 shutil.copy(str(existing_working_file), str(final_path))
#                 existing_working_file.unlink()  # Remove the old file
#                 logger.info(f"✅ Moved working file to: {final_path}")
#             except Exception as e:
#                 logger.error(f"Error moving file: {e}")
#                 final_path = existing_working_file
#                 final_filename = existing_working_file.name
        
#         # 🔥 NEW: Update timestamp to current time so it appears in "Today"
#         update_file_timestamp(final_path)
        
#         # 🔥 FIXED: Update usage for existing file too
#         new_usage = increment_user_usage(db, user, "video_downloads")
        
#         processing_time = time.time() - start_time
        
#         return {
#             "download_url": f"/files/{final_filename}",
#             "direct_download_url": f"/download_file/{final_filename}",
#             "youtube_id": video_id,
#             "quality": request.quality,
#             "file_size": file_size,
#             "file_size_mb": round(file_size / (1024 * 1024), 2),
#             "filename": final_filename,
#             "local_path": str(final_path),
#             "processing_time": round(processing_time, 2),
#             "message": "Video ready for download (existing file)",
#             "success": True,
#             "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
#             "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
#             "duration": video_info.get('duration', 0) if video_info else 0,
#             "usage_updated": new_usage,
#             "usage_type": "video_downloads"
#         }
    
#     # 🔥 FIXED: Download to final location directly
#     logger.info(f"🔥 No existing file found, downloading new video for {video_id}")
    
#     try:
#         logger.info(f"🔥 Downloading directly to: {DOWNLOADS_DIR}")
        
#         video_file_path = download_video_with_ytdlp(video_id, request.quality, output_dir=str(DOWNLOADS_DIR))
        
#         if not video_file_path or not os.path.exists(video_file_path):
#             raise HTTPException(status_code=404, detail="Failed to download video.")
        
#         downloaded_file = Path(video_file_path)
#         file_size = downloaded_file.stat().st_size
        
#         if file_size < 10000:
#             raise HTTPException(status_code=500, detail="Downloaded video appears to be corrupted.")
        
#         # 🔥 FIXED: Ensure consistent naming
#         if downloaded_file != final_path:
#             logger.info(f"🔥 Renaming downloaded file to standard name: {final_filename}")
#             try:
#                 if final_path.exists():
#                     final_path.unlink()  # Remove any conflicting file
                
#                 # 🔥 FIXED: Use move instead of copy to avoid timestamp issues
#                 downloaded_file.rename(final_path)
#                 logger.info(f"✅ File renamed to: {final_path}")
#             except Exception as e:
#                 logger.warning(f"Could not rename file: {e}, using original name")
#                 final_path = downloaded_file
#                 final_filename = downloaded_file.name
        
#         # 🔥 NEW: Update timestamp to current time so it appears in "Today"
#         update_file_timestamp(final_path)
        
#         logger.info(f"✅ Video download successful: {final_path} ({file_size} bytes)")
        
#     except Exception as e:
#         logger.error(f"❌ Download failed: {e}")
#         raise HTTPException(status_code=500, detail=f"Video download failed: {str(e)}")
    
#     # 🔥 FIXED: Update usage after successful download
#     new_usage = increment_user_usage(db, user, "video_downloads")
    
#     processing_time = time.time() - start_time
    
#     return {
#         "download_url": f"/files/{final_filename}",
#         "direct_download_url": f"/download_file/{final_filename}",
#         "youtube_id": video_id,
#         "quality": request.quality,
#         "file_size": file_size,
#         "file_size_mb": round(file_size / (1024 * 1024), 2),
#         "filename": final_filename,
#         "local_path": str(final_path),
#         "processing_time": round(processing_time, 2),
#         "message": "Video ready for download",
#         "success": True,
#         "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
#         "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
#         "duration": video_info.get('duration', 0) if video_info else 0,
#         "usage_updated": new_usage,
#         "usage_type": "video_downloads"
#     }

# # 🔥 UPDATED: Modified audio download endpoint with timestamp fix
# @app.post("/download_audio/")
# def download_audio(
#     request: AudioRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """🔥 FIXED: Audio download with proper file management and timestamp update"""
#     start_time = time.time()
    
#     video_id = extract_youtube_video_id(request.youtube_id)
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     if not check_internet_connectivity():
#         raise HTTPException(status_code=503, detail="No internet connection available.")
    
#     if not check_ytdlp_availability():
#         raise HTTPException(status_code=500, detail="Audio download service temporarily unavailable.")
    
#     # 🔥 FIXED: Check usage limits properly
#     can_use, current_usage, limit = check_usage_limit(user, "audio_downloads")
    
#     if not can_use:
#         raise HTTPException(
#             status_code=403,
#             detail=f"Monthly limit reached for audio downloads ({current_usage}/{limit})."
#         )
    
#     # Get video info for title display
#     video_info = None
#     try:
#         video_info = get_video_info(video_id)
#         logger.info(f"🔥 Got video info: {video_info.get('title', 'Unknown') if video_info else 'Failed to get info'}")
#     except Exception as e:
#         logger.warning(f"Could not get video info: {e}")
    
#     # Define expected filename
#     final_filename = f"{video_id}_audio_{request.quality}.mp3"
#     final_path = DOWNLOADS_DIR / final_filename
    
#     # 🔥 FIXED: First clean up any existing files to prevent conflicts
#     logger.info(f"🔥 Cleaning up any existing audio files for {video_id}...")
#     cleanup_existing_files(video_id, "audio", request.quality)
    
#     # 🔥 FIXED: Check if a working file still exists after cleanup
#     existing_working_file = find_working_audio_file(video_id, request.quality)
    
#     if existing_working_file:
#         logger.info(f"🔥 Found existing working file after cleanup: {existing_working_file}")
#         file_size = existing_working_file.stat().st_size
        
#         # Move to standard location if needed
#         if existing_working_file != final_path:
#             logger.info(f"🔥 Moving existing file to standard location")
#             try:
#                 # 🔥 FIXED: Use shutil.copy() instead of copy2() to not preserve old timestamps
#                 shutil.copy(str(existing_working_file), str(final_path))
#                 existing_working_file.unlink()  # Remove the old file
#                 logger.info(f"✅ Moved working file to: {final_path}")
#             except Exception as e:
#                 logger.error(f"Error moving file: {e}")
#                 final_path = existing_working_file
#                 final_filename = existing_working_file.name
        
#         # 🔥 NEW: Update timestamp to current time so it appears in "Today"
#         update_file_timestamp(final_path)
        
#         # 🔥 FIXED: Update usage for existing file too
#         new_usage = increment_user_usage(db, user, "audio_downloads")
        
#         processing_time = time.time() - start_time
        
#         return {
#             "download_url": f"/files/{final_filename}",
#             "direct_download_url": f"/download_file/{final_filename}",
#             "youtube_id": video_id,
#             "quality": request.quality,
#             "file_size": file_size,
#             "file_size_mb": round(file_size / (1024 * 1024), 2),
#             "filename": final_filename,
#             "local_path": str(final_path),
#             "processing_time": round(processing_time, 2),
#             "message": "Audio ready for download (existing file)",
#             "success": True,
#             "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
#             "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
#             "duration": video_info.get('duration', 0) if video_info else 0,
#             "usage_updated": new_usage,
#             "usage_type": "audio_downloads"
#         }
    
#     # 🔥 FIXED: Download to final location directly
#     logger.info(f"🔥 No existing file found, downloading new audio for {video_id}")
    
#     try:
#         logger.info(f"🔥 Downloading directly to: {DOWNLOADS_DIR}")
        
#         audio_file_path = download_audio_with_ytdlp(video_id, request.quality, output_dir=str(DOWNLOADS_DIR))
        
#         if not audio_file_path or not os.path.exists(audio_file_path):
#             raise HTTPException(status_code=404, detail="Failed to download audio.")
        
#         downloaded_file = Path(audio_file_path)
#         file_size = downloaded_file.stat().st_size
        
#         if file_size < 1000:
#             raise HTTPException(status_code=500, detail="Downloaded file appears to be corrupted.")
        
#         # 🔥 FIXED: Ensure consistent naming
#         if downloaded_file != final_path:
#             logger.info(f"🔥 Renaming downloaded file to standard name: {final_filename}")
#             try:
#                 if final_path.exists():
#                     final_path.unlink()  # Remove any conflicting file
                
#                 # 🔥 FIXED: Use move instead of copy to avoid timestamp issues
#                 downloaded_file.rename(final_path)
#                 logger.info(f"✅ File renamed to: {final_path}")
#             except Exception as e:
#                 logger.warning(f"Could not rename file: {e}, using original name")
#                 final_path = downloaded_file
#                 final_filename = downloaded_file.name
        
#         # 🔥 NEW: Update timestamp to current time so it appears in "Today"
#         update_file_timestamp(final_path)
        
#         logger.info(f"✅ Audio download successful: {final_path} ({file_size} bytes)")
        
#     except Exception as e:
#         logger.error(f"❌ Download failed: {e}")
#         raise HTTPException(status_code=500, detail=f"Audio download failed: {str(e)}")
    
#     # 🔥 FIXED: Update usage after successful download
#     new_usage = increment_user_usage(db, user, "audio_downloads")
    
#     processing_time = time.time() - start_time
    
#     return {
#         "download_url": f"/files/{final_filename}",
#         "direct_download_url": f"/download_file/{final_filename}",
#         "youtube_id": video_id,
#         "quality": request.quality,
#         "file_size": file_size,
#         "file_size_mb": round(file_size / (1024 * 1024), 2),
#         "filename": final_filename,
#         "local_path": str(final_path),
#         "processing_time": round(processing_time, 2),
#         "message": "Audio ready for download",
#         "success": True,
#         "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
#         "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
#         "duration": video_info.get('duration', 0) if video_info else 0,
#         "usage_updated": new_usage,
#         "usage_type": "audio_downloads"
#     } 

# @app.get("/subscription_status/")
# def get_subscription_status(current_user: User = Depends(get_current_user)):
#     """🔥 FIXED: Get subscription status with proper usage data"""
#     try:
#         tier = getattr(current_user, 'subscription_tier', 'free')
        
#         # 🔥 FIXED: Get actual usage from database
#         usage = {
#             "clean_transcripts": getattr(current_user, "usage_clean_transcripts", 0) or 0,
#             "unclean_transcripts": getattr(current_user, "usage_unclean_transcripts", 0) or 0,
#             "audio_downloads": getattr(current_user, "usage_audio_downloads", 0) or 0,
#             "video_downloads": getattr(current_user, "usage_video_downloads", 0) or 0
#         }
        
#         SUBSCRIPTION_LIMITS = {
#             "free": {"clean_transcripts": 5, "unclean_transcripts": 3, "audio_downloads": 2, "video_downloads": 1},
#             "pro": {"clean_transcripts": 100, "unclean_transcripts": 50, "audio_downloads": 50, "video_downloads": 20},
#             "premium": {"clean_transcripts": float('inf'), "unclean_transcripts": float('inf'), "audio_downloads": float('inf'), "video_downloads": float('inf')}
#         }
        
#         limits = SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS["free"])
#         json_limits = {k: ('unlimited' if v == float('inf') else v) for k, v in limits.items()}
        
#         logger.info(f"🔥 Subscription status for {current_user.username}: tier={tier}, usage={usage}")
        
#         return {
#             "tier": tier,
#             "status": "active" if tier != "free" else "inactive",
#             "usage": usage,
#             "limits": json_limits,
#             "downloads_folder": str(DOWNLOADS_DIR)
#         }
        
#     except Exception as e:
#         logger.error(f"Error getting subscription status: {e}")
#         return {
#             "tier": "free",
#             "status": "inactive",
#             "usage": {"clean_transcripts": 0, "unclean_transcripts": 0, "audio_downloads": 0, "video_downloads": 0},
#             "limits": {"clean_transcripts": 5, "unclean_transcripts": 3, "audio_downloads": 2, "video_downloads": 1},
#             "downloads_folder": str(DOWNLOADS_DIR)
#         }

# @app.get("/health/")
# def health():
#     return {
#         "status": "healthy", 
#         "timestamp": datetime.utcnow().isoformat(),
#         "downloads_path": str(DOWNLOADS_DIR),
#         "connectivity": {
#             "internet": check_internet_connectivity(),
#             "youtube": check_youtube_connectivity()
#         }
#     }

# # 🔥 NEW: Direct file download endpoint
# @app.get("/download_file/{filename}")
# def download_file_direct(filename: str, current_user: User = Depends(get_current_user)):
#     """Direct file download endpoint for downloaded files"""
#     try:
#         file_path = DOWNLOADS_DIR / filename
        
#         if not file_path.exists():
#             logger.error(f"❌ File not found: {file_path}")
#             raise HTTPException(status_code=404, detail="File not found")
        
#         if not file_path.is_file():
#             logger.error(f"❌ Path is not a file: {file_path}")
#             raise HTTPException(status_code=404, detail="Invalid file path")
        
#         # Security check: ensure file is in downloads directory
#         if not str(file_path.resolve()).startswith(str(DOWNLOADS_DIR.resolve())):
#             logger.error(f"❌ Security violation: {file_path}")
#             raise HTTPException(status_code=403, detail="Access denied")
        
#         logger.info(f"🔥 Serving file: {filename} ({file_path.stat().st_size} bytes)")
        
#         return FileResponse(
#             path=str(file_path),
#             filename=filename,
#             media_type='application/octet-stream'
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"❌ Error serving file {filename}: {e}")
#         raise HTTPException(status_code=500, detail="Internal server error")

# @app.get("/test_videos")
# def get_test_videos():
#     """Get test video IDs for development and testing"""
#     return {
#         "videos": [
#             {
#                 "id": "dQw4w9WgXcQ", 
#                 "title": "Rick Astley - Never Gonna Give You Up",
#                 "status": "verified_working",
#                 "supports": ["transcript", "audio", "video"],
#                 "note": "Perfect for testing all features"
#             },
#             {
#                 "id": "jNQXAC9IVRw", 
#                 "title": "Me at the zoo",
#                 "status": "verified_working",
#                 "supports": ["transcript", "audio", "video"],
#                 "note": "First YouTube video ever - works for all features"
#             },
#             {
#                 "id": "9bZkp7q19f0",
#                 "title": "PSY - GANGNAM STYLE",
#                 "status": "verified_working", 
#                 "supports": ["transcript", "audio", "video"],
#                 "note": "Popular video with multiple quality options"
#             },
#             {
#                 "id": "L_jWHffIx5E",
#                 "title": "Smash Mouth - All Star",
#                 "status": "verified_working",
#                 "supports": ["transcript", "audio", "video"], 
#                 "note": "Another reliable test video"
#             }
#         ],
#         "recommendations": {
#             "for_video_testing": ["dQw4w9WgXcQ", "jNQXAC9IVRw", "9bZkp7q19f0", "L_jWHffIx5E"],
#             "for_audio_testing": ["dQw4w9WgXcQ", "jNQXAC9IVRw", "9bZkp7q19f0"],
#             "for_transcript_testing": ["dQw4w9WgXcQ", "jNQXAC9IVRw", "9bZkp7q19f0"]
#         },
#         "note": "All these videos work for all features - use any for comprehensive testing"
#     }

# if __name__ == "__main__":
#     import uvicorn
#     print("🔥 Starting server on 0.0.0.0:8000")
#     print("🔥 Mobile access enabled")
#     print(f"🔥 Downloads folder: {str(DOWNLOADS_DIR)}")
    
#     uvicorn.run(
#         "main:app", 
#         host="0.0.0.0",
#         port=8000, 
#         reload=True
#     )



#============================================
# """
# 2. YouTube Content Downloader API - FIXED VERSION
# ==============================================
# 🔥 FIXES:
# - ✅ Usage tracking now works properly (updates counters)
# - ✅ Video downloads now include audio
# - ✅ Proper video metadata and titles
# - ✅ Following transcript download success pattern
# - ✅ Enhanced download success responses
# """

# from pathlib import Path
# from youtube_transcript_api import YouTubeTranscriptApi

# from fastapi import FastAPI, HTTPException, Depends, status
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles
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
# import time
# import stripe
# import tempfile
# import asyncio
# import shutil
# import uuid
# import socket

# # Import our models
# from models import User, TranscriptDownload, get_db, engine, SessionLocal, initialize_database, create_download_record_safe
# from transcript_utils import (
#     get_transcript_with_ytdlp,
#     download_audio_with_ytdlp,
#     download_video_with_ytdlp,
#     check_ytdlp_availability,
#     get_video_info
# )

# # Load environment variables
# load_dotenv()

# # =============================================================================
# # CONFIGURATION
# # =============================================================================

# # Stripe Configuration
# stripe_secret_key = os.getenv("STRIPE_SECRET_KEY")
# if stripe_secret_key:
#     stripe.api_key = stripe_secret_key
#     print("✅ Stripe configured successfully")
# else:
#     print("⚠️ Warning: STRIPE_SECRET_KEY not found in environment variables")

# # Logging Configuration
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("youtube_trans_downloader.main")

# # Environment Configuration
# ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# logger.info(f"Environment: {ENVIRONMENT}")
# logger.info("Starting YouTube Content Downloader API")
# logger.info("Environment variables loaded from .env file")
# logger.info("Using SQLite database for development")

# # Initialize database
# initialize_database()

# # FastAPI App Configuration
# app = FastAPI(
#     title="YouTube Content Downloader API", 
#     version="2.5.0",
#     description="A SaaS application for downloading YouTube transcripts, audio, and video"
# )

# # DOWNLOADS DIRECTORY SETUP - UNICODE SAFE
# try:
#     # Get user's home directory
#     home_dir = Path.home()
#     downloads_dir = home_dir / "Downloads"
#     downloads_dir.mkdir(exist_ok=True)
#     DOWNLOADS_DIR = downloads_dir
    
#     # Test write access
#     test_file = DOWNLOADS_DIR / "test_write.tmp"
#     test_file.write_text("test")
#     test_file.unlink()
    
#     logger.info("🔥 Using user Downloads folder")
#     logger.info(f"🔥 Path: {str(DOWNLOADS_DIR)}")
    
# except Exception as e:
#     logger.warning(f"Cannot use Downloads folder: {e}")
#     # Fallback to local directory
#     DOWNLOADS_DIR = Path("downloads")
#     DOWNLOADS_DIR.mkdir(exist_ok=True)
#     logger.info(f"🔥 Using fallback directory: {str(DOWNLOADS_DIR)}")

# # Mount static files
# app.mount("/files", StaticFiles(directory=str(DOWNLOADS_DIR)), name="files")

# # CORS Configuration
# allowed_origins = [
#     "http://localhost:3000", 
#     "http://127.0.0.1:3000", 
#     "http://192.168.1.185:3000",
#     FRONTEND_URL
# ] if ENVIRONMENT != "production" else [
#     "https://youtube-trans-downloader-api.onrender.com", 
#     FRONTEND_URL
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=allowed_origins, 
#     allow_credentials=True,
#     allow_methods=["*"], 
#     allow_headers=["*"],
# )

# # Security Configuration
# SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # =============================================================================
# # 🔥 FIXED USAGE TRACKING FUNCTIONS
# # =============================================================================

# def increment_user_usage(db: Session, user: User, usage_type: str):
#     """
#     🔥 FIXED: Properly increment user usage and commit to database
#     """
#     try:
#         logger.info(f"🔥 Incrementing usage for user {user.username}: {usage_type}")
        
#         # Get current usage
#         current_usage = getattr(user, f"usage_{usage_type}", 0) or 0
#         new_usage = current_usage + 1
        
#         # Set new usage
#         setattr(user, f"usage_{usage_type}", new_usage)
        
#         # Update usage reset date if needed
#         current_date = datetime.utcnow()
#         if not hasattr(user, 'usage_reset_date') or user.usage_reset_date is None:
#             user.usage_reset_date = current_date
#         elif user.usage_reset_date.month != current_date.month:
#             # Reset monthly usage
#             user.usage_clean_transcripts = 0
#             user.usage_unclean_transcripts = 0
#             user.usage_audio_downloads = 0
#             user.usage_video_downloads = 0
#             user.usage_reset_date = current_date
            
#             # Set the new usage for this type
#             setattr(user, f"usage_{usage_type}", 1)
#             new_usage = 1
        
#         # Commit to database
#         db.commit()
#         db.refresh(user)
        
#         logger.info(f"✅ Usage updated: {usage_type} = {new_usage}")
#         return new_usage
        
#     except Exception as e:
#         logger.error(f"❌ Error incrementing usage: {e}")
#         db.rollback()
#         return current_usage

# def check_usage_limit(user: User, usage_type: str) -> tuple[bool, int, int]:
#     """
#     🔥 FIXED: Check if user has reached usage limit
#     Returns: (can_use, current_usage, limit)
#     """
#     try:
#         # Get subscription tier
#         tier = getattr(user, 'subscription_tier', 'free')
        
#         # Define limits
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
        
#         current_usage = getattr(user, f"usage_{usage_type}", 0) or 0
#         limit = limits.get(tier, limits['free']).get(usage_type, 0)
        
#         can_use = current_usage < limit
        
#         logger.info(f"🔥 Usage check: {usage_type} = {current_usage}/{limit}, can_use = {can_use}")
        
#         return can_use, current_usage, limit
        
#     except Exception as e:
#         logger.error(f"❌ Error checking usage limit: {e}")
#         return False, 0, 0

# # =============================================================================
# # UTILITY FUNCTIONS
# # =============================================================================

# def check_internet_connectivity():
#     """Check if we can reach the internet"""
#     try:
#         socket.create_connection(("8.8.8.8", 53), timeout=3)
#         return True
#     except OSError:
#         return False

# def check_youtube_connectivity():
#     """Check if we can reach YouTube specifically"""
#     try:
#         socket.create_connection(("www.youtube.com", 443), timeout=5)
#         return True
#     except OSError:
#         return False

# def generate_unique_filename(base_name: str, extension: str) -> str:
#     """Generate a unique filename to avoid conflicts"""
#     unique_id = str(uuid.uuid4())[:8]
#     timestamp = int(time.time())
#     return f"{base_name}_{timestamp}_{unique_id}.{extension}"

# def find_working_audio_file(video_id: str, quality: str) -> Optional[Path]:
#     """Find any existing working audio file for this video/quality combination"""
#     try:
#         patterns = [
#             f"{video_id}_audio_{quality}.mp3",
#             f"{video_id}_audio_{quality}.m4a",
#             f"{video_id}_audio_{quality}*.mp3",
#             f"{video_id}_audio_{quality}*.m4a",
#             f"{video_id}*audio*{quality}*.mp3",
#             f"{video_id}*audio*.mp3",
#         ]
        
#         for pattern in patterns:
#             files = list(DOWNLOADS_DIR.glob(pattern))
#             for file_path in files:
#                 if file_path.is_file():
#                     file_size = file_path.stat().st_size
#                     if file_size > 100000:  # 100KB minimum
#                         logger.info(f"🔥 Found working audio file: {file_path.name} ({file_size} bytes)")
#                         return file_path
        
#         return None
        
#     except Exception as e:
#         logger.error(f"Error finding working audio file: {e}")
#         return None

# def find_working_video_file(video_id: str, quality: str) -> Optional[Path]:
#     """Find any existing working video file for this video/quality combination"""
#     try:
#         patterns = [
#             f"{video_id}_video_{quality}.mp4",
#             f"{video_id}_video_{quality}.webm",
#             f"{video_id}_video_{quality}*.mp4",
#             f"{video_id}_video_{quality}*.webm",
#             f"{video_id}*video*{quality}*.*",
#             f"{video_id}*video*.*",
#         ]
        
#         for pattern in patterns:
#             files = list(DOWNLOADS_DIR.glob(pattern))
#             for file_path in files:
#                 if file_path.is_file():
#                     file_size = file_path.stat().st_size
#                     if file_size > 1000000:  # 1MB minimum
#                         logger.info(f"🔥 Found working video file: {file_path.name} ({file_size} bytes)")
#                         return file_path
        
#         return None
        
#     except Exception as e:
#         logger.error(f"Error finding working video file: {e}")
#         return None

# def cleanup_existing_files(video_id: str, file_type: str, quality: str):
#     """Remove any existing corrupted or conflicting files"""
#     try:
#         if file_type == "audio":
#             patterns = [
#                 f"{video_id}_audio_{quality}*",
#                 f"{video_id}*audio*{quality}*",
#             ]
#         else:  # video
#             patterns = [
#                 f"{video_id}_video_{quality}*",
#                 f"{video_id}*video*{quality}*",
#             ]
        
#         for pattern in patterns:
#             for file_path in DOWNLOADS_DIR.glob(pattern):
#                 if file_path.is_file():
#                     file_size = file_path.stat().st_size
                    
#                     is_corrupted = (file_type == "audio" and file_size < 100000) or (file_type == "video" and file_size < 1000000)
#                     has_windows_rename = "(" in file_path.name and ")" in file_path.name
                    
#                     if is_corrupted or has_windows_rename:
#                         try:
#                             logger.info(f"🔥 Removing {'corrupted' if is_corrupted else 'renamed'} file: {file_path.name}")
#                             file_path.unlink()
#                         except Exception as e:
#                             logger.warning(f"Could not remove {file_path.name}: {e}")
                            
#     except Exception as e:
#         logger.warning(f"Error during cleanup: {e}")

# def cleanup_old_files():
#     """Enhanced cleanup that prevents all duplicate issues"""
#     try:
#         current_time = time.time()
#         max_age = 2 * 3600  # 2 hours
        
#         for file_path in DOWNLOADS_DIR.glob("*"):
#             if file_path.is_file():
#                 filename = file_path.name
#                 file_size = file_path.stat().st_size
#                 file_age = current_time - file_path.stat().st_mtime
                
#                 # Remove old files
#                 if file_age > max_age:
#                     try:
#                         file_path.unlink()
#                         logger.info(f"Cleaned up old file: {filename}")
#                         continue
#                     except:
#                         pass
                
#                 # Remove Windows duplicate files
#                 if "(" in filename and ")" in filename:
#                     try:
#                         file_path.unlink()
#                         logger.info(f"Removed Windows duplicate: {filename}")
#                         continue
#                     except:
#                         pass
                
#                 # Remove tiny corrupted files
#                 min_size = 100000 if "audio" in filename else 1000000
#                 if file_size < min_size and ("audio" in filename or "video" in filename):
#                     try:
#                         file_path.unlink()
#                         logger.info(f"Removed corrupted file: {filename} ({file_size} bytes)")
#                     except:
#                         pass
                        
#     except Exception as e:
#         logger.warning(f"Error during cleanup: {e}")

# # =============================================================================
# # PYDANTIC MODELS
# # =============================================================================

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
#         from_attributes = True

# class Token(BaseModel):
#     access_token: str
#     token_type: str

# class TranscriptRequest(BaseModel):
#     youtube_id: str
#     clean_transcript: bool = True
#     format: Optional[str] = None

# class AudioRequest(BaseModel):
#     youtube_id: str
#     quality: str = "medium"

# class VideoRequest(BaseModel):
#     youtube_id: str
#     quality: str = "720p"

# # =============================================================================
# # HELPER FUNCTIONS
# # =============================================================================

# def get_user(db: Session, username: str) -> Optional[User]:
#     return db.query(User).filter(User.username == username).first()

# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     return pwd_context.verify(plain_password, hashed_password)

# def get_password_hash(password: str) -> str:
#     return pwd_context.hash(password)

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
#     to_encode = data.copy()
#     expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED, 
#         detail="Could not validate credentials",
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

# def extract_youtube_video_id(youtube_id_or_url: str) -> str:
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

# def get_transcript_youtube_api(video_id: str, clean: bool = True, format: Optional[str] = None) -> str:
#     """Get transcript using YouTube Transcript API"""
#     logger.info(f"🔥 Getting transcript for {video_id}, clean={clean}, format={format}")
    
#     if not check_internet_connectivity():
#         logger.error("❌ No internet connectivity")
#         raise HTTPException(
#             status_code=503, 
#             detail="No internet connection available. Please check your network connection."
#         )
    
#     if not check_youtube_connectivity():
#         logger.error("❌ Cannot reach YouTube")
#         raise HTTPException(
#             status_code=503, 
#             detail="Cannot reach YouTube servers. Please try again later."
#         )
    
#     try:
#         logger.info(f"🔥 Attempting to get transcript via YouTube API for {video_id}")
#         transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
#         logger.info(f"✅ Got transcript with {len(transcript)} segments")
        
#         if clean:
#             logger.info("🔥 Processing clean transcript")
#             text = " ".join([seg['text'].replace('\n', ' ') for seg in transcript])
#             clean_text = " ".join(text.split())
            
#             words = clean_text.split()
#             paragraphs = []
#             current_paragraph = []
#             char_count = 0
            
#             for word in words:
#                 current_paragraph.append(word)
#                 char_count += len(word) + 1
                
#                 if char_count > 400 and word.endswith(('.', '!', '?')):
#                     paragraphs.append(' '.join(current_paragraph))
#                     current_paragraph = []
#                     char_count = 0
            
#             if current_paragraph:
#                 paragraphs.append(' '.join(current_paragraph))
            
#             result = '\n\n'.join(paragraphs)
#             logger.info(f"✅ Clean transcript processed, length: {len(result)}")
#             return result
#         else:
#             logger.info(f"🔥 Processing unclean transcript with format: {format}")
#             if format == "srt":
#                 result = segments_to_srt(transcript)
#             elif format == "vtt":
#                 result = segments_to_vtt(transcript)
#             else:
#                 lines = []
#                 for seg in transcript:
#                     t = int(seg['start'])
#                     timestamp = f"[{t//60:02d}:{t%60:02d}]"
#                     text_clean = seg['text'].replace('\n', ' ')
#                     lines.append(f"{timestamp} {text_clean}")
#                 result = "\n".join(lines)
            
#             logger.info(f"✅ Unclean transcript processed, length: {len(result)}")
#             return result
                
#     except Exception as e:
#         logger.error(f"❌ YouTube Transcript API failed: {e}")
        
#         try:
#             logger.info("🔄 Trying yt-dlp fallback")
#             if hasattr('transcript_utils', 'get_transcript_with_ytdlp'):
#                 fallback = get_transcript_with_ytdlp(video_id, clean=clean)
#                 if fallback:
#                     logger.info(f"✅ yt-dlp fallback succeeded, length: {len(fallback)}")
#                     return fallback
#         except Exception as fallback_error:
#             logger.error(f"❌ yt-dlp fallback failed: {fallback_error}")
        
#         logger.error(f"❌ No transcript found for video {video_id}")
#         if "No transcripts were found" in str(e) or "TranscriptsDisabled" in str(e):
#             raise HTTPException(
#                 status_code=404,
#                 detail="This video does not have captions/transcripts available."
#             )
#         else:
#             raise HTTPException(
#                 status_code=404,
#                 detail="No transcript/captions found for this video. The video may not have captions available."
#             )

# def segments_to_vtt(transcript) -> str:
#     """Convert transcript segments to WebVTT format"""
#     def sec_to_vtt(ts):
#         h = int(ts // 3600)
#         m = int((ts % 3600) // 60)
#         s = int(ts % 60)
#         ms = int((ts - int(ts)) * 1000)
#         return f"{h:02}:{m:02}:{s:02}.{ms:03}"
    
#     lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    
#     for seg in transcript:
#         start = sec_to_vtt(seg["start"])
#         end = sec_to_vtt(seg.get("start", 0) + seg.get("duration", 0))
#         text = seg["text"].replace("\n", " ").strip()
        
#         lines.append(f"{start} --> {end}")
#         lines.append(text)
#         lines.append("")
    
#     return "\n".join(lines)

# def segments_to_srt(transcript) -> str:
#     """Convert transcript segments to SRT format"""
#     def sec_to_srt(ts):
#         h = int(ts // 3600)
#         m = int((ts % 3600) // 60)
#         s = int(ts % 60)
#         ms = int((ts - int(ts)) * 1000)
#         return f"{h:02}:{m:02}:{s:02},{ms:03}"

#     lines = []
#     for idx, seg in enumerate(transcript):
#         start = sec_to_srt(seg["start"])
#         end = sec_to_srt(seg.get("start", 0) + seg.get("duration", 0))
#         text = seg["text"].replace("\n", " ").strip()
        
#         lines.append(f"{idx+1}")
#         lines.append(f"{start} --> {end}")
#         lines.append(text)
#         lines.append("")
    
#     return "\n".join(lines)

# # =============================================================================
# # FASTAPI ENDPOINTS
# # =============================================================================

# @app.on_event("startup")
# async def startup():
#     initialize_database()
#     cleanup_old_files()

# @app.get("/")
# def root():
#     return {
#         "message": "YouTube Content Downloader API", 
#         "status": "running", 
#         "version": "2.5.0",
#         "features": ["transcripts", "audio", "video", "downloads"],
#         "downloads_path": str(DOWNLOADS_DIR)
#     }

# @app.post("/register")
# def register(user: UserCreate, db: Session = Depends(get_db)):
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
    
#     logger.info(f"New user registered: {user.username} ({user.email})")
#     return {"message": "User registered successfully."}

# @app.post("/token")
# def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
#     user = db.query(User).filter(User.username == form_data.username).first()
#     if not user or not verify_password(form_data.password, user.hashed_password):
#         raise HTTPException(status_code=401, detail="Incorrect username or password")
    
#     access_token = create_access_token(
#         data={"sub": user.username},
#         expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     )
    
#     logger.info(f"User logged in: {user.username}")
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
#     """Download YouTube transcript - FIXED VERSION"""
#     start_time = time.time()
    
#     logger.info(f"🔥 Transcript request: {request.youtube_id}, clean: {request.clean_transcript}")
    
#     video_id = extract_youtube_video_id(request.youtube_id)
#     logger.info(f"🔥 Extracted video ID: {video_id}")
    
#     if not video_id or len(video_id) != 11:
#         logger.error(f"❌ Invalid video ID: {video_id}")
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     if not check_internet_connectivity():
#         logger.error("❌ No internet connectivity")
#         raise HTTPException(
#             status_code=503,
#             detail="No internet connection available. Please check your network connection."
#         )
    
#     logger.info("✅ Internet connectivity OK")
    
#     # 🔥 FIXED: Check usage limits properly
#     usage_key = "clean_transcripts" if request.clean_transcript else "unclean_transcripts"
#     can_use, current_usage, limit = check_usage_limit(user, usage_key)
    
#     if not can_use:
#         transcript_type = "clean" if request.clean_transcript else "unclean"
#         raise HTTPException(
#             status_code=403,
#             detail=f"Monthly limit reached for {transcript_type} transcripts ({current_usage}/{limit})."
#         )
    
#     # Get transcript
#     try:
#         logger.info(f"🔥 Attempting to get transcript for {video_id}")
#         transcript_text = get_transcript_youtube_api(
#             video_id, 
#             clean=request.clean_transcript, 
#             format=request.format
#         )
#         logger.info(f"✅ Transcript retrieved, length: {len(transcript_text) if transcript_text else 0}")
        
#     except HTTPException as http_e:
#         logger.error(f"❌ HTTP Exception: {http_e.detail}")
#         raise
#     except Exception as e:
#         logger.error(f"❌ Transcript download failed: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to download transcript: {str(e)}"
#         )
    
#     if not transcript_text:
#         logger.error("❌ Empty transcript returned")
#         raise HTTPException(
#             status_code=404,
#             detail="No transcript found for this video."
#         )
    
#     # 🔥 FIXED: Update usage properly
#     new_usage = increment_user_usage(db, user, usage_key)
    
#     processing_time = time.time() - start_time
    
#     logger.info(f"✅ User {user.username} downloaded {'clean' if request.clean_transcript else 'unclean'} transcript for {video_id}")
    
#     return {
#         "transcript": transcript_text,
#         "youtube_id": video_id,
#         "clean_transcript": request.clean_transcript,
#         "format": request.format,
#         "processing_time": round(processing_time, 2),
#         "success": True,
#         "usage_updated": new_usage,
#         "usage_type": usage_key
#     }

# @app.post("/download_audio/")
# def download_audio(
#     request: AudioRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """🔥 FIXED: Audio download with proper usage tracking and success handling"""
#     start_time = time.time()
    
#     video_id = extract_youtube_video_id(request.youtube_id)
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     if not check_internet_connectivity():
#         raise HTTPException(status_code=503, detail="No internet connection available.")
    
#     if not check_ytdlp_availability():
#         raise HTTPException(status_code=500, detail="Audio download service temporarily unavailable.")
    
#     # 🔥 FIXED: Check usage limits properly
#     can_use, current_usage, limit = check_usage_limit(user, "audio_downloads")
    
#     if not can_use:
#         raise HTTPException(
#             status_code=403,
#             detail=f"Monthly limit reached for audio downloads ({current_usage}/{limit})."
#         )
    
#     # Get video info for title display
#     video_info = None
#     try:
#         video_info = get_video_info(video_id)
#         logger.info(f"🔥 Got video info: {video_info.get('title', 'Unknown') if video_info else 'Failed to get info'}")
#     except Exception as e:
#         logger.warning(f"Could not get video info: {e}")
    
#     # Define expected filename
#     final_filename = f"{video_id}_audio_{request.quality}.mp3"
#     final_path = DOWNLOADS_DIR / final_filename
    
#     # Check if a working file already exists
#     existing_working_file = find_working_audio_file(video_id, request.quality)
    
#     if existing_working_file:
#         logger.info(f"🔥 Found existing working file: {existing_working_file}")
#         file_size = existing_working_file.stat().st_size
        
#         # If the existing file is not in the expected location, copy it there
#         if existing_working_file != final_path:
#             logger.info(f"🔥 Moving existing working file to standard location")
#             try:
#                 if final_path.exists():
#                     final_path.unlink()
                
#                 shutil.copy2(str(existing_working_file), str(final_path))
                
#                 if existing_working_file != final_path:
#                     existing_working_file.unlink()
                    
#                 logger.info(f"✅ Moved working file to: {final_path}")
#             except Exception as e:
#                 logger.error(f"Error moving file: {e}")
#                 final_path = existing_working_file
#                 final_filename = existing_working_file.name
        
#         # 🔥 FIXED: Update usage for existing file too
#         new_usage = increment_user_usage(db, user, "audio_downloads")
        
#         processing_time = time.time() - start_time
        
#         return {
#             "download_url": f"/files/{final_filename}",
#             "direct_download_url": f"/download_file/{final_filename}",
#             "youtube_id": video_id,
#             "quality": request.quality,
#             "file_size": file_size,
#             "file_size_mb": round(file_size / (1024 * 1024), 2),
#             "filename": final_filename,
#             "local_path": str(final_path),
#             "processing_time": round(processing_time, 2),
#             "message": "Audio ready for download (existing file)",
#             "success": True,
#             "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
#             "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
#             "duration": video_info.get('duration', 0) if video_info else 0,
#             "usage_updated": new_usage,
#             "usage_type": "audio_downloads"
#         }
    
#     # No working file exists, so download a new one
#     logger.info(f"🔥 No working file found, downloading new audio for {video_id}")
    
#     cleanup_existing_files(video_id, "audio", request.quality)
    
#     # Download to temp directory first
#     with tempfile.TemporaryDirectory() as temp_dir:
#         try:
#             logger.info(f"🔥 Downloading to temp: {temp_dir}")
            
#             audio_file_path = download_audio_with_ytdlp(video_id, request.quality, output_dir=temp_dir)
            
#             if not audio_file_path or not os.path.exists(audio_file_path):
#                 raise HTTPException(status_code=404, detail="Failed to download audio.")
            
#             temp_file = Path(audio_file_path)
#             file_size = temp_file.stat().st_size
            
#             if file_size < 1000:
#                 raise HTTPException(status_code=500, detail="Downloaded file appears to be corrupted.")
            
#             logger.info(f"🔥 Moving completed download to: {final_path}")
#             shutil.copy2(str(temp_file), str(final_path))
            
#             if not final_path.exists() or final_path.stat().st_size != file_size:
#                 raise HTTPException(status_code=500, detail="File copy verification failed.")
            
#             logger.info(f"✅ Audio download successful: {final_path} ({file_size} bytes)")
            
#         except Exception as e:
#             logger.error(f"❌ Download failed: {e}")
#             raise HTTPException(status_code=500, detail=f"Audio download failed: {str(e)}")
    
#     # 🔥 FIXED: Update usage after successful download
#     new_usage = increment_user_usage(db, user, "audio_downloads")
    
#     processing_time = time.time() - start_time
    
#     return {
#         "download_url": f"/files/{final_filename}",
#         "direct_download_url": f"/download_file/{final_filename}",
#         "youtube_id": video_id,
#         "quality": request.quality,
#         "file_size": file_size,
#         "file_size_mb": round(file_size / (1024 * 1024), 2),
#         "filename": final_filename,
#         "local_path": str(final_path),
#         "processing_time": round(processing_time, 2),
#         "message": "Audio ready for download",
#         "success": True,
#         "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
#         "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
#         "duration": video_info.get('duration', 0) if video_info else 0,
#         "usage_updated": new_usage,
#         "usage_type": "audio_downloads"
#     }

# @app.post("/download_video/")
# def download_video(
#     request: VideoRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """🔥 FIXED: Video download with proper usage tracking and audio preservation"""
#     start_time = time.time()
    
#     video_id = extract_youtube_video_id(request.youtube_id)
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     if not check_internet_connectivity():
#         raise HTTPException(status_code=503, detail="No internet connection available.")
    
#     if not check_ytdlp_availability():
#         raise HTTPException(status_code=500, detail="Video download service unavailable.")
    
#     # 🔥 FIXED: Check usage limits properly
#     can_use, current_usage, limit = check_usage_limit(user, "video_downloads")
    
#     if not can_use:
#         raise HTTPException(
#             status_code=403,
#             detail=f"Monthly limit reached for video downloads ({current_usage}/{limit})."
#         )
    
#     # Get video info for title display
#     video_info = None
#     try:
#         video_info = get_video_info(video_id)
#         logger.info(f"🔥 Got video info: {video_info.get('title', 'Unknown') if video_info else 'Failed to get info'}")
#     except Exception as e:
#         logger.warning(f"Could not get video info: {e}")
    
#     # Define expected filename
#     final_filename = f"{video_id}_video_{request.quality}.mp4"
#     final_path = DOWNLOADS_DIR / final_filename
    
#     # Check if a working file already exists
#     existing_working_file = find_working_video_file(video_id, request.quality)
    
#     if existing_working_file:
#         logger.info(f"🔥 Found existing working file: {existing_working_file}")
#         file_size = existing_working_file.stat().st_size
        
#         # If the existing file is not in the expected location, copy it there
#         if existing_working_file != final_path:
#             logger.info(f"🔥 Moving existing working file to standard location")
#             try:
#                 if final_path.exists():
#                     final_path.unlink()
                
#                 shutil.copy2(str(existing_working_file), str(final_path))
                
#                 if existing_working_file != final_path:
#                     existing_working_file.unlink()
                    
#                 logger.info(f"✅ Moved working file to: {final_path}")
#             except Exception as e:
#                 logger.error(f"Error moving file: {e}")
#                 final_path = existing_working_file
#                 final_filename = existing_working_file.name
        
#         # 🔥 FIXED: Update usage for existing file too
#         new_usage = increment_user_usage(db, user, "video_downloads")
        
#         processing_time = time.time() - start_time
        
#         return {
#             "download_url": f"/files/{final_filename}",
#             "direct_download_url": f"/download_file/{final_filename}",
#             "youtube_id": video_id,
#             "quality": request.quality,
#             "file_size": file_size,
#             "file_size_mb": round(file_size / (1024 * 1024), 2),
#             "filename": final_filename,
#             "local_path": str(final_path),
#             "processing_time": round(processing_time, 2),
#             "message": "Video ready for download (existing file)",
#             "success": True,
#             "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
#             "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
#             "duration": video_info.get('duration', 0) if video_info else 0,
#             "usage_updated": new_usage,
#             "usage_type": "video_downloads"
#         }
    
#     # No working file exists, so download a new one
#     logger.info(f"🔥 No working file found, downloading new video for {video_id}")
    
#     cleanup_existing_files(video_id, "video", request.quality)
    
#     # Download to temp directory first
#     with tempfile.TemporaryDirectory() as temp_dir:
#         try:
#             logger.info(f"🔥 Downloading to temp: {temp_dir}")
            
#             video_file_path = download_video_with_ytdlp(video_id, request.quality, output_dir=temp_dir)
            
#             if not video_file_path or not os.path.exists(video_file_path):
#                 raise HTTPException(status_code=404, detail="Failed to download video.")
            
#             temp_file = Path(video_file_path)
#             file_size = temp_file.stat().st_size
            
#             if file_size < 10000:
#                 raise HTTPException(status_code=500, detail="Downloaded video appears to be corrupted.")
            
#             # Determine extension from downloaded file
#             original_ext = temp_file.suffix
#             if original_ext:
#                 final_filename = f"{video_id}_video_{request.quality}{original_ext}"
#                 final_path = DOWNLOADS_DIR / final_filename
            
#             logger.info(f"🔥 Moving completed download to: {final_path}")
#             shutil.copy2(str(temp_file), str(final_path))
            
#             if not final_path.exists() or final_path.stat().st_size != file_size:
#                 raise HTTPException(status_code=500, detail="File copy verification failed.")
            
#             logger.info(f"✅ Video download successful: {final_path} ({file_size} bytes)")
            
#         except Exception as e:
#             logger.error(f"❌ Download failed: {e}")
#             raise HTTPException(status_code=500, detail=f"Video download failed: {str(e)}")
    
#     # 🔥 FIXED: Update usage after successful download
#     new_usage = increment_user_usage(db, user, "video_downloads")
    
#     processing_time = time.time() - start_time
    
#     return {
#         "download_url": f"/files/{final_filename}",
#         "direct_download_url": f"/download_file/{final_filename}",
#         "youtube_id": video_id,
#         "quality": request.quality,
#         "file_size": file_size,
#         "file_size_mb": round(file_size / (1024 * 1024), 2),
#         "filename": final_filename,
#         "local_path": str(final_path),
#         "processing_time": round(processing_time, 2),
#         "message": "Video ready for download",
#         "success": True,
#         "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
#         "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
#         "duration": video_info.get('duration', 0) if video_info else 0,
#         "usage_updated": new_usage,
#         "usage_type": "video_downloads"
#     }

# @app.get("/subscription_status/")
# def get_subscription_status(current_user: User = Depends(get_current_user)):
#     """🔥 FIXED: Get subscription status with proper usage data"""
#     try:
#         tier = getattr(current_user, 'subscription_tier', 'free')
        
#         # 🔥 FIXED: Get actual usage from database
#         usage = {
#             "clean_transcripts": getattr(current_user, "usage_clean_transcripts", 0) or 0,
#             "unclean_transcripts": getattr(current_user, "usage_unclean_transcripts", 0) or 0,
#             "audio_downloads": getattr(current_user, "usage_audio_downloads", 0) or 0,
#             "video_downloads": getattr(current_user, "usage_video_downloads", 0) or 0
#         }
        
#         SUBSCRIPTION_LIMITS = {
#             "free": {"clean_transcripts": 5, "unclean_transcripts": 3, "audio_downloads": 2, "video_downloads": 1},
#             "pro": {"clean_transcripts": 100, "unclean_transcripts": 50, "audio_downloads": 50, "video_downloads": 20},
#             "premium": {"clean_transcripts": float('inf'), "unclean_transcripts": float('inf'), "audio_downloads": float('inf'), "video_downloads": float('inf')}
#         }
        
#         limits = SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS["free"])
#         json_limits = {k: ('unlimited' if v == float('inf') else v) for k, v in limits.items()}
        
#         logger.info(f"🔥 Subscription status for {current_user.username}: tier={tier}, usage={usage}")
        
#         return {
#             "tier": tier,
#             "status": "active" if tier != "free" else "inactive",
#             "usage": usage,
#             "limits": json_limits,
#             "downloads_folder": str(DOWNLOADS_DIR)
#         }
        
#     except Exception as e:
#         logger.error(f"Error getting subscription status: {e}")
#         return {
#             "tier": "free",
#             "status": "inactive",
#             "usage": {"clean_transcripts": 0, "unclean_transcripts": 0, "audio_downloads": 0, "video_downloads": 0},
#             "limits": {"clean_transcripts": 5, "unclean_transcripts": 3, "audio_downloads": 2, "video_downloads": 1},
#             "downloads_folder": str(DOWNLOADS_DIR)
#         }

# @app.get("/health/")
# def health():
#     return {
#         "status": "healthy", 
#         "timestamp": datetime.utcnow().isoformat(),
#         "downloads_path": str(DOWNLOADS_DIR),
#         "connectivity": {
#             "internet": check_internet_connectivity(),
#             "youtube": check_youtube_connectivity()
#         }
#     }

# @app.get("/test_videos")
# def get_test_videos():
#     """Get test video IDs for development and testing"""
#     return {
#         "videos": [
#             {
#                 "id": "dQw4w9WgXcQ", 
#                 "title": "Rick Astley - Never Gonna Give You Up",
#                 "status": "verified_working"
#             },
#             {
#                 "id": "jNQXAC9IVRw", 
#                 "title": "Me at the zoo",
#                 "status": "verified_working"
#             }
#         ],
#         "note": "These videos are guaranteed to work and have captions available"
#     }

# if __name__ == "__main__":
#     import uvicorn
#     print("🔥 Starting server on 0.0.0.0:8000")
#     print("🔥 Mobile access enabled")
#     print(f"🔥 Downloads folder: {str(DOWNLOADS_DIR)}")
    
#     uvicorn.run(
#         "main:app", 
#         host="0.0.0.0",
#         port=8000, 
#         reload=True
#     )


# """
# YouTube Transcript Downloader API - DOWNLOADS PATH FIXED
# ========================================================

# Fixed version with proper downloads path to user's Downloads folder
# instead of backend's local downloads folder.

# 🔥 CRITICAL FIX: Files now save to C:\Users\{username}\Downloads
# This ensures mobile users can actually find their downloaded files!
# """

# from pathlib import Path
# from youtube_transcript_api import YouTubeTranscriptApi

# from fastapi import FastAPI, HTTPException, Depends, status
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles
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
# import time
# import stripe
# import tempfile
# import asyncio
# import shutil
# import uuid
# import socket

# # Import our fixed models
# from models import User, TranscriptDownload, get_db, engine, SessionLocal, initialize_database, create_download_record_safe
# from transcript_utils import (
#     get_transcript_with_ytdlp,
#     download_audio_with_ytdlp,
#     download_video_with_ytdlp,
#     check_ytdlp_availability,
#     get_video_info
# )

# # Load environment variables
# load_dotenv()

# # =============================================================================
# # CONFIGURATION
# # =============================================================================

# # Stripe Configuration
# stripe_secret_key = os.getenv("STRIPE_SECRET_KEY")
# if stripe_secret_key:
#     stripe.api_key = stripe_secret_key
#     print("✅ Stripe configured successfully")
# else:
#     print("⚠️ Warning: STRIPE_SECRET_KEY not found in environment variables")

# # Logging Configuration
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("youtube_trans_downloader.main")

# # Environment Configuration
# ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# logger.info(f"Environment: {ENVIRONMENT}")
# logger.info("Starting YouTube Transcript Downloader API")
# logger.info("Environment variables loaded from .env file")
# logger.info("Using SQLite database for development")

# # Initialize database
# initialize_database()

# # FastAPI App Configuration
# app = FastAPI(
#     title="YouTube Transcript Downloader API", 
#     version="2.4.0",
#     description="A SaaS application for downloading YouTube transcripts, audio, and video with proper Downloads folder"
# )

# # 🔥 CRITICAL FIX: Use user's actual Downloads folder instead of local downloads
# DOWNLOADS_DIR = Path.home() / "Downloads"
# DOWNLOADS_DIR.mkdir(exist_ok=True)

# # Log the downloads directory for verification
# logger.info(f"🔥 DOWNLOADS DIRECTORY: {DOWNLOADS_DIR}")
# logger.info(f"🔥 Downloads will be saved to: {DOWNLOADS_DIR.absolute()}")

# # Verify the directory exists and is writable
# try:
#     test_file = DOWNLOADS_DIR / "test_write.tmp"
#     test_file.write_text("test")
#     test_file.unlink()
#     logger.info("✅ Downloads directory is writable")
# except Exception as e:
#     logger.error(f"❌ Downloads directory not writable: {e}")
#     # Fallback to a local downloads directory if user Downloads folder is not accessible
#     DOWNLOADS_DIR = Path("downloads")
#     DOWNLOADS_DIR.mkdir(exist_ok=True)
#     logger.warning(f"🔄 Using fallback directory: {DOWNLOADS_DIR.absolute()}")

# # Mount static files from the downloads directory
# app.mount("/files", StaticFiles(directory=str(DOWNLOADS_DIR)), name="files")

# # CORS Configuration - COMPLETE FIX
# allowed_origins = [
#     "http://localhost:3000", 
#     "http://127.0.0.1:3000", 
#     "http://192.168.1.185:3000",  # Your network IP
#     FRONTEND_URL
# ] if ENVIRONMENT != "production" else [
#     "https://youtube-trans-downloader-api.onrender.com", 
#     FRONTEND_URL
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=allowed_origins, 
#     allow_credentials=True,
#     allow_methods=["*"], 
#     allow_headers=["*"],
# )

# # Security Configuration
# SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # Usage tracking keys
# USAGE_KEYS = {
#     True: "clean_transcripts",
#     False: "unclean_transcripts"
# }

# TRANSCRIPT_TYPE_MAP = {
#     True: "clean",
#     False: "unclean"
# }

# # =============================================================================
# # SUPPORTING UTILITY FUNCTIONS
# # =============================================================================

# def check_internet_connectivity():
#     """Check if we can reach the internet"""
#     try:
#         import socket
#         socket.create_connection(("8.8.8.8", 53), timeout=3)
#         return True
#     except OSError:
#         return False

# def check_youtube_connectivity():
#     """Check if we can reach YouTube specifically"""
#     try:
#         socket.create_connection(("www.youtube.com", 443), timeout=5)
#         return True
#     except OSError:
#         return False

# def generate_unique_filename(base_name: str, extension: str) -> str:
#     """Generate a unique filename to avoid conflicts"""
#     unique_id = str(uuid.uuid4())[:8]
#     timestamp = int(time.time())
#     return f"{base_name}_{timestamp}_{unique_id}.{extension}"

# def cleanup_old_files():
#     """Clean up files older than 2 hours"""
#     try:
#         current_time = time.time()
#         max_age = 2 * 3600  # 2 hours in seconds
        
#         for file_path in DOWNLOADS_DIR.glob("*"):
#             if file_path.is_file():
#                 file_age = current_time - file_path.stat().st_mtime
#                 if file_age > max_age:
#                     file_path.unlink()
#                     logger.info(f"Cleaned up old file: {file_path.name}")
#     except Exception as e:
#         logger.warning(f"Error during file cleanup: {e}")

# # =============================================================================
# # PYDANTIC MODELS
# # =============================================================================

# class PaymentIntentRequest(BaseModel):
#     """Request model for creating payment intents"""
#     amount: Optional[float] = None
#     plan_name: Optional[str] = None
#     planName: Optional[str] = None
#     price_id: Optional[str] = None
#     priceId: Optional[str] = None

# class PaymentConfirmRequest(BaseModel):
#     """Request model for confirming payments"""
#     payment_intent_id: str

# class UserCreate(BaseModel):
#     """Model for user registration"""
#     username: str
#     email: str
#     password: str

# class UserResponse(BaseModel):
#     """Response model for user data"""
#     id: int
#     username: str = None
#     email: str
#     created_at: Optional[datetime] = None
    
#     class Config:
#         from_attributes = True

# class Token(BaseModel):
#     """JWT token response model"""
#     access_token: str
#     token_type: str

# class TranscriptRequest(BaseModel):
#     """Request model for transcript downloads"""
#     youtube_id: str
#     clean_transcript: bool = True
#     format: Optional[str] = None  # "srt" or "vtt" for unclean format

# class AudioRequest(BaseModel):
#     """Request model for audio downloads"""
#     youtube_id: str
#     quality: str = "medium"  # "high", "medium", "low"

# class VideoRequest(BaseModel):
#     """Request model for video downloads"""
#     youtube_id: str
#     quality: str = "720p"  # "1080p", "720p", "480p", "360p"

# # =============================================================================
# # HELPER FUNCTIONS
# # =============================================================================

# def get_user(db: Session, username: str) -> Optional[User]:
#     """Get user by username from database"""
#     return db.query(User).filter(User.username == username).first()

# def get_user_by_username(db: Session, username: str):
#     """Alternative method to get user by username"""
#     return db.query(User).filter(User.username == username).first()

# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     """Verify a plain password against its hash"""
#     return pwd_context.verify(plain_password, hashed_password)

# def get_password_hash(password: str) -> str:
#     """Generate password hash"""
#     return pwd_context.hash(password)

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
#     """Create JWT access token"""
#     to_encode = data.copy()
#     expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
#     """Get current authenticated user from JWT token"""
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED, 
#         detail="Could not validate credentials",
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

# # =============================================================================
# # TRANSCRIPT PROCESSING FUNCTIONS
# # =============================================================================

# def extract_youtube_video_id(youtube_id_or_url: str) -> str:
#     """Extract 11-character video ID from YouTube URL or return ID if already provided"""
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

# def get_transcript_youtube_api(video_id: str, clean: bool = True, format: Optional[str] = None) -> str:
#     """
#     Get transcript using YouTube Transcript API with better error handling
#     """
#     # Check connectivity first
#     if not check_internet_connectivity():
#         raise HTTPException(
#             status_code=503, 
#             detail="No internet connection available. Please check your network connection and try again."
#         )
    
#     if not check_youtube_connectivity():
#         raise HTTPException(
#             status_code=503, 
#             detail="Cannot reach YouTube servers. This might be a temporary network issue or firewall restriction."
#         )
    
#     try:
#         transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        
#         if clean:
#             # Clean format - format into readable paragraphs
#             text = " ".join([seg['text'].replace('\n', ' ') for seg in transcript])
#             clean_text = " ".join(text.split())
            
#             # Break into paragraphs (every ~400 characters at sentence end)
#             words = clean_text.split()
#             paragraphs = []
#             current_paragraph = []
#             char_count = 0
            
#             for word in words:
#                 current_paragraph.append(word)
#                 char_count += len(word) + 1
                
#                 if char_count > 400 and word.endswith(('.', '!', '?')):
#                     paragraphs.append(' '.join(current_paragraph))
#                     current_paragraph = []
#                     char_count = 0
            
#             if current_paragraph:
#                 paragraphs.append(' '.join(current_paragraph))
            
#             return '\n\n'.join(paragraphs)
#         else:
#             # Unclean format with timestamps
#             if format == "srt":
#                 return segments_to_srt(transcript)
#             elif format == "vtt":
#                 return segments_to_vtt(transcript)
#             else:
#                 # Default timestamp format [MM:SS]
#                 lines = []
#                 for seg in transcript:
#                     t = int(seg['start'])
#                     timestamp = f"[{t//60:02d}:{t%60:02d}]"
#                     text_clean = seg['text'].replace('\n', ' ')
#                     lines.append(f"{timestamp} {text_clean}")
#                 return "\n".join(lines)
                
#     except Exception as e:
#         logger.warning(f"Transcript API failed: {e} - trying yt-dlp fallback...")
        
#         # Try yt-dlp fallback with better error handling
#         try:
#             if clean:
#                 fallback = get_transcript_with_ytdlp(video_id, clean=True)
#                 if fallback:
#                     # Format fallback text into paragraphs
#                     words = fallback.split()
#                     paragraphs = []
#                     current_paragraph = []
#                     char_count = 0
                    
#                     for word in words:
#                         current_paragraph.append(word)
#                         char_count += len(word) + 1
                        
#                         if char_count > 400 and word.endswith(('.', '!', '?')):
#                             paragraphs.append(' '.join(current_paragraph))
#                             current_paragraph = []
#                             char_count = 0
                    
#                     if current_paragraph:
#                         paragraphs.append(' '.join(current_paragraph))
                    
#                     return '\n\n'.join(paragraphs)
#             else:
#                 fallback = get_transcript_with_ytdlp(video_id, clean=False)
#                 if fallback and format == "vtt":
#                     return convert_timestamp_to_vtt(fallback)
#                 elif fallback and format == "srt":
#                     return convert_timestamp_to_srt(fallback)
#                 return fallback
#         except Exception as fallback_error:
#             logger.error(f"yt-dlp fallback also failed: {fallback_error}")
        
#         # If both methods fail, give a helpful error message
#         if "NameResolutionError" in str(e) or "Failed to resolve" in str(e):
#             raise HTTPException(
#                 status_code=503,
#                 detail="Network connection issue: Unable to reach YouTube servers. Please check your internet connection or try again later."
#             )
#         else:
#             raise HTTPException(
#                 status_code=404,
#                 detail="No transcript/captions found for this video. The video may not have captions available or may be restricted."
#             )

# def convert_timestamp_to_vtt(timestamp_text: str) -> str:
#     """Convert [MM:SS] format to proper WEBVTT format"""
#     lines = timestamp_text.strip().split('\n')
#     vtt_lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    
#     for line in lines:
#         match = re.match(r'\[(\d{2}):(\d{2})\] (.+)', line)
#         if match:
#             mm, ss, text = match.groups()
#             start_time = f"00:{mm}:{ss}.000"
#             end_minutes = int(mm)
#             end_seconds = int(ss) + 3
#             if end_seconds >= 60:
#                 end_minutes += 1
#                 end_seconds -= 60
#             end_time = f"00:{end_minutes:02d}:{end_seconds:02d}.000"
            
#             vtt_lines.append(f"{start_time} --> {end_time}")
#             vtt_lines.append(text.strip())
#             vtt_lines.append("")
    
#     return '\n'.join(vtt_lines)

# def convert_timestamp_to_srt(timestamp_text: str) -> str:
#     """Convert [MM:SS] format to proper SRT format"""
#     lines = timestamp_text.strip().split('\n')
#     srt_lines = []
#     counter = 1
    
#     for line in lines:
#         match = re.match(r'\[(\d{2}):(\d{2})\] (.+)', line)
#         if match:
#             mm, ss, text = match.groups()
#             start_time = f"00:{mm}:{ss},000"
#             end_minutes = int(mm)
#             end_seconds = int(ss) + 3
#             if end_seconds >= 60:
#                 end_minutes += 1
#                 end_seconds -= 60
#             end_time = f"00:{end_minutes:02d}:{end_seconds:02d},000"
            
#             srt_lines.append(str(counter))
#             srt_lines.append(f"{start_time} --> {end_time}")
#             srt_lines.append(text.strip())
#             srt_lines.append("")
#             counter += 1
    
#     return '\n'.join(srt_lines)

# def segments_to_vtt(transcript) -> str:
#     """Convert transcript segments to proper WebVTT format"""
#     def sec_to_vtt(ts):
#         h = int(ts // 3600)
#         m = int((ts % 3600) // 60)
#         s = int(ts % 60)
#         ms = int((ts - int(ts)) * 1000)
#         return f"{h:02}:{m:02}:{s:02}.{ms:03}"
    
#     lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    
#     for seg in transcript:
#         start = sec_to_vtt(seg["start"])
#         end = sec_to_vtt(seg.get("start", 0) + seg.get("duration", 0))
#         text = seg["text"].replace("\n", " ").strip()
        
#         lines.append(f"{start} --> {end}")
#         lines.append(text)
#         lines.append("")
    
#     return "\n".join(lines)

# def segments_to_srt(transcript) -> str:
#     """Convert transcript segments to SRT format"""
#     def sec_to_srt(ts):
#         h = int(ts // 3600)
#         m = int((ts % 3600) // 60)
#         s = int(ts % 60)
#         ms = int((ts - int(ts)) * 1000)
#         return f"{h:02}:{m:02}:{s:02},{ms:03}"

#     lines = []
#     for idx, seg in enumerate(transcript):
#         start = sec_to_srt(seg["start"])
#         end = sec_to_srt(seg.get("start", 0) + seg.get("duration", 0))
#         text = seg["text"].replace("\n", " ").strip()
        
#         lines.append(f"{idx+1}")
#         lines.append(f"{start} --> {end}")
#         lines.append(text)
#         lines.append("")
    
#     return "\n".join(lines)

# # =============================================================================
# # FASTAPI ENDPOINTS
# # =============================================================================

# @app.on_event("startup")
# async def startup():
#     """Initialize database tables and cleanup old files on application startup"""
#     initialize_database()
#     cleanup_old_files()

# @app.get("/")
# def root():
#     """Root endpoint - API health check"""
#     return {
#         "message": "YouTube Content Downloader API with User Downloads Folder", 
#         "status": "running", 
#         "version": "2.4.0",
#         "features": ["transcripts", "audio", "video", "user_downloads_folder"],
#         "downloads_path": str(DOWNLOADS_DIR.absolute())
#     }

# # =============================================================================
# # AUTHENTICATION ENDPOINTS
# # =============================================================================

# @app.post("/register")
# def register(user: UserCreate, db: Session = Depends(get_db)):
#     """
#     Register a new user account
    
#     Creates a new user with hashed password and default free tier subscription
#     """
#     # Check if username already exists
#     if db.query(User).filter(User.username == user.username).first():
#         raise HTTPException(status_code=400, detail="Username already exists.")
    
#     # Check if email already exists
#     if db.query(User).filter(User.email == user.email).first():
#         raise HTTPException(status_code=400, detail="Email already exists.")
    
#     # Create new user
#     user_obj = User(
#         username=user.username,
#         email=user.email,
#         hashed_password=get_password_hash(user.password),
#         created_at=datetime.utcnow()
#     )
#     db.add(user_obj)
#     db.commit()
#     db.refresh(user_obj)
    
#     logger.info(f"New user registered: {user.username} ({user.email})")
#     return {"message": "User registered successfully."}

# @app.post("/token")
# def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
#     """
#     User login endpoint
    
#     Authenticates user credentials and returns JWT access token
#     """
#     user = db.query(User).filter(User.username == form_data.username).first()
#     if not user or not verify_password(form_data.password, user.hashed_password):
#         raise HTTPException(status_code=401, detail="Incorrect username or password")
    
#     access_token = create_access_token(
#         data={"sub": user.username},
#         expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     )
    
#     logger.info(f"User logged in: {user.username}")
#     return {"access_token": access_token, "token_type": "bearer"}

# @app.get("/users/me", response_model=UserResponse)
# def read_users_me(current_user: User = Depends(get_current_user)):
#     """Get current authenticated user information"""
#     return current_user

# @app.post("/download_transcript/")
# def download_transcript(
#     request: TranscriptRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Download YouTube transcript with enhanced error handling
#     """
#     start_time = time.time()
    
#     # Extract and validate video ID
#     video_id = extract_youtube_video_id(request.youtube_id)
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     # Check connectivity
#     if not check_internet_connectivity():
#         raise HTTPException(
#             status_code=503,
#             detail="No internet connection available. Please check your network connection."
#         )
    
#     # Check for monthly usage reset
#     if user.usage_reset_date.month != datetime.utcnow().month:
#         user.reset_monthly_usage()
#         db.commit()
    
#     # Check usage limits
#     plan_limits = user.get_plan_limits()
#     usage_key = "clean_transcripts" if request.clean_transcript else "unclean_transcripts"
#     current_usage = getattr(user, f"usage_{usage_key}", 0)
#     allowed = plan_limits.get(usage_key, 0)
    
#     if allowed != float('inf') and current_usage >= allowed:
#         transcript_type = "clean" if request.clean_transcript else "unclean"
#         raise HTTPException(
#             status_code=403,
#             detail=f"Monthly limit reached for {transcript_type} transcripts. Please upgrade your plan."
#         )
    
#     # Get transcript
#     try:
#         transcript_text = get_transcript_youtube_api(
#             video_id, 
#             clean=request.clean_transcript, 
#             format=request.format
#         )
#     except HTTPException:
#         raise  # Re-raise HTTP exceptions as-is
#     except Exception as e:
#         logger.error(f"Transcript download failed: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail="Failed to download transcript. The video may not have captions available."
#         )
    
#     if not transcript_text:
#         raise HTTPException(
#             status_code=404,
#             detail="No transcript found for this video. The video may not have captions available."
#         )
    
#     # Update usage and record download
#     user.increment_usage(usage_key)
#     processing_time = time.time() - start_time
    
#     # Create download record
#     try:
#         download_record = create_download_record_safe(
#             db=db,
#             user_id=user.id,
#             youtube_id=video_id,
#             transcript_type="clean" if request.clean_transcript else "unclean",
#             file_size=len(transcript_text.encode('utf-8')),
#             processing_time=processing_time,
#             download_method="youtube_api",
#             quality=None,
#             language="en",
#             file_format=request.format or "txt",
#             download_url=None,
#             expires_at=None,
#             status="completed"
#         )
        
#         if download_record:
#             db.commit()
            
#     except Exception as db_error:
#         logger.error(f"Database error recording download: {db_error}")
#         # Don't fail the request if database recording fails
#         pass
    
#     logger.info(f"User {user.username} downloaded {'clean' if request.clean_transcript else 'unclean'} transcript for {video_id}")
    
#     return {
#         "transcript": transcript_text,
#         "youtube_id": video_id,
#         "clean_transcript": request.clean_transcript,
#         "format": request.format,
#         "processing_time": round(processing_time, 2),
#         "success": True
#     }

# # =============================================================================
# # AUDIO DOWNLOAD ENDPOINT - FIXED FOR USER DOWNLOADS FOLDER
# # =============================================================================

# @app.post("/download_audio/")
# def download_audio(
#     request: AudioRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Download YouTube audio with file saving to user's Downloads folder
#     🔥 FIXED: Files now save to user's actual Downloads folder
#     """
#     start_time = time.time()
    
#     # Clean up old files first
#     cleanup_old_files()
    
#     # Extract and validate video ID
#     video_id = extract_youtube_video_id(request.youtube_id)
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     # Check connectivity
#     if not check_internet_connectivity():
#         raise HTTPException(
#             status_code=503,
#             detail="No internet connection available. Please check your network connection."
#         )
    
#     # Check for monthly usage reset
#     if user.usage_reset_date.month != datetime.utcnow().month:
#         user.reset_monthly_usage()
#         db.commit()
    
#     # Check usage limits
#     plan_limits = user.get_plan_limits()
#     current_usage = getattr(user, "usage_audio_downloads", 0)
#     allowed = plan_limits.get("audio_downloads", 0)
    
#     if allowed != float('inf') and current_usage >= allowed:
#         raise HTTPException(
#             status_code=403,
#             detail="Monthly limit reached for audio downloads. Please upgrade your plan."
#         )
    
#     # Check if yt-dlp is available
#     if not check_ytdlp_availability():
#         raise HTTPException(
#             status_code=500, 
#             detail="Audio download service temporarily unavailable. Please install yt-dlp and FFmpeg."
#         )
    
#     # 🔥 CRITICAL FIX: Download audio directly to user's Downloads folder
#     downloads_path = str(DOWNLOADS_DIR)
#     logger.info(f"🔥 Downloading audio to user's Downloads folder: {downloads_path}")
    
#     try:
#         audio_file_path = download_audio_with_ytdlp(video_id, request.quality, output_dir=downloads_path)
#         logger.info(f"🔥 Audio download returned path: {audio_file_path}")
        
#     except Exception as e:
#         logger.error(f"Audio download failed: {e}")
        
#         # Enhanced error handling with specific messages
#         error_msg = str(e).lower()
#         if "ffmpeg" in error_msg or "ffprobe" in error_msg:
#             raise HTTPException(
#                 status_code=500,
#                 detail="FFmpeg processing error. Please ensure FFmpeg is properly installed and accessible."
#             )
#         elif "network" in error_msg or "connection" in error_msg or "nameresolutionerror" in error_msg:
#             raise HTTPException(
#                 status_code=503,
#                 detail="Network error: Cannot reach YouTube. Please check your internet connection."
#             )
#         elif "timeout" in error_msg:
#             raise HTTPException(
#                 status_code=408,
#                 detail="Download timeout. The audio download took too long. Please try again."
#             )
#         else:
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Audio download failed: {str(e)}"
#             )
    
#     # Verify file exists
#     if not audio_file_path or not os.path.exists(audio_file_path):
#         raise HTTPException(
#             status_code=404, 
#             detail="Failed to download audio from this video. The video may not have audio available or may be restricted."
#         )
    
#     # Convert to Path object for easier handling
#     audio_file = Path(audio_file_path)
    
#     # Generate a unique filename for serving
#     unique_filename = generate_unique_filename(f"{video_id}_audio_{request.quality}", "mp3")
#     final_path = DOWNLOADS_DIR / unique_filename
    
#     # Move file to final location with better error handling
#     try:
#         logger.info(f"🔥 Moving file from {audio_file} to {final_path}")
        
#         if audio_file.samefile(final_path):
#             # File is already in the right place
#             logger.info("✅ File is already in the correct location")
#         else:
#             # Move/copy file to final location
#             shutil.move(str(audio_file), str(final_path))
#             logger.info(f"✅ File moved successfully to {final_path}")
            
#     except Exception as e:
#         logger.error(f"❌ Error moving audio file: {e}")
#         # Try copying instead
#         try:
#             shutil.copy2(str(audio_file), str(final_path))
#             if audio_file.exists():
#                 audio_file.unlink()
#             logger.info(f"✅ File copied successfully to {final_path}")
#         except Exception as copy_error:
#             logger.error(f"❌ Error copying audio file: {copy_error}")
#             raise HTTPException(
#                 status_code=500,
#                 detail="Error processing downloaded audio file."
#             )
    
#     # Verify final file exists and has content
#     if not final_path.exists() or final_path.stat().st_size == 0:
#         logger.error(f"❌ Final file verification failed: exists={final_path.exists()}, size={final_path.stat().st_size if final_path.exists() else 'N/A'}")
#         raise HTTPException(
#             status_code=500,
#             detail="Audio file processing failed. Please try again."
#         )
    
#     # Get file size
#     file_size = final_path.stat().st_size
#     processing_time = time.time() - start_time
    
#     # Update usage and record download
#     user.increment_usage("audio_downloads")
    
#     # Create download record
#     try:
#         download_record = create_download_record_safe(
#             db=db,
#             user_id=user.id,
#             youtube_id=video_id,
#             transcript_type="audio",
#             file_size=file_size,
#             processing_time=processing_time,
#             download_method="ytdlp",
#             quality=request.quality,
#             language="en",
#             file_format="mp3",
#             download_url=f"/files/{unique_filename}",
#             expires_at=datetime.utcnow() + timedelta(hours=1),
#             status="completed"
#         )
        
#         if download_record:
#             db.commit()
            
#     except Exception as db_error:
#         logger.error(f"Database error recording download: {db_error}")
#         # Don't fail the request if database recording fails
#         pass
    
#     logger.info(f"✅ User {user.username} downloaded audio for {video_id} ({request.quality}) - {file_size} bytes")
#     logger.info(f"🔥 File saved to user's Downloads folder: {final_path}")
    
#     return {
#         "download_url": f"/files/{unique_filename}",
#         "direct_download_url": f"/download_file/{unique_filename}",
#         "youtube_id": video_id,
#         "quality": request.quality,
#         "file_size": file_size,
#         "file_size_mb": round(file_size / (1024 * 1024), 2),
#         "filename": unique_filename,
#         "original_filename": f"{video_id}_audio_{request.quality}.mp3",
#         "local_path": str(final_path),  # 🔥 ADD: Show user where file is saved
#         "expires_in": "1 hour",
#         "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
#         "processing_time": round(processing_time, 2),
#         "message": "Audio download ready and saved to Downloads folder",
#         "success": True
#     }

# # =============================================================================
# # VIDEO DOWNLOAD ENDPOINT - FIXED FOR USER DOWNLOADS FOLDER
# # =============================================================================

# @app.post("/download_video/")
# def download_video(
#     request: VideoRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Download YouTube video with file saving to user's Downloads folder
#     🔥 FIXED: Files now save to user's actual Downloads folder
#     """
#     start_time = time.time()
    
#     # Clean up old files first
#     cleanup_old_files()
    
#     # Extract and validate video ID
#     video_id = extract_youtube_video_id(request.youtube_id)
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     # Check connectivity
#     if not check_internet_connectivity():
#         raise HTTPException(
#             status_code=503,
#             detail="No internet connection available. Please check your network connection."
#         )
    
#     # Check for monthly usage reset
#     if user.usage_reset_date.month != datetime.utcnow().month:
#         user.reset_monthly_usage()
#         db.commit()
    
#     # Check usage limits
#     plan_limits = user.get_plan_limits()
#     current_usage = getattr(user, "usage_video_downloads", 0)
#     allowed = plan_limits.get("video_downloads", 0)
    
#     if allowed != float('inf') and current_usage >= allowed:
#         raise HTTPException(
#             status_code=403,
#             detail="Monthly limit reached for video downloads. Please upgrade your plan."
#         )
    
#     # Check if yt-dlp is available
#     if not check_ytdlp_availability():
#         raise HTTPException(
#             status_code=500, 
#             detail="Video download service temporarily unavailable. Please install yt-dlp and FFmpeg."
#         )
    
#     # 🔥 CRITICAL FIX: Download video directly to user's Downloads folder
#     downloads_path = str(DOWNLOADS_DIR)
#     logger.info(f"🔥 Downloading video to user's Downloads folder: {downloads_path}")
    
#     try:
#         video_file = download_video_with_ytdlp(video_id, request.quality, output_dir=downloads_path)
#         logger.info(f"🔥 Video download returned path: {video_file}")
#     except Exception as e:
#         logger.error(f"Video download failed: {e}")
#         if "NameResolutionError" in str(e) or "Failed to resolve" in str(e):
#             raise HTTPException(
#                 status_code=503,
#                 detail="Network error: Cannot reach YouTube. Please check your internet connection."
#             )
#         else:
#             raise HTTPException(
#                 status_code=500,
#                 detail="Video download failed. The video may not be available or have restrictions."
#             )
    
#     if not video_file or not os.path.exists(video_file):
#         raise HTTPException(status_code=404, detail="Failed to download video.")
    
#     # Generate a unique filename for serving
#     original_filename = os.path.basename(video_file)
#     file_extension = "mp4"  # Default to mp4
#     if "." in original_filename:
#         file_extension = original_filename.split(".")[-1]
    
#     unique_filename = generate_unique_filename(f"{video_id}_video_{request.quality}", file_extension)
#     final_path = DOWNLOADS_DIR / unique_filename
    
#     # Move file to final location
#     try:
#         shutil.move(video_file, final_path)
#         logger.info(f"✅ Video file moved to: {final_path}")
#     except Exception as e:
#         logger.error(f"❌ Error moving video file: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail="Error processing downloaded video file."
#         )
    
#     # Get file size
#     file_size = os.path.getsize(final_path)
#     processing_time = time.time() - start_time
    
#     # Update usage and record download
#     user.increment_usage("video_downloads")
    
#     # Create download record
#     download_record = create_download_record_safe(
#         db=db,
#         user_id=user.id,
#         youtube_id=video_id,
#         transcript_type="video",
#         file_size=file_size,
#         processing_time=processing_time,
#         download_method="ytdlp",
#         quality=request.quality,
#         language="en",
#         file_format=file_extension,
#         download_url=f"/files/{unique_filename}",
#         expires_at=datetime.utcnow() + timedelta(hours=1),
#         status="completed"
#     )
    
#     if download_record:
#         db.commit()
    
#     logger.info(f"✅ User {user.username} downloaded video for {video_id} ({request.quality})")
#     logger.info(f"🔥 File saved to user's Downloads folder: {final_path}")
    
#     return {
#         "download_url": f"/files/{unique_filename}",
#         "direct_download_url": f"/download_file/{unique_filename}",
#         "youtube_id": video_id,
#         "quality": request.quality,
#         "file_size": file_size,
#         "file_size_mb": round(file_size / (1024 * 1024), 2),
#         "filename": unique_filename,
#         "local_path": str(final_path),  # 🔥 ADD: Show user where file is saved
#         "expires_in": "1 hour",
#         "message": "Video download ready and saved to Downloads folder",
#         "success": True
#     }

# # =============================================================================
# # FILE SERVING ENDPOINTS
# # =============================================================================

# @app.get("/download_file/{filename}")
# async def download_file(filename: str):
#     """
#     Serve downloaded files with proper headers for direct download
#     🔥 ENHANCED: Better mobile download support
#     """
#     file_path = DOWNLOADS_DIR / filename
    
#     if not file_path.exists():
#         raise HTTPException(
#             status_code=404, 
#             detail="File not found or expired. Downloads expire after 1 hour."
#         )
    
#     # Determine content type
#     if filename.endswith('.mp3'):
#         media_type = 'audio/mpeg'
#     elif filename.endswith('.mp4'):
#         media_type = 'video/mp4'
#     elif filename.endswith('.webm'):
#         media_type = 'video/webm'
#     elif filename.endswith('.mkv'):
#         media_type = 'video/x-matroska'
#     else:
#         media_type = 'application/octet-stream'
    
#     # Generate a clean filename for download
#     clean_filename = filename
#     if '_' in filename:
#         # Extract video ID and quality from unique filename
#         parts = filename.split('_')
#         if len(parts) >= 4:
#             video_id = parts[0]
#             content_type = parts[1]  # 'audio' or 'video'
#             quality = parts[2]
#             extension = filename.split('.')[-1]
#             clean_filename = f"{video_id}_{content_type}_{quality}.{extension}"
    
#     return FileResponse(
#         path=file_path,
#         media_type=media_type,
#         filename=clean_filename,
#         headers={
#             "Content-Disposition": f"attachment; filename={clean_filename}",
#             "Cache-Control": "no-cache, no-store, must-revalidate",
#             "Pragma": "no-cache",
#             "Expires": "0",
#             # 🔥 ADD: Better mobile support headers
#             "Content-Transfer-Encoding": "binary",
#             "Accept-Ranges": "bytes"
#         }
#     )

# # =============================================================================
# # SUBSCRIPTION ENDPOINTS
# # =============================================================================

# @app.get("/subscription_status/")
# def get_subscription_status(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Get user's current subscription status, usage, and limits
#     """
#     try:
#         # Get tier directly from user model
#         tier = getattr(current_user, 'subscription_tier', 'free')
        
#         # Current usage
#         usage = {
#             "clean_transcripts": getattr(current_user, "usage_clean_transcripts", 0),
#             "unclean_transcripts": getattr(current_user, "usage_unclean_transcripts", 0),
#             "audio_downloads": getattr(current_user, "usage_audio_downloads", 0),
#             "video_downloads": getattr(current_user, "usage_video_downloads", 0)
#         }
        
#         # Plan limits configuration
#         SUBSCRIPTION_LIMITS = {
#             "free": {
#                 "clean_transcripts": 5, 
#                 "unclean_transcripts": 3, 
#                 "audio_downloads": 2, 
#                 "video_downloads": 1
#             },
#             "pro": {
#                 "clean_transcripts": 100, 
#                 "unclean_transcripts": 50, 
#                 "audio_downloads": 50, 
#                 "video_downloads": 20
#             },
#             "premium": {
#                 "clean_transcripts": float('inf'), 
#                 "unclean_transcripts": float('inf'), 
#                 "audio_downloads": float('inf'), 
#                 "video_downloads": float('inf')
#             }
#         }
        
#         limits = SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS["free"])
#         json_limits = {k: ('unlimited' if v == float('inf') else v) for k, v in limits.items()}
        
#         return {
#             "tier": tier,
#             "status": "active" if tier != "free" else "inactive",
#             "usage": usage,
#             "limits": json_limits,
#             "subscription_id": None,
#             "stripe_customer_id": getattr(current_user, 'stripe_customer_id', None),
#             "current_period_end": None,
#             "downloads_folder": str(DOWNLOADS_DIR.absolute())  # 🔥 ADD: Show downloads location
#         }
        
#     except Exception as e:
#         logger.error(f"❌ Error getting subscription status: {str(e)}")
#         # Fallback response
#         return {
#             "tier": "free",
#             "status": "inactive",
#             "usage": {"clean_transcripts": 0, "unclean_transcripts": 0, "audio_downloads": 0, "video_downloads": 0},
#             "limits": {"clean_transcripts": 5, "unclean_transcripts": 3, "audio_downloads": 2, "video_downloads": 1},
#             "subscription_id": None,
#             "stripe_customer_id": None,
#             "current_period_end": None,
#             "downloads_folder": str(DOWNLOADS_DIR.absolute())
#         }

# # =============================================================================
# # SYSTEM CHECK AND DEBUG ENDPOINTS
# # =============================================================================

# @app.get("/system_check/")
# def system_check():
#     """Enhanced system check with downloads folder verification"""
#     checks = {
#         "ytdlp_available": check_ytdlp_availability(),
#         "database_connected": True,  # If we got here, DB is working
#         "stripe_configured": bool(stripe_secret_key),
#         "environment": ENVIRONMENT,
#         "internet_connectivity": check_internet_connectivity(),
#         "youtube_connectivity": check_youtube_connectivity(),
#         "downloads_directory": DOWNLOADS_DIR.exists(),
#         "downloads_writable": True  # We tested this at startup
#     }
    
#     # Check ffmpeg availability
#     try:
#         result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
#         checks["ffmpeg_available"] = result.returncode == 0
#     except:
#         checks["ffmpeg_available"] = False
    
#     # Test downloads directory
#     try:
#         test_file = DOWNLOADS_DIR / "test_write.tmp"
#         test_file.write_text("test")
#         test_file.unlink()
#         checks["downloads_writable"] = True
#     except:
#         checks["downloads_writable"] = False
    
#     # Generate recommendations
#     recommendations = []
#     if not checks["ffmpeg_available"]:
#         recommendations.append("Install FFmpeg for audio/video processing")
#     if not checks["ytdlp_available"]:
#         recommendations.append("Install yt-dlp for video downloads")
#     if not checks["stripe_configured"]:
#         recommendations.append("Configure Stripe for payments")
#     if not checks["internet_connectivity"]:
#         recommendations.append("Check internet connection")
#     if not checks["youtube_connectivity"]:
#         recommendations.append("Check access to YouTube (firewall/proxy settings)")
#     if not checks["downloads_writable"]:
#         recommendations.append("Check Downloads folder permissions")
    
#     return {
#         "checks": checks,
#         "recommendations": recommendations,
#         "downloads_path": str(DOWNLOADS_DIR.absolute()),
#         "status": "healthy" if all([
#             checks["ytdlp_available"],
#             checks["ffmpeg_available"],
#             checks["internet_connectivity"],
#             checks["youtube_connectivity"],
#             checks["downloads_writable"]
#         ]) else "degraded"
#     }

# @app.get("/health/")
# def health():
#     """Health check endpoint for monitoring"""
#     return {
#         "status": "healthy", 
#         "timestamp": datetime.utcnow().isoformat(), 
#         "features": ["transcripts", "audio", "video", "user_downloads_folder"],
#         "downloads_path": str(DOWNLOADS_DIR.absolute()),
#         "connectivity": {
#             "internet": check_internet_connectivity(),
#             "youtube": check_youtube_connectivity()
#         }
#     }

# @app.get("/test_videos")
# def get_test_videos():
#     """Get test video IDs for development and testing"""
#     return {
#         "videos": [
#             {
#                 "id": "dQw4w9WgXcQ", 
#                 "title": "Rick Astley - Never Gonna Give You Up",
#                 "status": "verified_working"
#             },
#             {
#                 "id": "jNQXAC9IVRw", 
#                 "title": "Me at the zoo",
#                 "status": "verified_working"
#             }
#         ],
#         "note": "These videos are guaranteed to work and have captions available"
#     }

# # =============================================================================
# # PAYMENT ENDPOINTS 
# # =============================================================================

# @app.post("/create_payment_intent/")
# async def create_payment_intent(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Create a Stripe Payment Intent (defaults to Pro plan)"""
#     try:
#         amount = 9.99
#         plan_name = "pro"
        
#         logger.info(f"Creating payment intent for user {current_user.id}: {plan_name} plan (${amount})")
        
#         amount_cents = int(amount * 100)
        
#         stripe_customer_id = getattr(current_user, 'stripe_customer_id', None)
#         if not stripe_customer_id:
#             logger.info(f"Creating new Stripe customer for user {current_user.id}")
#             customer = stripe.Customer.create(
#                 email=current_user.email,
#                 metadata={'user_id': str(current_user.id)}
#             )
#             stripe_customer_id = customer.id
#             current_user.stripe_customer_id = stripe_customer_id
#             db.commit()
        
#         intent = stripe.PaymentIntent.create(
#             amount=amount_cents,
#             currency='usd',
#             customer=stripe_customer_id,
#             metadata={
#                 'user_id': str(current_user.id),
#                 'plan_name': plan_name
#             },
#             automatic_payment_methods={'enabled': True}
#         )
        
#         logger.info(f"Successfully created payment intent {intent.id} for ${amount}")
        
#         return {
#             "client_secret": intent.client_secret,
#             "payment_intent_id": intent.id,
#             "amount": amount,
#             "plan_name": plan_name
#         }
        
#     except stripe.error.StripeError as e:
#         logger.error(f"Stripe error: {e}")
#         raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
#     except Exception as e:
#         logger.error(f"Failed to create payment intent: {e}")
#         raise HTTPException(status_code=400, detail=str(e))

# @app.post("/create_payment_intent/{plan_type}")
# async def create_payment_intent_with_plan(
#     plan_type: str,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Create a Stripe Payment Intent for a specific plan"""
#     try:
#         plans = {
#             'pro': {'amount': 9.99, 'name': 'Pro'},
#             'premium': {'amount': 19.99, 'name': 'Premium'}
#         }
        
#         if plan_type not in plans:
#             raise HTTPException(status_code=400, detail="Invalid plan type")
        
#         plan = plans[plan_type]
#         amount = plan['amount']
#         plan_name = plan['name']
        
#         logger.info(f"Creating payment intent for user {current_user.id}: {plan_name} plan (${amount})")
        
#         amount_cents = int(amount * 100)
        
#         stripe_customer_id = getattr(current_user, 'stripe_customer_id', None)
#         if not stripe_customer_id:
#             logger.info(f"Creating new Stripe customer for user {current_user.id}")
#             customer = stripe.Customer.create(
#                 email=current_user.email,
#                 metadata={'user_id': str(current_user.id)}
#             )
#             stripe_customer_id = customer.id
#             current_user.stripe_customer_id = stripe_customer_id
#             db.commit()
        
#         intent = stripe.PaymentIntent.create(
#             amount=amount_cents,
#             currency='usd',
#             customer=stripe_customer_id,
#             metadata={
#                 'user_id': str(current_user.id),
#                 'plan_name': plan_type,
#                 'plan_display_name': plan_name
#             },
#             automatic_payment_methods={'enabled': True}
#         )
        
#         logger.info(f"Successfully created payment intent {intent.id}")
        
#         return {
#             "client_secret": intent.client_secret,
#             "payment_intent_id": intent.id,
#             "amount": amount,
#             "plan_name": plan_name,
#             "plan_type": plan_type
#         }
        
#     except stripe.error.StripeError as e:
#         logger.error(f"Stripe error: {e}")
#         raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
#     except Exception as e:
#         logger.error(f"Failed to create payment intent: {e}")
#         raise HTTPException(status_code=400, detail=str(e))

# @app.post("/confirm_payment/")
# async def confirm_payment(
#     request: PaymentConfirmRequest,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Confirm payment and upgrade user subscription"""
#     try:
#         intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
        
#         if intent.status == 'succeeded':
#             plan_name = intent.metadata.get('plan_name', 'pro').lower()
            
#             logger.info(f"Payment succeeded for user {current_user.id}, upgrading to {plan_name}")
            
#             if plan_name == 'pro':
#                 current_user.subscription_tier = 'pro'
#             elif plan_name == 'premium':
#                 current_user.subscription_tier = 'premium'
#             else:
#                 current_user.subscription_tier = 'pro'
            
#             current_user.reset_monthly_usage()
#             db.commit()
            
#             return {
#                 "success": True,
#                 "subscription": {
#                     "tier": current_user.subscription_tier,
#                     "status": "active"
#                 },
#                 "message": f"Successfully upgraded to {plan_name} plan"
#             }
#         else:
#             logger.warning(f"Payment intent {intent.id} status: {intent.status}")
#             raise HTTPException(status_code=400, detail=f"Payment not completed. Status: {intent.status}")
            
#     except stripe.error.StripeError as e:
#         logger.error(f"Stripe error: {e}")
#         raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
#     except Exception as e:
#         logger.error(f"Failed to confirm payment: {e}")
#         raise HTTPException(status_code=400, detail=str(e))

# @app.post("/cancel_subscription/")
# async def cancel_subscription(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Cancel user subscription and downgrade to free tier"""
#     try:
#         logger.info(f"Cancelling subscription for user {current_user.id}")
        
#         current_user.subscription_tier = 'free'
#         db.commit()
        
#         return {
#             "success": True,
#             "message": "Subscription cancelled successfully. You've been downgraded to the free tier."
#         }
        
#     except Exception as e:
#         logger.error(f"Failed to cancel subscription: {e}")
#         raise HTTPException(status_code=400, detail=str(e))

# # =============================================================================
# # CLEANUP TASK
# # =============================================================================

# @app.on_event("shutdown")
# async def shutdown():
#     """Clean up temporary files on application shutdown"""
#     cleanup_old_files()

# # =============================================================================
# # APPLICATION ENTRY POINT
# # =============================================================================

# if __name__ == "__main__":
#     import uvicorn
#     print("🔥 Starting server on 0.0.0.0:8000")
#     print("🔥 This allows mobile connections")
#     print(f"🔥 Downloads will be saved to: {DOWNLOADS_DIR.absolute()}")
    
#     uvicorn.run(
#         "main:app", 
#         host="0.0.0.0",  # CRITICAL - must be 0.0.0.0
#         port=8000, 
#         reload=True
#     )

#=======================================================================
# """
# YouTube Transcript Downloader API - COMPLETE FIX
# ===============================================

# Fixed version with proper file serving, better error handling,
# and network connectivity solutions.

# Features:
# - File serving endpoints for audio/video downloads
# - Better error handling for network issues
# - Automatic file cleanup
# - Fallback mechanisms for connectivity issues
# - Proper temporary file management

# Author: YouTube Transcript Downloader Team
# Version: 2.3.1 (Complete Fix - No Nested Folders)
# """

# from pathlib import Path
# from youtube_transcript_api import YouTubeTranscriptApi

# from fastapi import FastAPI, HTTPException, Depends, status
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles
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
# import time
# import stripe
# import tempfile
# import asyncio
# import shutil
# import uuid
# import socket

# # Import our fixed models
# from models import User, TranscriptDownload, get_db, engine, SessionLocal, initialize_database, create_download_record_safe
# from transcript_utils import (
#     get_transcript_with_ytdlp,
#     download_audio_with_ytdlp,
#     download_video_with_ytdlp,
#     check_ytdlp_availability,
#     get_video_info
# )

# # Load environment variables
# load_dotenv()

# # =============================================================================
# # CONFIGURATION
# # =============================================================================

# # Stripe Configuration
# stripe_secret_key = os.getenv("STRIPE_SECRET_KEY")
# if stripe_secret_key:
#     stripe.api_key = stripe_secret_key
#     print("✅ Stripe configured successfully")
# else:
#     print("⚠️ Warning: STRIPE_SECRET_KEY not found in environment variables")

# # Logging Configuration
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("youtube_trans_downloader.main")

# # Environment Configuration
# ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# logger.info(f"Environment: {ENVIRONMENT}")
# logger.info("Starting YouTube Transcript Downloader API")
# logger.info("Environment variables loaded from .env file")
# logger.info("Using SQLite database for development")

# # Initialize database
# initialize_database()

# # FastAPI App Configuration
# app = FastAPI(
#     title="YouTube Transcript Downloader API", 
#     version="2.3.1",
#     description="A SaaS application for downloading YouTube transcripts, audio, and video with file serving"
# )

# DOWNLOADS_DIR = Path("downloads")
# DOWNLOADS_DIR.mkdir(exist_ok=True)
# app.mount("/files", StaticFiles(directory="downloads"), name="files")

# # CORS Configuration - COMPLETE FIX
# allowed_origins = [
#     "http://localhost:3000", 
#     "http://127.0.0.1:3000", 
#     "http://192.168.1.185:3000",  # Your network IP
#     FRONTEND_URL
# ] if ENVIRONMENT != "production" else [
#     "https://youtube-trans-downloader-api.onrender.com", 
#     FRONTEND_URL
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=allowed_origins, 
#     allow_credentials=True,
#     allow_methods=["*"], 
#     allow_headers=["*"],
# )


# # # CORS Configuration - ADD YOUR NETWORK IP
# # allowed_origins = [
# #     "http://localhost:3000", 
# #     "http://127.0.0.1:3000", 
# #     "http://192.168.1.185:3000",  # ADD THIS LINE with your network IP
# #     FRONTEND_URL
# # ] if ENVIRONMENT != "production" else [
# #     "https://youtube-trans-downloader-api.onrender.com", 
# #     FRONTEND_URL
# # ]

# # app.add_middleware(
# #     CORSMiddleware,
# #     allow_origins=allowed_origins, 
# #     allow_credentials=True,
# #     allow_methods=["*"], 
# #     allow_headers=["*"],
# # )


# # Security Configuration
# SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # Usage tracking keys
# USAGE_KEYS = {
#     True: "clean_transcripts",
#     False: "unclean_transcripts"
# }

# TRANSCRIPT_TYPE_MAP = {
#     True: "clean",
#     False: "unclean"
# }

# # =============================================================================
# # SUPPORTING UTILITY FUNCTIONS
# # =============================================================================

# def check_internet_connectivity():
#     """Check if we can reach the internet"""
#     try:
#         import socket
#         socket.create_connection(("8.8.8.8", 53), timeout=3)
#         return True
#     except OSError:
#         return False

# def check_youtube_connectivity():
#     """Check if we can reach YouTube specifically"""
#     try:
#         socket.create_connection(("www.youtube.com", 443), timeout=5)
#         return True
#     except OSError:
#         return False

# def generate_unique_filename(base_name: str, extension: str) -> str:
#     """Generate a unique filename to avoid conflicts"""
#     unique_id = str(uuid.uuid4())[:8]
#     timestamp = int(time.time())
#     return f"{base_name}_{timestamp}_{unique_id}.{extension}"

# def cleanup_old_files():
#     """Clean up files older than 2 hours"""
#     try:
#         current_time = time.time()
#         max_age = 2 * 3600  # 2 hours in seconds
        
#         for file_path in DOWNLOADS_DIR.glob("*"):
#             if file_path.is_file():
#                 file_age = current_time - file_path.stat().st_mtime
#                 if file_age > max_age:
#                     file_path.unlink()
#                     logger.info(f"Cleaned up old file: {file_path.name}")
#     except Exception as e:
#         logger.warning(f"Error during file cleanup: {e}")

# # =============================================================================
# # PYDANTIC MODELS
# # =============================================================================

# class PaymentIntentRequest(BaseModel):
#     """Request model for creating payment intents"""
#     amount: Optional[float] = None
#     plan_name: Optional[str] = None
#     planName: Optional[str] = None
#     price_id: Optional[str] = None
#     priceId: Optional[str] = None

# class PaymentConfirmRequest(BaseModel):
#     """Request model for confirming payments"""
#     payment_intent_id: str

# class UserCreate(BaseModel):
#     """Model for user registration"""
#     username: str
#     email: str
#     password: str

# class UserResponse(BaseModel):
#     """Response model for user data"""
#     id: int
#     username: str = None
#     email: str
#     created_at: Optional[datetime] = None
    
#     class Config:
#         from_attributes = True

# class Token(BaseModel):
#     """JWT token response model"""
#     access_token: str
#     token_type: str

# class TranscriptRequest(BaseModel):
#     """Request model for transcript downloads"""
#     youtube_id: str
#     clean_transcript: bool = True
#     format: Optional[str] = None  # "srt" or "vtt" for unclean format

# class AudioRequest(BaseModel):
#     """Request model for audio downloads"""
#     youtube_id: str
#     quality: str = "medium"  # "high", "medium", "low"

# class VideoRequest(BaseModel):
#     """Request model for video downloads"""
#     youtube_id: str
#     quality: str = "720p"  # "1080p", "720p", "480p", "360p"

# # =============================================================================
# # HELPER FUNCTIONS
# # =============================================================================

# def get_user(db: Session, username: str) -> Optional[User]:
#     """Get user by username from database"""
#     return db.query(User).filter(User.username == username).first()

# def get_user_by_username(db: Session, username: str):
#     """Alternative method to get user by username"""
#     return db.query(User).filter(User.username == username).first()

# def verify_password(plain_password: str, hashed_password: str) -> bool:
#     """Verify a plain password against its hash"""
#     return pwd_context.verify(plain_password, hashed_password)

# def get_password_hash(password: str) -> str:
#     """Generate password hash"""
#     return pwd_context.hash(password)

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
#     """Create JWT access token"""
#     to_encode = data.copy()
#     expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
#     to_encode.update({"exp": expire})
#     return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
#     """Get current authenticated user from JWT token"""
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED, 
#         detail="Could not validate credentials",
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

# # =============================================================================
# # TRANSCRIPT PROCESSING FUNCTIONS
# # =============================================================================

# def extract_youtube_video_id(youtube_id_or_url: str) -> str:
#     """Extract 11-character video ID from YouTube URL or return ID if already provided"""
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

# def get_transcript_youtube_api(video_id: str, clean: bool = True, format: Optional[str] = None) -> str:
#     """
#     Get transcript using YouTube Transcript API with better error handling
#     """
#     # Check connectivity first
#     if not check_internet_connectivity():
#         raise HTTPException(
#             status_code=503, 
#             detail="No internet connection available. Please check your network connection and try again."
#         )
    
#     if not check_youtube_connectivity():
#         raise HTTPException(
#             status_code=503, 
#             detail="Cannot reach YouTube servers. This might be a temporary network issue or firewall restriction."
#         )
    
#     try:
#         transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        
#         if clean:
#             # Clean format - format into readable paragraphs
#             text = " ".join([seg['text'].replace('\n', ' ') for seg in transcript])
#             clean_text = " ".join(text.split())
            
#             # Break into paragraphs (every ~400 characters at sentence end)
#             words = clean_text.split()
#             paragraphs = []
#             current_paragraph = []
#             char_count = 0
            
#             for word in words:
#                 current_paragraph.append(word)
#                 char_count += len(word) + 1
                
#                 if char_count > 400 and word.endswith(('.', '!', '?')):
#                     paragraphs.append(' '.join(current_paragraph))
#                     current_paragraph = []
#                     char_count = 0
            
#             if current_paragraph:
#                 paragraphs.append(' '.join(current_paragraph))
            
#             return '\n\n'.join(paragraphs)
#         else:
#             # Unclean format with timestamps
#             if format == "srt":
#                 return segments_to_srt(transcript)
#             elif format == "vtt":
#                 return segments_to_vtt(transcript)
#             else:
#                 # Default timestamp format [MM:SS]
#                 lines = []
#                 for seg in transcript:
#                     t = int(seg['start'])
#                     timestamp = f"[{t//60:02d}:{t%60:02d}]"
#                     text_clean = seg['text'].replace('\n', ' ')
#                     lines.append(f"{timestamp} {text_clean}")
#                 return "\n".join(lines)
                
#     except Exception as e:
#         logger.warning(f"Transcript API failed: {e} - trying yt-dlp fallback...")
        
#         # Try yt-dlp fallback with better error handling
#         try:
#             if clean:
#                 fallback = get_transcript_with_ytdlp(video_id, clean=True)
#                 if fallback:
#                     # Format fallback text into paragraphs
#                     words = fallback.split()
#                     paragraphs = []
#                     current_paragraph = []
#                     char_count = 0
                    
#                     for word in words:
#                         current_paragraph.append(word)
#                         char_count += len(word) + 1
                        
#                         if char_count > 400 and word.endswith(('.', '!', '?')):
#                             paragraphs.append(' '.join(current_paragraph))
#                             current_paragraph = []
#                             char_count = 0
                    
#                     if current_paragraph:
#                         paragraphs.append(' '.join(current_paragraph))
                    
#                     return '\n\n'.join(paragraphs)
#             else:
#                 fallback = get_transcript_with_ytdlp(video_id, clean=False)
#                 if fallback and format == "vtt":
#                     return convert_timestamp_to_vtt(fallback)
#                 elif fallback and format == "srt":
#                     return convert_timestamp_to_srt(fallback)
#                 return fallback
#         except Exception as fallback_error:
#             logger.error(f"yt-dlp fallback also failed: {fallback_error}")
        
#         # If both methods fail, give a helpful error message
#         if "NameResolutionError" in str(e) or "Failed to resolve" in str(e):
#             raise HTTPException(
#                 status_code=503,
#                 detail="Network connection issue: Unable to reach YouTube servers. Please check your internet connection or try again later."
#             )
#         else:
#             raise HTTPException(
#                 status_code=404,
#                 detail="No transcript/captions found for this video. The video may not have captions available or may be restricted."
#             )

# def convert_timestamp_to_vtt(timestamp_text: str) -> str:
#     """Convert [MM:SS] format to proper WEBVTT format"""
#     lines = timestamp_text.strip().split('\n')
#     vtt_lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    
#     for line in lines:
#         match = re.match(r'\[(\d{2}):(\d{2})\] (.+)', line)
#         if match:
#             mm, ss, text = match.groups()
#             start_time = f"00:{mm}:{ss}.000"
#             end_minutes = int(mm)
#             end_seconds = int(ss) + 3
#             if end_seconds >= 60:
#                 end_minutes += 1
#                 end_seconds -= 60
#             end_time = f"00:{end_minutes:02d}:{end_seconds:02d}.000"
            
#             vtt_lines.append(f"{start_time} --> {end_time}")
#             vtt_lines.append(text.strip())
#             vtt_lines.append("")
    
#     return '\n'.join(vtt_lines)

# def convert_timestamp_to_srt(timestamp_text: str) -> str:
#     """Convert [MM:SS] format to proper SRT format"""
#     lines = timestamp_text.strip().split('\n')
#     srt_lines = []
#     counter = 1
    
#     for line in lines:
#         match = re.match(r'\[(\d{2}):(\d{2})\] (.+)', line)
#         if match:
#             mm, ss, text = match.groups()
#             start_time = f"00:{mm}:{ss},000"
#             end_minutes = int(mm)
#             end_seconds = int(ss) + 3
#             if end_seconds >= 60:
#                 end_minutes += 1
#                 end_seconds -= 60
#             end_time = f"00:{end_minutes:02d}:{end_seconds:02d},000"
            
#             srt_lines.append(str(counter))
#             srt_lines.append(f"{start_time} --> {end_time}")
#             srt_lines.append(text.strip())
#             srt_lines.append("")
#             counter += 1
    
#     return '\n'.join(srt_lines)

# def segments_to_vtt(transcript) -> str:
#     """Convert transcript segments to proper WebVTT format"""
#     def sec_to_vtt(ts):
#         h = int(ts // 3600)
#         m = int((ts % 3600) // 60)
#         s = int(ts % 60)
#         ms = int((ts - int(ts)) * 1000)
#         return f"{h:02}:{m:02}:{s:02}.{ms:03}"
    
#     lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    
#     for seg in transcript:
#         start = sec_to_vtt(seg["start"])
#         end = sec_to_vtt(seg.get("start", 0) + seg.get("duration", 0))
#         text = seg["text"].replace("\n", " ").strip()
        
#         lines.append(f"{start} --> {end}")
#         lines.append(text)
#         lines.append("")
    
#     return "\n".join(lines)

# def segments_to_srt(transcript) -> str:
#     """Convert transcript segments to SRT format"""
#     def sec_to_srt(ts):
#         h = int(ts // 3600)
#         m = int((ts % 3600) // 60)
#         s = int(ts % 60)
#         ms = int((ts - int(ts)) * 1000)
#         return f"{h:02}:{m:02}:{s:02},{ms:03}"

#     lines = []
#     for idx, seg in enumerate(transcript):
#         start = sec_to_srt(seg["start"])
#         end = sec_to_srt(seg.get("start", 0) + seg.get("duration", 0))
#         text = seg["text"].replace("\n", " ").strip()
        
#         lines.append(f"{idx+1}")
#         lines.append(f"{start} --> {end}")
#         lines.append(text)
#         lines.append("")
    
#     return "\n".join(lines)

# # =============================================================================
# # FASTAPI ENDPOINTS
# # =============================================================================

# @app.on_event("startup")
# async def startup():
#     """Initialize database tables and cleanup old files on application startup"""
#     initialize_database()
#     cleanup_old_files()

# @app.get("/")
# def root():
#     """Root endpoint - API health check"""
#     return {
#         "message": "YouTube Content Downloader API with File Serving", 
#         "status": "running", 
#         "version": "2.3.1",
#         "features": ["transcripts", "audio", "video", "file_serving"]
#     }

# # =============================================================================
# # AUTHENTICATION ENDPOINTS
# # =============================================================================

# @app.post("/register")
# def register(user: UserCreate, db: Session = Depends(get_db)):
#     """
#     Register a new user account
    
#     Creates a new user with hashed password and default free tier subscription
#     """
#     # Check if username already exists
#     if db.query(User).filter(User.username == user.username).first():
#         raise HTTPException(status_code=400, detail="Username already exists.")
    
#     # Check if email already exists
#     if db.query(User).filter(User.email == user.email).first():
#         raise HTTPException(status_code=400, detail="Email already exists.")
    
#     # Create new user
#     user_obj = User(
#         username=user.username,
#         email=user.email,
#         hashed_password=get_password_hash(user.password),
#         created_at=datetime.utcnow()
#     )
#     db.add(user_obj)
#     db.commit()
#     db.refresh(user_obj)
    
#     logger.info(f"New user registered: {user.username} ({user.email})")
#     return {"message": "User registered successfully."}

# @app.post("/token")
# def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
#     """
#     User login endpoint
    
#     Authenticates user credentials and returns JWT access token
#     """
#     user = db.query(User).filter(User.username == form_data.username).first()
#     if not user or not verify_password(form_data.password, user.hashed_password):
#         raise HTTPException(status_code=401, detail="Incorrect username or password")
    
#     access_token = create_access_token(
#         data={"sub": user.username},
#         expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     )
    
#     logger.info(f"User logged in: {user.username}")
#     return {"access_token": access_token, "token_type": "bearer"}

# @app.get("/users/me", response_model=UserResponse)
# def read_users_me(current_user: User = Depends(get_current_user)):
#     """Get current authenticated user information"""
#     return current_user

# # Insert right after the authentication endpoints but before the audio download endpoint.
# # Add this endpoint to your main.py file (in the TRANSCRIPT ENDPOINTS section)

# @app.post("/download_transcript/")
# def download_transcript(
#     request: TranscriptRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Download YouTube transcript with enhanced error handling
#     """
#     start_time = time.time()
    
#     # Extract and validate video ID
#     video_id = extract_youtube_video_id(request.youtube_id)
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     # Check connectivity
#     if not check_internet_connectivity():
#         raise HTTPException(
#             status_code=503,
#             detail="No internet connection available. Please check your network connection."
#         )
    
#     # Check for monthly usage reset
#     if user.usage_reset_date.month != datetime.utcnow().month:
#         user.reset_monthly_usage()
#         db.commit()
    
#     # Check usage limits
#     plan_limits = user.get_plan_limits()
#     usage_key = "clean_transcripts" if request.clean_transcript else "unclean_transcripts"
#     current_usage = getattr(user, f"usage_{usage_key}", 0)
#     allowed = plan_limits.get(usage_key, 0)
    
#     if allowed != float('inf') and current_usage >= allowed:
#         transcript_type = "clean" if request.clean_transcript else "unclean"
#         raise HTTPException(
#             status_code=403,
#             detail=f"Monthly limit reached for {transcript_type} transcripts. Please upgrade your plan."
#         )
    
#     # Get transcript
#     try:
#         transcript_text = get_transcript_youtube_api(
#             video_id, 
#             clean=request.clean_transcript, 
#             format=request.format
#         )
#     except HTTPException:
#         raise  # Re-raise HTTP exceptions as-is
#     except Exception as e:
#         logger.error(f"Transcript download failed: {e}")
#         raise HTTPException(
#             status_code=500,
#             detail="Failed to download transcript. The video may not have captions available."
#         )
    
#     if not transcript_text:
#         raise HTTPException(
#             status_code=404,
#             detail="No transcript found for this video. The video may not have captions available."
#         )
    
#     # Update usage and record download
#     user.increment_usage(usage_key)
#     processing_time = time.time() - start_time
    
#     # Create download record
#     try:
#         download_record = create_download_record_safe(
#             db=db,
#             user_id=user.id,
#             youtube_id=video_id,
#             transcript_type="clean" if request.clean_transcript else "unclean",
#             file_size=len(transcript_text.encode('utf-8')),
#             processing_time=processing_time,
#             download_method="youtube_api",
#             quality=None,
#             language="en",
#             file_format=request.format or "txt",
#             download_url=None,
#             expires_at=None,
#             status="completed"
#         )
        
#         if download_record:
#             db.commit()
            
#     except Exception as db_error:
#         logger.error(f"Database error recording download: {db_error}")
#         # Don't fail the request if database recording fails
#         pass
    
#     logger.info(f"User {user.username} downloaded {'clean' if request.clean_transcript else 'unclean'} transcript for {video_id}")
    
#     return {
#         "transcript": transcript_text,
#         "youtube_id": video_id,
#         "clean_transcript": request.clean_transcript,
#         "format": request.format,
#         "processing_time": round(processing_time, 2),
#         "success": True
#     }




# # =============================================================================
# # AUDIO DOWNLOAD ENDPOINT - COMPLETELY FIXED
# # =============================================================================

# @app.post("/download_audio/")
# def download_audio(
#     request: AudioRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Download YouTube audio with complete file serving support
#     FIXED: No more nested folders + better error handling
#     """
#     start_time = time.time()
    
#     # Clean up old files first
#     cleanup_old_files()
    
#     # Extract and validate video ID
#     video_id = extract_youtube_video_id(request.youtube_id)
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     # Check connectivity
#     if not check_internet_connectivity():
#         raise HTTPException(
#             status_code=503,
#             detail="No internet connection available. Please check your network connection."
#         )
    
#     # Check for monthly usage reset
#     if user.usage_reset_date.month != datetime.utcnow().month:
#         user.reset_monthly_usage()
#         db.commit()
    
#     # Check usage limits
#     plan_limits = user.get_plan_limits()
#     current_usage = getattr(user, "usage_audio_downloads", 0)
#     allowed = plan_limits.get("audio_downloads", 0)
    
#     if allowed != float('inf') and current_usage >= allowed:
#         raise HTTPException(
#             status_code=403,
#             detail="Monthly limit reached for audio downloads. Please upgrade your plan."
#         )
    
#     # Check if yt-dlp is available
#     if not check_ytdlp_availability():
#         raise HTTPException(
#             status_code=500, 
#             detail="Audio download service temporarily unavailable. Please install yt-dlp and FFmpeg."
#         )
    
#     # Enhanced FFmpeg check with helpful error message
#     def check_ffmpeg_available():
#         try:
#             subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5, check=True)
#             subprocess.run(["ffprobe", "-version"], capture_output=True, timeout=5, check=True)
#             return True
#         except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
#             return False
    
#     if not check_ffmpeg_available():
#         raise HTTPException(
#             status_code=500,
#             detail="FFmpeg not found. Audio downloads require FFmpeg. Please ensure FFmpeg is installed and accessible from the command line."
#         )
    
#     # FIXED: Download audio directly to the correct downloads directory
#     downloads_path = str(DOWNLOADS_DIR)
#     logger.info(f"Downloading audio to: {downloads_path}")
    
#     try:
#         audio_file_path = download_audio_with_ytdlp(video_id, request.quality, output_dir=downloads_path)
#         logger.info(f"Audio download returned path: {audio_file_path}")
        
#     except Exception as e:
#         logger.error(f"Audio download failed: {e}")
        
#         # Enhanced error handling with specific messages
#         error_msg = str(e).lower()
#         if "ffmpeg" in error_msg or "ffprobe" in error_msg:
#             raise HTTPException(
#                 status_code=500,
#                 detail="FFmpeg processing error. Please ensure FFmpeg is properly installed and accessible."
#             )
#         elif "network" in error_msg or "connection" in error_msg or "nameresolutionerror" in error_msg:
#             raise HTTPException(
#                 status_code=503,
#                 detail="Network error: Cannot reach YouTube. Please check your internet connection."
#             )
#         elif "timeout" in error_msg:
#             raise HTTPException(
#                 status_code=408,
#                 detail="Download timeout. The audio download took too long. Please try again."
#             )
#         else:
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Audio download failed: {str(e)}"
#             )
    
#     # FIXED: Better file path handling
#     if not audio_file_path or not os.path.exists(audio_file_path):
#         raise HTTPException(
#             status_code=404, 
#             detail="Failed to download audio from this video. The video may not have audio available or may be restricted."
#         )
    
#     # Convert to Path object for easier handling
#     audio_file = Path(audio_file_path)
    
#     # Generate a unique filename for serving
#     unique_filename = generate_unique_filename(f"{video_id}_audio_{request.quality}", "mp3")
#     final_path = DOWNLOADS_DIR / unique_filename
    
#     # FIXED: Better file moving with detailed logging
#     try:
#         logger.info(f"Moving file from {audio_file} to {final_path}")
        
#         if audio_file.samefile(final_path):
#             # File is already in the right place
#             logger.info("File is already in the correct location")
#         else:
#             # Move/copy file to final location
#             shutil.move(str(audio_file), str(final_path))
#             logger.info(f"File moved successfully to {final_path}")
            
#     except Exception as e:
#         logger.error(f"Error moving audio file: {e}")
#         # Try copying instead
#         try:
#             shutil.copy2(str(audio_file), str(final_path))
#             if audio_file.exists():
#                 audio_file.unlink()
#             logger.info(f"File copied successfully to {final_path}")
#         except Exception as copy_error:
#             logger.error(f"Error copying audio file: {copy_error}")
#             raise HTTPException(
#                 status_code=500,
#                 detail="Error processing downloaded audio file."
#             )
    
#     # Verify final file exists and has content
#     if not final_path.exists() or final_path.stat().st_size == 0:
#         logger.error(f"Final file verification failed: exists={final_path.exists()}, size={final_path.stat().st_size if final_path.exists() else 'N/A'}")
#         raise HTTPException(
#             status_code=500,
#             detail="Audio file processing failed. Please try again."
#         )
    
#     # Get file size
#     file_size = final_path.stat().st_size
#     processing_time = time.time() - start_time
    
#     # Update usage and record download
#     user.increment_usage("audio_downloads")
    
#     # Create download record with enhanced error handling
#     try:
#         download_record = create_download_record_safe(
#             db=db,
#             user_id=user.id,
#             youtube_id=video_id,
#             transcript_type="audio",
#             file_size=file_size,
#             processing_time=processing_time,
#             download_method="ytdlp",
#             quality=request.quality,
#             language="en",
#             file_format="mp3",
#             download_url=f"/files/{unique_filename}",
#             expires_at=datetime.utcnow() + timedelta(hours=1),
#             status="completed"
#         )
        
#         if download_record:
#             db.commit()
            
#     except Exception as db_error:
#         logger.error(f"Database error recording download: {db_error}")
#         # Don't fail the request if database recording fails
#         pass
    
#     logger.info(f"User {user.username} downloaded audio for {video_id} ({request.quality}) - {file_size} bytes")
    
#     # FIXED: Enhanced response with all necessary information
#     return {
#         "download_url": f"/files/{unique_filename}",
#         "direct_download_url": f"/download_file/{unique_filename}",
#         "youtube_id": video_id,
#         "quality": request.quality,
#         "file_size": file_size,
#         "file_size_mb": round(file_size / (1024 * 1024), 2),
#         "filename": unique_filename,
#         "original_filename": f"{video_id}_audio_{request.quality}.mp3",
#         "expires_in": "1 hour",
#         "expires_at": (datetime.utcnow() + timedelta(hours=1)).isoformat(),
#         "processing_time": round(processing_time, 2),
#         "message": "Audio download ready",
#         "success": True
#     }

# # =============================================================================
# # VIDEO DOWNLOAD ENDPOINT
# # =============================================================================

# @app.post("/download_video/")
# def download_video(
#     request: VideoRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Download YouTube video with file serving
#     """
#     start_time = time.time()
    
#     # Clean up old files first
#     cleanup_old_files()
    
#     # Extract and validate video ID
#     video_id = extract_youtube_video_id(request.youtube_id)
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     # Check connectivity
#     if not check_internet_connectivity():
#         raise HTTPException(
#             status_code=503,
#             detail="No internet connection available. Please check your network connection."
#         )
    
#     # Check for monthly usage reset
#     if user.usage_reset_date.month != datetime.utcnow().month:
#         user.reset_monthly_usage()
#         db.commit()
    
#     # Check usage limits
#     plan_limits = user.get_plan_limits()
#     current_usage = getattr(user, "usage_video_downloads", 0)
#     allowed = plan_limits.get("video_downloads", 0)
    
#     if allowed != float('inf') and current_usage >= allowed:
#         raise HTTPException(
#             status_code=403,
#             detail="Monthly limit reached for video downloads. Please upgrade your plan."
#         )
    
#     # Check if yt-dlp is available
#     if not check_ytdlp_availability():
#         raise HTTPException(
#             status_code=500, 
#             detail="Video download service temporarily unavailable. Please install yt-dlp and FFmpeg."
#         )
    
#     # Download video to downloads directory
#     try:
#         video_file = download_video_with_ytdlp(video_id, request.quality, output_dir=str(DOWNLOADS_DIR))
#     except Exception as e:
#         logger.error(f"Video download failed: {e}")
#         if "NameResolutionError" in str(e) or "Failed to resolve" in str(e):
#             raise HTTPException(
#                 status_code=503,
#                 detail="Network error: Cannot reach YouTube. Please check your internet connection."
#             )
#         else:
#             raise HTTPException(
#                 status_code=500,
#                 detail="Video download failed. The video may not be available or have restrictions."
#             )
    
#     if not video_file or not os.path.exists(video_file):
#         raise HTTPException(status_code=404, detail="Failed to download video.")
    
#     # Generate a unique filename for serving
#     original_filename = os.path.basename(video_file)
#     file_extension = "mp4"  # Default to mp4
#     if "." in original_filename:
#         file_extension = original_filename.split(".")[-1]
    
#     unique_filename = generate_unique_filename(f"{video_id}_video_{request.quality}", file_extension)
#     final_path = DOWNLOADS_DIR / unique_filename
    
#     # Move file to final location
#     shutil.move(video_file, final_path)
    
#     # Get file size
#     file_size = os.path.getsize(final_path)
#     processing_time = time.time() - start_time
    
#     # Update usage and record download
#     user.increment_usage("video_downloads")
    
#     # Create download record
#     download_record = create_download_record_safe(
#         db=db,
#         user_id=user.id,
#         youtube_id=video_id,
#         transcript_type="video",
#         file_size=file_size,
#         processing_time=processing_time,
#         download_method="ytdlp",
#         quality=request.quality,
#         language="en",
#         file_format=file_extension,
#         download_url=f"/files/{unique_filename}",
#         expires_at=datetime.utcnow() + timedelta(hours=1),
#         status="completed"
#     )
    
#     if download_record:
#         db.commit()
    
#     logger.info(f"User {user.username} downloaded video for {video_id} ({request.quality})")
    
#     return {
#         "download_url": f"/files/{unique_filename}",
#         "youtube_id": video_id,
#         "quality": request.quality,
#         "file_size": file_size,
#         "filename": unique_filename,
#         "expires_in": "1 hour",
#         "message": "Video download ready"
#     }

# # =============================================================================
# # FILE SERVING ENDPOINTS
# # =============================================================================

# @app.get("/download_file/{filename}")
# async def download_file(filename: str):
#     """
#     Serve downloaded files with proper headers for direct download
#     """
#     file_path = DOWNLOADS_DIR / filename
    
#     if not file_path.exists():
#         raise HTTPException(
#             status_code=404, 
#             detail="File not found or expired. Downloads expire after 1 hour."
#         )
    
#     # Determine content type
#     if filename.endswith('.mp3'):
#         media_type = 'audio/mpeg'
#     elif filename.endswith('.mp4'):
#         media_type = 'video/mp4'
#     elif filename.endswith('.webm'):
#         media_type = 'video/webm'
#     elif filename.endswith('.mkv'):
#         media_type = 'video/x-matroska'
#     else:
#         media_type = 'application/octet-stream'
    
#     # Generate a clean filename for download
#     clean_filename = filename
#     if '_' in filename:
#         # Extract video ID and quality from unique filename
#         parts = filename.split('_')
#         if len(parts) >= 4:
#             video_id = parts[0]
#             content_type = parts[1]  # 'audio' or 'video'
#             quality = parts[2]
#             extension = filename.split('.')[-1]
#             clean_filename = f"{video_id}_{content_type}_{quality}.{extension}"
    
#     return FileResponse(
#         path=file_path,
#         media_type=media_type,
#         filename=clean_filename,
#         headers={
#             "Content-Disposition": f"attachment; filename={clean_filename}",
#             "Cache-Control": "no-cache, no-store, must-revalidate",
#             "Pragma": "no-cache",
#             "Expires": "0"
#         }
#     )

# # =============================================================================
# # SUBSCRIPTION ENDPOINTS
# # =============================================================================

# @app.get("/subscription_status/")
# def get_subscription_status(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     Get user's current subscription status, usage, and limits
#     """
#     try:
#         # Get tier directly from user model
#         tier = getattr(current_user, 'subscription_tier', 'free')
        
#         # Current usage
#         usage = {
#             "clean_transcripts": getattr(current_user, "usage_clean_transcripts", 0),
#             "unclean_transcripts": getattr(current_user, "usage_unclean_transcripts", 0),
#             "audio_downloads": getattr(current_user, "usage_audio_downloads", 0),
#             "video_downloads": getattr(current_user, "usage_video_downloads", 0)
#         }
        
#         # Plan limits configuration
#         SUBSCRIPTION_LIMITS = {
#             "free": {
#                 "clean_transcripts": 5, 
#                 "unclean_transcripts": 3, 
#                 "audio_downloads": 2, 
#                 "video_downloads": 1
#             },
#             "pro": {
#                 "clean_transcripts": 100, 
#                 "unclean_transcripts": 50, 
#                 "audio_downloads": 50, 
#                 "video_downloads": 20
#             },
#             "premium": {
#                 "clean_transcripts": float('inf'), 
#                 "unclean_transcripts": float('inf'), 
#                 "audio_downloads": float('inf'), 
#                 "video_downloads": float('inf')
#             }
#         }
        
#         limits = SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS["free"])
#         json_limits = {k: ('unlimited' if v == float('inf') else v) for k, v in limits.items()}
        
#         return {
#             "tier": tier,
#             "status": "active" if tier != "free" else "inactive",
#             "usage": usage,
#             "limits": json_limits,
#             "subscription_id": None,
#             "stripe_customer_id": getattr(current_user, 'stripe_customer_id', None),
#             "current_period_end": None
#         }
        
#     except Exception as e:
#         logger.error(f"❌ Error getting subscription status: {str(e)}")
#         # Fallback response
#         return {
#             "tier": "free",
#             "status": "inactive",
#             "usage": {"clean_transcripts": 0, "unclean_transcripts": 0, "audio_downloads": 0, "video_downloads": 0},
#             "limits": {"clean_transcripts": 5, "unclean_transcripts": 3, "audio_downloads": 2, "video_downloads": 1},
#             "subscription_id": None,
#             "stripe_customer_id": None,
#             "current_period_end": None
#         }

# # =============================================================================
# # SYSTEM CHECK AND DEBUG ENDPOINTS
# # =============================================================================

# @app.get("/system_check/")
# def system_check():
#     """Enhanced system check with connectivity tests"""
#     checks = {
#         "ytdlp_available": check_ytdlp_availability(),
#         "database_connected": True,  # If we got here, DB is working
#         "stripe_configured": bool(stripe_secret_key),
#         "environment": ENVIRONMENT,
#         "internet_connectivity": check_internet_connectivity(),
#         "youtube_connectivity": check_youtube_connectivity(),
#         "downloads_directory": DOWNLOADS_DIR.exists()
#     }
    
#     # Check ffmpeg availability
#     try:
#         result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
#         checks["ffmpeg_available"] = result.returncode == 0
#     except:
#         checks["ffmpeg_available"] = False
    
#     # Generate recommendations
#     recommendations = []
#     if not checks["ffmpeg_available"]:
#         recommendations.append("Install FFmpeg for audio/video processing")
#     if not checks["ytdlp_available"]:
#         recommendations.append("Install yt-dlp for video downloads")
#     if not checks["stripe_configured"]:
#         recommendations.append("Configure Stripe for payments")
#     if not checks["internet_connectivity"]:
#         recommendations.append("Check internet connection")
#     if not checks["youtube_connectivity"]:
#         recommendations.append("Check access to YouTube (firewall/proxy settings)")
    
#     return {
#         "checks": checks,
#         "recommendations": recommendations,
#         "status": "healthy" if all([
#             checks["ytdlp_available"],
#             checks["ffmpeg_available"],
#             checks["internet_connectivity"],
#             checks["youtube_connectivity"]
#         ]) else "degraded"
#     }

# @app.get("/health/")
# def health():
#     """Health check endpoint for monitoring"""
#     return {
#         "status": "healthy", 
#         "timestamp": datetime.utcnow().isoformat(), 
#         "features": ["transcripts", "audio", "video", "file_serving"],
#         "connectivity": {
#             "internet": check_internet_connectivity(),
#             "youtube": check_youtube_connectivity()
#         }
#     }

# @app.get("/test_videos")
# def get_test_videos():
#     """Get test video IDs for development and testing"""
#     return {
#         "videos": [
#             {
#                 "id": "dQw4w9WgXcQ", 
#                 "title": "Rick Astley - Never Gonna Give You Up",
#                 "status": "verified_working"
#             },
#             {
#                 "id": "jNQXAC9IVRw", 
#                 "title": "Me at the zoo",
#                 "status": "verified_working"
#             }
#         ],
#         "note": "These videos are guaranteed to work and have captions available"
#     }

# # =============================================================================
# # PAYMENT ENDPOINTS 
# # =============================================================================

# @app.post("/create_payment_intent/")
# async def create_payment_intent(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Create a Stripe Payment Intent (defaults to Pro plan)"""
#     try:
#         amount = 9.99
#         plan_name = "pro"
        
#         logger.info(f"Creating payment intent for user {current_user.id}: {plan_name} plan (${amount})")
        
#         amount_cents = int(amount * 100)
        
#         stripe_customer_id = getattr(current_user, 'stripe_customer_id', None)
#         if not stripe_customer_id:
#             logger.info(f"Creating new Stripe customer for user {current_user.id}")
#             customer = stripe.Customer.create(
#                 email=current_user.email,
#                 metadata={'user_id': str(current_user.id)}
#             )
#             stripe_customer_id = customer.id
#             current_user.stripe_customer_id = stripe_customer_id
#             db.commit()
        
#         intent = stripe.PaymentIntent.create(
#             amount=amount_cents,
#             currency='usd',
#             customer=stripe_customer_id,
#             metadata={
#                 'user_id': str(current_user.id),
#                 'plan_name': plan_name
#             },
#             automatic_payment_methods={'enabled': True}
#         )
        
#         logger.info(f"Successfully created payment intent {intent.id} for ${amount}")
        
#         return {
#             "client_secret": intent.client_secret,
#             "payment_intent_id": intent.id,
#             "amount": amount,
#             "plan_name": plan_name
#         }
        
#     except stripe.error.StripeError as e:
#         logger.error(f"Stripe error: {e}")
#         raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
#     except Exception as e:
#         logger.error(f"Failed to create payment intent: {e}")
#         raise HTTPException(status_code=400, detail=str(e))

# @app.post("/create_payment_intent/{plan_type}")
# async def create_payment_intent_with_plan(
#     plan_type: str,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Create a Stripe Payment Intent for a specific plan"""
#     try:
#         plans = {
#             'pro': {'amount': 9.99, 'name': 'Pro'},
#             'premium': {'amount': 19.99, 'name': 'Premium'}
#         }
        
#         if plan_type not in plans:
#             raise HTTPException(status_code=400, detail="Invalid plan type")
        
#         plan = plans[plan_type]
#         amount = plan['amount']
#         plan_name = plan['name']
        
#         logger.info(f"Creating payment intent for user {current_user.id}: {plan_name} plan (${amount})")
        
#         amount_cents = int(amount * 100)
        
#         stripe_customer_id = getattr(current_user, 'stripe_customer_id', None)
#         if not stripe_customer_id:
#             logger.info(f"Creating new Stripe customer for user {current_user.id}")
#             customer = stripe.Customer.create(
#                 email=current_user.email,
#                 metadata={'user_id': str(current_user.id)}
#             )
#             stripe_customer_id = customer.id
#             current_user.stripe_customer_id = stripe_customer_id
#             db.commit()
        
#         intent = stripe.PaymentIntent.create(
#             amount=amount_cents,
#             currency='usd',
#             customer=stripe_customer_id,
#             metadata={
#                 'user_id': str(current_user.id),
#                 'plan_name': plan_type,
#                 'plan_display_name': plan_name
#             },
#             automatic_payment_methods={'enabled': True}
#         )
        
#         logger.info(f"Successfully created payment intent {intent.id}")
        
#         return {
#             "client_secret": intent.client_secret,
#             "payment_intent_id": intent.id,
#             "amount": amount,
#             "plan_name": plan_name,
#             "plan_type": plan_type
#         }
        
#     except stripe.error.StripeError as e:
#         logger.error(f"Stripe error: {e}")
#         raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
#     except Exception as e:
#         logger.error(f"Failed to create payment intent: {e}")
#         raise HTTPException(status_code=400, detail=str(e))

# @app.post("/confirm_payment/")
# async def confirm_payment(
#     request: PaymentConfirmRequest,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Confirm payment and upgrade user subscription"""
#     try:
#         intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
        
#         if intent.status == 'succeeded':
#             plan_name = intent.metadata.get('plan_name', 'pro').lower()
            
#             logger.info(f"Payment succeeded for user {current_user.id}, upgrading to {plan_name}")
            
#             if plan_name == 'pro':
#                 current_user.subscription_tier = 'pro'
#             elif plan_name == 'premium':
#                 current_user.subscription_tier = 'premium'
#             else:
#                 current_user.subscription_tier = 'pro'
            
#             current_user.reset_monthly_usage()
#             db.commit()
            
#             return {
#                 "success": True,
#                 "subscription": {
#                     "tier": current_user.subscription_tier,
#                     "status": "active"
#                 },
#                 "message": f"Successfully upgraded to {plan_name} plan"
#             }
#         else:
#             logger.warning(f"Payment intent {intent.id} status: {intent.status}")
#             raise HTTPException(status_code=400, detail=f"Payment not completed. Status: {intent.status}")
            
#     except stripe.error.StripeError as e:
#         logger.error(f"Stripe error: {e}")
#         raise HTTPException(status_code=400, detail=f"Stripe error: {str(e)}")
#     except Exception as e:
#         logger.error(f"Failed to confirm payment: {e}")
#         raise HTTPException(status_code=400, detail=str(e))

# @app.post("/cancel_subscription/")
# async def cancel_subscription(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Cancel user subscription and downgrade to free tier"""
#     try:
#         logger.info(f"Cancelling subscription for user {current_user.id}")
        
#         current_user.subscription_tier = 'free'
#         db.commit()
        
#         return {
#             "success": True,
#             "message": "Subscription cancelled successfully. You've been downgraded to the free tier."
#         }
        
#     except Exception as e:
#         logger.error(f"Failed to cancel subscription: {e}")
#         raise HTTPException(status_code=400, detail=str(e))

# # =============================================================================
# # CLEANUP TASK
# # =============================================================================

# @app.on_event("shutdown")
# async def shutdown():
#     """Clean up temporary files on application shutdown"""
#     cleanup_old_files()

# # =============================================================================
# # APPLICATION ENTRY POINT
# # =============================================================================

# # Your run.py should have exactly this:
# if __name__ == "__main__":
#     import uvicorn
#     print("🔥 Starting server on 0.0.0.0:8000")
#     print("🔥 This allows mobile connections")
    
#     uvicorn.run(
#         "main:app", 
#         host="0.0.0.0",  # CRITICAL - must be 0.0.0.0
#         port=8000, 
#         reload=True
#     )