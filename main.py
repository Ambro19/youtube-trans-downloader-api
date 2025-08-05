"""
YouTube Content Downloader API - UNICODE SAFE VERSION
========================================================
Fixed version with proper downloads path to user's Downloads folder
without any Unicode escape issues.
"""

from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
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
    version="2.4.1",
    description="A SaaS application for downloading YouTube transcripts, audio, and video"
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

# CORS Configuration
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
)

# Security Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

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

# 🔥 NEW: Isolated download functions that prevent any file conflicts
def download_audio_with_ytdlp_isolated(video_id: str, quality: str, output_dir: str, output_name: str) -> str:
    """Download audio with completely isolated temp files"""
    quality_settings = {
        "high": {"format": "bestaudio[ext=m4a]/bestaudio[ext=mp3]/bestaudio", "audio_quality": "0"},
        "medium": {"format": "bestaudio[abr<=128]/bestaudio[ext=m4a]/bestaudio", "audio_quality": "2"},
        "low": {"format": "bestaudio[abr<=96]/bestaudio[ext=m4a]/bestaudio", "audio_quality": "5"}
    }
    
    settings = quality_settings.get(quality, quality_settings["medium"])
    output_template = f"{output_name}.%(ext)s"
    
    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "mp3",
        "--audio-quality", settings["audio_quality"],
        "--format", settings["format"],
        "--output", output_template,
        "--no-playlist",
        "--no-warnings",
        "--prefer-ffmpeg",
        f"https://www.youtube.com/watch?v={video_id}"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=output_dir, check=False)
    
    if result.returncode != 0:
        logger.error(f"yt-dlp failed: {result.stderr}")
        raise Exception(f"Download failed: {result.stderr}")
    
    # Find the downloaded file
    output_path = Path(output_dir)
    for file_path in output_path.glob(f"{output_name}*"):
        if file_path.is_file() and file_path.stat().st_size > 1000:
            return str(file_path)
    
    raise Exception("No audio file found after download")

def download_video_with_ytdlp_isolated(video_id: str, quality: str, output_dir: str, output_name: str) -> str:
    """Download video with completely isolated temp files"""
    output_template = f"{output_name}.%(ext)s"
    
    cmd = [
        "yt-dlp",
        "--output", output_template,
        "--merge-output-format", "mp4",
        "--no-playlist",
        "--no-warnings",
        f"https://www.youtube.com/watch?v={video_id}"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=output_dir, check=False)
    
    if result.returncode != 0:
        logger.error(f"yt-dlp failed: {result.stderr}")
        raise Exception(f"Download failed: {result.stderr}")
    
    # Find the downloaded file
    output_path = Path(output_dir)
    for file_path in output_path.glob(f"{output_name}*"):
        if file_path.is_file() and file_path.stat().st_size > 10000:
            return str(file_path)
    
    raise Exception("No video file found after download")

# 🔥 NEW HELPER FUNCTIONS - Add these to your main.py

def find_working_audio_file(video_id: str, quality: str) -> Optional[Path]:
    """Find any existing working audio file for this video/quality combination"""
    try:
        # Search patterns in order of preference
        patterns = [
            f"{video_id}_audio_{quality}.mp3",           # Exact match
            f"{video_id}_audio_{quality}.m4a",           # Alternative format
            f"{video_id}_audio_{quality}*.mp3",          # With extra stuff
            f"{video_id}_audio_{quality}*.m4a",          # Alternative with extra
            f"{video_id}*audio*{quality}*.mp3",          # Fuzzy match
            f"{video_id}*audio*.mp3",                    # Any audio for this video
        ]
        
        for pattern in patterns:
            files = list(DOWNLOADS_DIR.glob(pattern))
            for file_path in files:
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    # Consider files > 100KB as potentially working
                    if file_size > 100000:  # 100KB minimum
                        logger.info(f"🔥 Found working audio file: {file_path.name} ({file_size} bytes)")
                        return file_path
        
        return None
        
    except Exception as e:
        logger.error(f"Error finding working audio file: {e}")
        return None

def find_working_video_file(video_id: str, quality: str) -> Optional[Path]:
    """Find any existing working video file for this video/quality combination"""
    try:
        # Search patterns in order of preference
        patterns = [
            f"{video_id}_video_{quality}.mp4",           # Exact match
            f"{video_id}_video_{quality}.webm",          # Alternative format
            f"{video_id}_video_{quality}*.mp4",          # With extra stuff
            f"{video_id}_video_{quality}*.webm",         # Alternative with extra
            f"{video_id}*video*{quality}*.*",            # Fuzzy match
            f"{video_id}*video*.*",                      # Any video for this video
        ]
        
        for pattern in patterns:
            files = list(DOWNLOADS_DIR.glob(pattern))
            for file_path in files:
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    # Consider files > 1MB as potentially working videos
                    if file_size > 1000000:  # 1MB minimum
                        logger.info(f"🔥 Found working video file: {file_path.name} ({file_size} bytes)")
                        return file_path
        
        return None
        
    except Exception as e:
        logger.error(f"Error finding working video file: {e}")
        return None


def cleanup_existing_files(video_id: str, file_type: str, quality: str):
    """Remove any existing corrupted or conflicting files"""
    try:
        # Patterns to clean up
        if file_type == "audio":
            patterns = [
                f"{video_id}_audio_{quality}*",
                f"{video_id}*audio*{quality}*",
            ]
        else:  # video
            patterns = [
                f"{video_id}_video_{quality}*",
                f"{video_id}*video*{quality}*",
            ]
        
        for pattern in patterns:
            for file_path in DOWNLOADS_DIR.glob(pattern):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    
                    # Remove corrupted files (too small) or files with (1) in name
                    is_corrupted = (file_type == "audio" and file_size < 100000) or (file_type == "video" and file_size < 1000000)
                    has_windows_rename = "(" in file_path.name and ")" in file_path.name
                    
                    if is_corrupted or has_windows_rename:
                        try:
                            logger.info(f"🔥 Removing {'corrupted' if is_corrupted else 'renamed'} file: {file_path.name}")
                            file_path.unlink()
                        except Exception as e:
                            logger.warning(f"Could not remove {file_path.name}: {e}")
                            
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")

# 🔥 ENHANCED cleanup function - Replace your existing one
def cleanup_old_files():
    """Enhanced cleanup that prevents all duplicate issues"""
    try:
        current_time = time.time()
        max_age = 2 * 3600  # 2 hours
        
        for file_path in DOWNLOADS_DIR.glob("*"):
            if file_path.is_file():
                filename = file_path.name
                file_size = file_path.stat().st_size
                file_age = current_time - file_path.stat().st_mtime
                
                # Remove old files
                if file_age > max_age:
                    try:
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {filename}")
                        continue
                    except:
                        pass
                
                # Remove Windows duplicate files (with parentheses)
                if "(" in filename and ")" in filename:
                    try:
                        file_path.unlink()
                        logger.info(f"Removed Windows duplicate: {filename}")
                        continue
                    except:
                        pass
                
                # Remove tiny corrupted files
                min_size = 100000 if "audio" in filename else 1000000  # 100KB for audio, 1MB for video
                if file_size < min_size and ("audio" in filename or "video" in filename):
                    try:
                        file_path.unlink()
                        logger.info(f"Removed corrupted file: {filename} ({file_size} bytes)")
                    except:
                        pass
                        
    except Exception as e:
        logger.warning(f"Error during cleanup: {e}")

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
            # Clean format - readable paragraphs
            text = " ".join([seg['text'].replace('\n', ' ') for seg in transcript])
            clean_text = " ".join(text.split())
            
            # Break into paragraphs
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
            # Unclean format with timestamps
            if format == "srt":
                result = segments_to_srt(transcript)
            elif format == "vtt":
                result = segments_to_vtt(transcript)
            else:
                # Default timestamp format [MM:SS]
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
        logger.error(f"❌ Exception type: {type(e).__name__}")
        
        # Try yt-dlp fallback
        try:
            logger.info("🔄 Trying yt-dlp fallback")
            if hasattr('transcript_utils', 'get_transcript_with_ytdlp'):
                fallback = get_transcript_with_ytdlp(video_id, clean=clean)
                if fallback:
                    logger.info(f"✅ yt-dlp fallback succeeded, length: {len(fallback)}")
                    return fallback
        except Exception as fallback_error:
            logger.error(f"❌ yt-dlp fallback failed: {fallback_error}")
        
        # If both methods fail
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
# FASTAPI ENDPOINTS
# =============================================================================

@app.on_event("startup")
async def startup():
    initialize_database()
    cleanup_old_files()

@app.get("/")
def root():
    return {
        "message": "YouTube Content Downloader API", 
        "status": "running", 
        "version": "2.4.1",
        "features": ["transcripts", "audio", "video", "downloads"],
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
    """Download YouTube transcript"""
    start_time = time.time()
    
    logger.info(f"🔥 Transcript request: {request.youtube_id}, clean: {request.clean_transcript}")
    
    # Extract and validate video ID
    video_id = extract_youtube_video_id(request.youtube_id)
    logger.info(f"🔥 Extracted video ID: {video_id}")
    
    if not video_id or len(video_id) != 11:
        logger.error(f"❌ Invalid video ID: {video_id}")
        raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
    # Check connectivity
    if not check_internet_connectivity():
        logger.error("❌ No internet connectivity")
        raise HTTPException(
            status_code=503,
            detail="No internet connection available. Please check your network connection."
        )
    
    logger.info("✅ Internet connectivity OK")
    
    # Check for monthly usage reset
    current_month = datetime.utcnow().month
    if hasattr(user, 'usage_reset_date') and user.usage_reset_date.month != current_month:
        if hasattr(user, 'reset_monthly_usage'):
            user.reset_monthly_usage()
            db.commit()
    
    # Check usage limits (simplified for now)
    usage_key = "clean_transcripts" if request.clean_transcript else "unclean_transcripts"
    current_usage = getattr(user, f"usage_{usage_key}", 0)
    logger.info(f"🔥 Current usage for {usage_key}: {current_usage}")
    
    # Basic limits check
    if current_usage >= 100:  # Basic limit
        transcript_type = "clean" if request.clean_transcript else "unclean"
        raise HTTPException(
            status_code=403,
            detail=f"Monthly limit reached for {transcript_type} transcripts."
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
        raise  # Re-raise HTTP exceptions as-is
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
    
    # Update usage (simplified)
    if hasattr(user, 'increment_usage'):
        user.increment_usage(usage_key)
        db.commit()
    
    processing_time = time.time() - start_time
    
    logger.info(f"✅ User {user.username} downloaded {'clean' if request.clean_transcript else 'unclean'} transcript for {video_id}")
    
    return {
        "transcript": transcript_text,
        "youtube_id": video_id,
        "clean_transcript": request.clean_transcript,
        "format": request.format,
        "processing_time": round(processing_time, 2),
        "success": True
    }


# # 🔥 DEFINITIVE FIX - Replace BOTH download endpoints in main.py with these versions
# # This fix prevents Windows from creating "(1)" renamed files completely

# @app.post("/download_audio/")
# def download_audio(
#     request: AudioRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     start_time = time.time()
    
#     video_id = extract_youtube_video_id(request.youtube_id)
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     if not check_internet_connectivity():
#         raise HTTPException(status_code=503, detail="No internet connection available.")
    
#     if not check_ytdlp_availability():
#         raise HTTPException(status_code=500, detail="Audio download service temporarily unavailable.")
    
#     # 🔥 CRITICAL FIX: Define final filename FIRST and ensure it's completely clean
#     final_filename = f"{video_id}_audio_{request.quality}.mp3"
#     final_path = DOWNLOADS_DIR / final_filename
    
#     # 🔥 STEP 1: Remove any existing files that could cause conflicts
#     logger.info(f"🔥 Preparing download for: {final_filename}")
    
#     # Remove ALL possible conflicting files
#     conflicting_patterns = [
#         f"{video_id}_audio_{request.quality}*",  # Any file with this base name
#         f"{video_id}_audio_*",                   # Any audio file for this video
#     ]
    
#     for pattern in conflicting_patterns:
#         for existing_file in DOWNLOADS_DIR.glob(pattern):
#             try:
#                 logger.info(f"🔥 Removing existing file: {existing_file.name}")
#                 existing_file.unlink()
#             except Exception as e:
#                 logger.warning(f"Could not remove {existing_file.name}: {e}")
    
#     # 🔥 STEP 2: Use completely isolated temp directory with unique name
#     temp_base = tempfile.gettempdir()
#     unique_temp_dir = Path(temp_base) / f"ytdl_{video_id}_{uuid.uuid4().hex[:8]}"
#     unique_temp_dir.mkdir(exist_ok=True)
    
#     try:
#         logger.info(f"🔥 Using isolated temp directory: {unique_temp_dir}")
        
#         # 🔥 STEP 3: Download to temp with unique filename to prevent ANY conflicts
#         temp_output_name = f"temp_{uuid.uuid4().hex[:8]}_audio"
        
#         # Modified download function call
#         audio_file_path = download_audio_with_ytdlp_isolated(
#             video_id, 
#             request.quality, 
#             output_dir=str(unique_temp_dir),
#             output_name=temp_output_name
#         )
        
#         if not audio_file_path or not os.path.exists(audio_file_path):
#             raise HTTPException(status_code=404, detail="Failed to download audio.")
        
#         # 🔥 STEP 4: Verify temp file is good
#         temp_file = Path(audio_file_path)
#         file_size = temp_file.stat().st_size
        
#         if file_size < 1000:
#             raise HTTPException(status_code=500, detail="Downloaded file appears to be corrupted.")
        
#         # 🔥 STEP 5: Copy to final location with atomic operation
#         logger.info(f"🔥 Moving verified file to final location: {final_path}")
        
#         # Ensure target directory exists
#         final_path.parent.mkdir(exist_ok=True)
        
#         # Copy file (this is safer than move for cross-filesystem operations)
#         shutil.copy2(str(temp_file), str(final_path))
        
#         # Verify final file
#         if not final_path.exists() or final_path.stat().st_size != file_size:
#             raise HTTPException(status_code=500, detail="File copy verification failed.")
        
#         logger.info(f"✅ Audio download successful: {final_path} ({file_size} bytes)")
        
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
#             "message": "Audio ready for download",
#             "success": True
#         }
        
#     finally:
#         # 🔥 CLEANUP: Always remove temp directory
#         try:
#             shutil.rmtree(str(unique_temp_dir), ignore_errors=True)
#         except:
#             pass


# 🔥 COMPLETE FINAL FIX - Replace these functions in your main.py
@app.post("/download_audio/")
def download_audio(
    request: AudioRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    start_time = time.time()
    
    video_id = extract_youtube_video_id(request.youtube_id)
    if not video_id or len(video_id) != 11:
        raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
    if not check_internet_connectivity():
        raise HTTPException(status_code=503, detail="No internet connection available.")
    
    if not check_ytdlp_availability():
        raise HTTPException(status_code=500, detail="Audio download service temporarily unavailable.")
    
    # 🔥 STEP 1: Get video info for title display
    video_info = None
    try:
        video_info = get_video_info(video_id)
        logger.info(f"🔥 Got video info: {video_info.get('title', 'Unknown') if video_info else 'Failed to get info'}")
    except Exception as e:
        logger.warning(f"Could not get video info: {e}")
    
    # 🔥 STEP 2: Define expected filename
    final_filename = f"{video_id}_audio_{request.quality}.mp3"
    final_path = DOWNLOADS_DIR / final_filename
    
    # 🔥 STEP 3: Check if a WORKING file already exists
    existing_working_file = find_working_audio_file(video_id, request.quality)
    
    if existing_working_file:
        logger.info(f"🔥 Found existing working file: {existing_working_file}")
        file_size = existing_working_file.stat().st_size
        
        # If the existing file is not in the expected location, copy it there
        if existing_working_file != final_path:
            logger.info(f"🔥 Moving existing working file to standard location")
            try:
                # Remove any corrupted file at the target location
                if final_path.exists():
                    final_path.unlink()
                
                # Copy the working file to the standard location
                shutil.copy2(str(existing_working_file), str(final_path))
                
                # Remove the old file to prevent duplicates
                if existing_working_file != final_path:
                    existing_working_file.unlink()
                    
                logger.info(f"✅ Moved working file to: {final_path}")
            except Exception as e:
                logger.error(f"Error moving file: {e}")
                # If move fails, just use the existing file
                final_path = existing_working_file
                final_filename = existing_working_file.name
        
        processing_time = time.time() - start_time
        
        # Return the existing working file
        return {
            "download_url": f"/files/{final_filename}",
            "direct_download_url": f"/download_file/{final_filename}",
            "youtube_id": video_id,
            "quality": request.quality,
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "filename": final_filename,
            "local_path": str(final_path),
            "processing_time": round(processing_time, 2),
            "message": "Audio ready for download (existing file)",
            "success": True,
            "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
            "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
            "duration": video_info.get('duration', 0) if video_info else 0
        }
    
    # 🔥 STEP 4: No working file exists, so download a new one
    logger.info(f"🔥 No working file found, downloading new audio for {video_id}")
    
    # Remove any existing files (including corrupted ones)
    cleanup_existing_files(video_id, "audio", request.quality)
    
    # Download to temp directory first
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            logger.info(f"🔥 Downloading to temp: {temp_dir}")
            
            audio_file_path = download_audio_with_ytdlp(video_id, request.quality, output_dir=temp_dir)
            
            if not audio_file_path or not os.path.exists(audio_file_path):
                raise HTTPException(status_code=404, detail="Failed to download audio.")
            
            temp_file = Path(audio_file_path)
            file_size = temp_file.stat().st_size
            
            # Verify file is not corrupted
            if file_size < 1000:
                raise HTTPException(status_code=500, detail="Downloaded file appears to be corrupted.")
            
            # Move to final location
            logger.info(f"🔥 Moving completed download to: {final_path}")
            shutil.copy2(str(temp_file), str(final_path))
            
            # Verify final file
            if not final_path.exists() or final_path.stat().st_size != file_size:
                raise HTTPException(status_code=500, detail="File copy verification failed.")
            
            logger.info(f"✅ Audio download successful: {final_path} ({file_size} bytes)")
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            raise HTTPException(status_code=500, detail=f"Audio download failed: {str(e)}")
    
    processing_time = time.time() - start_time
    
    return {
        "download_url": f"/files/{final_filename}",
        "direct_download_url": f"/download_file/{final_filename}",
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
        "duration": video_info.get('duration', 0) if video_info else 0
    }


# @app.post("/download_video/")
# def download_video(
#     request: VideoRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     start_time = time.time()
    
#     video_id = extract_youtube_video_id(request.youtube_id)
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
#     if not check_internet_connectivity():
#         raise HTTPException(status_code=503, detail="No internet connection available.")
    
#     if not check_ytdlp_availability():
#         raise HTTPException(status_code=500, detail="Video download service unavailable.")
    
#     # 🔥 CRITICAL FIX: Define final filename FIRST and ensure it's completely clean
#     final_filename = f"{video_id}_video_{request.quality}.mp4"
#     final_path = DOWNLOADS_DIR / final_filename
    
#     # 🔥 STEP 1: Remove any existing files that could cause conflicts
#     logger.info(f"🔥 Preparing download for: {final_filename}")
    
#     # Remove ALL possible conflicting files
#     conflicting_patterns = [
#         f"{video_id}_video_{request.quality}*",  # Any file with this base name
#         f"{video_id}_video_*",                   # Any video file for this video
#         f"{video_id}*.mp4",                      # Any mp4 for this video
#         f"{video_id}*.webm",                     # Any webm for this video
#     ]
    
#     for pattern in conflicting_patterns:
#         for existing_file in DOWNLOADS_DIR.glob(pattern):
#             try:
#                 logger.info(f"🔥 Removing existing file: {existing_file.name}")
#                 existing_file.unlink()
#             except Exception as e:
#                 logger.warning(f"Could not remove {existing_file.name}: {e}")
    
#     # 🔥 STEP 2: Use completely isolated temp directory with unique name
#     temp_base = tempfile.gettempdir()
#     unique_temp_dir = Path(temp_base) / f"ytdl_{video_id}_{uuid.uuid4().hex[:8]}"
#     unique_temp_dir.mkdir(exist_ok=True)
    
#     try:
#         logger.info(f"🔥 Using isolated temp directory: {unique_temp_dir}")
        
#         # 🔥 STEP 3: Download to temp with unique filename to prevent ANY conflicts
#         temp_output_name = f"temp_{uuid.uuid4().hex[:8]}_video"
        
#         # Modified download function call
#         video_file_path = download_video_with_ytdlp_isolated(
#             video_id, 
#             request.quality, 
#             output_dir=str(unique_temp_dir),
#             output_name=temp_output_name
#         )
        
#         if not video_file_path or not os.path.exists(video_file_path):
#             raise HTTPException(status_code=404, detail="Failed to download video.")
        
#         # 🔥 STEP 4: Verify temp file is good
#         temp_file = Path(video_file_path)
#         file_size = temp_file.stat().st_size
        
#         if file_size < 10000:
#             raise HTTPException(status_code=500, detail="Downloaded video appears to be corrupted.")
        
#         # 🔥 STEP 5: Copy to final location with atomic operation
#         logger.info(f"🔥 Moving verified file to final location: {final_path}")
        
#         # Ensure target directory exists
#         final_path.parent.mkdir(exist_ok=True)
        
#         # Copy file (this is safer than move for cross-filesystem operations)
#         shutil.copy2(str(temp_file), str(final_path))
        
#         # Verify final file
#         if not final_path.exists() or final_path.stat().st_size != file_size:
#             raise HTTPException(status_code=500, detail="File copy verification failed.")
        
#         logger.info(f"✅ Video download successful: {final_path} ({file_size} bytes)")
        
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
#             "message": "Video ready for download",
#             "success": True
#         }
        
#     finally:
#         # 🔥 CLEANUP: Always remove temp directory
#         try:
#             shutil.rmtree(str(unique_temp_dir), ignore_errors=True)
#         except:
#             pass


@app.post("/download_video/")
def download_video(
    request: VideoRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    start_time = time.time()
    
    video_id = extract_youtube_video_id(request.youtube_id)
    if not video_id or len(video_id) != 11:
        raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
    if not check_internet_connectivity():
        raise HTTPException(status_code=503, detail="No internet connection available.")
    
    if not check_ytdlp_availability():
        raise HTTPException(status_code=500, detail="Video download service unavailable.")
    
    # 🔥 STEP 1: Get video info for title display
    video_info = None
    try:
        video_info = get_video_info(video_id)
        logger.info(f"🔥 Got video info: {video_info.get('title', 'Unknown') if video_info else 'Failed to get info'}")
    except Exception as e:
        logger.warning(f"Could not get video info: {e}")
    
    # 🔥 STEP 2: Define expected filename
    final_filename = f"{video_id}_video_{request.quality}.mp4"
    final_path = DOWNLOADS_DIR / final_filename
    
    # 🔥 STEP 3: Check if a WORKING file already exists
    existing_working_file = find_working_video_file(video_id, request.quality)
    
    if existing_working_file:
        logger.info(f"🔥 Found existing working file: {existing_working_file}")
        file_size = existing_working_file.stat().st_size
        
        # If the existing file is not in the expected location, copy it there
        if existing_working_file != final_path:
            logger.info(f"🔥 Moving existing working file to standard location")
            try:
                # Remove any corrupted file at the target location
                if final_path.exists():
                    final_path.unlink()
                
                # Copy the working file to the standard location
                shutil.copy2(str(existing_working_file), str(final_path))
                
                # Remove the old file to prevent duplicates
                if existing_working_file != final_path:
                    existing_working_file.unlink()
                    
                logger.info(f"✅ Moved working file to: {final_path}")
            except Exception as e:
                logger.error(f"Error moving file: {e}")
                # If move fails, just use the existing file
                final_path = existing_working_file
                final_filename = existing_working_file.name
        
        processing_time = time.time() - start_time
        
        # Return the existing working file
        return {
            "download_url": f"/files/{final_filename}",
            "direct_download_url": f"/download_file/{final_filename}",
            "youtube_id": video_id,
            "quality": request.quality,
            "file_size": file_size,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "filename": final_filename,
            "local_path": str(final_path),
            "processing_time": round(processing_time, 2),
            "message": "Video ready for download (existing file)",
            "success": True,
            "title": video_info.get('title', 'Unknown Title') if video_info else 'Unknown Title',
            "uploader": video_info.get('uploader', 'Unknown') if video_info else 'Unknown',
            "duration": video_info.get('duration', 0) if video_info else 0
        }
    
    # 🔥 STEP 4: No working file exists, so download a new one
    logger.info(f"🔥 No working file found, downloading new video for {video_id}")
    
    # Remove any existing files (including corrupted ones)
    cleanup_existing_files(video_id, "video", request.quality)
    
    # Download to temp directory first
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            logger.info(f"🔥 Downloading to temp: {temp_dir}")
            
            video_file_path = download_video_with_ytdlp(video_id, request.quality, output_dir=temp_dir)
            
            if not video_file_path or not os.path.exists(video_file_path):
                raise HTTPException(status_code=404, detail="Failed to download video.")
            
            temp_file = Path(video_file_path)
            file_size = temp_file.stat().st_size
            
            # Verify file is not corrupted
            if file_size < 10000:
                raise HTTPException(status_code=500, detail="Downloaded video appears to be corrupted.")
            
            # Determine extension from downloaded file
            original_ext = temp_file.suffix
            if original_ext:
                final_filename = f"{video_id}_video_{request.quality}{original_ext}"
                final_path = DOWNLOADS_DIR / final_filename
            
            # Move to final location
            logger.info(f"🔥 Moving completed download to: {final_path}")
            shutil.copy2(str(temp_file), str(final_path))
            
            # Verify final file
            if not final_path.exists() or final_path.stat().st_size != file_size:
                raise HTTPException(status_code=500, detail="File copy verification failed.")
            
            logger.info(f"✅ Video download successful: {final_path} ({file_size} bytes)")
            
        except Exception as e:
            logger.error(f"❌ Download failed: {e}")
            raise HTTPException(status_code=500, detail=f"Video download failed: {str(e)}")
    
    processing_time = time.time() - start_time
    
    return {
        "download_url": f"/files/{final_filename}",
        "direct_download_url": f"/download_file/{final_filename}",
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
        "duration": video_info.get('duration', 0) if video_info else 0
    }


@app.get("/subscription_status/")
def get_subscription_status(current_user: User = Depends(get_current_user)):
    try:
        tier = getattr(current_user, 'subscription_tier', 'free')
        
        usage = {
            "clean_transcripts": getattr(current_user, "usage_clean_transcripts", 0),
            "unclean_transcripts": getattr(current_user, "usage_unclean_transcripts", 0),
            "audio_downloads": getattr(current_user, "usage_audio_downloads", 0),
            "video_downloads": getattr(current_user, "usage_video_downloads", 0)
        }
        
        SUBSCRIPTION_LIMITS = {
            "free": {"clean_transcripts": 5, "unclean_transcripts": 3, "audio_downloads": 2, "video_downloads": 1},
            "pro": {"clean_transcripts": 100, "unclean_transcripts": 50, "audio_downloads": 50, "video_downloads": 20},
            "premium": {"clean_transcripts": float('inf'), "unclean_transcripts": float('inf'), "audio_downloads": float('inf'), "video_downloads": float('inf')}
        }
        
        limits = SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS["free"])
        json_limits = {k: ('unlimited' if v == float('inf') else v) for k, v in limits.items()}
        
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

@app.get("/health/")
def health():
    return {
        "status": "healthy", 
        "timestamp": datetime.utcnow().isoformat(),
        "downloads_path": str(DOWNLOADS_DIR),
        "connectivity": {
            "internet": check_internet_connectivity(),
            "youtube": check_youtube_connectivity()
        }
    }

@app.get("/test_videos")
def get_test_videos():
    """Get test video IDs for development and testing"""
    return {
        "videos": [
            {
                "id": "dQw4w9WgXcQ", 
                "title": "Rick Astley - Never Gonna Give You Up",
                "status": "verified_working"
            },
            {
                "id": "jNQXAC9IVRw", 
                "title": "Me at the zoo",
                "status": "verified_working"
            }
        ],
        "note": "These videos are guaranteed to work and have captions available"
    }

if __name__ == "__main__":
    import uvicorn
    print("🔥 Starting server on 0.0.0.0:8000")
    print("🔥 Mobile access enabled")
    print(f"🔥 Downloads folder: {str(DOWNLOADS_DIR)}")
    
    uvicorn.run(
        "main:app", 
        host="0.0.0.0",
        port=8000, 
        reload=True
    )


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