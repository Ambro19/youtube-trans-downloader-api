##======= DO NOT CHANGE THIS FINAL MAIN.PY ACTIVATED: 7/16/25 @ 10:00 PM. USE AND KEEP IT ==

# # main.py (COMPLETE PATCH) - unified usage, robust /subscription_status/, download history ready
# PATCHED fallback: yt-dlp retry + robust .json3 loader. Replaces get_transcript_with_ytdlp()
# PATCHED: Robust yt-dlp fallback + fix VTT fallback + proper TranscriptDownload import

from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi
from database import TranscriptDownload  # ✅ Fix NameError in /download_transcript

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
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

#================
# PATCH: Add endpoints for admin/debugging transcript extraction/logs

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from database import get_db, TranscriptDownload # ✅ CORRECT location for model
from transcript_utils import get_transcript_with_ytdlp  # ✅ Fixed circular import



# --- Import all database models ---
from database import engine, SessionLocal, get_db, create_tables
from database import engine, SessionLocal, get_db  # Make sure this points to your db
from models import User, create_tables  # Use your models.py User model!
from database import SessionLocal, get_db  # Make sure this points to your db

load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("youtube_trans_downloader.main")

# --- App & CORS ---
app = FastAPI(title="YouTubeTransDownloader API", version="2.0.0 (user-usage)")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
allowed_origins = [
    "http://localhost:3000", "http://127.0.0.1:3000", FRONTEND_URL
] if ENVIRONMENT != "production" else [
    "https://youtube-trans-downloader-api.onrender.com", FRONTEND_URL
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins, allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# --- Security ---
SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- Helper Functions ---
def get_user(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()

def get_user_by_username(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta if expires_delta else timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials",
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

# --- Pydantic Models ---
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
    format: Optional[str] = None  # "srt" or "vtt" when unclean

# --- Transcript logic ---
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

# Replace these functions in your main.py
def get_transcript_youtube_api(video_id: str, clean: bool = True, format: Optional[str] = None) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        
        if clean:
            # Clean format - format into readable paragraphs
            text = " ".join([seg['text'].replace('\n', ' ') for seg in transcript])
            clean_text = " ".join(text.split())  # Remove extra spaces
            
            # Break into paragraphs (every ~500 characters at sentence end)
            words = clean_text.split()
            paragraphs = []
            current_paragraph = []
            char_count = 0
            
            for word in words:
                current_paragraph.append(word)
                char_count += len(word) + 1
                
                # Break into new paragraph if we've reached ~500 chars and hit sentence end
                if char_count > 400 and word.endswith(('.', '!', '?')):
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
                    char_count = 0
            
            # Add remaining words
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
            
            return '\n\n'.join(paragraphs)
        else:
            # Unclean format with timestamps
            if format == "srt":
                return segments_to_srt(transcript)
            elif format == "vtt":
                return segments_to_vtt(transcript)
            else:
                # Default timestamp format [MM:SS]
                lines = []
                for seg in transcript:
                    t = int(seg['start'])
                    timestamp = f"[{t//60:02d}:{t%60:02d}]"
                    text_clean = seg['text'].replace('\n', ' ')
                    lines.append(f"{timestamp} {text_clean}")
                return "\n".join(lines)
    except Exception as e:
        logger.warning(f"Transcript API failed: {e} - trying yt-dlp fallback...")
        # For fallback, we need to handle the format properly
        if clean:
            fallback = get_transcript_with_ytdlp(video_id, clean=True)
            if fallback:
                # Format fallback text into paragraphs too
                words = fallback.split()
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
                
                return '\n\n'.join(paragraphs)
        else:
            # For unclean fallback, we need to convert to proper format
            fallback = get_transcript_with_ytdlp(video_id, clean=False)
            if fallback and format == "vtt":
                # Convert [MM:SS] format to proper VTT
                return convert_timestamp_to_vtt(fallback)
            elif fallback and format == "srt":
                # Convert [MM:SS] format to proper SRT
                return convert_timestamp_to_srt(fallback)
            return fallback
        
        logger.error(f"All transcript methods failed for video: {video_id}")
        return None


# # Also update the get_transcript_youtube_api function to handle formats better
# def get_transcript_youtube_api(video_id: str, clean: bool = True, format: Optional[str] = None) -> str:
#     try:
#         from youtube_transcript_api import YouTubeTranscriptApi
#         transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        
#         if clean:
#             # Clean format - just text
#             text = " ".join([seg['text'].replace('\n', ' ') for seg in transcript])
#             return " ".join(text.split())
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
#         fallback = get_transcript_with_ytdlp(video_id, clean)
#         if fallback:
#             return fallback
#         else:
#             logger.error(f"yt-dlp fallback also failed for video: {video_id}")
#             return None

def convert_timestamp_to_vtt(timestamp_text: str) -> str:
    """Convert [MM:SS] format to proper WEBVTT format"""
    lines = timestamp_text.strip().split('\n')
    vtt_lines = [
        "WEBVTT",
        "Kind: captions", 
        "Language: en",
        ""
    ]
    
    for line in lines:
        # Match [MM:SS] pattern
        match = re.match(r'\[(\d{2}):(\d{2})\] (.+)', line)
        if match:
            mm, ss, text = match.groups()
            start_time = f"00:{mm}:{ss}.000"
            # Estimate end time (add ~3 seconds)
            end_minutes = int(mm)
            end_seconds = int(ss) + 3
            if end_seconds >= 60:
                end_minutes += 1
                end_seconds -= 60
            end_time = f"00:{end_minutes:02d}:{end_seconds:02d}.000"
            
            vtt_lines.append(f"{start_time} --> {end_time}")
            vtt_lines.append(text.strip())
            vtt_lines.append("")
    
    return '\n'.join(vtt_lines)

def convert_timestamp_to_srt(timestamp_text: str) -> str:
    """Convert [MM:SS] format to proper SRT format"""
    lines = timestamp_text.strip().split('\n')
    srt_lines = []
    counter = 1
    
    for line in lines:
        # Match [MM:SS] pattern
        match = re.match(r'\[(\d{2}):(\d{2})\] (.+)', line)
        if match:
            mm, ss, text = match.groups()
            start_time = f"00:{mm}:{ss},000"
            # Estimate end time (add ~3 seconds)
            end_minutes = int(mm)
            end_seconds = int(ss) + 3
            if end_seconds >= 60:
                end_minutes += 1
                end_seconds -= 60
            end_time = f"00:{end_minutes:02d}:{end_seconds:02d},000"
            
            srt_lines.append(str(counter))
            srt_lines.append(f"{start_time} --> {end_time}")
            srt_lines.append(text.strip())
            srt_lines.append("")
            counter += 1
    
    return '\n'.join(srt_lines)


# fallback for unavailable transcript
def get_transcript_with_ytdlp(video_id: str, clean=True, retries=3, wait_sec=1) -> str:
    try:
        output_vtt = f"{video_id}.en.vtt"
        output_json3 = f"{video_id}.en.json3"
        url = f"https://www.youtube.com/watch?v={video_id}"

        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-sub",
            "--sub-lang", "en",
            "--sub-format", "json3/vtt",
            "--output", "%(id)s",
            url
        ]
        subprocess.run(cmd, capture_output=True, check=False)

        for _ in range(retries):
            if os.path.exists(output_json3):
                with open(output_json3, encoding="utf8") as f:
                    data = json.load(f)
                os.remove(output_json3)

                blocks = []
                for e in data.get("events", []):
                    if "segs" in e and "tStartMs" in e:
                        text = ''.join(s.get("utf8", '') for s in e["segs"] if s.get("utf8"))
                        if text.strip():
                            sec = int(e["tStartMs"] // 1000)
                            ts = f"[{sec//60:02d}:{sec%60:02d}]"
                            blocks.append(f"{ts} {text.strip()}" if not clean else text.strip())
                return "\n".join(blocks) if blocks else None

            time.sleep(wait_sec)

        if os.path.exists(output_vtt):
            with open(output_vtt, encoding="utf8") as f:
                vtt_raw = f.read()
            os.remove(output_vtt)
            return vtt_raw.strip() if vtt_raw else None

        raise FileNotFoundError(f"yt-dlp output not found: {output_json3} or .vtt")

    except Exception as e:
        logger.error(f"yt-dlp fallback failed: {e}")
        return None


def segments_to_vtt(transcript):
    """Convert segments to proper WebVTT format."""
    def sec_to_vtt(ts):
        h = int(ts // 3600)
        m = int((ts % 3600) // 60)
        s = int(ts % 60)
        ms = int((ts - int(ts)) * 1000)
        return f"{h:02}:{m:02}:{s:02}.{ms:03}"
    
    lines = [
        "WEBVTT",
        "Kind: captions", 
        "Language: en",
        ""
    ]
    
    for seg in transcript:
        start = sec_to_vtt(seg["start"])
        end = sec_to_vtt(seg.get("start", 0) + seg.get("duration", 0))
        text = seg["text"].replace("\n", " ").strip()
        
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    
    return "\n".join(lines)

def segments_to_srt(transcript):
    """Convert segments to SRT format."""
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

# # Updated sections for main.py - Fix VTT format and improve transcript handling

# def segments_to_vtt(transcript):
#     """Convert segments to proper WebVTT format."""
#     def sec_to_vtt(ts):
#         h = int(ts // 3600)
#         m = int((ts % 3600) // 60)
#         s = int(ts % 60)
#         ms = int((ts - int(ts)) * 1000)
#         return f"{h:02}:{m:02}:{s:02}.{ms:03}"
    
#     lines = [
#         "WEBVTT",
#         "Kind: captions", 
#         "Language: en",
#         ""  # Empty line after headers
#     ]
    
#     for seg in transcript:
#         start = sec_to_vtt(seg["start"])
#         end = sec_to_vtt(seg.get("start", 0) + seg.get("duration", 0))
#         text = seg["text"].replace("\n", " ").strip()
        
#         # Add timestamp line
#         lines.append(f"{start} --> {end}")
#         # Add text line
#         lines.append(text)
#         # Add empty line between cue blocks
#         lines.append("")
    
#     return "\n".join(lines)

# def segments_to_srt(transcript):
#     """Convert segments to SRT format."""
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
#         lines.append("")  # Empty line between subtitles
    
#     return "\n".join(lines)


# --- Usage keys mapping ---
USAGE_KEYS = {
    True: "clean_transcripts",
    False: "unclean_transcripts"
}

TRANSCRIPT_TYPE_MAP = {
    True: "clean",
    False: "unclean"
}

# --- FastAPI Endpoints ---

@app.on_event("startup")
def startup():
    create_tables(engine)

@app.get("/")
def root():
    return {"message": "YouTube Transcript Downloader API", "status": "running"}

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
    video_id = extract_youtube_video_id(request.youtube_id)
    if not video_id or len(video_id) != 11:
        raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    usage_key = USAGE_KEYS[request.clean_transcript]
    transcript_type = TRANSCRIPT_TYPE_MAP[request.clean_transcript]
    if user.usage_reset_date.month != datetime.utcnow().month:
        user.reset_monthly_usage()
        db.commit()
    plan_limits = user.get_plan_limits()
    current_usage = getattr(user, f"usage_{usage_key}", 0)
    allowed = plan_limits[usage_key]
    if allowed != float('inf') and current_usage >= allowed:
        raise HTTPException(
            status_code=403,
            detail=f"Monthly limit reached for {usage_key.replace('_',' ')}. Please upgrade your plan."
        )
    transcript = get_transcript_youtube_api(
        video_id,
        clean=request.clean_transcript,
        format=request.format if not request.clean_transcript else None
    )
    if not transcript or len(transcript.strip()) < 10:
        raise HTTPException(status_code=404, detail="No transcript/captions found for this video.")
    user.increment_usage(usage_key)
    db.add(TranscriptDownload(
        user_id=user.id,
        youtube_id=video_id,
        transcript_type=transcript_type,
        created_at=datetime.utcnow()
    ))
    db.commit()
    logger.info(f"User {user.username} downloaded transcript for {video_id} ({usage_key})")
    return {
        "transcript": transcript,
        "youtube_id": video_id,
        "message": "Transcript downloaded successfully"
    }

@app.get("/subscription_status/")
def get_subscription_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # --- Try to load subscription from DB (graceful fallback if not found) ---
        try:
            subscription = db.query(Subscription).filter(
                Subscription.user_id == current_user.id
            ).first()
        except Exception as e:
            subscription = None

        # --- Derive current tier/status/expiry from User+Subscription ---
        now = datetime.utcnow()
        if not subscription:
            tier = getattr(current_user, 'subscription_tier', 'free')
            status = "inactive"
            expiry_date = None
        else:
            if hasattr(subscription, 'expiry_date') and subscription.expiry_date and subscription.expiry_date < now:
                tier = "free"
                status = "expired"
                expiry_date = subscription.expiry_date
            else:
                tier = getattr(subscription, 'tier', 'free') or 'free'
                status = getattr(subscription, 'status', 'active') if tier != "free" else "inactive"
                expiry_date = getattr(subscription, 'expiry_date', None)

        # Usage and limits (ALWAYS provide these keys for frontend!)
        usage = {
            "clean_transcripts": getattr(current_user, "usage_clean_transcripts", 0),
            "unclean_transcripts": getattr(current_user, "usage_unclean_transcripts", 0),
            "audio_downloads": getattr(current_user, "usage_audio_downloads", 0),
            "video_downloads": getattr(current_user, "usage_video_downloads", 0)
        }
        # Map plan limits for UI
        SUBSCRIPTION_LIMITS = {
            "free":     {"clean_transcripts": 5, "unclean_transcripts": 3, "audio_downloads": 2, "video_downloads": 1},
            "pro":      {"clean_transcripts": 100, "unclean_transcripts": 50, "audio_downloads": 50, "video_downloads": 20},
            "premium":  {"clean_transcripts": float('inf'), "unclean_transcripts": float('inf'), "audio_downloads": float('inf'), "video_downloads": float('inf')}
        }
        limits = SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS["free"])
        json_limits = {k: ('unlimited' if v == float('inf') else v) for k, v in limits.items()}
        return {
            "tier": tier,
            "status": status,
            "usage": usage,
            "limits": json_limits,
            "subscription_id": getattr(subscription, 'payment_id', None) if subscription else None,
            "stripe_customer_id": getattr(current_user, 'stripe_customer_id', None),
            "current_period_end": expiry_date.isoformat() if expiry_date else None
        }
    except Exception as e:
        logger.error(f"❌ Error getting subscription status: {str(e)}")
        # Fallback: return all zeros + free tier
        return {
            "tier": "free",
            "status": "inactive",
            "usage": {
                "clean_transcripts": 0,
                "unclean_transcripts": 0,
                "audio_downloads": 0,
                "video_downloads": 0
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

@app.get("/health/")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
[]
@app.get("/test_videos")
def get_test_videos():
    return {
        "videos": [
            {"id": "dQw4w9WgXcQ", "title": "Rick Astley - Never Gonna Give You Up"},
            {"id": "jNQXAC9IVRw", "title": "Me at the zoo"}
        ]
    }

#===========================
@app.get("/debug_transcript/{video_id}")
def debug_transcript(video_id: str):
    """Try to fetch transcript directly from yt-dlp and return raw result."""
    result = get_transcript_with_ytdlp(video_id, clean=False)
    if not result:
        raise HTTPException(status_code=404, detail="Transcript not found or parsing failed")
    return {"video_id": video_id, "raw": result[:3000]}  # Limit long output

@app.get("/downloads/")
def recent_downloads(limit: int = 10, db: Session = Depends(get_db)):
    """Admin route to return recent transcript downloads."""
    downloads = db.query(TranscriptDownload).order_by(TranscriptDownload.created_at.desc()).limit(limit).all()
    return [
        {
            "id": d.id,
            "user_id": d.user_id,
            "youtube_id": d.youtube_id,
            "transcript_type": d.transcript_type,
            "created_at": d.created_at,
            "lang": d.language,
            "method": d.download_method,
        }
        for d in downloads
    ]
#===========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

## =========================
##==== FINAL MAIN.PY--ACTIVATED: 7/12/25 @ 11:49 PM SO KEEP IT!!=========

# # main.py (COMPLETE PATCH) - unified usage, robust /subscription_status/, download history ready

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

# # --- Import all database models ---

# from models import User, TranscriptDownload, SubscriptionHistory
# from database import engine, SessionLocal, get_db, create_tables

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
#         from_attributes = True

# class Token(BaseModel):
#     access_token: str
#     token_type: str

# class TranscriptRequest(BaseModel):
#     youtube_id: str
#     clean_transcript: bool = True

# # --- Transcript logic ---
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
#               #  lines.append(f"{timestamp} {seg['text'].replace('\n', ' ')}")
#                 text_clean = seg['text'].replace('\n', ' ')
#             return "\n".join(lines)
#     except Exception as e:
#         print(f"Transcript API failed: {e} - trying yt-dlp fallback...")
#         yt_dlp_transcript = get_transcript_with_ytdlp(video_id, clean=clean)
#         if yt_dlp_transcript:
#             return yt_dlp_transcript
#         return None

# def get_transcript_with_ytdlp(video_id, clean=True):
#     try:
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
#         os.remove(json_path)
#         return "\n".join(text_blocks) if text_blocks else None
#     except Exception as e:
#         print("yt-dlp fallback error:", e)
#         return None

# # --- Usage keys mapping ---
# USAGE_KEYS = {
#     True: "clean_transcripts",
#     False: "unclean_transcripts"
# }

# TRANSCRIPT_TYPE_MAP = {
#     True: "clean",
#     False: "unclean"
# }

# # --- FastAPI Endpoints ---

# @app.on_event("startup")
# def startup():
#     create_tables()

# @app.get("/")
# def root():
#     return {"message": "YouTube Transcript Downloader API", "status": "running"}

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
#     usage_key = USAGE_KEYS[request.clean_transcript]
#     transcript_type = TRANSCRIPT_TYPE_MAP[request.clean_transcript]
#     # Usage logic
#     if user.usage_reset_date.month != datetime.utcnow().month:
#         user.reset_monthly_usage()
#         db.commit()
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
#         raise HTTPException(status_code=404, detail="No transcript/captions found for this video.")
#     # Increment usage and save history
#     user.increment_usage(usage_key)
#     db.add(TranscriptDownload(
#         user_id=user.id,
#         youtube_id=video_id,
#         transcript_type=transcript_type,
#         created_at=datetime.utcnow()
#     ))
#     db.commit()
#     logger.info(f"User {user.username} downloaded transcript for {video_id} ({usage_key})")
#     return {
#         "transcript": transcript,
#         "youtube_id": video_id,
#         "message": "Transcript downloaded successfully"
#     }

# @app.get("/subscription_status/")
# def get_subscription_status(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     try:
#         # --- Try to load subscription from DB (graceful fallback if not found) ---
#         try:
#             subscription = db.query(Subscription).filter(
#                 Subscription.user_id == current_user.id
#             ).first()
#         except Exception as e:
#             subscription = None

#         # --- Derive current tier/status/expiry from User+Subscription ---
#         now = datetime.utcnow()
#         if not subscription:
#             tier = getattr(current_user, 'subscription_tier', 'free')
#             status = "inactive"
#             expiry_date = None
#         else:
#             if hasattr(subscription, 'expiry_date') and subscription.expiry_date and subscription.expiry_date < now:
#                 tier = "free"
#                 status = "expired"
#                 expiry_date = subscription.expiry_date
#             else:
#                 tier = getattr(subscription, 'tier', 'free') or 'free'
#                 status = getattr(subscription, 'status', 'active') if tier != "free" else "inactive"
#                 expiry_date = getattr(subscription, 'expiry_date', None)

#         # Usage and limits (ALWAYS provide these keys for frontend!)
#         usage = {
#             "clean_transcripts": getattr(current_user, "usage_clean_transcripts", 0),
#             "unclean_transcripts": getattr(current_user, "usage_unclean_transcripts", 0),
#             "audio_downloads": getattr(current_user, "usage_audio_downloads", 0),
#             "video_downloads": getattr(current_user, "usage_video_downloads", 0)
#         }
#         # Map plan limits for UI
#         SUBSCRIPTION_LIMITS = {
#             "free":     {"clean_transcripts": 5, "unclean_transcripts": 3, "audio_downloads": 2, "video_downloads": 1},
#             "pro":      {"clean_transcripts": 100, "unclean_transcripts": 50, "audio_downloads": 50, "video_downloads": 20},
#             "premium":  {"clean_transcripts": float('inf'), "unclean_transcripts": float('inf'), "audio_downloads": float('inf'), "video_downloads": float('inf')}
#         }
#         limits = SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS["free"])
#         json_limits = {k: ('unlimited' if v == float('inf') else v) for k, v in limits.items()}
#         return {
#             "tier": tier,
#             "status": status,
#             "usage": usage,
#             "limits": json_limits,
#             "subscription_id": getattr(subscription, 'payment_id', None) if subscription else None,
#             "stripe_customer_id": getattr(current_user, 'stripe_customer_id', None),
#             "current_period_end": expiry_date.isoformat() if expiry_date else None
#         }
#     except Exception as e:
#         logger.error(f"❌ Error getting subscription status: {str(e)}")
#         # Fallback: return all zeros + free tier
#         return {
#             "tier": "free",
#             "status": "inactive",
#             "usage": {
#                 "clean_transcripts": 0,
#                 "unclean_transcripts": 0,
#                 "audio_downloads": 0,
#                 "video_downloads": 0
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

