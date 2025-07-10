##========= ACTIVATED: 7/9/25 @ 10:10 PM =========
# main.py (robust, usage tracked via User model, ready for TranscriptDownload history)
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import Response
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

from database import engine, SessionLocal, get_db, Subscription, TranscriptDownload  # Import models for later history!
from models import User, create_tables

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

# --- Transcript logic (youtube-transcript-api only, no yt-dlp) ---

def extract_youtube_video_id(youtube_id_or_url: str) -> str:
    # Accept raw ID, youtu.be/..., youtube.com/watch?v=...
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

def get_transcript_youtube_api(video_id: str, clean: bool = True) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        if clean:
            text = " ".join([seg['text'].replace('\n', ' ') for seg in transcript])
            return " ".join(text.split())
        else:
            lines = []
            for seg in transcript:
                t = int(seg['start'])
                timestamp = f"[{t//60:02d}:{t%60:02d}]"
                text_clean = seg['text'].replace('\n', ' ')
                lines.append(f"{timestamp} {text_clean}")
            return "\n".join(lines)
    except Exception as e:
        print(f"Transcript API failed: {e} - trying yt-dlp fallback...")
        yt_dlp_transcript = get_transcript_with_ytdlp(video_id, clean=clean)
        if yt_dlp_transcript:
            return yt_dlp_transcript
        return None

def get_transcript_with_ytdlp(video_id, clean=True):
    """
    Use yt-dlp to extract captions as a fallback.
    Returns transcript as a string, or None.
    """
    try:
        cmd = [
            "yt-dlp",
            "--skip-download",
            "--write-auto-subs",
            "--sub-lang", "en",
            "--sub-format", "json3",
            "--output", "%(id)s",
            f"https://www.youtube.com/watch?v={video_id}"
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        json_path = f"{video_id}.en.json3"
        if not os.path.exists(json_path):
            return None
        with open(json_path, encoding="utf8") as f:
            data = json.load(f)
        text_blocks = []
        for event in data.get("events", []):
            if "segs" in event and "tStartMs" in event:
                text = "".join([seg.get("utf8", "") for seg in event["segs"]]).strip()
                if text:
                    if clean:
                        text_blocks.append(text)
                    else:
                        start_sec = int(event["tStartMs"] // 1000)
                        timestamp = f"[{start_sec // 60:02d}:{start_sec % 60:02d}]"
                        text_blocks.append(f"{timestamp} {text}")
        os.remove(json_path)  # Clean up
        return "\n".join(text_blocks) if text_blocks else None
    except Exception as e:
        print("yt-dlp fallback error:", e)
        return None

def get_demo_content(clean=True):
    if clean:
        return "This is demo transcript content. The real YouTube transcript could not be downloaded."
    else:
        return "[00:00] This is demo transcript content. The real YouTube transcript could not be downloaded."

# --- Usage keys for User model (matches your models.py) ---
USAGE_KEYS = {
    True: "clean_transcripts",
    False: "unclean_transcripts"
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
    # Check for existing username/email
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
    # Monthly reset if needed
    if user.usage_reset_date.month != datetime.utcnow().month:
        user.reset_monthly_usage()
        db.commit()
    # Check subscription and usage limit
    plan_limits = user.get_plan_limits()
    current_usage = getattr(user, f"usage_{usage_key}", 0)
    allowed = plan_limits[usage_key]
    if allowed != float('inf') and current_usage >= allowed:
        raise HTTPException(
            status_code=403,
            detail=f"Monthly limit reached for {usage_key.replace('_',' ')}. Please upgrade your plan."
        )

    transcript = get_transcript_youtube_api(video_id, clean=request.clean_transcript)
    if not transcript or len(transcript.strip()) < 10:
        raise HTTPException(status_code=404, detail="No transcript/captions found for this video.")

    # Increment usage counter and save
    user.increment_usage(usage_key)
    db.commit()
    db.refresh(user)

    # For future: Add TranscriptDownload record here if download history is required

    # Return transcript and updated usage info for immediate frontend refresh
    usage_status = {
        "clean_transcripts": user.usage_clean_transcripts,
        "unclean_transcripts": user.usage_unclean_transcripts,
        "audio_downloads": user.usage_audio_downloads,
        "video_downloads": user.usage_video_downloads,
    }
    logger.info(f"User {user.username} downloaded transcript for {video_id} ({usage_key})")
    return {
        "transcript": transcript,
        "youtube_id": video_id,
        "message": "Transcript downloaded successfully",
        "usage": usage_status,
        "limits": plan_limits
    }

@app.get("/subscription_status/")
async def get_subscription_status_ultra_safe(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Query subscription info
        subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id
        ).first()
        if not subscription:
            tier = "free"
            status = "inactive"
            expiry_date = None
        else:
            tier = subscription.tier if subscription.tier else "free"
            expiry_date = getattr(subscription, 'expiry_date', None)
            status = "active" if (subscription.status == 'active' and (not expiry_date or expiry_date > datetime.now())) else "inactive"
        plan_limits = current_user.get_plan_limits()
        usage = current_user.get_current_usage()
        json_limits = {k: ('unlimited' if v == float('inf') else v) for k, v in plan_limits.items()}
        return {
            "tier": tier,
            "status": status,
            "usage": usage,
            "limits": json_limits,
            "subscription_id": subscription.payment_id if subscription and hasattr(subscription, 'payment_id') else None,
            "stripe_customer_id": getattr(current_user, 'stripe_customer_id', None),
            "current_period_end": expiry_date.isoformat() if expiry_date else None
        }
    except Exception as e:
        logger.error(f"❌ Error getting subscription status: {str(e)}")
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

@app.get("/health/")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/test_videos")
def get_test_videos():
    return {
        "videos": [
            {"id": "dQw4w9WgXcQ", "title": "Rick Astley - Never Gonna Give You Up"},
            {"id": "jNQXAC9IVRw", "title": "Me at the zoo"}
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
