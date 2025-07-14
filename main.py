# ================== START: DeepSeek main.py ==========================

# main.py - Integrated version with all plan functionality
from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
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
from enum import Enum
from sqlalchemy import extract
import random
import asyncio
#from database import Base, engine, SessionLocal, get_db

# --- Import all database models ---
from database import engine, SessionLocal, get_db, Base
from database import User, Subscription, TranscriptDownload, PaymentHistory

load_dotenv()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("youtube_trans_downloader.main")

# --- App & CORS ---
app = FastAPI(title="YouTubeTransDownloader API", version="2.1.0 (plan-integrated)")
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

# --- Plan Definitions ---
class PlanType(str, Enum):
    FREE = "free"
    PRO = "pro"
    PREMIUM = "premium"

PLAN_FEATURES = {
    PlanType.FREE: {
        "monthly_downloads": 5,
        "formats": ["txt", "srt"],
        "processing_priority": 3,
        "support": "community",
        "transcript_cleaning": False,
        "batch_processing": False,
        "api_access": False,
        "processing_delay": (2, 5),
        "price": 0
    },
    PlanType.PRO: {
        "monthly_downloads": 100,
        "formats": ["txt", "srt", "vtt"],
        "processing_priority": 2,
        "support": "email",
        "transcript_cleaning": True,
        "batch_processing": False,
        "api_access": False,
        "processing_delay": (0.5, 1),
        "price": 9.99
    },
    PlanType.PREMIUM: {
        "monthly_downloads": float('inf'),
        "formats": ["txt", "srt", "vtt", "json", "csv", "xml"],
        "processing_priority": 1,
        "support": "priority",
        "transcript_cleaning": True,
        "batch_processing": True,
        "api_access": True,
        "processing_delay": (0.1, 0.3),
        "price": 19.99
    }
}

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

async def apply_processing_delay(plan_type: PlanType):
    """Simulate processing delay based on user's plan"""
    delay_range = PLAN_FEATURES[plan_type]["processing_delay"]
    delay = random.uniform(delay_range[0], delay_range[1])
    await asyncio.sleep(delay)

def check_user_limits(user: User, db: Session, transcript_type: str) -> bool:
    """Check if user has reached their monthly download limit"""
    subscription = db.query(Subscription).filter(
        Subscription.user_id == user.id,
        Subscription.status == 'active'
    ).first()
    
    if not subscription or subscription.tier == PlanType.PREMIUM:
        return False
    
    current_month = datetime.utcnow().month
    downloads_this_month = db.query(TranscriptDownload).filter(
        TranscriptDownload.user_id == user.id,
        extract('month', TranscriptDownload.created_at) == current_month
    ).count()
    
    plan_limit = PLAN_FEATURES[PlanType(subscription.tier)]["monthly_downloads"]
    return downloads_this_month >= plan_limit

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
        os.remove(json_path)
        return "\n".join(text_blocks) if text_blocks else None
    except Exception as e:
        print("yt-dlp fallback error:", e)
        return None

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
    format: str = "txt"

class BatchDownloadRequest(BaseModel):
    urls: List[str]
    clean_transcript: bool = True

# --- FastAPI Endpoints ---

@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)

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
async def download_transcript(
    request: TranscriptRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    video_id = extract_youtube_video_id(request.youtube_id)
    if not video_id or len(video_id) != 11:
        raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    
    # Get user's subscription
    subscription = db.query(Subscription).filter(
        Subscription.user_id == current_user.id,
        Subscription.status == 'active'
    ).first()
    
    if not subscription:
        plan_type = PlanType.FREE
    else:
        plan_type = PlanType(subscription.tier)
    
    # Check download limits
    if check_user_limits(current_user, db, "clean" if request.clean_transcript else "unclean"):
        raise HTTPException(
            status_code=403,
            detail=f"You've reached your monthly download limit for {plan_type.value} plan."
        )
    
    # Check format availability
    if request.format not in PLAN_FEATURES[plan_type]["formats"]:
        raise HTTPException(
            status_code=403,
            detail=f"{request.format.upper()} format not available for {plan_type.value} plan."
        )
    
    # Apply processing delay based on plan
    background_tasks.add_task(apply_processing_delay, plan_type)
    
    # Get transcript
    transcript = get_transcript_youtube_api(video_id, clean=request.clean_transcript)
    if not transcript or len(transcript.strip()) < 10:
        raise HTTPException(status_code=404, detail="No transcript/captions found for this video.")
    
    # Format the transcript based on requested format
    formatted_transcript = transcript
    if request.format == "json":
        formatted_transcript = {
            "video_id": video_id,
            "transcript": transcript.split('\n'),
            "metadata": {
                "format": "json",
                "clean": request.clean_transcript,
                "generated_at": datetime.utcnow().isoformat()
            }
        }
    elif request.format == "xml":
        formatted_transcript = f"""<?xml version="1.0" encoding="UTF-8"?>
<transcript video_id="{video_id}">
{"".join([f"<line>{line}</line>" for line in transcript.split('\n')])}
</transcript>"""
    elif request.format == "csv":
        lines = transcript.split('\n')
        formatted_transcript = "start_time,text\n" + "\n".join([
            f"{i},{line}" for i, line in enumerate(lines)
        ])
    
    # Save download record
    download = TranscriptDownload(
        user_id=current_user.id,
        youtube_id=video_id,
        transcript_type='clean' if request.clean_transcript else 'unclean',
        format=request.format,
        created_at=datetime.utcnow()
    )
    db.add(download)
    db.commit()
    
    return {
        "transcript": formatted_transcript,
        "format": request.format,
        "youtube_id": video_id,
        "message": "Transcript downloaded successfully",
        "plan": plan_type.value,
        "remaining_downloads": (
            "unlimited" if plan_type == PlanType.PREMIUM else
            PLAN_FEATURES[plan_type]["monthly_downloads"] - db.query(TranscriptDownload).filter(
                TranscriptDownload.user_id == current_user.id,
                extract('month', TranscriptDownload.created_at) == datetime.utcnow().month
            ).count()
        )
    }

@app.post("/batch_download/")
async def batch_download(
    request: BatchDownloadRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Check if user is premium
    subscription = db.query(Subscription).filter(
        Subscription.user_id == current_user.id,
        Subscription.status == 'active',
        Subscription.tier == PlanType.PREMIUM
    ).first()
    
    if not subscription:
        raise HTTPException(
            status_code=403,
            detail="Batch processing is only available for premium users."
        )
    
    # Limit batch size
    if len(request.urls) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum batch size is 50 URLs."
        )
    
    # Process each URL
    results = []
    for url in request.urls:
        try:
            video_id = extract_youtube_video_id(url)
            transcript = get_transcript_youtube_api(video_id, clean=request.clean_transcript)
            
            if transcript:
                download = TranscriptDownload(
                    user_id=current_user.id,
                    youtube_id=video_id,
                    transcript_type='clean' if request.clean_transcript else 'unclean',
                    created_at=datetime.utcnow()
                )
                db.add(download)
                results.append({
                    "url": url,
                    "video_id": video_id,
                    "status": "success",
                    "transcript": transcript[:100] + "..."  # Return preview
                })
            else:
                results.append({
                    "url": url,
                    "video_id": video_id,
                    "status": "failed",
                    "error": "No transcript available"
                })
        except Exception as e:
            results.append({
                "url": url,
                "status": "failed",
                "error": str(e)
            })
    
    db.commit()
    return {"results": results}

@app.get("/subscription_status/")
def get_subscription_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    subscription = db.query(Subscription).filter(
        Subscription.user_id == current_user.id
    ).first()
    
    if not subscription:
        tier = PlanType.FREE
        status = "inactive"
        expiry_date = None
    else:
        tier = PlanType(subscription.tier)
        status = subscription.status
        expiry_date = subscription.current_period_end
    
    # Get current month's downloads
    current_month = datetime.utcnow().month
    downloads_this_month = db.query(TranscriptDownload).filter(
        TranscriptDownload.user_id == current_user.id,
        extract('month', TranscriptDownload.created_at) == current_month
    ).count()
    
    return {
        "tier": tier.value,
        "status": status,
        "downloads_used": downloads_this_month,
        "downloads_allowed": (
            "unlimited" if tier == PlanType.PREMIUM else 
            PLAN_FEATURES[tier]["monthly_downloads"]
        ),
        "features": PLAN_FEATURES[tier],
        "current_period_end": expiry_date.isoformat() if expiry_date else None,
        "stripe_customer_id": current_user.stripe_customer_id
    }

@app.get("/plans/")
def get_available_plans():
    return {
        "plans": {
            plan.value: {
                **PLAN_FEATURES[plan],
                "name": plan.value.capitalize(),
                "key": plan.value
            }
            for plan in PlanType
        }
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

# ================== END: DeepSeek main.py ==========================

# ##========= ACTIVATED: 7/9/25 @ 10:10 PM =========
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
#     format: Optional[str] = None  # "srt" or "vtt" when unclean

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

# # def get_transcript_youtube_api(video_id: str, clean: bool = True) -> str:
# #     try:
# #         from youtube_transcript_api import YouTubeTranscriptApi
# #         transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
# #         if clean:
# #             text = " ".join([seg['text'].replace('\n', ' ') for seg in transcript])
# #             return " ".join(text.split())
# #         else:
# #             lines = []
# #             for seg in transcript:
# #                 t = int(seg['start'])
# #                 timestamp = f"[{t//60:02d}:{t%60:02d}]"
# #               #  lines.append(f"{timestamp} {seg['text'].replace('\n', ' ')}")
# #                 text_clean = seg['text'].replace('\n', ' ')
# #             return "\n".join(lines)
# #     except Exception as e:
# #         print(f"Transcript API failed: {e} - trying yt-dlp fallback...")
# #         yt_dlp_transcript = get_transcript_with_ytdlp(video_id, clean=clean)
# #         if yt_dlp_transcript:
# #             return yt_dlp_transcript
# #         return None

# def get_transcript_youtube_api(video_id: str, clean: bool = True, format: Optional[str] = None) -> str:
#     try:
#         from youtube_transcript_api import YouTubeTranscriptApi
#         transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
#         if clean:
#             text = " ".join([seg['text'].replace('\n', ' ') for seg in transcript])
#             return " ".join(text.split())
#         else:
#             if format == "srt":
#                 return segments_to_srt(transcript)
#             elif format == "vtt":
#                 return segments_to_vtt(transcript)
#             else:
#                 # fallback: plain timestamped text
#                 lines = []
#                 for seg in transcript:
#                     t = int(seg['start'])
#                     timestamp = f"[{t//60:02d}:{t%60:02d}]"
#                     text_clean = seg['text'].replace('\n', ' ')
#                     lines.append(f"{timestamp} {text_clean}")
#                 return "\n".join(lines)
#     except Exception as e:
#         print(f"Transcript API failed: {e} - trying yt-dlp fallback...")
#         # Fallback to yt-dlp: Not shown for brevity, but should follow similar SRT/VTT logic
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

# # Formatters
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
#         text = seg["text"].replace("\n", " ")
#         lines.append(f"{idx+1}\n{start} --> {end}\n{text}\n")
#     return "\n".join(lines)

# def segments_to_vtt(transcript):
#     """Convert segments to WebVTT format."""
#     def sec_to_vtt(ts):
#         h = int(ts // 3600)
#         m = int((ts % 3600) // 60)
#         s = int(ts % 60)
#         ms = int((ts - int(ts)) * 1000)
#         return f"{h:02}:{m:02}:{s:02}.{ms:03}"
#     lines = ["WEBVTT\n"]
#     for seg in transcript:
#         start = sec_to_vtt(seg["start"])
#         end = sec_to_vtt(seg.get("start", 0) + seg.get("duration", 0))
#         text = seg["text"].replace("\n", " ")
#         lines.append(f"{start} --> {end}\n{text}\n")
#     return "\n".join(lines)

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

# # @app.post("/download_transcript/")
# # def download_transcript(
# #     request: TranscriptRequest,
# #     user: User = Depends(get_current_user),
# #     db: Session = Depends(get_db)
# # ):
# #     video_id = extract_youtube_video_id(request.youtube_id)
# #     if not video_id or len(video_id) != 11:
# #         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
# #     usage_key = USAGE_KEYS[request.clean_transcript]
# #     transcript_type = TRANSCRIPT_TYPE_MAP[request.clean_transcript]
# #     # Usage logic
# #     if user.usage_reset_date.month != datetime.utcnow().month:
# #         user.reset_monthly_usage()
# #         db.commit()
# #     plan_limits = user.get_plan_limits()
# #     current_usage = getattr(user, f"usage_{usage_key}", 0)
# #     allowed = plan_limits[usage_key]
# #     if allowed != float('inf') and current_usage >= allowed:
# #         raise HTTPException(
# #             status_code=403,
# #             detail=f"Monthly limit reached for {usage_key.replace('_',' ')}. Please upgrade your plan."
# #         )
# #     transcript = get_transcript_youtube_api(video_id, clean=request.clean_transcript)
# #     if not transcript or len(transcript.strip()) < 10:
# #         raise HTTPException(status_code=404, detail="No transcript/captions found for this video.")
# #     # Increment usage and save history
# #     user.increment_usage(usage_key)
# #     db.add(TranscriptDownload(
# #         user_id=user.id,
# #         youtube_id=video_id,
# #         transcript_type=transcript_type,
# #         created_at=datetime.utcnow()
# #     ))
# #     db.commit()
# #     logger.info(f"User {user.username} downloaded transcript for {video_id} ({usage_key})")
# #     return {
# #         "transcript": transcript,
# #         "youtube_id": video_id,
# #         "message": "Transcript downloaded successfully"
# #     }

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
#     transcript = get_transcript_youtube_api(
#         video_id,
#         clean=request.clean_transcript,
#         format=request.format if not request.clean_transcript else None
#     )
#     if not transcript or len(transcript.strip()) < 10:
#         raise HTTPException(status_code=404, detail="No transcript/captions found for this video.")
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
#             {"id": "jNQXAC9IVRw", "title": "Me at the zoo"}
#         ]
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

#=========================
# ##==== FINAL MAIN.PY--ACTIVATED: 7/12/25 @ 11:49 PM SO KEEP IT!!=========

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
#             {"id": "jNQXAC9IVRw", "title": "Me at the zoo"}
#         ]
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

