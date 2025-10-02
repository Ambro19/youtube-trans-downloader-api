# Adds middlewares, webhook signature verification + idempotency, and a background cleanup thread. 
# (Everything else preserved.)

# backend/main.py
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import re, time, socket, mimetypes, logging, jwt, threading #os
from collections import deque, defaultdict

#Newly added import
from security_headers import SecurityHeadersMiddleware
from rate_limit import RateLimitMiddleware, rules_for_env

from models import engine  # add this
from db_migrations import run_startup_migrations
import os

# ✅ use the shared dependency (no circular import)
from auth_deps import get_current_user
# Stripe webhook helpers
from webhook_handler import handle_stripe_webhook, fix_existing_premium_users

from fastapi import FastAPI, HTTPException, Depends, status, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
#from starlette.middleware.base import BaseHTTPMiddleware

from pydantic import BaseModel
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError, IntegrityError
from sqlalchemy import delete as sqla_delete

### ###
import os, time, threading
from collections import defaultdict, deque
from typing import Optional, Dict, Deque
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from starlette.requests import Request
from starlette.responses import JSONResponse
### ###
from dotenv import load_dotenv, find_dotenv
load_dotenv()
load_dotenv(dotenv_path=find_dotenv(".env.local"), override=True)
load_dotenv(dotenv_path=find_dotenv(".env"),       override=False)

#------------ YouTube API ----------------
from youtube_transcript_api import YouTubeTranscriptApi
from security_headers import SecurityHeadersMiddleware

APP_ENV = os.getenv("APP_ENV", os.getenv("ENV", "development")).lower()
IS_PROD = APP_ENV == "production"


# Stripe optional
stripe = None
try:
    import stripe as _stripe  # type: ignore
    if os.getenv("STRIPE_SECRET_KEY"):
        _stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        stripe = _stripe
except Exception:
    stripe = None

# Local modules
from payment import router as payment_router
#from payment import router as billing_router   # the FastAPI router for /billing/*
from subscription_sync import sync_user_subscription_from_stripe

from models import User, TranscriptDownload, Subscription, get_db, initialize_database
from transcript_utils import (
    get_transcript_with_ytdlp,
    download_audio_with_ytdlp,
    download_video_with_ytdlp,
    check_ytdlp_availability,
    get_video_info,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("youtube_trans_downloader")

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
FILE_RETENTION_DAYS = int(os.getenv("FILE_RETENTION_DAYS", "7"))

# ---------- App ----------
app = FastAPI(title="YouTube Content Downloader API", version="3.3.0")

# Optional timestamp normalizer
try:
    from timestamp_patch import EnsureUtcZMiddleware  # type: ignore
    app.add_middleware(EnsureUtcZMiddleware)
except Exception as e:
    logger.info("Timestamp middleware not loaded: %s", e)

# ---------- Security Headers Middleware ----------
class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Sane security headers with optional CSP/HSTS. Accepts kwargs so add_middleware works.
    - In dev, pass csp=None so it doesn't block hot-reload/assets.
    - HSTS only makes sense behind HTTPS.
    """
    def __init__(
        self,
        app: ASGIApp,
        *,
        csp: Optional[str] = None,
        hsts: bool = False,
        hsts_max_age: int = 31536000,
        hsts_preload: bool = False,
        referrer_policy: str = "no-referrer",
        x_frame_options: str = "DENY",
        permissions_policy: Optional[str] = "geolocation=(), microphone=(), camera=()",
        server_header: Optional[str] = "YCD",
        apply_csp_to_api_only: bool = True,  # skip CSP on HTML/_spa to avoid breaking SPA
    ) -> None:
        super().__init__(app)
        self.csp = csp
        self.hsts = hsts
        self.hsts_max_age = int(hsts_max_age)
        self.hsts_preload = bool(hsts_preload)
        self.referrer_policy = referrer_policy
        self.x_frame_options = x_frame_options
        self.permissions_policy = permissions_policy
        self.server_header = server_header
        self.apply_csp_to_api_only = apply_csp_to_api_only

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Always-useful headers
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        if self.x_frame_options:
            response.headers["X-Frame-Options"] = self.x_frame_options
        if self.referrer_policy:
            response.headers["Referrer-Policy"] = self.referrer_policy
        # Don't emit CORP by default; it can break cross-origin API usage.
        # response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        response.headers.setdefault("X-XSS-Protection", "0")
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")

        if self.server_header is not None:
            response.headers["Server"] = self.server_header

        # HSTS only if explicitly enabled and the request is HTTPS (or forwarded as HTTPS)
        if self.hsts:
            scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
            if (scheme or "").lower() == "https":
                hsts_value = f"max-age={self.hsts_max_age}; includeSubDomains"
                if self.hsts_preload:
                    hsts_value += "; preload"
                response.headers["Strict-Transport-Security"] = hsts_value

        # CSP (skip for HTML & _spa if configured)
        if self.csp:
            ct = (response.headers.get("content-type", "") or "").split(";")[0].strip().lower()
            if self.apply_csp_to_api_only:
                if not ct.startswith("text/html") and not request.url.path.startswith("/_spa"):
                    response.headers["Content-Security-Policy"] = self.csp
            else:
                response.headers["Content-Security-Policy"] = self.csp

        if self.permissions_policy:
            response.headers["Permissions-Policy"] = self.permissions_policy

        return response

# One, clear registration with supported kwargs
DEV_CSP = None
PROD_CSP = (
    "default-src 'self'; "
    "img-src 'self' data: blob:; "
    "media-src 'self' data: blob:; "
    "font-src 'self' data:; "
    "style-src 'self' 'unsafe-inline'; "
    "script-src 'self'; "
    "connect-src 'self' https://api.stripe.com; "
    "frame-ancestors 'none'; "
    "base-uri 'none'; "
)

app.add_middleware(
    SecurityHeadersMiddleware,
    csp=(PROD_CSP if IS_PROD else DEV_CSP),
    hsts=IS_PROD,
    hsts_max_age=63072000,
    hsts_preload=False,
    referrer_policy="no-referrer",
    x_frame_options="DENY",
    permissions_policy="geolocation=(), microphone=(), camera=()",
    server_header="YCD",
    apply_csp_to_api_only=True,
)

# ---------- Simple In-Memory Rate Limiter ----------
class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple sliding-window limiter per client IP.
    For production scale, back with Redis.
    Env overrides:
      RL_ENABLED=true|false
      RL_DEFAULT_WINDOW_SEC=60, RL_DEFAULT_MAX=120
      RL_AUTH_WINDOW_SEC=60, RL_AUTH_MAX=10
      RL_DL_WINDOW_SEC=60, RL_DL_MAX=40
    """
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.now = time.time
        self.buckets: Dict[str, Deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

        self.enabled = os.getenv("RL_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

        self.default_window = int(os.getenv("RL_DEFAULT_WINDOW_SEC", "60"))
        self.default_max    = int(os.getenv("RL_DEFAULT_MAX", "120"))

        self.auth_window = int(os.getenv("RL_AUTH_WINDOW_SEC", "60"))
        self.auth_max    = int(os.getenv("RL_AUTH_MAX", "10"))

        self.dl_window   = int(os.getenv("RL_DL_WINDOW_SEC", "60"))
        self.dl_max      = int(os.getenv("RL_DL_MAX", "40"))

    def key_for(self, request: Request) -> tuple[str, int, int]:
        ip = (request.client.host if request.client else "unknown")
        path = request.url.path
        if path.startswith(("/token", "/register", "/webhook/stripe")):
            return (f"AUTH:{ip}", self.auth_window, self.auth_max)
        if path.startswith(("/download_", "/download-file/")):
            return (f"DL:{ip}", self.dl_window, self.dl_max)
        return (f"GEN:{ip}", self.default_window, self.default_max)

    async def dispatch(self, request: Request, call_next):
        if not self.enabled:
            return await call_next(request)

        key, window, limit = self.key_for(request)
        now = self.now()
        with self.lock:
            q = self.buckets[key]
            while q and now - q[0] > window:
                q.popleft()
            if len(q) >= limit:
                retry_after = max(1, int(window - (now - q[0])))
                return JSONResponse(
                    {"detail": "Too Many Requests"},
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                )
            q.append(now)

        return await call_next(request)

# Register exactly once
app.add_middleware(RateLimitMiddleware)

# ---------- Routers ----------
from batch import router as batch_router
try:
    app.include_router(batch_router, tags=["batch"])
except Exception as e:
    logger.error("Could not include batch routes: %s", e)

try:
    from activity import router as activity_router
    app.include_router(activity_router, tags=["activity"])
    logger.info("✅ Activity tracking router loaded")
except Exception as e:
    logger.warning("Could not include activity routes: %s", e)

try:
    app.include_router(payment_router, tags=["payments"])
except Exception as e:
    logger.error("Could not include payment routes: %s", e)

# ---------- CORS ----------
allowed_origins = (
    ["http://localhost:3000", "http://127.0.0.1:3000", "http://192.168.1.185:3000", FRONTEND_URL]
    if ENVIRONMENT != "production" else [FRONTEND_URL]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in allowed_origins if o],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "Content-Type", "Content-Length", "Content-Range"],
)

# -------------------- Downloads dir -------------------------
USE_SYSTEM_DOWNLOADS = os.getenv("USE_SYSTEM_DOWNLOADS", "false").lower() == "true"
if USE_SYSTEM_DOWNLOADS:
    try:
        d = Path.home() / "Downloads"
        d.mkdir(exist_ok=True)
        (d / ".__test_write.tmp").write_text("ok", encoding="utf-8")
        (d / ".__test_write.tmp").unlink(missing_ok=True)
        DOWNLOADS_DIR = d
        logger.info("Using system Downloads folder: %s", DOWNLOADS_DIR)
    except Exception as e:
        DOWNLOADS_DIR = Path(__file__).resolve().parent / "server_files"
        DOWNLOADS_DIR.mkdir(exist_ok=True)
        logger.info("Falling back to server_files: %s (reason: %s)", DOWNLOADS_DIR, e)
else:
    DOWNLOADS_DIR = Path(__file__).resolve().parent / "server_files"
    DOWNLOADS_DIR.mkdir(exist_ok=True)
    logger.info("Using private cache folder: %s", DOWNLOADS_DIR)

app.mount("/files", StaticFiles(directory=str(DOWNLOADS_DIR)), name="files")

# -------------------- Auth / helpers ----------------------------------------
SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def is_mobile_request(request: Request) -> bool:
    ua = (request.headers.get("user-agent") or "").lower()
    return any(p in ua for p in ["android","iphone","ipad","ipod","blackberry","windows phone","mobile","webos","opera mini"])

def get_safe_filename(filename: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", filename)[:120]

def get_mobile_mime_type(path: str, file_type: str) -> str:
    if file_type == "audio" or path.endswith((".mp3",".m4a",".aac")): return "audio/mpeg"
    if file_type == "video" or path.endswith((".mp4",".m4v",".mov")): return "video/mp4"
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = dict(data)
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_access_token_for_mobile(username: str) -> str:
    return create_access_token({"sub": username}, timedelta(hours=2))

def verify_password(plain: str, hashed: str) -> bool: return pwd_context.verify(plain, hashed)
def get_password_hash(pw: str) -> str: return pwd_context.hash(pw)

def get_user(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()

def canonical_account(user: User) -> Dict[str, Any]:
    return {"username": (user.username or "").strip(), "email": (user.email or "").strip().lower()}

def check_internet() -> bool:
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3); return True
    except OSError:
        return False

def check_youtube() -> bool:
    try:
        socket.create_connection(("www.youtube.com", 443), timeout=5); return True
    except OSError:
        return False

def extract_youtube_video_id(text: str) -> str:
    pats = [r'(?:youtube\.com\/watch\?v=)([^&\n?#]+)', r'(?:youtu\.be\/)([^&\n?#]+)',
            r'(?:youtube\.com\/embed\/)([^&\n?#]+)', r'(?:youtube\.com\/shorts\/)([^&\n?#]+)',
            r'[?&]v=([^&\n?#]+)']
    for p in pats:
        m = re.search(p, text)
        if m: return m.group(1)[:11]
    return text.strip()[:11]

# -------------------- Transcript helpers (unchanged below this line except where noted) ----
def segments_to_vtt(transcript) -> str:
    def sec_to_vtt(ts: float) -> str:
        h = int(ts // 3600); m = int((ts % 3600) // 60); s = int(ts % 60); ms = int((ts - int(ts)) * 1000)
        return f"{h:02}:{m:02}:{s:02}.{ms:03}"
    lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    for seg in transcript:
        start = sec_to_vtt(seg.get("start", 0))
        end = sec_to_vtt(seg.get("start", 0) + seg.get("duration", 0))
        text = (seg.get("text") or "").replace("\n", " ").strip()
        lines.append(f"{start} --> {end}"); lines.append(text); lines.append("")
    return "\n".join(lines)

def segments_to_srt(transcript) -> str:
    def sec_to_srt(ts: float) -> str:
        h = int(ts // 3600); m = int((ts % 3600) // 60); s = int(ts % 60); ms = int((ts - int(ts)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    out = []
    for i, seg in enumerate(transcript, start=1):
        start = sec_to_srt(seg.get("start", 0))
        end = sec_to_srt(seg.get("start", 0) + seg.get("duration", 0))
        text = (seg.get("text") or "").replace("\n", " ").strip()
        out.append(str(i)); out.append(f"{start} --> {end}"); out.append(text); out.append("")
    return "\n".join(out)

# -------------------- Transcript helpers ------------------------------------
def get_transcript_youtube_api(video_id: str, clean: bool = True, fmt: Optional[str] = None) -> str:
    """
    Try YouTubeTranscriptApi first, then fall back to yt-dlp helpers.
    Returns:
      - clean=True  -> plain text (paragraphized)
      - clean=False + fmt='srt' -> SRT text
      - clean=False + fmt='vtt' -> VTT text
      - clean=False (default)   -> timestamped [MM:SS] lines
    """
    logger.info("Transcript for %s (clean=%s, fmt=%s)", video_id, clean, fmt)

    # quick connectivity checks (keep the same helpers you already have)
    if not check_internet():
        raise HTTPException(status_code=503, detail="No internet connection available.")
    if not check_youtube():
        raise HTTPException(status_code=503, detail="Cannot reach YouTube right now.")

    # 1) Primary path: YouTubeTranscriptApi
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])

        if clean:
            # collapse to clean paragraphs (~400+ chars per para on sentence boundary)
            text = " ".join((seg.get("text") or "").replace("\n", " ") for seg in transcript)
            text = " ".join(text.split())
            out, cur, chars = [], [], 0
            for word in text.split():
                cur.append(word); chars += len(word) + 1
                if chars > 400 and word.endswith((".", "!", "?")):
                    out.append(" ".join(cur)); cur, chars = [], 0
            if cur:
                out.append(" ".join(cur))
            return "\n\n".join(out)

        # not clean: choose format or default timestamped lines
        if (fmt or "").lower() == "srt":
            return segments_to_srt(transcript)
        if (fmt or "").lower() == "vtt":
            return segments_to_vtt(transcript)

        # default: timestamped [MM:SS] lines
        lines = []
        for seg in transcript:
            t = int(seg.get("start", 0))
            timestamp = f"[{t//60:02d}:{t%60:02d}]"
            txt = (seg.get("text") or "").replace("\n", " ")
            lines.append(f"{timestamp} {txt}")
        return "\n".join(lines)

    except Exception as e:
        logger.warning("YouTubeTranscriptApi failed: %s — falling back to yt-dlp", e)

        # 2) Fallback: yt-dlp-based extraction (json3/vtt/srt)
        # clean=True -> plain text; clean=False + VTT available -> VTT text;
        # clean=False + only SRT -> timestamped lines.
        fb = get_transcript_with_ytdlp(video_id, clean=clean)
        if fb:
            # If the caller explicitly asked for VTT and our fallback produced VTT,
            # great; otherwise we just return what the fallback produced.
            return fb

        # If we get here, nothing worked
        msg = str(e)
        if "No transcripts were found" in msg or "TranscriptsDisabled" in msg:
            raise HTTPException(status_code=404, detail="This video has no captions/transcripts.")
        raise HTTPException(status_code=404, detail="No transcript/captions found for this video.")

# -------------------- Usage limits / records --------------------------------
def increment_user_usage(db: Session, user: User, usage_type: str) -> int:
    current = getattr(user, f"usage_{usage_type}", 0) or 0
    new_val = current + 1
    setattr(user, f"usage_{usage_type}", new_val)

    now = datetime.utcnow()
    if not getattr(user, "usage_reset_date", None):
        user.usage_reset_date = now
    elif user.usage_reset_date.month != now.month:
        user.usage_clean_transcripts = 0
        user.usage_unclean_transcripts = 0
        user.usage_audio_downloads = 0
        user.usage_video_downloads = 0
        user.usage_reset_date = now
        setattr(user, f"usage_{usage_type}", 1)
        new_val = 1

    db.commit(); db.refresh(user)
    return new_val

def check_usage_limit(user: User, usage_type: str) -> Tuple[bool, int, int]:
    tier = getattr(user, "subscription_tier", "free")
    limits = {
        "free":   {"clean_transcripts": 5,  "unclean_transcripts": 3,  "audio_downloads": 2,  "video_downloads": 1},
        "pro":    {"clean_transcripts": 100,"unclean_transcripts": 50, "audio_downloads": 50, "video_downloads": 20},
        "premium":{"clean_transcripts": float("inf"), "unclean_transcripts": float("inf"),
                   "audio_downloads": float("inf"), "video_downloads": float("inf")},
    }
    cur = getattr(user, f"usage_{usage_type}", 0) or 0
    limit = limits.get(tier, limits["free"]).get(usage_type, 0)
    return cur < limit, cur, limit

def create_download_record(db: Session, user: User, kind: str, youtube_id: str, **kw):
    try:
        rec = TranscriptDownload(
            user_id=user.id,
            youtube_id=youtube_id,
            transcript_type=kind,
            quality=kw.get("quality", "default"),
            file_format=kw.get("file_format", "txt"),
            file_size=kw.get("file_size", 0),
            processing_time=kw.get("processing_time", 0),
            created_at=datetime.utcnow(),
        )
        db.add(rec); db.commit(); db.refresh(rec)
        return rec
    except Exception as e:
        logger.error("Create download record failed: %s", e)
        db.rollback()
        return None

# ENHANCED: Activity classification helper (unchanged)
def classify_activity_by_format(transcript_type: str, file_format: str) -> Tuple[str, str, str, str]:
    t_type = (transcript_type or "").lower(); f_format = (file_format or "").lower()
    if f_format == 'srt': return ("Generated SRT Transcript", "🕒", "SRT transcript", "transcript")
    if f_format == 'vtt': return ("Generated VTT Transcript", "🕒", "VTT transcript", "transcript")
    if f_format == 'txt':
        if 'unclean' in t_type or 'timestamped' in t_type: return ("Generated Timestamped Transcript", "🕒", "Timestamped transcript", "transcript")
        return ("Generated Clean Transcript", "📄", "Clean text transcript", "transcript")
    if f_format in ['mp3','m4a','aac','wav']: return ("Downloaded Audio File", "🎵", f"{f_format.upper()} file", "audio")
    if f_format in ['mp4','mkv','avi','mov']: return ("Downloaded Video File", "🎬", f"{f_format.upper()} file", "video")
    if 'audio' in t_type: return ("Downloaded Audio File", "🎵", "Audio file", "audio")
    if 'video' in t_type: return ("Downloaded Video File", "🎬", "Video file", "video")
    if 'clean' in t_type: return ("Generated Clean Transcript", "📄", "Clean transcript", "transcript")
    if 'unclean' in t_type: return ("Generated Timestamped Transcript", "🕒", "Timestamped transcript", "transcript")
    return ("Downloaded Content", "📁", "Content", "general")

# -------------------- FIXED Account Deletion Helpers -------------------
def delete_user_subscription(db: Session, user: User) -> bool:
    """
    Delete user subscription with proper error handling for missing database columns.
    """
    try:
        if stripe and getattr(user, "subscription_tier", "free") != "free":
            # Try to find and cancel Stripe subscription
            try:
                subscription = (
                    db.query(Subscription)
                    .filter(Subscription.user_id == user.id)
                    .order_by(Subscription.created_at.desc())
                    .first()
                )
                
                if subscription:
                    # Try to get stripe_subscription_id - handle missing column gracefully
                    stripe_sub_id = None
                    try:
                        stripe_sub_id = getattr(subscription, 'stripe_subscription_id', None)
                    except (AttributeError, OperationalError):
                        logger.warning("Subscription table missing stripe_subscription_id column")
                    
                    # Cancel in Stripe if we have the ID
                    if stripe_sub_id:
                        try:
                            stripe.Subscription.delete(stripe_sub_id)
                            logger.info(f"Cancelled Stripe subscription {stripe_sub_id} for user {user.id}")
                        except Exception as e:
                            logger.warning(f"Failed to cancel Stripe subscription: {e}")
                    
                    # Update subscription status
                    subscription.status = "cancelled"
                    subscription.cancelled_at = datetime.utcnow()
                    db.commit()
                    
            except OperationalError as e:
                if "no such column" in str(e).lower():
                    logger.warning(f"Database column missing during subscription deletion: {e}")
                    # Continue with deletion even if we can't access Stripe info
                else:
                    raise
        
        # Delete all subscription records for this user
        try:
            deleted_count = db.query(Subscription).filter(Subscription.user_id == user.id).delete()
            logger.info(f"Deleted {deleted_count} subscription records for user {user.id}")
            db.commit()
        except Exception as e:
            logger.warning(f"Error deleting subscription records: {e}")
            db.rollback()
            
        return True
        
    except Exception as e:
        logger.error(f"Error deleting user subscription: {e}")
        db.rollback()
        return False

def delete_user_data(db: Session, user: User) -> bool:
    """
    Delete user data and files with proper error handling.
    """
    try:
        # Delete download records
        try:
            deleted_count = db.query(TranscriptDownload).filter(TranscriptDownload.user_id == user.id).delete()
            logger.info(f"Deleted {deleted_count} download records for user {user.id}")
            db.commit()
        except Exception as e:
            logger.warning(f"Error deleting download records: {e}")
            db.rollback()
        
        # Delete user files
        try:
            user_files = list(DOWNLOADS_DIR.glob(f"*{user.id}*"))
            deleted_files = 0
            for file_path in user_files:
                try:
                    file_path.unlink(missing_ok=True)
                    deleted_files += 1
                except Exception as e:
                    logger.warning(f"Error deleting file {file_path}: {e}")
            logger.info(f"Deleted {deleted_files} user files for user {user.id}")
        except Exception as e:
            logger.warning(f"Error during file cleanup: {e}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error deleting user data: {e}")
        return False

# -------------------- Schemas -----------------------------------------------
class UserCreate(BaseModel): username: str; email: str; password: str
class UserResponse(BaseModel):
    id: int; username: Optional[str]=None; email: str; created_at: Optional[datetime]=None
    class Config: from_attributes = True
class Token(BaseModel): access_token: str; token_type: str
class TranscriptRequest(BaseModel): youtube_id: str; clean_transcript: bool=True; format: Optional[str]=None
class AudioRequest(BaseModel): youtube_id: str; quality: str="medium"
class VideoRequest(BaseModel): youtube_id: str; quality: str="720p"
class CancelRequest(BaseModel): at_period_end: Optional[bool] = True
class DeleteAccountResponse(BaseModel): message: str; deleted_at: str; user_email: str
class ChangePasswordRequest(BaseModel):
    # current_password: Optional[str] = None  # optional when must_change_password = True
    # new_password: str
    current_password: str
    new_password: str
    
# -------------------- Startup ------------------------------------------------
def _cleanup_stale_files_loop():
    keep_days = max(1, FILE_RETENTION_DAYS)
    suffixes = {".mp3",".m4a",".aac",".mp4",".mkv",".mov",".txt",".srt",".vtt"}
    while True:
        try:
            cutoff = time.time() - keep_days * 86400
            for p in DOWNLOADS_DIR.iterdir():
                try:
                    if not p.is_file(): continue
                    if p.suffix.lower() not in suffixes: continue
                    if p.stat().st_mtime < cutoff:
                        p.unlink(missing_ok=True)
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"stale-files cleanup error: {e}")
        time.sleep(12 * 3600)

@app.on_event("startup")
async def on_startup():
    initialize_database()
    run_startup_migrations(engine)    # <-- add this line
    # Spawn cleanup thread
    threading.Thread(target=_cleanup_stale_files_loop, daemon=True).start()
    logger.info("Environment: %s", ENVIRONMENT)
    logger.info("Backend started")

    # Debug routes list (optional)
    try:
        for r in app.routes:
            methods = getattr(r, "methods", None)
            path = getattr(r, "path", None)
            if path:
                logger.info("ROUTE %-18s %s", (",".join(sorted(methods)) if methods else ""), path)
    except Exception as e:
        logger.debug("route table print failed: %s", e)

# -------------------- Routes -------------------------------------------------
@app.get("/")
def root():
    return {"message": "YouTube Content Downloader API", "status": "running",
            "version": "3.3.0", "features": ["transcripts","audio","video","mobile","history","payments"],
            "downloads_path": str(DOWNLOADS_DIR)}

# -------------------- Users password change ----------------
@app.post("/users/change_password")
def change_password(req: ChangePasswordRequest,
                    current_user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    """
    - If current_user.must_change_password == True → current_password is optional.
    - Otherwise we require and verify current_password.
    """
    must_change = bool(getattr(current_user, "must_change_password", False))

    if not must_change:
        if not req.current_password:
            raise HTTPException(status_code=400, detail="Current password is required")
        if not verify_password(req.current_password, current_user.hashed_password):
            raise HTTPException(status_code=401, detail="Current password is incorrect")

    # Basic sanity on new password
    if not req.new_password or len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")

    # Update password + clear flag
    current_user.hashed_password = get_password_hash(req.new_password)
    if hasattr(current_user, "must_change_password"):
        current_user.must_change_password = False
    current_user.updated_at = datetime.utcnow() if hasattr(current_user, "updated_at") else None

    try:
        db.commit()
        db.refresh(current_user)
        logger.info(f"🔐 Password changed for {current_user.username} (must_change_password cleared={not must_change})")
        return {"status": "ok", "message": "Password updated", "must_change_password": False}
    except Exception as e:
        db.rollback()
        logger.error(f"change_password failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to update password")

# -------------------- User password change ----------------
@app.post("/user/change_password")
def change_password(
    req: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Verify old password
    if not verify_password(req.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    # Set new hash
    current_user.hashed_password = get_password_hash(req.new_password)
    # Clear the must-change flag
    try:
        current_user.must_change_password = False
    except Exception:
        pass

    db.commit(); db.refresh(current_user)
    logger.info("🔑 Password changed for user %s", current_user.username)
    return {"status": "ok"}

# -------------------- User Register (find-or-create Stripe customer) --------------------
@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    username = (user.username or "").strip()
    email = (user.email or "").strip().lower()

    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=400, detail="Username already exists.")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email already exists.")

    obj = User(
        username=username,
        email=email,
        hashed_password=get_password_hash(user.password),
        created_at=datetime.utcnow(),
        subscription_tier="free",
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)

    logger.info(f"✅ Registered new user: {username} ({email})")

    # --- Stripe: reuse existing customer by email if present; else create one ---
    if stripe:
        try:
            customer = None

            # Prefer the Search API (works in test & prod; falls back if unavailable)
            try:
                found = stripe.Customer.search(
                    query=f"email:'{email}' AND -deleted:'true'",
                    limit=1,
                )
                if getattr(found, "data", []):
                    customer = found.data[0]
            except Exception as e:
                logger.debug("Stripe customer.search unavailable/failure: %s", e)

            # Fallback: list by email
            if not customer:
                try:
                    listed = stripe.Customer.list(email=email, limit=1)
                    if getattr(listed, "data", []):
                        customer = listed.data[0]
                except Exception as e:
                    logger.debug("Stripe customer.list failure: %s", e)

            if customer:
                # Link to existing Stripe customer and enrich metadata (best-effort)
                obj.stripe_customer_id = customer["id"]
                try:
                    stripe.Customer.modify(
                        customer["id"],
                        name=username or customer.get("name"),
                        metadata={**(customer.get("metadata") or {}), "app_user_id": str(obj.id)},
                    )
                except Exception as e:
                    logger.debug("Stripe customer.modify warning: %s", e)
                db.commit()
                db.refresh(obj)
                logger.info("🔗 Linked existing Stripe customer %s to user %s", obj.stripe_customer_id, username)
            else:
                # Create a fresh customer (idempotent on our user id)
                created = stripe.Customer.create(
                    email=email,
                    name=username,
                    metadata={"app_user_id": str(obj.id)},
                    idempotency_key=f"user_register_{obj.id}",
                )
                obj.stripe_customer_id = created["id"]
                db.commit()
                db.refresh(obj)
                logger.info("✅ Created Stripe customer %s for user %s", obj.stripe_customer_id, username)

        except Exception as e:
            # Never block registration on Stripe issues
            logger.warning("Stripe customer link/create failed for %s: %s", email, e)

    return {
        "message": "User registered successfully.",
        "account": canonical_account(obj),
        "stripe_customer_id": getattr(obj, "stripe_customer_id", None),
    }

# -------------------- token endpoint --------------------
@app.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    username_input = form.username.strip()
    password_input = form.password

    user = db.query(User).filter(User.username == username_input).first()

    if not user:
        logger.warning(f"❌ Login failed: user not found for username='{username_input}'")
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    if not verify_password(password_input, user.hashed_password):
        logger.warning(f"❌ Login failed: password mismatch for username='{username_input}'")
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    logger.info(f"✅ Login successful for: {user.username}")

    token = create_access_token(
        {"sub": user.username},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    # ✅ include flag so UI can redirect to /change-password if required
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": canonical_account(user),
        "must_change_password": bool(getattr(user, "must_change_password", False)),
    }

@app.get("/users/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

# -------------------- Account Deletion (schema-tolerant) --------------------
@app.delete("/user/delete-account")
def delete_account(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    uid = int(current_user.id)
    email = (current_user.email or "unknown@unknown.com")
    warnings: list[str] = []

    # 1) Cancel Stripe subs by customer id (no DB reads)
    try:
        if stripe and getattr(current_user, "stripe_customer_id", None):
            try:
                subs = stripe.Subscription.list(customer=current_user.stripe_customer_id, limit=100)
                for sub in getattr(subs, "data", []):
                    try:
                        stripe.Subscription.delete(sub.id)
                    except Exception as e:
                        warnings.append(f"stripe cancel {sub.id}: {str(e)[:60]}")
            except Exception as e:
                warnings.append(f"stripe list: {str(e)[:60]}")
    except Exception as e:
        warnings.append(f"stripe: {str(e)[:60]}")

    # 2) Delete related DB rows WITHOUT selecting (no created_at touch)
    try:
        db.execute(sqla_delete(Subscription).where(Subscription.user_id == uid))
        db.execute(sqla_delete(TranscriptDownload).where(TranscriptDownload.user_id == uid))
        db.execute(sqla_delete(User).where(User.id == uid))
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"DB delete failed for user {uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete account")

    # 3) Best-effort file cleanup (after commit)
    try:
        deleted_files = 0
        for p in DOWNLOADS_DIR.glob(f"*{uid}*"):
            try:
                p.unlink(missing_ok=True)
                deleted_files += 1
            except Exception:
                pass
        if deleted_files == 0:
            logger.info(f"no cached files found for user {uid}")
    except Exception as e:
        warnings.append(f"file cleanup: {str(e)[:60]}")

    msg = "Account deleted successfully."
    if warnings:
        msg = f"Account deleted with warnings: {'; '.join(warnings)}"

    return {
        "message": msg,
        "deleted_at": datetime.utcnow().isoformat(),
        "user_email": email,
    }

# -------------------- Stripe webhook (verified + idempotent) ----------------
_IDEMP_STORE = {}
_IDEMP_TTL_SEC = 24 * 3600
_IDEMP_LOCK = threading.Lock()

def _idemp_seen(event_id: str) -> bool:
    now = time.time()
    with _IDEMP_LOCK:
        # purge old
        for k, ts in list(_IDEMP_STORE.items()):
            if now - ts > _IDEMP_TTL_SEC:
                _IDEMP_STORE.pop(k, None)
        if event_id in _IDEMP_STORE:
            return True
        _IDEMP_STORE[event_id] = now
        return False

@app.post("/webhook/stripe")
async def stripe_webhook_endpoint(request: Request):
    if not stripe or not os.getenv("STRIPE_SECRET_KEY"):
        raise HTTPException(status_code=503, detail="Stripe is not configured")
    secret = os.getenv("STRIPE_WEBHOOK_SECRET")
    if not secret:
        raise HTTPException(status_code=500, detail="Webhook secret not configured")

    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload=payload, sig_header=sig, secret=secret)  # type: ignore
    except Exception as e:
        logger.warning(f"Stripe webhook signature verification failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")

    if not event or not event.get("id"):
        raise HTTPException(status_code=400, detail="Invalid event payload")

    if _idemp_seen(event["id"]):
        logger.info(f"Stripe webhook duplicate event {event['id']} ignored")
        return {"status": "ok", "duplicate": True}

    # Attach verified event so handler can reuse
    request.state.verified_event = event
    # Delegate to existing handler (keeps your current logic)
    result = await handle_stripe_webhook(request) if hasattr(handle_stripe_webhook, "__call__") else {"status":"ok"}
    return result

# -------------------- Subscription management -------------------------------
def _latest_subscription(db: Session, user_id: int) -> Optional[Subscription]:
    try:
        return (
            db.query(Subscription)
            .filter(Subscription.user_id == user_id)
            .order_by(Subscription.created_at.desc())
            .first()
        )
    except Exception:
        return None

@app.post("/subscription/cancel")
def cancel_subscription(req: CancelRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if (current_user.subscription_tier or "free") == "free":
        raise HTTPException(status_code=400, detail="No active subscription to cancel.")

    sub = _latest_subscription(db, current_user.id)
    at_period_end = True if req.at_period_end is None else bool(req.at_period_end)

    stripe_updated = False
    if stripe and sub and hasattr(sub, 'stripe_subscription_id'):
        stripe_sub_id = getattr(sub, 'stripe_subscription_id', None)
        if stripe_sub_id:
            try:
                if at_period_end:
                    stripe.Subscription.modify(stripe_sub_id, cancel_at_period_end=True)
                    stripe_updated = True
                else:
                    stripe.Subscription.delete(stripe_sub_id)
                    stripe_updated = True
            except Exception as e:
                logger.warning("Stripe cancel failed for user %s: %s", current_user.username, e)

    if at_period_end:
        if sub:
            note = f"cancel_at_period_end=true; updated={datetime.utcnow().isoformat()}"
            sub.extra_data = (sub.extra_data or "") + ("\n" if sub.extra_data else "") + note
        result = {"status": "scheduled_cancellation", "at_period_end": True}
    else:
        if sub:
            sub.status = "cancelled"; sub.cancelled_at = datetime.utcnow()
        current_user.subscription_tier = "free"
        result = {"status": "cancelled", "at_period_end": False, "tier": "free"}

    try:
        db.commit(); db.refresh(current_user); db.refresh(sub) if sub else None
    except Exception:
        db.rollback()

    result.update({"stripe_updated": stripe_updated})
    return result

# -------------------- Transcript / Audio / Video (unchanged APIs) -----------
@app.post("/download_transcript")
@app.post("/download_transcript/")
def download_transcript(req: TranscriptRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    start = time.time()
    vid = extract_youtube_video_id(req.youtube_id)
    if not vid or len(vid) != 11: raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    if not check_internet(): raise HTTPException(status_code=503, detail="No internet connection available.")

    if req.format in ['srt', 'vtt']:
        usage_key = "unclean_transcripts"; file_format = req.format
    elif req.clean_transcript:
        usage_key = "clean_transcripts"; file_format = "txt"
    else:
        usage_key = "unclean_transcripts"; file_format = "txt"
    ok, used, limit = check_usage_limit(user, usage_key)
    if not ok:
        type_name = "SRT transcript" if req.format == 'srt' else "VTT transcript" if req.format == 'vtt' else "clean transcript" if req.clean_transcript else "timestamped transcript"
        raise HTTPException(status_code=403, detail=f"Monthly limit reached for {type_name} ({used}/{limit}).")

    text = get_transcript_youtube_api(vid, clean=req.clean_transcript, fmt=req.format)
    if not text: raise HTTPException(status_code=404, detail="No transcript found for this video.")

    new_usage = increment_user_usage(db, user, usage_key)
    proc = time.time() - start
    rec = create_download_record(db=db, user=user, kind=usage_key, youtube_id=vid, file_format=file_format, file_size=len(text), processing_time=proc)
    return {"transcript": text, "youtube_id": vid, "clean_transcript": req.clean_transcript, "format": req.format,
            "processing_time": round(proc, 2), "success": True, "usage_updated": new_usage, "usage_type": usage_key,
            "download_record_id": rec.id if rec else None, "account": canonical_account(user)}

def _touch_now(p: Path):
    try:
        now = time.time(); os.utime(p, (now, now))
    except Exception as e:
        logger.warning("Could not set mtime: %s", e)

@app.post("/download_audio/")
def download_audio(req: AudioRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    start = time.time()
    vid = extract_youtube_video_id(req.youtube_id)
    if not vid or len(vid) != 11: raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    if not check_internet(): raise HTTPException(status_code=503, detail="No internet connection available.")
    if not check_ytdlp_availability(): raise HTTPException(status_code=500, detail="Audio download service temporarily unavailable.")

    ok, used, limit = check_usage_limit(user, "audio_downloads")
    if not ok: raise HTTPException(status_code=403, detail=f"Monthly limit reached for audio downloads ({used}/{limit}).")

    info = {}
    try: info = get_video_info(vid) or {}
    except Exception as e: logger.warning("get_video_info failed: %s", e)

    final_name = f"{vid}_audio_{req.quality}.mp3"
    final_path = DOWNLOADS_DIR / final_name

    path = download_audio_with_ytdlp(vid, req.quality, output_dir=str(DOWNLOADS_DIR))
    if not path or not os.path.exists(path): raise HTTPException(status_code=404, detail="Failed to download audio.")
    downloaded = Path(path)
    fsize = downloaded.stat().st_size
    if fsize < 1000: raise HTTPException(status_code=500, detail="Downloaded audio appears corrupted.")

    if downloaded != final_path:
        try:
            if final_path.exists(): final_path.unlink()
            downloaded.rename(final_path)
        except Exception as e:
            logger.warning("Rename failed, using original name: %s", e)
            final_path = downloaded; final_name = downloaded.name; fsize = final_path.stat().st_size

    _touch_now(final_path)
    new_usage = increment_user_usage(db, user, "audio_downloads")
    proc = time.time() - start
    rec = create_download_record(db=db, user=user, kind="audio_downloads", youtube_id=vid, quality=req.quality, file_format="mp3", file_size=fsize, processing_time=proc)
    token = create_access_token_for_mobile(user.username)
    direct_url = f"/download-file/audio/{final_name}?auth={token}"
    return {"download_url": f"/files/{final_name}", "direct_download_url": direct_url, "youtube_id": vid, "quality": req.quality,
            "file_size": fsize, "file_size_mb": round(fsize / (1024 * 1024), 2), "filename": final_name, "local_path": str(final_path),
            "processing_time": round(proc, 2), "message": "Audio ready for download", "success": True,
            "title": info.get("title", "Unknown Title"), "uploader": info.get("uploader", "Unknown"), "duration": info.get("duration", 0),
            "usage_updated": new_usage, "usage_type": "audio_downloads",
            "download_record_id": rec.id if rec else None, "account": canonical_account(user)}

@app.post("/download_video/")
def download_video(req: VideoRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    start = time.time()
    vid = extract_youtube_video_id(req.youtube_id)
    if not vid or len(vid) != 11: raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    if not check_internet(): raise HTTPException(status_code=503, detail="No internet connection available.")
    if not check_ytdlp_availability(): raise HTTPException(status_code=500, detail="Video download service unavailable.")

    ok, used, limit = check_usage_limit(user, "video_downloads")
    if not ok: raise HTTPException(status_code=403, detail=f"Monthly limit reached for video downloads ({used}/{limit}).")

    info = {}
    try: info = get_video_info(vid) or {}
    except Exception as e: logger.warning("get_video_info failed: %s", e)

    final_name = f"{vid}_video_{req.quality}.mp4"
    final_path = DOWNLOADS_DIR / final_name

    path = download_video_with_ytdlp(vid, req.quality, output_dir=str(DOWNLOADS_DIR))
    if not path or not os.path.exists(path): raise HTTPException(status_code=404, detail="Failed to download video.")
    downloaded = Path(path)
    fsize = downloaded.stat().st_size
    if fsize < 10_000: raise HTTPException(status_code=500, detail="Downloaded video appears corrupted.")

    if downloaded != final_path:
        try:
            if final_path.exists(): final_path.unlink()
            downloaded.rename(final_path)
        except Exception as e:
            logger.warning("Rename failed, using original name: %s", e)
            final_path = downloaded; final_name = downloaded.name; fsize = final_path.stat().st_size

    _touch_now(final_path)
    new_usage = increment_user_usage(db, user, "video_downloads")
    proc = time.time() - start
    rec = create_download_record(db=db, user=user, kind="video_downloads", youtube_id=vid, quality=req.quality, file_format="mp4", file_size=fsize, processing_time=proc)
    token = create_access_token_for_mobile(user.username)
    direct_url = f"/download-file/video/{final_name}?auth={token}"
    return {"download_url": f"/files/{final_name}", "direct_download_url": direct_url, "youtube_id": vid, "quality": req.quality,
            "file_size": fsize, "file_size_mb": round(fsize / (1024 * 1024), 2), "filename": final_name, "local_path": str(final_path),
            "processing_time": round(proc, 2), "message": "Video ready for download", "success": True,
            "title": info.get("title", "Unknown Title"), "uploader": info.get("uploader", "Unknown"), "duration": info.get("duration", 0),
            "usage_updated": new_usage, "usage_type": "video_downloads",
            "download_record_id": rec.id if rec else None, "account": canonical_account(user)}

# -------------------- Secure file delivery ----------------------------------
@app.get("/download-file/{file_type}/{filename}")
async def download_file(request: Request, file_type: str, filename: str, auth: Optional[str] = Query(None), db: Session = Depends(get_db)):
    if file_type not in {"audio", "video"}: raise HTTPException(status_code=400, detail="Invalid file type")

    user = None
    if auth:
        try:
            payload = jwt.decode(auth, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username: user = get_user(db, username)
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid token")

    if not user:
        ah = request.headers.get("authorization", "")
        if ah.lower().startswith("bearer "):
            try:
                payload = jwt.decode(ah.split(" ",1)[1], SECRET_KEY, algorithms=[ALGORITHM])
                username = payload.get("sub")
                if username: user = get_user(db, username)
            except Exception:
                pass

    if not user: raise HTTPException(status_code=401, detail="Authentication required")

    file_path = (DOWNLOADS_DIR / filename).resolve()
    if not file_path.exists(): raise HTTPException(status_code=404, detail="File not found")
    if not str(file_path).startswith(str(DOWNLOADS_DIR.resolve())):
        raise HTTPException(status_code=403, detail="Access denied")

    size = file_path.stat().st_size
    mime = get_mobile_mime_type(str(file_path), file_type)
    safe_name = get_safe_filename(filename)

    if is_mobile_request(request):
        def gen():
            with open(file_path, "rb") as f:
                while True:
                    chunk = f.read(8192)
                    if not chunk: break
                    yield chunk
        headers = {
            "Content-Type": mime,
            "Content-Disposition": f'attachment; filename="{safe_name}"; filename*=UTF-8\'\'{safe_name}',
            "Content-Length": str(size),
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Accept-Ranges": "bytes",
            "X-Content-Type-Options": "nosniff",
        }
        return StreamingResponse(gen(), media_type=mime, headers=headers)

    headers = {"Content-Disposition": f'attachment; filename="{safe_name}"', "Content-Length": str(size), "Accept-Ranges": "bytes"}
    return FileResponse(path=str(file_path), media_type=mime, headers=headers, filename=safe_name)

# -------------------- Activity / Status / SPA (unchanged) -------------------
@app.get("/user/download-history")
def get_download_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(TranscriptDownload).filter(TranscriptDownload.user_id == current_user.id).order_by(TranscriptDownload.created_at.desc()).limit(50).all()
    hist = [{
        "id": d.id, "type": d.transcript_type, "video_id": d.youtube_id, "quality": d.quality or "default",
        "file_format": d.file_format or "unknown", "file_size": d.file_size or 0, "downloaded_at": d.created_at.isoformat() if d.created_at else None,
        "processing_time": d.processing_time or 0, "status": getattr(d, "status", "completed"), "language": getattr(d, "language", "en"),
    } for d in rows]
    return {"downloads": hist, "total_count": len(hist), "account": canonical_account(current_user)}

@app.get("/user/recent-activity")
@app.get("/user/recent-activity/")
def get_recent_activity(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(TranscriptDownload).filter(TranscriptDownload.user_id == current_user.id).order_by(TranscriptDownload.created_at.desc()).limit(15).all()
    activities = []
    for d in rows:
        action, icon, desc_prefix, category = classify_activity_by_format(d.transcript_type or "", d.file_format or "txt")
        description = f"{desc_prefix} for video {d.youtube_id}"
        if d.quality and d.quality != 'default': description += f" ({d.quality})"
        if d.file_size:
            size_mb = d.file_size / (1024 * 1024)
            description += f" - {size_mb:.1f}MB" if size_mb >= 1 else f" - {d.file_size / 1024:.0f}KB"
        activities.append({
            "id": d.id, "action": action, "description": description, "timestamp": d.created_at.isoformat() if d.created_at else None,
            "icon": icon, "type": d.transcript_type, "video_id": d.youtube_id, "file_format": d.file_format or "txt",
            "file_size": d.file_size, "quality": d.quality, "status": getattr(d, "status", "completed"),
            "category": category, "processing_time": d.processing_time
        })
    if not activities:
        activities.append({
            "id": 0, "action": "Account created", "description": f"Welcome to the app, {current_user.username}!",
            "timestamp": current_user.created_at.isoformat() if current_user.created_at else None, "type": "auth", "icon": "🎉", "category": "system"
        })
    return {"activities": activities, "total_count": len(activities), "account": canonical_account(current_user), "fetched_at": datetime.utcnow().isoformat()}

#---------------------------- Subscription status endpoint ---------------------------
@app.get("/subscription_status")
@app.get("/subscription_status/")
def subscription_status(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # ✅ Sync with Stripe if ?sync=1 is provided
    if request.query_params.get("sync") == "1":
        try:
            sync_user_subscription_from_stripe(current_user, db)
        except Exception as e:
            logger.warning(f"Stripe sync skipped (non-fatal): {e}")

    # ✅ Determine current tier and usage
    tier = getattr(current_user, "subscription_tier", "free") or "free"
    usage = {
        "clean_transcripts": getattr(current_user, "usage_clean_transcripts", 0) or 0,
        "unclean_transcripts": getattr(current_user, "usage_unclean_transcripts", 0) or 0,
        "audio_downloads": getattr(current_user, "usage_audio_downloads", 0) or 0,
        "video_downloads": getattr(current_user, "usage_video_downloads", 0) or 0,
    }

    # ✅ Define limits per plan
    PLAN_LIMITS = {
        "free":    {"clean_transcripts": 5,   "unclean_transcripts": 3,  "audio_downloads": 2,  "video_downloads": 1},
        "pro":     {"clean_transcripts": 100, "unclean_transcripts": 50, "audio_downloads": 50, "video_downloads": 20},
        "premium": {"clean_transcripts": float("inf"), "unclean_transcripts": float("inf"),
                    "audio_downloads": float("inf"),    "video_downloads": float("inf")},
    }

    limits = PLAN_LIMITS.get(tier, PLAN_LIMITS["free"])
    limits_display = {k: ("unlimited" if v == float("inf") else v) for k, v in limits.items()}

    return {
        "tier": tier,
        "status": "active" if tier != "free" else "inactive",
        "usage": usage,
        "limits": limits_display,
        "downloads_folder": str(DOWNLOADS_DIR),
        "account": canonical_account(current_user),
    }

#---------------------------- health endpoint ---------------------------
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": ENVIRONMENT,
        "services": {
            "youtube_api": "available",
            "stripe": "configured" if os.getenv("STRIPE_SECRET_KEY") else "not_configured",
            "file_system": "accessible",
            "yt_dlp": "available" if check_ytdlp_availability() else "unavailable",
        },
        "downloads_path": str(DOWNLOADS_DIR),
    }

#---------------------------- debug/users endpoint ---------------------------
@app.get("/debug/users")
def debug_users(db: Session = Depends(get_db)):
    if os.getenv("ENVIRONMENT") != "development":
        raise HTTPException(status_code=404, detail="Not found")
    users = db.query(User).all()
    return {"total_users": len(users),
            "users": [{"id": u.id, "username": u.username, "email": (u.email or '').strip().lower(),
                       "created_at": u.created_at.isoformat() if u.created_at else None,
                       "subscription_tier": getattr(u, "subscription_tier", "free"),
                       "is_active": getattr(u, "is_active", True)} for u in users]}

FRONTEND_BUILD = (Path(__file__).resolve().parents[1] / "frontend" / "build")
if FRONTEND_BUILD.exists():
    app.mount("/_spa", StaticFiles(directory=str(FRONTEND_BUILD), html=True), name="spa")
    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_catch_all(full_path: str):
        if full_path.startswith(("download_", "user/", "subscription_status", "health", "token", "register", "files/", "debug/")):
            raise HTTPException(status_code=404, detail="Not found")
        index_file = FRONTEND_BUILD / "index.html"
        if index_file.exists():
            return HTMLResponse(index_file.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="Frontend not built")

if __name__ == "__main__":
    import uvicorn
    print(f"Starting server on 0.0.0.0:8000 — files: {DOWNLOADS_DIR}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

