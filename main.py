from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
import os, re, time, socket, mimetypes, logging, jwt

from auth_deps import get_current_user # add near the other imports
from fastapi import FastAPI, HTTPException, Depends, status, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from pydantic import BaseModel
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from dotenv import load_dotenv, find_dotenv
load_dotenv()
# Load local overrides first, then base .env (local wins)
load_dotenv(dotenv_path=find_dotenv(".env.local"), override=True)
load_dotenv(dotenv_path=find_dotenv(".env"), override=False)

from youtube_transcript_api import YouTubeTranscriptApi

# ---------------- Optional Stripe (for cancellation/portal/checkout orchestration)
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

app = FastAPI(title="YouTube Content Downloader API", version="3.2.0")

# Optional timestamp normalizer
try:
    from timestamp_patch import EnsureUtcZMiddleware  # type: ignore
    app.add_middleware(EnsureUtcZMiddleware)
except Exception as e:
    logger.info("Timestamp middleware not loaded: %s", e)

# in backend/main.py (near the payments include)
from batch import router as batch_router

try:
    app.include_router(batch_router, tags=["batch"])
except Exception as e:
    logger.error("Could not include batch routes: %s", e)

# Payments
try:
    app.include_router(payment_router, tags=["payments"])
except Exception as e:
    logger.error("Could not include payment routes: %s", e)

# CORS
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

# -------------------- Downloads dir (no duplicates) -------------------------
# Default: private server cache folder (prevents double files showing in Windows Downloads)
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

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    cred_exc = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials",
                             headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username: raise cred_exc
    except Exception:
        raise cred_exc
    user = get_user(db, username)
    if not user: raise cred_exc
    return user

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

# -------------------- Transcript helpers ------------------------------------
def segments_to_vtt(transcript) -> str:
    def sec_to_vtt(ts: float) -> str:
        h = int(ts // 3600)
        m = int((ts % 3600) // 60)
        s = int(ts % 60)
        ms = int((ts - int(ts)) * 1000)
        return f"{h:02}:{m:02}:{s:02}.{ms:03}"
    lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    for seg in transcript:
        start = sec_to_vtt(seg.get("start", 0))
        end = sec_to_vtt(seg.get("start", 0) + seg.get("duration", 0))
        text = (seg.get("text") or "").replace("\n", " ").strip()
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)

def segments_to_srt(transcript) -> str:
    def sec_to_srt(ts: float) -> str:
        h = int(ts // 3600)
        m = int((ts % 3600) // 60)
        s = int(ts % 60)
        ms = int((ts - int(ts)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    out = []
    for i, seg in enumerate(transcript, start=1):
        start = sec_to_srt(seg.get("start", 0))
        end = sec_to_srt(seg.get("start", 0) + seg.get("duration", 0))
        text = (seg.get("text") or "").replace("\n", " ").strip()
        out.append(str(i))
        out.append(f"{start} --> {end}")
        out.append(text)
        out.append("")
    return "\n".join(out)

def get_transcript_youtube_api(video_id: str, clean: bool = True, fmt: Optional[str] = None) -> str:
    logger.info("Transcript for %s (clean=%s, fmt=%s)", video_id, clean, fmt)

    if not check_internet():
        raise HTTPException(status_code=503, detail="No internet connection available.")
    if not check_youtube():
        raise HTTPException(status_code=503, detail="Cannot reach YouTube right now.")

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
    except Exception as e:
        logger.error("YouTubeTranscriptApi failed: %s", e)
        # Try yt-dlp fallback
        try:
            fb = get_transcript_with_ytdlp(video_id, clean=clean)
            if fb:
                return fb
        except Exception as ee:
            logger.error("yt-dlp fallback also failed: %s", ee)
        # Bubble user-friendly error
        if "No transcripts were found" in str(e) or "TranscriptsDisabled" in str(e):
            raise HTTPException(status_code=404, detail="This video has no captions/transcripts.")
        raise HTTPException(status_code=404, detail="No transcript/captions found for this video.")

    if clean:
        text = " ".join((seg.get("text") or "").replace("\n", " ") for seg in transcript)
        text = " ".join(text.split())
        # paragraphize ~400+ chars on sentence boundaries
        out, cur, chars = [], [], 0
        for word in text.split():
            cur.append(word); chars += len(word) + 1
            if chars > 400 and word.endswith((".", "!", "?")):
                out.append(" ".join(cur)); cur, chars = [], 0
        if cur: out.append(" ".join(cur))
        return "\n\n".join(out)

    # Unclean
    if fmt == "srt":
        return segments_to_srt(transcript)
    if fmt == "vtt":
        return segments_to_vtt(transcript)
    # default "timestamped lines"
    lines = []
    for seg in transcript:
        t = int(seg.get("start", 0))
        timestamp = f"[{t//60:02d}:{t%60:02d}]"
        txt = (seg.get("text") or "").replace("\n", " ")
        lines.append(f"{timestamp} {txt}")
    return "\n".join(lines)

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

    db.commit()
    db.refresh(user)
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

# -------------------- Schemas -----------------------------------------------
class UserCreate(BaseModel): username: str; email: str; password: str
class UserResponse(BaseModel):
    id: int; username: Optional[str]=None; email: str; created_at: Optional[datetime]=None
    class Config: from_attributes = True
class Token(BaseModel): access_token: str; token_type: str
class TranscriptRequest(BaseModel): youtube_id: str; clean_transcript: bool=True; format: Optional[str]=None
class AudioRequest(BaseModel): youtube_id: str; quality: str="medium"
class VideoRequest(BaseModel): youtube_id: str; quality: str="720p"
class CancelRequest(BaseModel):
    at_period_end: Optional[bool] = True  # default: schedule cancel at period end

# -------------------- Startup ------------------------------------------------
@app.on_event("startup")
async def on_startup():
    initialize_database()
    logger.info("Environment: %s", ENVIRONMENT)
    logger.info("Backend started")
    # --- DEBUG: print route table on startup ---
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
            "version": "3.2.0", "features": ["transcripts","audio","video","mobile","history","payments"],
            "downloads_path": str(DOWNLOADS_DIR)}

@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.username == user.username).first():
        raise HTTPException(status_code=400, detail="Username already exists.")
    if db.query(User).filter(User.email == user.email).first():
        raise HTTPException(status_code=400, detail="Email already exists.")
    obj = User(username=user.username, email=(user.email or "").strip().lower(),
               hashed_password=get_password_hash(user.password), created_at=datetime.utcnow())
    db.add(obj); db.commit(); db.refresh(obj)
    return {"message": "User registered successfully.", "account": canonical_account(obj)}

@app.post("/token")
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    token = create_access_token({"sub": user.username}, timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    return {"access_token": token, "token_type": "bearer", "user": canonical_account(user)}

@app.get("/users/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

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
    """
    Schedule (default) or perform immediate cancellation.
    - If Stripe is configured & we have a subscription id -> reflect in Stripe too.
    - Always works even without Stripe by updating local records/downgrading.
    """
    if (current_user.subscription_tier or "free") == "free":
        raise HTTPException(status_code=400, detail="No active subscription to cancel.")

    sub = _latest_subscription(db, current_user.id)
    at_period_end = True if req.at_period_end is None else bool(req.at_period_end)

    # Try to update in Stripe if possible
    stripe_updated = False
    if stripe and sub and sub.stripe_subscription_id:
        try:
            if at_period_end:
                stripe.Subscription.modify(sub.stripe_subscription_id, cancel_at_period_end=True)
                stripe_updated = True
            else:
                stripe.Subscription.delete(sub.stripe_subscription_id)
                stripe_updated = True
        except Exception as e:
            logger.warning("Stripe cancel failed for user %s: %s", current_user.username, e)

    # Update local database regardless
    if at_period_end:
        # Keep plan active until period end — record intent in extra_data
        if sub:
            note = f"cancel_at_period_end=true; updated={datetime.utcnow().isoformat()}"
            sub.extra_data = (sub.extra_data or "") + ("\n" if sub.extra_data else "") + note
        result = {"status": "scheduled_cancellation", "at_period_end": True}
    else:
        # Immediate cancel: mark subscription and downgrade user
        if sub:
            sub.status = "cancelled"
            sub.cancelled_at = datetime.utcnow()
        current_user.subscription_tier = "free"
        result = {"status": "cancelled", "at_period_end": False, "tier": "free"}

    try:
        db.commit()
        db.refresh(current_user)
        if sub:
            db.refresh(sub)
    except Exception:
        db.rollback()

    result.update({"stripe_updated": stripe_updated})
    return result

# -------------------- Transcript --------------------------------------------
@app.post("/download_transcript")
@app.post("/download_transcript/")
def download_transcript(req: TranscriptRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    start = time.time()
    vid = extract_youtube_video_id(req.youtube_id)
    if not vid or len(vid) != 11:
        raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    if not check_internet():
        raise HTTPException(status_code=503, detail="No internet connection available.")

    usage_key = "clean_transcripts" if req.clean_transcript else "unclean_transcripts"
    ok, used, limit = check_usage_limit(user, usage_key)
    if not ok:
        tname = "clean" if req.clean_transcript else "unclean"
        raise HTTPException(status_code=403, detail=f"Monthly limit reached for {tname} transcripts ({used}/{limit}).")

    text = get_transcript_youtube_api(vid, clean=req.clean_transcript, fmt=req.format)
    if not text:
        raise HTTPException(status_code=404, detail="No transcript found for this video.")

    new_usage = increment_user_usage(db, user, usage_key)
    proc = time.time() - start
    rec = create_download_record(
        db=db, user=user, kind=usage_key, youtube_id=vid,
        file_format=(req.format if not req.clean_transcript else "txt"),
        file_size=len(text), processing_time=proc
    )
    return {
        "transcript": text,
        "youtube_id": vid,
        "clean_transcript": req.clean_transcript,
        "format": req.format,
        "processing_time": round(proc, 2),
        "success": True,
        "usage_updated": new_usage,
        "usage_type": usage_key,
        "download_record_id": rec.id if rec else None,
        "account": canonical_account(user),
    }

# -------------------- Audio --------------------------------------------------
def _touch_now(p: Path):
    try:
        now = time.time()
        os.utime(p, (now, now))
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
            final_path = downloaded
            final_name = downloaded.name
            fsize = final_path.stat().st_size

    _touch_now(final_path)  # <-- force Today on Windows

    new_usage = increment_user_usage(db, user, "audio_downloads")
    proc = time.time() - start
    rec = create_download_record(db=db, user=user, kind="audio_downloads", youtube_id=vid,
                                 quality=req.quality, file_format="mp3", file_size=fsize, processing_time=proc)

    token = create_access_token_for_mobile(user.username)
    direct_url = f"/download-file/audio/{final_name}?auth={token}"

    return {
        "download_url": f"/files/{final_name}",
        "direct_download_url": direct_url,
        "youtube_id": vid, "quality": req.quality,
        "file_size": fsize, "file_size_mb": round(fsize / (1024 * 1024), 2),
        "filename": final_name, "local_path": str(final_path),
        "processing_time": round(proc, 2), "message": "Audio ready for download", "success": True,
        "title": info.get("title", "Unknown Title"), "uploader": info.get("uploader", "Unknown"),
        "duration": info.get("duration", 0),
        "usage_updated": new_usage, "usage_type": "audio_downloads",
        "download_record_id": rec.id if rec else None, "account": canonical_account(user),
    }

# -------------------- Video --------------------------------------------------
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
            final_path = downloaded
            final_name = downloaded.name
            fsize = final_path.stat().st_size

    _touch_now(final_path)  # <-- force Today on Windows

    new_usage = increment_user_usage(db, user, "video_downloads")
    proc = time.time() - start
    rec = create_download_record(db=db, user=user, kind="video_downloads", youtube_id=vid,
                                 quality=req.quality, file_format="mp4", file_size=fsize, processing_time=proc)

    token = create_access_token_for_mobile(user.username)
    direct_url = f"/download-file/video/{final_name}?auth={token}"

    return {
        "download_url": f"/files/{final_name}",
        "direct_download_url": direct_url,
        "youtube_id": vid, "quality": req.quality,
        "file_size": fsize, "file_size_mb": round(fsize / (1024 * 1024), 2),
        "filename": final_name, "local_path": str(final_path),
        "processing_time": round(proc, 2), "message": "Video ready for download", "success": True,
        "title": info.get("title", "Unknown Title"), "uploader": info.get("uploader", "Unknown"),
        "duration": info.get("duration", 0),
        "usage_updated": new_usage, "usage_type": "video_downloads",
        "download_record_id": rec.id if rec else None, "account": canonical_account(user),
    }

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

# -------------------- History / activity / subscription ---------------------
@app.get("/user/download-history")
def get_download_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(TranscriptDownload).filter(TranscriptDownload.user_id == current_user.id).order_by(TranscriptDownload.created_at.desc()).limit(50).all()
    hist = [{
        "id": d.id,
        "type": d.transcript_type,
        "video_id": d.youtube_id,
        "quality": d.quality or "default",
        "file_format": d.file_format or "unknown",
        "file_size": d.file_size or 0,
        "downloaded_at": d.created_at.isoformat() if d.created_at else None,
        "processing_time": d.processing_time or 0,
        "status": getattr(d, "status", "completed"),
        "language": getattr(d, "language", "en"),
    } for d in rows]
    return {"downloads": hist, "total_count": len(hist), "account": canonical_account(current_user)}

@app.get("/user/recent-activity")
@app.get("/user/recent-activity/")
def get_recent_activity(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.query(TranscriptDownload).filter(TranscriptDownload.user_id == current_user.id).order_by(TranscriptDownload.created_at.desc()).limit(10).all()
    activities = []
    for d in rows:
        t = d.transcript_type
        if t == "clean_transcripts":
            action, icon, desc = "Generated clean transcript", "📄", f"Clean transcript for {d.youtube_id}"
        elif t == "unclean_transcripts":
            action, icon, desc = "Generated timestamped transcript", "🕒", f"Timestamped transcript for {d.youtube_id}"
        elif t == "audio_downloads":
            action, icon, desc = "Downloaded audio file", "🎵", f"{(d.quality or 'unknown').title()} MP3 from {d.youtube_id}"
        elif t == "video_downloads":
            action, icon, desc = "Downloaded video file", "🎬", f"{d.quality or 'unknown'} MP4 from {d.youtube_id}"
        else:
            action, icon, desc = f"Downloaded {t}", "📁", f"Content from {d.youtube_id}"
        activities.append({
            "id": d.id, "action": action, "description": desc,
            "timestamp": d.created_at.isoformat() if d.created_at else None,
            "type": "download", "icon": icon, "video_id": d.youtube_id, "file_size": d.file_size
        })
    if not activities:
        activities.append({
            "id": 0, "action": "Account created",
            "description": f"Welcome to the app, {current_user.username}!",
            "timestamp": current_user.created_at.isoformat() if current_user.created_at else None,
            "type": "auth", "icon": "🎉"
        })
    return {"activities": activities, "total_count": len(activities), "account": canonical_account(current_user)}

@app.get("/subscription_status")
@app.get("/subscription_status/")
def subscription_status(current_user: User = Depends(get_current_user)):
    tier = getattr(current_user, "subscription_tier", "free")
    usage = {
        "clean_transcripts": getattr(current_user, "usage_clean_transcripts", 0) or 0,
        "unclean_transcripts": getattr(current_user, "usage_unclean_transcripts", 0) or 0,
        "audio_downloads": getattr(current_user, "usage_audio_downloads", 0) or 0,
        "video_downloads": getattr(current_user, "usage_video_downloads", 0) or 0,
    }
    LIM = {
        "free": {"clean_transcripts": 5, "unclean_transcripts": 3, "audio_downloads": 2, "video_downloads": 1},
        "pro": {"clean_transcripts": 100, "unclean_transcripts": 50, "audio_downloads": 50, "video_downloads": 20},
        "premium": {"clean_transcripts": float("inf"), "unclean_transcripts": float("inf"), "audio_downloads": float("inf"), "video_downloads": float("inf")},
    }.get(tier, {})
    limits = {k: ("unlimited" if v == float("inf") else v) for k, v in LIM.items()}
    return {"tier": tier, "status": ("active" if tier != "free" else "inactive"), "usage": usage, "limits": limits, "downloads_folder": str(DOWNLOADS_DIR), "account": canonical_account(current_user)}

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

@app.get("/debug/users")
def debug_users(db: Session = Depends(get_db)):
    if os.getenv("ENVIRONMENT") != "development":
        raise HTTPException(status_code=404, detail="Not found")
    users = db.query(User).all()
    return {
        "total_users": len(users),
        "users": [{
            "id": u.id,
            "username": u.username,
            "email": (u.email or "").strip().lower(),
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "subscription_tier": getattr(u, "subscription_tier", "free"),
            "is_active": getattr(u, "is_active", True),
        } for u in users]
    }

# -------------------- SPA serving (updated with API path guarding) ----------
FRONTEND_BUILD = (Path(__file__).resolve().parents[1] / "frontend" / "build")
if FRONTEND_BUILD.exists():
    app.mount("/_spa", StaticFiles(directory=str(FRONTEND_BUILD), html=True), name="spa")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_catch_all(full_path: str):
        # Do NOT handle paths that start with API-ish prefixes
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


