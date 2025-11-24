# backend/main.py — FIXED: Transcript handling for FetchedTranscriptSnippet objects
# Key fixes:
# 1. All segment helper functions now handle both dict and object types
# 2. Better error handling with graceful fallbacks
# 3. More descriptive error messages
# 4. Improved yt-dlp integration
# 5. Removed duplicate /health route; added HEAD handler only
# 6. Avoided shadowing the 'stripe' module in top-level imports
# 7. Removed duplicate OAuth2PasswordRequestForm import

from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple
import re, time, socket, mimetypes, logging, jwt, threading, os
from collections import defaultdict, deque

from dotenv import load_dotenv, find_dotenv
load_dotenv()
load_dotenv(dotenv_path=find_dotenv(".env.local"), override=True)
load_dotenv(dotenv_path=find_dotenv(".env"),       override=False)

from fastapi import FastAPI, HTTPException, Depends, Request, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse, ORJSONResponse
from fastapi.staticfiles import StaticFiles
from passlib.context import CryptContext
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from sqlalchemy.orm import Session
from sqlalchemy import delete as sqla_delete
from sqlalchemy.exc import OperationalError, IntegrityError
import smtplib, ssl
from email.message import EmailMessage
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from pydantic import BaseModel, EmailStr
from email_utils import send_password_reset_email
from auth_utils import get_password_hash, verify_password, validate_password_strength
from subscription_sync import sync_user_subscription_from_stripe
from youtube_transcript_api import YouTubeTranscriptApi
from transcript_fetcher import get_transcript_smart as get_transcript_youtube_api

try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
    )
except Exception:
    class YouTubeTranscriptApi: pass
    class TranscriptsDisabled(Exception): pass
    class NoTranscriptFound(Exception): pass

try:
    from youtube_transcript_api import CouldNotRetrieveTranscript
except Exception:
    class CouldNotRetrieveTranscript(Exception): pass

# --- JWT secret (keep this) ---
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY env var is required. "
        'Generate one with:  python -c "import secrets; print(secrets.token_urlsafe(64))"'
    )

RESET_TOKEN_TTL_SECONDS = int(os.getenv("RESET_TOKEN_TTL_SECONDS", "3600"))
serializer = URLSafeTimedSerializer(SECRET_KEY)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg2" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

import logging
logger = logging.getLogger("youtube_trans_downloader")
logger.setLevel(logging.INFO)

driver = "sqlite" if DATABASE_URL.startswith("sqlite") else "postgres"
logger.info(f"✅ Config OK — using database driver: {driver}")

from models import (
    User, TranscriptDownload, Subscription, get_db, initialize_database, engine
)
from db_migrations import run_startup_migrations
from auth_deps import get_current_user
from webhook_handler import handle_stripe_webhook, fix_existing_premium_users
from payment import router as payment_router

try:
    from activity import router as activity_router
except Exception:
    activity_router = None

try:
    from batch import router as batch_router
except Exception:
    batch_router = None

from transcript_utils import (
    get_transcript_with_ytdlp,
    download_audio_with_ytdlp,
    download_video_with_ytdlp,
    check_ytdlp_availability,
    get_video_info,
)

CONTACT_TO = os.getenv("CONTACT_RECIPIENT", "onetechly@gmail.com")
CONTACT_FROM = os.getenv("CONTACT_FROM", "no-reply@onetechly.com")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("youtube_trans_downloader")

APP_ENV = os.getenv("APP_ENV", os.getenv("ENV", "development")).lower()
IS_PROD = APP_ENV == "production"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
FILE_RETENTION_DAYS = int(os.getenv("FILE_RETENTION_DAYS", "7"))

# Stripe: configure conditionally without shadowing the module name in imports
stripe = None
try:
    import stripe as _stripe
    if os.getenv("STRIPE_SECRET_KEY"):
        _stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        stripe = _stripe
except Exception:
    stripe = None

app = FastAPI(
    title="YouTube Content Downloader API",
    version="3.4.2",
    default_response_class=ORJSONResponse,
)

def ensure_stripe_customer_for_user(user: "User", db: Session) -> None:
    """
    Ensure this app user has a Stripe customer id.

    - If one already exists, do nothing.
    - Otherwise, try to find a Stripe customer by email.
    - If not found, create one.
    """
    if not stripe or not os.getenv("STRIPE_SECRET_KEY"):
        return

    if getattr(user, "stripe_customer_id", None):
        # Already linked
        return

    email = (user.email or "").strip().lower()
    username = (user.username or "").strip() or None

    if not email:
        return  # nothing we can do

    try:
        customer = None

        # 1) Try Customer.search (newer API)
        try:
            found = stripe.Customer.search(
                query=f"email:'{email}' AND -deleted:'true'",
                limit=1,
            )
            if getattr(found, "data", []):
                customer = found.data[0]
        except Exception as e:
            logger.debug("ensure_stripe_customer_for_user: customer.search failed: %s", e)

        # 2) Fallback to Customer.list
        if not customer:
            try:
                listed = stripe.Customer.list(email=email, limit=1)
                if getattr(listed, "data", []):
                    customer = listed.data[0]
            except Exception as e:
                logger.debug("ensure_stripe_customer_for_user: customer.list failed: %s", e)

        if customer:
            # Backfill existing Stripe customer
            user.stripe_customer_id = customer["id"]
            try:
                stripe.Customer.modify(
                    customer["id"],
                    name=username or customer.get("name"),
                    metadata={
                        **(customer.get("metadata") or {}),
                        "app_user_id": str(user.id),
                    },
                )
            except Exception:
                pass
            db.commit()
            db.refresh(user)
            logger.info("🔄 Linked existing Stripe customer %s to user %s", customer["id"], email)
            return

        # 3) No customer found → create a new one
        created = stripe.Customer.create(
            email=email,
            name=username,
            metadata={"app_user_id": str(user.id)},
            idempotency_key=f"user_backfill_{user.id}",
        )
        user.stripe_customer_id = created["id"]
        db.commit()
        db.refresh(user)
        logger.info("✅ Created Stripe customer %s for user %s", created["id"], email)

    except Exception as e:
        logger.warning("ensure_stripe_customer_for_user failed for %s: %s", email, e)
        db.rollback()

try:
    from timestamp_patch import EnsureUtcZMiddleware
    app.add_middleware(EnsureUtcZMiddleware)
except Exception as e:
    logger.info("Timestamp middleware not loaded: %s", e)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
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
        apply_csp_to_api_only: bool = True,
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
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers["X-Frame-Options"] = self.x_frame_options
        response.headers["Referrer-Policy"] = self.referrer_policy
        response.headers.setdefault("X-XSS-Protection", "0")
        response.headers.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        if self.server_header is not None:
            response.headers["Server"] = self.server_header
        if self.hsts:
            scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
            if (scheme or "").lower() == "https":
                hsts_value = f"max-age={self.hsts_max_age}; includeSubDomains"
                if self.hsts_preload:
                    hsts_value += "; preload"
                response.headers["Strict-Transport-Security"] = hsts_value
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

DEV_CSP = None

PROD_CSP = (
    "default-src 'self'; "
    "img-src 'self' data: blob:; "
    "media-src 'self' data: blob:; "
    "font-src 'self' data:; "
    "style-src 'self' 'unsafe-inline'; "
    # ✅ allow both Render subdomain and your custom API domain
    "connect-src 'self' https://api.onetechly.com https://youtube-trans-downloader-api.onrender.com https://api.stripe.com; "
    # (optional, if Stripe Elements/Checkout script is used)
    "script-src 'self' https://js.stripe.com; "
    "frame-src https://js.stripe.com; "
    "frame-ancestors 'none'; "
    "base-uri 'none'; "
)
app.add_middleware(
    SecurityHeadersMiddleware,
    csp=(PROD_CSP if IS_PROD else DEV_CSP),
    hsts=IS_PROD,
    hsts_max_age=63072000,
    referrer_policy="no-referrer",
    x_frame_options="DENY",
    permissions_policy="geolocation=(), microphone=(), camera=()",
    server_header="YCD",
    apply_csp_to_api_only=True,
)

class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.now = time.time
        self.buckets: Dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()
        self.enabled = os.getenv("RL_ENABLED", "true").lower() in {"1","true","yes","on"}
        self.default_window = int(os.getenv("RL_DEFAULT_WINDOW_SEC", "60"))
        self.default_max    = int(os.getenv("RL_DEFAULT_MAX", "120"))
        self.auth_window    = int(os.getenv("RL_AUTH_WINDOW_SEC", "60"))
        self.auth_max       = int(os.getenv("RL_AUTH_MAX", "10"))
        self.dl_window      = int(os.getenv("RL_DL_WINDOW_SEC", "60"))
        self.dl_max         = int(os.getenv("RL_DL_MAX", "40"))

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
                return ORJSONResponse(
                    {"detail": "Too Many Requests"},
                    status_code=429,
                    headers={"Retry-After": str(retry_after)},
                )
            q.append(now)
        return await call_next(request)

app.add_middleware(RateLimitMiddleware)

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

PUBLIC_ORIGINS = [
    "https://onetechly.com",
    "https://www.onetechly.com",
    "https://api.onetechly.com",   # add this
]

DEV_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://192.168.1.185:3000",
]

if ENVIRONMENT != "production":
    allow_origins = PUBLIC_ORIGINS + DEV_ORIGINS + ([FRONTEND_URL] if FRONTEND_URL else [])
else:
    allow_origins = PUBLIC_ORIGINS + ([FRONTEND_URL] if FRONTEND_URL else [])

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in allow_origins if o],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Disposition",
        "Content-Type",
        "Content-Length",
        "Content-Range",
    ],
)

print("✅ CORS enabled for origins:", allow_origins)

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

SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = dict(data)
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def create_access_token_for_mobile(username: str) -> str:
    return create_access_token({"sub": username}, timedelta(hours=2))

def get_user(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()

def canonical_account(user: User) -> Dict[str, Any]:
    return {"username": (user.username or "").strip(), "email": (user.email or "").strip().lower()}

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

_YT_ID_RE = re.compile(r'(?<![\w-])([A-Za-z0-9_-]{11})(?![\w-])')

def extract_youtube_video_id(text: str) -> str:
    t = (text or "").strip()
    pats = [
        r'(?:youtube\.com/watch\?[^#\s]*[?&]v=)([^&\n?#]{11})',
        r'(?:youtu\.be/)([^&\n?#/]{11})',
        r'(?:youtube\.com/embed/)([^&\n?#/]{11})',
        r'(?:youtube\.com/shorts/)([^&\n?#/]{11})',
    ]
    for p in pats:
        m = re.search(p, t)
        if m:
            cand = m.group(1)
            if _YT_ID_RE.fullmatch(cand):
                return cand
    m = _YT_ID_RE.search(t)
    return m.group(1) if m else ""

# ============================================================================
# FIXED: Transcript Helper Functions
# ============================================================================
# These functions now properly handle both dict and FetchedTranscriptSnippet objects

def _get_segment_value(seg: Any, key: str, default: Any = None) -> Any:
    if isinstance(seg, dict):
        return seg.get(key, default)
    else:
        return getattr(seg, key, default)

def _sec_to_vtt(ts: float) -> str:
    h = int(ts // 3600)
    m = int((ts % 3600) // 60)
    s = int(ts % 60)
    ms = int((ts - int(ts)) * 1000)
    return f"{h:02}:{m:02}:{s:02}.{ms:03}"

def segments_to_vtt(transcript) -> str:
    lines = ["WEBVTT", "Kind: captions", "Language: en", ""]
    try:
        for seg in transcript:
            start = _get_segment_value(seg, "start", 0)
            duration = _get_segment_value(seg, "duration", 0)
            text = _get_segment_value(seg, "text", "")
            if not text:
                continue
            start_ts = _sec_to_vtt(start)
            end_ts = _sec_to_vtt(start + duration)
            text_clean = text.replace("\n", " ").strip()
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(text_clean)
            lines.append("")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"segments_to_vtt error: {e}")
        raise

def segments_to_srt(transcript) -> str:
    def sec_to_srt(ts: float) -> str:
        h = int(ts // 3600)
        m = int((ts % 3600) // 60)
        s = int(ts % 60)
        ms = int((ts - int(ts)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    out = []
    try:
        for i, seg in enumerate(transcript, start=1):
            start = _get_segment_value(seg, "start", 0)
            duration = _get_segment_value(seg, "duration", 0)
            text = _get_segment_value(seg, "text", "")
            if not text:
                continue
            start_ts = sec_to_srt(start)
            end_ts = sec_to_srt(start + duration)
            text_clean = text.replace("\n", " ").strip()
            out.append(str(i))
            out.append(f"{start_ts} --> {end_ts}")
            out.append(text_clean)
            out.append("")
        return "\n".join(out)
    except Exception as e:
        logger.warning(f"segments_to_srt error: {e}")
        raise

_EN_PRIORITY = ["en", "en-US", "en-GB", "en-CA", "en-AU", "en-IE", "en-NZ"]

def _clean_plain_blocks(blocks: list[str]) -> str:
    out, cur, chars = [], [], 0
    for word in " ".join(blocks).split():
        cur.append(word)
        chars += len(word) + 1
        if chars > 400 and word.endswith((".", "!", "?")):
            out.append(" ".join(cur))
            cur, chars = [], 0
    if cur:
        out.append(" ".join(cur))
    return "\n\n".join(out)

def _format_timestamped(segments):
    lines = []
    try:
        for seg in segments:
            start = _get_segment_value(seg, "start", 0)
            text = _get_segment_value(seg, "text", "")
            if not text:
                continue
            t = int(start)
            text_clean = text.replace("\n", " ")
            lines.append(f"[{t // 60:02d}:{t % 60:02d}] {text_clean}")
        return "\n".join(lines)
    except Exception as e:
        logger.warning(f"_format_timestamped error: {e}")
        raise

# ============================================================================
# End of Fixed Transcript Functions
# ============================================================================

PLAN_LIMITS = {
    "free":    {"clean_transcripts": 5,   "unclean_transcripts": 3,  "audio_downloads": 2,  "video_downloads": 1},
    "pro":     {"clean_transcripts": 100, "unclean_transcripts": 50, "audio_downloads": 50, "video_downloads": 20},
    "premium": {"clean_transcripts": float("inf"), "unclean_transcripts": float("inf"),
                "audio_downloads": float("inf"),    "video_downloads": float("inf")},
}

def _now_utc() -> datetime:
    return datetime.utcnow().replace(tzinfo=timezone.utc)

def _to_naive_utc(d: Optional[datetime]) -> Optional[datetime]:
    if not d: return None
    if d.tzinfo is None: return d
    return d.astimezone(timezone.utc).replace(tzinfo=None)

def _compute_next_reset_date(user: User) -> datetime:
    anchor = _to_naive_utc(getattr(user, "subscription_started_at", None)) \
             or _to_naive_utc(getattr(user, "updated_at", None)) \
             or _to_naive_utc(getattr(user, "created_at", None)) \
             or datetime.utcnow()
    today = datetime.utcnow()
    day = min(anchor.day, 28)
    this_month_anniv = today.replace(day=day, hour=0, minute=0, second=0, microsecond=0)
    if today < this_month_anniv:
        return this_month_anniv
    y, m = (today.year + (today.month // 12), ((today.month % 12) + 1))
    return this_month_anniv.replace(year=y, month=m)

def _reset_usage_counters(user: User):
    user.usage_clean_transcripts = 0
    user.usage_unclean_transcripts = 0
    user.usage_audio_downloads = 0
    user.usage_video_downloads = 0

def ensure_monthly_reset_and_tier(db: Session, user: User) -> datetime:
    try:
        sub = (
            db.query(Subscription)
            .filter(Subscription.user_id == user.id)
            .order_by(Subscription.created_at.desc())
            .first()
        )
        if sub and getattr(sub, "status", "active") not in ("active", "trialing"):
            if user.subscription_tier != "free":
                logger.info("⬇️ Auto-downgrade %s -> free (status=%s)", user.email, sub.status)
            user.subscription_tier = "free"
    except Exception as e:
        logger.debug("subscription check failed: %s", e)

    next_reset = getattr(user, "next_usage_reset_at", None)
    now = datetime.utcnow()
    if not next_reset:
        next_reset = _compute_next_reset_date(user)
        user.next_usage_reset_at = next_reset

    if now >= next_reset:
        logger.info("🔁 Usage reset for %s (was due %s)", user.email, next_reset.isoformat())
        _reset_usage_counters(user)
        try:
            day = min((next_reset.day or 1), 28)
        except Exception:
            day = min((_compute_next_reset_date(user).day or 1), 28)
        nxt_month = (next_reset.year + (next_reset.month // 12), ((next_reset.month % 12) + 1))
        user.next_usage_reset_at = next_reset.replace(year=nxt_month[0], month=nxt_month[1], day=day)

    db.commit()
    db.refresh(user)
    return getattr(user, "next_usage_reset_at", _compute_next_reset_date(user))

def increment_user_usage(db: Session, user: User, usage_type: str) -> int:
    ensure_monthly_reset_and_tier(db, user)
    current = getattr(user, f"usage_{usage_type}", 0) or 0
    new_val = current + 1
    setattr(user, f"usage_{usage_type}", new_val)
    try:
        db.commit(); db.refresh(user)
    except Exception:
        db.rollback()
        raise
    return new_val

def check_usage_limit(user: User, usage_type: str) -> Tuple[bool, int, int]:
    tier = getattr(user, "subscription_tier", "free") or "free"
    limits = PLAN_LIMITS.get(tier, PLAN_LIMITS["free"])
    cur = getattr(user, f"usage_{usage_type}", 0) or 0
    limit = limits.get(usage_type, 0)
    return cur < limit, cur, (int(limit) if limit != float("inf") else int(1e12))

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

def classify_activity_by_format(transcript_type: str, file_format: str) -> tuple[str, str, str, str]:
    t_type = (transcript_type or "").lower()
    f_format = (file_format or "").lower()

    if f_format in ["mp3", "m4a", "aac", "wav"]:
        return ("Downloaded Audio File", "🎵", f"{f_format.upper()} file", "audio")
    if f_format in ["mp4", "mkv", "avi", "mov"]:
        return ("Downloaded Video File", "🎬", f"{f_format.upper()} file", "video")

    if "audio" in t_type:
        return ("Downloaded Audio File", "🎵", "Audio file", "audio")
    if "video" in t_type:
        return ("Downloaded Video File", "🎬", "Video file", "video")
    if f_format == "srt":
        return ("Generated SRT Transcript", "🕒", "SRT transcript", "transcript")
    if f_format == "vtt":
        return ("Generated VTT Transcript", "🕒", "VTT transcript", "transcript")
    if "clean" in t_type:
        return ("Generated Clean Transcript", "📄", "Clean text transcript", "transcript")
    if "unclean" in t_type or "timestamped" in t_type:
        return ("Generated Timestamped Transcript", "🕒", "Timestamped transcript", "transcript")

    return ("Downloaded Content", "📁", "Content", "general")

def delete_user_subscription(db: Session, user: User) -> bool:
    try:
        if stripe and getattr(user, "subscription_tier", "free") != "free":
            try:
                subscription = (
                    db.query(Subscription)
                    .filter(Subscription.user_id == user.id)
                    .order_by(Subscription.created_at.desc())
                    .first()
                )
                if subscription:
                    stripe_sub_id = getattr(subscription, 'stripe_subscription_id', None)
                    if stripe_sub_id:
                        try:
                            stripe.Subscription.delete(stripe_sub_id)
                            logger.info(f"Cancelled Stripe subscription {stripe_sub_id} for user {user.id}")
                        except Exception as e:
                            logger.warning(f"Failed to cancel Stripe subscription: {e}")
                    subscription.status = "cancelled"
                    subscription.cancelled_at = datetime.utcnow()
                    db.commit()
            except OperationalError as e:
                logger.warning(f"DB issue during subscription deletion: {e}")
        try:
            deleted_count = db.query(Subscription).filter(Subscription.user_id == user.id).delete()
            logger.info(f"Deleted {deleted_count} subscription rows for user {user.id}")
            db.commit()
        except Exception as e:
            logger.warning(f"Error deleting subscription rows: {e}")
            db.rollback()
        return True
    except Exception as e:
        logger.error(f"Error deleting user subscription: {e}")
        db.rollback()
        return False

def delete_user_data(db: Session, user: User) -> bool:
    try:
        try:
            deleted_count = db.query(TranscriptDownload).filter(TranscriptDownload.user_id == user.id).delete()
            logger.info(f"Deleted {deleted_count} download rows for user {user.id}")
            db.commit()
        except Exception as e:
            logger.warning(f"Error deleting download rows: {e}")
            db.rollback()
        try:
            user_files = list(DOWNLOADS_DIR.glob(f"*{user.id}*"))
            for file_path in user_files:
                try:
                    file_path.unlink(missing_ok=True)
                except Exception as e:
                    logger.warning(f"Error deleting file {file_path}: {e}")
        except Exception as e:
            logger.warning(f"File cleanup error: {e}")
        return True
    except Exception as e:
        logger.error(f"Error deleting user data: {e}")
        return False

def _send_contact_email(name: str, email: str, message: str) -> None:
    subject = f"[OneTechly] New Contact Message from {name or 'Unknown'}"
    body = (
        f"Name: {name}\n"
        f"Email: {email}\n\n"
        f"Message:\n{message}\n\n"
        f"— Sent {datetime.utcnow().isoformat()}Z"
    )

    sg_key = os.getenv("SENDGRID_API_KEY")
    if sg_key:
        try:
            import requests, json
            resp = requests.post(
                "https://api.sendgrid.com/v3/mail/send",
                headers={
                    "Authorization": f"Bearer {sg_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "personalizations": [{"to": [{"email": CONTACT_TO}]}],
                    "from": {"email": CONTACT_FROM},
                    "subject": subject,
                    "content": [{"type": "text/plain", "value": body}],
                }),
                timeout=10,
            )
            if resp.status_code not in (200, 202):
                raise RuntimeError(f"SendGrid error {resp.status_code}: {resp.text[:300]}")
            return
        except Exception as e:
            logger.warning(f"SendGrid failed, falling back to SMTP: {e}")

    smtp_user = os.getenv("GMAIL_SMTP_USERNAME")
    smtp_pass = os.getenv("GMAIL_APP_PASSWORD") or os.getenv("GMAIL_SMTP_PASSWORD")
    if smtp_user and smtp_pass:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = CONTACT_FROM or smtp_user
        msg["To"] = CONTACT_TO
        msg.set_content(body)
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context, timeout=15) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return

    raise RuntimeError("No mail provider configured (SENDGRID_API_KEY or Gmail SMTP creds required).")

class UserCreate(BaseModel):
    username: str; email: str; password: str

class UserResponse(BaseModel):
    id: int; username: Optional[str]=None; email: str; created_at: Optional[datetime]=None
    class Config: from_attributes = True

class Token(BaseModel): access_token: str; token_type: str
class TranscriptRequest(BaseModel): youtube_id: str; clean_transcript: bool=True; format: Optional[str]=None
class AudioRequest(BaseModel): youtube_id: str; quality: str="medium"
class VideoRequest(BaseModel): youtube_id: str; quality: str="720p"
class CancelRequest(BaseModel): at_period_end: Optional[bool] = True
class DeleteAccountResponse(BaseModel): message: str; deleted_at: str; user_email: str

# 1. Add new helper models
#----Architecture: client-side helper system (“YCD Desktop Helper”)---------
#Add these just after your other Pydantic models (near CancelRequest, DeleteAccountResponse, etc.):

class HelperTask(BaseModel):
    type: str                 # "transcript" | "audio" | "video"
    youtube_id: str
    clean_transcript: Optional[bool] = None
    format: Optional[str] = None       # "srt" | "vtt" | None
    quality: Optional[str] = None      # "high"/"medium"/"low" or "1080p"/"720p"/...

class HelperUsageReport(BaseModel):
    youtube_id: str
    usage_type: str                    # "clean_transcripts", "unclean_transcripts", "audio_downloads", "video_downloads"
    file_format: Optional[str] = "txt" # "txt", "srt", "vtt", "mp3", "mp4", ...
    quality: Optional[str] = "default"
    file_size: Optional[int] = 0
    processing_time: Optional[float] = 0.0

#------------------End of (“YCD Desktop Helper”) --------------------------

class ChangePasswordRequest(BaseModel):
    current_password: str
    new_password: str
class LoginJSON(BaseModel):
    username: str
    password: str
class ForgotPasswordIn(BaseModel):
    email: EmailStr

class ResetPasswordIn(BaseModel):
    token: str
    new_password: str

class ContactMessage(BaseModel):
    name: str
    email: str
    message: str

_email_re = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_contact_hits: Dict[str, deque] = defaultdict(deque)
_contact_lock = threading.Lock()

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

# ==============================================================================
# MAIN.PY UPDATES - ADD THESE TWO FUNCTIONS
# ==============================================================================
# Add these functions after your imports section, before @app.on_event("startup")

def _sanitize_stripe_key():
    """
    Remove whitespace/newlines from Stripe key.
    
    FIXES: InvalidHeader error caused by newlines in STRIPE_SECRET_KEY
    """
    key = os.getenv("STRIPE_SECRET_KEY")
    if key:
        clean_key = key.strip()
        if clean_key != key:
            os.environ["STRIPE_SECRET_KEY"] = clean_key
            logger.info("✅ Sanitized STRIPE_SECRET_KEY (removed %d whitespace chars)", 
                       len(key) - len(clean_key))
        
        # Re-initialize Stripe with clean key
        try:
            import stripe as _stripe
            _stripe.api_key = clean_key
            global stripe
            stripe = _stripe
            logger.info("✅ Stripe configured with clean API key")
        except Exception as e:
            logger.warning("⚠️  Could not reinitialize Stripe: %s", e)

def _hydrate_youtube_cookies_to_tmp() -> None:
    """
    Normalize cookie env so yt-dlp always gets a writable, ephemeral file.
    
    Priority:
      1. YT_COOKIES_B64 (preferred) -> decode to /tmp/yt-dlp/cookies.txt
      2. YT_COOKIES_FILE (fallback) -> copy to /tmp if in read-only mount
    
    Also sets cache dirs to /tmp to avoid permission issues.
    
    FIXES: Cookie file not accessible, YouTube bot detection
    """
    import base64
    import tempfile
    import shutil
    
    # Keep yt-dlp caches ephemeral (writable)
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("YTDLP_HOME", "/tmp/yt-dlp")
    os.environ.setdefault("YT_DLP_DIR", "/tmp/yt-dlp")
    
    # Strategy 1: Decode YT_COOKIES_B64 env var
    b64 = (os.getenv("YT_COOKIES_B64") or "").strip()
    if b64:
        try:
            # Decode base64
            raw = base64.b64decode(b64)
            
            # Normalize line endings for Netscape cookie file format
            raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            
            # Write to /tmp (guaranteed writable)
            target_dir = Path("/tmp/yt-dlp")
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / "cookies.txt"
            
            with open(target, "wb") as f:
                f.write(raw)
            
            # Verify
            if target.exists() and target.stat().st_size > 10:
                os.environ["YT_COOKIES_FILE"] = str(target)
                logger.info("✅ Decoded YT_COOKIES_B64 to %s (%d bytes)", 
                           target, len(raw))
                return
            else:
                logger.warning("⚠️  Decoded cookies file is empty or missing")
                
        except Exception as e:
            logger.error("❌ Failed to decode YT_COOKIES_B64: %s", e, exc_info=True)
    
    # Strategy 2: Copy existing YT_COOKIES_FILE if in read-only location
    fpath = (os.getenv("YT_COOKIES_FILE") or "").strip()
    if fpath and os.path.exists(fpath):
        try:
            # If mounted in read-only location, copy to /tmp
            if fpath.startswith(("/etc/", "/run/")):
                target_dir = Path("/tmp/yt-dlp")
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / "cookies.txt"
                
                shutil.copyfile(fpath, target)
                os.environ["YT_COOKIES_FILE"] = str(target)
                logger.info("✅ Copied YT_COOKIES_FILE from %s to %s", fpath, target)
            else:
                # Already writable
                logger.info("✅ Using YT_COOKIES_FILE: %s", fpath)
        except Exception as e:
            logger.warning("⚠️  Could not copy YT_COOKIES_FILE to /tmp: %s", e)
    else:
        if not b64 and not fpath:
            logger.warning("⚠️  No YouTube cookies configured (YT_COOKIES_B64 or YT_COOKIES_FILE)")
            logger.warning("⚠️  Transcript downloads may be blocked by YouTube")

# ==============================================================================
# REPLACE YOUR EXISTING @app.on_event("startup") WITH THIS
# ==============================================================================

@app.on_event("startup")
async def on_startup():
    """
    Application startup tasks.
    
    CRITICAL ORDER:
    1. Fix environment variables (Stripe + cookies) FIRST
    2. Initialize database
    3. Start background tasks
    4. Log startup info
    """
    # Step 1: Fix environment variables BEFORE anything else
    _sanitize_stripe_key()          # Fix Stripe newline issue
    _hydrate_youtube_cookies_to_tmp()  # Fix YouTube cookie access
    
    # Step 2: Initialize database
    initialize_database()
    run_startup_migrations(engine)
    
    # Step 3: Start background cleanup task
    threading.Thread(target=_cleanup_stale_files_loop, daemon=True).start()
    
    # Step 4: Log startup information
    logger.info("=" * 60)
    logger.info("🚀 YouTube Content Downloader API Starting")
    logger.info("=" * 60)
    logger.info("📝 Environment: %s", ENVIRONMENT)
    logger.info("📁 Downloads directory: %s", DOWNLOADS_DIR)
    logger.info("🗄️  Database: %s", "PostgreSQL" if "postgres" in DATABASE_URL else "SQLite")
    
    # Log service availability
    cookie_file = os.getenv("YT_COOKIES_FILE")
    logger.info("🍪 Cookies configured: %s", bool(cookie_file and os.path.exists(cookie_file)))
    if cookie_file and os.path.exists(cookie_file):
        logger.info("   └─ Path: %s", cookie_file)
        logger.info("   └─ Size: %d bytes", os.path.getsize(cookie_file))
    
    logger.info("💳 Stripe configured: %s", bool(stripe and os.getenv("STRIPE_SECRET_KEY")))
    logger.info("🎬 yt-dlp available: %s", check_ytdlp_availability())
    logger.info("🌐 Internet: %s", check_internet())
    
    logger.info("=" * 60)
    logger.info("✅ Backend started successfully")
    logger.info("=" * 60)

@app.get("/")
def root():
    return {"message": "YouTube Content Downloader API", "status": "running",
            "version": "3.4.2", "features": ["transcripts","audio","video","mobile","history","payments"],
            "downloads_path": str(DOWNLOADS_DIR)}

#------------------------New register endpoint with debbug (DeepSeek)-------------------------
@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    # Enhanced debug logging
    logger.info(f"🔵 REGISTRATION STARTED - Username: {user.username}, Email: {user.email}")
    
    username = (user.username or "").strip()
    email = (user.email or "").strip().lower()

    logger.info(f"🔵 Processing registration - Username: '{username}', Email: '{email}'")

    # Check for existing username with debug
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        logger.warning(f"❌ REGISTRATION FAILED - Username already exists: {username}")
        raise HTTPException(status_code=400, detail="Username already exists.")
    
    # Check for existing email with debug  
    existing_email = db.query(User).filter(User.email == email).first()
    if existing_email:
        logger.warning(f"❌ REGISTRATION FAILED - Email already exists: {email}")
        raise HTTPException(status_code=400, detail="Email already exists.")

    logger.info(f"🔵 No existing user found - Creating new user account")

    # Create user object
    try:
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
        logger.info(f"✅ DATABASE SUCCESS - Registered new user: {username} ({email}) with ID: {obj.id}")
    except Exception as e:
        logger.error(f"❌ DATABASE ERROR - Failed to create user {username}: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error during registration")

    # Stripe integration with enhanced logging
    if stripe:
        try:
            logger.info(f"🔵 Starting Stripe integration for user {obj.id}")
            customer = None
            
            # Search for existing Stripe customer
            try:
                logger.info(f"🔵 Searching for existing Stripe customer with email: {email}")
                found = stripe.Customer.search(query=f"email:'{email}' AND -deleted:'true'", limit=1)
                if getattr(found, "data", []): 
                    customer = found.data[0]
                    logger.info(f"✅ Found existing Stripe customer: {customer['id']}")
            except Exception as e:
                logger.warning(f"⚠️ Stripe customer search failed for {email}: {str(e)}")

            # Fallback: list customers
            if not customer:
                try:
                    logger.info(f"🔵 Fallback: listing customers with email: {email}")
                    listed = stripe.Customer.list(email=email, limit=1)
                    if getattr(listed, "data", []): 
                        customer = listed.data[0]
                        logger.info(f"✅ Found existing Stripe customer via list: {customer['id']}")
                except Exception as e:
                    logger.warning(f"⚠️ Stripe customer list failed for {email}: {str(e)}")

            # Update existing customer or create new one
            if customer:
                logger.info(f"🔵 Updating existing Stripe customer: {customer['id']}")
                obj.stripe_customer_id = customer["id"]
                try:
                    stripe.Customer.modify(
                        customer["id"], 
                        name=username or customer.get("name"),
                        metadata={**(customer.get("metadata") or {}), "app_user_id": str(obj.id)}
                    )
                    logger.info(f"✅ Successfully updated Stripe customer: {customer['id']}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to update Stripe customer {customer['id']}: {str(e)}")
                
                db.commit()
                db.refresh(obj)
            else:
                logger.info(f"🔵 Creating new Stripe customer for user {obj.id}")
                try:
                    created = stripe.Customer.create(
                        email=email, 
                        name=username, 
                        metadata={"app_user_id": str(obj.id)},
                        idempotency_key=f"user_register_{obj.id}",
                    )
                    obj.stripe_customer_id = created["id"]
                    db.commit()
                    db.refresh(obj)
                    logger.info(f"✅ Successfully created new Stripe customer: {created['id']}")
                except Exception as e:
                    logger.error(f"❌ Stripe customer creation failed for {email}: {str(e)}")
                    # Don't fail registration if Stripe fails
                    logger.info("🟡 Continuing registration without Stripe integration")

        except Exception as e:
            logger.error(f"❌ STRIPE INTEGRATION FAILED for {email}: {str(e)}")
            # Continue without failing the registration
            logger.info("🟡 Registration continuing despite Stripe failure")

    logger.info(f"🎉 REGISTRATION COMPLETED SUCCESSFULLY - User: {username} ({email})")
    
    return {
        "message": "User registered successfully.", 
        "account": canonical_account(obj),
        "stripe_customer_id": getattr(obj, "stripe_customer_id", None)
    }

@app.post("/auth/forgot-password")
def forgot_password(payload: ForgotPasswordIn):
    from models import SessionLocal
    db = SessionLocal()
    try:
        def get_user_by_email(db, email):
            return db.query(User).filter(User.email == email).first()
        user = get_user_by_email(db, payload.email)
        if user:
            token = serializer.dumps({"email": payload.email})
            reset_link = f"{FRONTEND_URL}/reset?token={token}"
            try:
                send_password_reset_email(payload.email, reset_link)
            except Exception as e:
                logger.exception("Failed to send reset email; link still valid")
        return {"ok": True}
    finally:
        db.close()

@app.post("/auth/reset-password")
def reset_password(payload: ResetPasswordIn):
    try:
        data = serializer.loads(payload.token, max_age=RESET_TOKEN_TTL_SECONDS)
        email = data.get("email")
    except SignatureExpired:
        raise HTTPException(status_code=400, detail="Reset link expired")
    except BadSignature:
        raise HTTPException(status_code=400, detail="Reset link invalid")

    from models import SessionLocal
    db = SessionLocal()
    try:
        def get_user_by_email(db, email):
            return db.query(User).filter(User.email == email).first()
        user = get_user_by_email(db, email)
        if not user:
            raise HTTPException(status_code=400, detail="Reset link invalid")

        user.hashed_password = get_password_hash(payload.new_password)
        db.add(user)
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@app.post("/token")
def token_login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    username_input = (form.username or "").strip()
    password_input = form.password or ""

    user = db.query(User).filter(User.username == username_input).first()
    if not user or not verify_password(password_input, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    # 🔹 Backfill Stripe customer id if missing
    ensure_stripe_customer_for_user(user, db)

    token = create_access_token(
        {"sub": user.username},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": canonical_account(user),
        "must_change_password": bool(getattr(user, "must_change_password", False)),
    }


@app.post("/token_json")
def token_login_json(req: LoginJSON, db: Session = Depends(get_db)):
    username_input = (req.username or "").strip()
    password_input = req.password or ""

    user = db.query(User).filter(User.username == username_input).first()
    if not user or not verify_password(password_input, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    # 🔹 Backfill Stripe customer id if missing
    ensure_stripe_customer_for_user(user, db)

    token = create_access_token(
        {"sub": user.username},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {
        "access_token": token,
        "token_type": "bearer",
        "user": canonical_account(user),
        "must_change_password": bool(getattr(user, "must_change_password", False)),
    }

@app.get("/users/me", response_model=UserResponse)
def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/user/change_password")
def change_password(req: ChangePasswordRequest, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if not verify_password(req.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    if not req.new_password or len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")
    current_user.hashed_password = get_password_hash(req.new_password)
    try:
        current_user.must_change_password = False
    except Exception:
        pass
    db.commit(); db.refresh(current_user)
    logger.info("🔑 Password changed for user %s", current_user.username)
    return {"status": "ok"}

@app.delete("/user/delete-account")
def delete_account(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    uid = int(current_user.id)
    email = (current_user.email or "unknown@unknown.com")

    try:
        if stripe and getattr(current_user, "stripe_customer_id", None):
            subs = stripe.Subscription.list(customer=current_user.stripe_customer_id, limit=100)
            for sub in getattr(subs, "data", []):
                try:
                    stripe.Subscription.delete(sub.id)
                except Exception:
                    pass
    except Exception:
        pass

    try:
        db.execute(sqla_delete(Subscription).where(Subscription.user_id == uid))
        db.execute(sqla_delete(TranscriptDownload).where(TranscriptDownload.user_id == uid))
        db.execute(sqla_delete(User).where(User.id == uid))
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"DB delete failed for user {uid}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete account")

    try:
        for p in DOWNLOADS_DIR.glob(f"*{uid}*"):
            try: p.unlink(missing_ok=True)
            except Exception: pass
    except Exception:
        pass

    return {"message": "Account deleted successfully.", "deleted_at": datetime.utcnow().isoformat(), "user_email": email}

_IDEMP_STORE: Dict[str, float] = {}
_IDEMP_TTL_SEC = 24 * 3600
_IDEMP_LOCK = threading.Lock()

def _idemp_seen(event_id: str) -> bool:
    now = time.time()
    with _IDEMP_LOCK:
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
        event = stripe.Webhook.construct_event(payload=payload, sig_header=sig, secret=secret)
    except Exception as e:
        logger.warning(f"Stripe webhook signature verification failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")

    if not event or not event.get("id"):
        raise HTTPException(status_code=400, detail="Invalid event payload")

    if _idemp_seen(event["id"]):
        logger.info(f"Stripe webhook duplicate event {event['id']} ignored")
        return {"status": "ok", "duplicate": True}

    request.state.verified_event = event
    result = await handle_stripe_webhook(request) if hasattr(handle_stripe_webhook, "__call__") else {"status":"ok"}
    return result

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

@app.get("/debug/probe/{video_id}")
def debug_probe(video_id: str):
    result = {
        "video_id": video_id,
        "internet": check_internet(),
        "youtube_reachable": check_youtube(),
        "strategies": {"direct_en": None, "list_pick": None, "translate_en": None, "yt_dlp": None},
        "errors": {}
    }

    try:
        seg = YouTubeTranscriptApi.get_transcript(video_id, languages=["en", "en-US", "en-GB"])
        txt = " ".join(_get_segment_value(s, "text", "") for s in seg)
        result["strategies"]["direct_en"] = {"segments": len(seg), "preview": txt[:200]}
    except Exception as e:
        result["errors"]["direct_en"] = str(e)

    try:
        listing = YouTubeTranscriptApi.list_transcripts(video_id)
        authored = generated = None
        for t in listing:
            lang = (getattr(t, "language_code", "") or "").lower()
            is_gen = bool(getattr(t, "is_generated", False))
            if lang in {"en", "en-us", "en-gb"} and not is_gen and authored is None:
                authored = t
            if lang in {"en", "en-us", "en-gb"} and is_gen and generated is None:
                generated = t
        picked = authored or generated
        if picked:
            seg = picked.fetch()
            txt = " ".join(_get_segment_value(s, "text", "") for s in seg)
            result["strategies"]["list_pick"] = {
                "picked": "authored_en" if picked is authored else "generated_en",
                "segments": len(seg),
                "preview": txt[:200],
            }
        else:
            result["strategies"]["list_pick"] = "no_english_track"
    except Exception as e:
        result["errors"]["list_pick"] = str(e)

    try:
        listing = YouTubeTranscriptApi.list_transcripts(video_id)
        candidate = None
        for t in listing:
            if getattr(t, "is_generated", False) and "en" in (getattr(t, "translation_languages", []) or []):
                candidate = t; break
        if candidate:
            seg = candidate.translate("en").fetch()
            txt = " ".join(_get_segment_value(s, "text", "") for s in seg)
            result["strategies"]["translate_en"] = {"segments": len(seg), "preview": txt[:200]}
        else:
            result["strategies"]["translate_en"] = "no_generated_translatable_to_en"
    except Exception as e:
        result["errors"]["translate_en"] = str(e)

    try:
        fb = get_transcript_with_ytdlp(video_id, clean=True)
        result["strategies"]["yt_dlp"] = {"found": bool(fb), "preview": (fb or "")[:200]}
    except Exception as e:
        result["errors"]["yt_dlp"] = str(e)

    return result

class _AuthForm(BaseModel):
    pass

def _touch_now(p: Path):
    try:
        now = time.time(); os.utime(p, (now, now))
    except Exception as e:
        logger.warning("Could not set mtime: %s", e)

#2. Add a new endpoint for helper usage reporting
#----Architecture: client-side helper system (“YCD Desktop Helper”)---------
#Put this somewhere after those models (before your download endpoints is fine):

@app.post("/helper/report_usage")
def helper_report_usage(
    report: HelperUsageReport,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Called by the frontend AFTER YCD Desktop Helper successfully finishes
    a download on the user's device.

    This keeps:
      - usage counters
      - download history
    in sync with your subscription logic.
    """
    ensure_monthly_reset_and_tier(db, current_user)

    usage_type = (report.usage_type or "").strip()
    if usage_type not in {
        "clean_transcripts",
        "unclean_transcripts",
        "audio_downloads",
        "video_downloads",
    }:
        raise HTTPException(status_code=400, detail="Invalid usage_type")

    ok, used, limit = check_usage_limit(current_user, usage_type)
    if not ok:
        label = (
            "clean transcripts" if usage_type == "clean_transcripts"
            else "timestamped transcripts" if usage_type == "unclean_transcripts"
            else "audio downloads" if usage_type == "audio_downloads"
            else "video downloads" if usage_type == "video_downloads"
            else "usage"
        )
        raise HTTPException(
            status_code=403,
            detail=f"Monthly limit reached for {label} ({used}/{limit}).",
        )

    new_usage = increment_user_usage(db, current_user, usage_type)
    rec = create_download_record(
        db=db,
        user=current_user,
        kind=usage_type,
        youtube_id=report.youtube_id,
        quality=report.quality or "default",
        file_format=report.file_format or "txt",
        file_size=report.file_size or 0,
        processing_time=report.processing_time or 0.0,
    )

    return {
        "ok": True,
        "usage_updated": new_usage,
        "usage_type": usage_type,
        "download_record_id": rec.id if rec else None,
        "account": canonical_account(current_user),
    }

#------------------End of (“YCD Desktop Helper”) --------------------------

# 👉 IMPORTANT: DO NOT DELETE THIS FUNCTION
# @app.post("/download_transcript")
# @app.post("/download_transcript/")
# def download_transcript(req: TranscriptRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
#     """
#     Download transcript with fixed error handling.
#     Now properly handles FetchedTranscriptSnippet objects.
#     """
#     ensure_monthly_reset_and_tier(db, user)
#     start = time.time()
#     vid = extract_youtube_video_id(req.youtube_id)
#     if not vid or len(vid) != 11: 
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
#     if not check_internet(): 
#         raise HTTPException(status_code=503, detail="No internet connection available.")

#     if req.format in ['srt', 'vtt']:
#         usage_key = "unclean_transcripts"
#         file_format = req.format
#     elif req.clean_transcript:
#         usage_key = "clean_transcripts"
#         file_format = "txt"
#     else:
#         usage_key = "unclean_transcripts"
#         file_format = "txt"
        
#     ok, used, limit = check_usage_limit(user, usage_key)
#     if not ok:
#         type_name = "SRT transcript" if req.format == 'srt' else "VTT transcript" if req.format == 'vtt' else "clean transcript" if req.clean_transcript else "timestamped transcript"
#         raise HTTPException(status_code=403, detail=f"Monthly limit reached for {type_name} ({used}/{limit}).")

#     # Get transcript with proper error handling
#     try:
#         text = get_transcript_youtube_api(vid, clean=req.clean_transcript, fmt=req.format)
#     except Exception as e:
#         error_msg = str(e)
#         logger.error(f"Transcript fetch failed for {vid}: {error_msg}")
        
#         # Check if it's a cloud provider IP block
#         if "blocking" in error_msg.lower() or "cloud provider" in error_msg.lower():
#             raise HTTPException(
#                 status_code=503,
#                 detail="YouTube is temporarily blocking transcript access from our servers. This video may not have captions available, or please try again in a few minutes. Note: Some videos work better than others depending on YouTube's restrictions."
#             )
        
#         # Check if no captions available
#         if "no captions" in error_msg.lower() or "not have captions" in error_msg.lower():
#             raise HTTPException(
#                 status_code=404,
#                 detail="This video does not have captions/transcripts available. Please try a different video."
#             )
        
#         # Generic error
#         raise HTTPException(
#             status_code=404,
#             detail=f"Could not retrieve transcript for this video. The video may not have captions available, or YouTube may be blocking access."
#         )
    
#     if not text:
#         raise HTTPException(status_code=404, detail="No transcript found for this video.")

#     new_usage = increment_user_usage(db, user, usage_key)
#     proc = time.time() - start
#     rec = create_download_record(
#         db=db, 
#         user=user, 
#         kind=usage_key, 
#         youtube_id=vid, 
#         file_format=file_format, 
#         file_size=len(text), 
#         processing_time=proc
#     )
    
#     return {
#         "transcript": text, 
#         "youtube_id": vid, 
#         "clean_transcript": req.clean_transcript, 
#         "format": req.format,
#         "processing_time": round(proc, 2), 
#         "success": True, 
#         "usage_updated": new_usage, 
#         "usage_type": usage_key,
#         "download_record_id": rec.id if rec else None, 
#         "account": canonical_account(user)
#     }

# 3. Update /download_transcript to return helper task when YouTube blocks cloud IPs
#----Architecture: client-side helper system (“YCD Desktop Helper”)---------
# Replace the existing download_transcript function with this version:

@app.post("/download_transcript")
@app.post("/download_transcript/")
def download_transcript(
    req: TranscriptRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Download transcript with robust fallback:

    1. Try transcript_fetcher.get_transcript_smart (YouTubeTranscriptApi, etc.)
    2. If that fails or returns empty -> try yt-dlp + cookies via get_transcript_with_ytdlp
    3. Only return "no captions" 404 if BOTH strategies fail and the error really is "no captions".

    This makes production behave much closer to local dev, where yt-dlp + cookies
    is already known to succeed for videos like `_ExYGT7S3E4`.
    """
    ensure_monthly_reset_and_tier(db, user)
    start = time.time()

    vid = extract_youtube_video_id(req.youtube_id)
    if not vid or len(vid) != 11:
        raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")

    if not check_internet():
        raise HTTPException(
            status_code=503,
            detail="No internet connection available."
        )

    # Work out usage bucket + nominal file format
    if req.format in ["srt", "vtt"]:
        usage_key = "unclean_transcripts"
        file_format = req.format
    elif req.clean_transcript:
        usage_key = "clean_transcripts"
        file_format = "txt"
    else:
        usage_key = "unclean_transcripts"
        file_format = "txt"

    ok, used, limit = check_usage_limit(user, usage_key)
    if not ok:
        type_name = (
            "SRT transcript" if req.format == "srt"
            else "VTT transcript" if req.format == "vtt"
            else "clean transcript" if req.clean_transcript
            else "timestamped transcript"
        )
        raise HTTPException(
            status_code=403,
            detail=f"Monthly limit reached for {type_name} ({used}/{limit}).",
        )

    text: Optional[str] = None
    primary_error: Optional[Exception] = None

    # ------------------------------------------------------------------
    # 1) Primary strategy: transcript_fetcher.get_transcript_smart
    # ------------------------------------------------------------------
    try:
        text = get_transcript_youtube_api(
            vid,
            clean=req.clean_transcript,
            fmt=req.format,
        )
    except Exception as e:
        primary_error = e
        logger.error(
            "Primary transcript fetch failed for %s: %s",
            vid,
            str(e),
        )

    # If we got an empty string from primary, treat as failure too
    if text is not None and isinstance(text, str) and not text.strip():
        logger.warning(
            "Primary transcript fetch for %s returned empty text; will try yt-dlp fallback",
            vid,
        )
        text = None

    # ------------------------------------------------------------------
    # 2) Fallback strategy: yt-dlp + cookies (get_transcript_with_ytdlp)
    # ------------------------------------------------------------------
    used_fallback = False
    fallback_error: Optional[Exception] = None

    if text is None and check_ytdlp_availability():
        cookie_file = os.getenv("YT_COOKIES_FILE")
        logger.info(
            "Attempting yt-dlp transcript fallback for %s (cookies=%s, clean=%s, fmt=%s)",
            vid,
            bool(cookie_file and os.path.exists(cookie_file or "")),
            req.clean_transcript,
            req.format,
        )
        try:
            # Try with the most complete signature first
            try:
                text = get_transcript_with_ytdlp(
                    vid,
                    clean=req.clean_transcript,
                    fmt=req.format,
                )
            except TypeError:
                # Older helper that only accepts (video_id, clean)
                text = get_transcript_with_ytdlp(
                    vid,
                    clean=req.clean_transcript,
                )

            used_fallback = True
            if not text or (isinstance(text, str) and not text.strip()):
                logger.warning(
                    "yt-dlp fallback for %s returned empty text",
                    vid,
                )
                text = None
        except Exception as e:
            fallback_error = e
            logger.error(
                "yt-dlp transcript fallback failed for %s: %s",
                vid,
                str(e),
            )

    # ------------------------------------------------------------------
    # 3) If still no transcript, classify error and raise HTTPException
    # ------------------------------------------------------------------
    if not text:
        # Prefer fallback error if it exists, otherwise the primary one
        error_to_report = fallback_error or primary_error
        error_msg = (str(error_to_report) if error_to_report else "").lower()

        # Emit one consolidated log line so Render logs show the whole story
        logger.error(
            "Transcript completely failed for %s (used_fallback=%s, "
            "primary_error=%r, fallback_error=%r)",
            vid,
            used_fallback,
            primary_error,
            fallback_error,
        )

        # Heuristic classification based on message text
        if "blocking" in error_msg or "cloud provider" in error_msg:
            # YouTube explicitly blocking data-center IPs
            raise HTTPException(
                status_code=503,
                detail=(
                    "YouTube is temporarily blocking transcript access from our servers. "
                    "This video may still be playable in your browser, but captions "
                    "cannot be fetched right now. Please try again later or try a "
                    "different video."
                ),
            )

        if "no captions" in error_msg or "not have captions" in error_msg or "no transcript found" in error_msg:
            # Only say "no captions" if ALL strategies failed
            raise HTTPException(
                status_code=404,
                detail=(
                    "This video does not have captions/transcripts available from "
                    "YouTube's APIs. Please try a different video."
                ),
            )

        # Generic catch-all
        raise HTTPException(
            status_code=502,
            detail=(
                "Could not retrieve transcript for this video. The video may not have "
                "captions available, or YouTube may be blocking access from our servers."
            ),
        )

    # ------------------------------------------------------------------
    # 4) Success path: record usage + return transcript text
    # ------------------------------------------------------------------
    new_usage = increment_user_usage(db, user, usage_key)
    proc = time.time() - start

    rec = create_download_record(
        db=db,
        user=user,
        kind=usage_key,
        youtube_id=vid,
        file_format=file_format,
        file_size=len(text or ""),
        processing_time=proc,
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
        "fallback_used": used_fallback,
    }

#------------------End of (“YCD Desktop Helper”) --------------------------

# 4. Update /download_audio/ to offer helper task when YouTube blocks audio
#----Architecture: client-side helper system (“YCD Desktop Helper”)---------
# Replace your existing download_audio function with this version:

@app.post("/download_audio/")
def download_audio(
    req: AudioRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Download audio with yt-dlp when possible.

    If YouTube returns 403 / bot-detection errors to our cloud IP,
    we return a helper_task so the frontend can use YCD Desktop Helper
    to download audio locally on the user's device.
    """
    ensure_monthly_reset_and_tier(db, user)
    start = time.time()

    vid = extract_youtube_video_id(req.youtube_id)
    if not vid or len(vid) != 11:
        raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    if not check_internet():
        raise HTTPException(status_code=503, detail="No internet connection available.")
    if not check_ytdlp_availability():
        raise HTTPException(status_code=500, detail="Audio download service temporarily unavailable.")

    ok, used, limit = check_usage_limit(user, "audio_downloads")
    if not ok:
        raise HTTPException(
            status_code=403,
            detail=f"Monthly limit reached for audio downloads ({used}/{limit}).",
        )

    info: Dict[str, Any] = {}
    try:
        info = get_video_info(vid) or {}
    except Exception as e:
        logger.warning("get_video_info failed: %s", e)

    final_name = f"{vid}_audio_{req.quality}.mp3"
    final_path = DOWNLOADS_DIR / final_name

    try:
        path = download_audio_with_ytdlp(
            vid,
            req.quality,
            output_dir=str(DOWNLOADS_DIR),
        )
    except Exception as e:
        error_msg = str(e)
        logger.error("Audio download failed for %s: %s", vid, error_msg)
        lower = error_msg.lower()

        # 👉 1) Cloud IP blocked / bot-detection / 403 → use helper
        if (
            "403" in lower
            or "forbidden" in lower
            or "sign in to confirm" in lower
            or "bot" in lower
        ):
            logger.warning(
                "YouTube appears to be blocking audio download for %s from cloud IP; "
                "returning helper_task for desktop helper.",
                vid,
            )
            return {
                "success": False,
                "needs_local_helper": True,
                "reason": (
                    "YouTube is temporarily blocking audio downloads from our servers. "
                    "Use the YCD Desktop Helper to complete this download directly "
                    "from your device."
                ),
                "helper_task": {
                    "type": "audio",
                    "youtube_id": vid,
                    "quality": req.quality,
                },
                "usage_type": "audio_downloads",
            }

        # 👉 2) No audio streams available
        if "available" in lower or "formats" in lower:
            raise HTTPException(
                status_code=404,
                detail="No audio streams available for this video. It may be restricted or unavailable in your region.",
            )

        # 👉 3) Timeout
        if "timeout" in lower:
            raise HTTPException(
                status_code=504,
                detail="Download timed out. Please try again.",
            )

        # 👉 4) Generic error
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download audio: {error_msg[:200]}",
        )

    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Failed to download audio.")

    downloaded = Path(path)
    fsize = downloaded.stat().st_size
    if fsize < 1000:
        raise HTTPException(status_code=500, detail="Downloaded audio appears corrupted.")

    if downloaded != final_path:
        try:
            if final_path.exists():
                final_path.unlink()
            downloaded.rename(final_path)
        except Exception as e:
            logger.warning("Rename failed, using original name: %s", e)
            final_path = downloaded
            final_name = downloaded.name
            fsize = final_path.stat().st_size

    _touch_now(final_path)
    new_usage = increment_user_usage(db, user, "audio_downloads")
    proc = time.time() - start
    rec = create_download_record(
        db=db,
        user=user,
        kind="audio_downloads",
        youtube_id=vid,
        quality=req.quality,
        file_format="mp3",
        file_size=fsize,
        processing_time=proc,
    )

    token = create_access_token_for_mobile(user.username)
    direct_url = f"/download-file/audio/{final_name}?auth={token}"

    return {
        "download_url": f"/files/{final_name}",
        "direct_download_url": direct_url,
        "youtube_id": vid,
        "quality": req.quality,
        "file_size": fsize,
        "file_size_mb": round(fsize / (1024 * 1024), 2),
        "filename": final_name,
        "local_path": str(final_path),
        "processing_time": round(proc, 2),
        "message": "Audio ready for download",
        "success": True,
        "title": info.get("title", "Unknown Title"),
        "uploader": info.get("uploader", "Unknown"),
        "duration": info.get("duration", 0),
        "usage_updated": new_usage,
        "usage_type": "audio_downloads",
        "download_record_id": rec.id if rec else None,
        "account": canonical_account(user),
    }

#------------------End of (“YCD Desktop Helper”) --------------------------

# 👉 IMPORTANT: DO NOT DELETE THIS FUNCTION
# @app.post("/download_audio/")
# def download_audio(req: AudioRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
#     ensure_monthly_reset_and_tier(db, user)
#     start = time.time()
#     vid = extract_youtube_video_id(req.youtube_id)
#     if not vid or len(vid) != 11: 
#         raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
#     if not check_internet(): 
#         raise HTTPException(status_code=503, detail="No internet connection available.")
#     if not check_ytdlp_availability(): 
#         raise HTTPException(status_code=500, detail="Audio download service temporarily unavailable.")

#     ok, used, limit = check_usage_limit(user, "audio_downloads")
#     if not ok: 
#         raise HTTPException(status_code=403, detail=f"Monthly limit reached for audio downloads ({used}/{limit}).")

#     info = {}
#     try: 
#         info = get_video_info(vid) or {}
#     except Exception as e: 
#         logger.warning("get_video_info failed: %s", e)

#     final_name = f"{vid}_audio_{req.quality}.mp3"
#     final_path = DOWNLOADS_DIR / final_name

#     try:
#         path = download_audio_with_ytdlp(vid, req.quality, output_dir=str(DOWNLOADS_DIR))
#     except Exception as e:
#         error_msg = str(e)
#         logger.error(f"Audio download failed for {vid}: {error_msg}")
        
#         if "403" in error_msg or "Forbidden" in error_msg:
#             raise HTTPException(
#                 status_code=503, 
#                 detail="YouTube is temporarily blocking audio downloads. This video may have restrictions. Please try again in a few minutes or try a different video."
#             )
#         elif "available" in error_msg.lower() or "formats" in error_msg.lower():
#             raise HTTPException(
#                 status_code=404,
#                 detail="No audio streams available for this video. It may be restricted or unavailable in your region."
#             )
#         elif "timeout" in error_msg.lower():
#             raise HTTPException(
#                 status_code=504,
#                 detail="Download timed out. Please try again."
#             )
#         else:
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Failed to download audio: {error_msg[:200]}"
#             )

#     if not path or not os.path.exists(path): 
#         raise HTTPException(status_code=404, detail="Failed to download audio.")
    
#     downloaded = Path(path)
#     fsize = downloaded.stat().st_size
#     if fsize < 1000: 
#         raise HTTPException(status_code=500, detail="Downloaded audio appears corrupted.")

#     if downloaded != final_path:
#         try:
#             if final_path.exists(): final_path.unlink()
#             downloaded.rename(final_path)
#         except Exception as e:
#             logger.warning("Rename failed, using original name: %s", e)
#             final_path = downloaded; final_name = downloaded.name; fsize = final_path.stat().st_size

#     _touch_now(final_path)
#     new_usage = increment_user_usage(db, user, "audio_downloads")
#     proc = time.time() - start
#     rec = create_download_record(db=db, user=user, kind="audio_downloads", youtube_id=vid, quality=req.quality, file_format="mp3", file_size=fsize, processing_time=proc)
#     token = create_access_token_for_mobile(user.username)
#     direct_url = f"/download-file/audio/{final_name}?auth={token}"
    
#     return {"download_url": f"/files/{final_name}", "direct_download_url": direct_url, "youtube_id": vid, "quality": req.quality,
#             "file_size": fsize, "file_size_mb": round(fsize / (1024 * 1024), 2), "filename": final_name, "local_path": str(final_path),
#             "processing_time": round(proc, 2), "message": "Audio ready for download", "success": True,
#             "title": info.get("title", "Unknown Title"), "uploader": info.get("uploader", "Unknown"), "duration": info.get("duration", 0),
#             "usage_updated": new_usage, "usage_type": "audio_downloads",
#             "download_record_id": rec.id if rec else None, "account": canonical_account(user)}
#-------------------------------------------------------

# 5. Update /download_video/ similarly
#----Architecture: client-side helper system (“YCD Desktop Helper”)---------
# Replace your existing download_video function with this:
@app.post("/download_video/")
def download_video(
    req: VideoRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Download video with yt-dlp when possible.

    If YouTube returns 403 / bot-detection errors to our cloud IP,
    we return a helper_task so the frontend can use YCD Desktop Helper
    to download video locally on the user's device.
    """
    ensure_monthly_reset_and_tier(db, user)
    start = time.time()

    vid = extract_youtube_video_id(req.youtube_id)
    if not vid or len(vid) != 11:
        raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
    if not check_internet():
        raise HTTPException(status_code=503, detail="No internet connection available.")
    if not check_ytdlp_availability():
        raise HTTPException(status_code=500, detail="Video download service unavailable.")

    ok, used, limit = check_usage_limit(user, "video_downloads")
    if not ok:
        raise HTTPException(
            status_code=403,
            detail=f"Monthly limit reached for video downloads ({used}/{limit}).",
        )

    info: Dict[str, Any] = {}
    try:
        info = get_video_info(vid) or {}
    except Exception as e:
        logger.warning("get_video_info failed: %s", e)

    final_name = f"{vid}_video_{req.quality}.mp4"
    final_path = DOWNLOADS_DIR / final_name

    try:
        path = download_video_with_ytdlp(
            vid,
            req.quality,
            output_dir=str(DOWNLOADS_DIR),
        )
    except Exception as e:
        error_msg = str(e)
        logger.error("Video download failed for %s: %s", vid, error_msg)
        lower = error_msg.lower()

        # 👉 1) Cloud IP blocked / bot-detection / 403 → use helper
        if (
            "403" in lower
            or "forbidden" in lower
            or "sign in to confirm" in lower
            or "bot" in lower
        ):
            logger.warning(
                "YouTube appears to be blocking video download for %s from cloud IP; "
                "returning helper_task for desktop helper.",
                vid,
            )
            return {
                "success": False,
                "needs_local_helper": True,
                "reason": (
                    "YouTube is temporarily blocking video downloads from our servers. "
                    "Use the YCD Desktop Helper to complete this download directly "
                    "from your device."
                ),
                "helper_task": {
                    "type": "video",
                    "youtube_id": vid,
                    "quality": req.quality,
                },
                "usage_type": "video_downloads",
            }

        # 👉 2) No video formats available
        if "available" in lower or "formats" in lower:
            raise HTTPException(
                status_code=404,
                detail="No video streams available for this video. It may be restricted or unavailable in your region.",
            )

        # 👉 3) Timeout
        if "timeout" in lower:
            raise HTTPException(
                status_code=504,
                detail="Download timed out. Please try again.",
            )

        # 👉 4) Generic error
        raise HTTPException(
            status_code=500,
            detail=f"Failed to download video: {error_msg[:200]}",
        )

    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Failed to download video.")

    downloaded = Path(path)
    fsize = downloaded.stat().st_size
    if fsize < 10_000:
        raise HTTPException(status_code=500, detail="Downloaded video appears corrupted.")

    if downloaded != final_path:
        try:
            if final_path.exists():
                final_path.unlink()
            downloaded.rename(final_path)
        except Exception as e:
            logger.warning("Rename failed, using original name: %s", e)
            final_path = downloaded
            final_name = downloaded.name
            fsize = final_path.stat().st_size

    _touch_now(final_path)
    new_usage = increment_user_usage(db, user, "video_downloads")
    proc = time.time() - start
    rec = create_download_record(
        db=db,
        user=user,
        kind="video_downloads",
        youtube_id=vid,
        quality=req.quality,
        file_format="mp4",
        file_size=fsize,
        processing_time=proc,
    )

    token = create_access_token_for_mobile(user.username)
    direct_url = f"/download-file/video/{final_name}?auth={token}"

    return {
        "download_url": f"/files/{final_name}",
        "direct_download_url": direct_url,
        "youtube_id": vid,
        "quality": req.quality,
        "file_size": fsize,
        "file_size_mb": round(fsize / (1024 * 1024), 2),
        "filename": final_name,
        "local_path": str(final_path),
        "processing_time": round(proc, 2),
        "message": "Video ready for download",
        "success": True,
        "title": info.get("title", "Unknown Title"),
        "uploader": info.get("uploader", "Unknown"),
        "duration": info.get("duration", 0),
        "usage_updated": new_usage,
        "usage_type": "video_downloads",
        "download_record_id": rec.id if rec else None,
        "account": canonical_account(user),
    }

#------------------End of (“YCD Desktop Helper”) --------------------------

# #----Download video endpoint-----
# # 👉 IMPORTANT: DO NOT DELETE THIS FUNCTION

# @app.post("/download_video/")
# def download_video(req: VideoRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
#     ensure_monthly_reset_and_tier(db, user)
#     start = time.time()
#     vid = extract_youtube_video_id(req.youtube_id)
#     if not vid or len(vid) != 11: raise HTTPException(status_code=400, detail="Invalid YouTube video ID.")
#     if not check_internet(): raise HTTPException(status_code=503, detail="No internet connection available.")
#     if not check_ytdlp_availability(): raise HTTPException(status_code=500, detail="Video download service unavailable.")

#     ok, used, limit = check_usage_limit(user, "video_downloads")
#     if not ok: raise HTTPException(status_code=403, detail=f"Monthly limit reached for video downloads ({used}/{limit}).")

#     info = {}
#     try: info = get_video_info(vid) or {}
#     except Exception as e: logger.warning("get_video_info failed: %s", e)

#     final_name = f"{vid}_video_{req.quality}.mp4"
#     final_path = DOWNLOADS_DIR / final_name

#     path = download_video_with_ytdlp(vid, req.quality, output_dir=str(DOWNLOADS_DIR))
#     if not path or not os.path.exists(path): raise HTTPException(status_code=404, detail="Failed to download video.")
#     downloaded = Path(path)
#     fsize = downloaded.stat().st_size
#     if fsize < 10_000: raise HTTPException(status_code=500, detail="Downloaded video appears corrupted.")

#     if downloaded != final_path:
#         try:
#             if final_path.exists(): final_path.unlink()
#             downloaded.rename(final_path)
#         except Exception as e:
#             logger.warning("Rename failed, using original name: %s", e)
#             final_path = downloaded; final_name = downloaded.name; fsize = final_path.stat().st_size

#     _touch_now(final_path)
#     new_usage = increment_user_usage(db, user, "video_downloads")
#     proc = time.time() - start
#     rec = create_download_record(db=db, user=user, kind="video_downloads", youtube_id=vid, quality=req.quality, file_format="mp4", file_size=fsize, processing_time=proc)
#     token = create_access_token_for_mobile(user.username)
#     direct_url = f"/download-file/video/{final_name}?auth={token}"
#     return {"download_url": f"/files/{final_name}", "direct_download_url": direct_url, "youtube_id": vid, "quality": req.quality,
#             "file_size": fsize, "file_size_mb": round(fsize / (1024 * 1024), 2), "filename": final_name, "local_path": str(final_path),
#             "processing_time": round(proc, 2), "message": "Video ready for download", "success": True,
#             "title": info.get("title", "Unknown Title"), "uploader": info.get("uploader", "Unknown"), "duration": info.get("duration", 0),
#             "usage_updated": new_usage, "usage_type": "video_downloads",
#             "download_record_id": rec.id if rec else None, "account": canonical_account(user)}

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

@app.get("/subscription_status")
@app.get("/subscription_status/")
def subscription_status(request: Request, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    if request.query_params.get("sync") == "1":
        try:
            sync_user_subscription_from_stripe(current_user, db)
        except Exception as e:
            logger.warning(f"Stripe sync skipped (non-fatal): {e}")

    next_reset = ensure_monthly_reset_and_tier(db, current_user)

    tier = (getattr(current_user, "subscription_tier", "free") or "free").lower()
    usage = {
        "clean_transcripts":   getattr(current_user, "usage_clean_transcripts", 0) or 0,
        "unclean_transcripts": getattr(current_user, "usage_unclean_transcripts", 0) or 0,
        "audio_downloads":     getattr(current_user, "usage_audio_downloads", 0) or 0,
        "video_downloads":     getattr(current_user, "usage_video_downloads", 0) or 0,
    }

    limits = PLAN_LIMITS.get(tier, PLAN_LIMITS["free"])
    limits_display = {k: ("unlimited" if v == float("inf") else v) for k, v in limits.items()}

    return {
        "tier": tier,
        "status": "active" if tier != "free" else "inactive",
        "usage": usage,
        "limits": limits_display,
        "next_reset": next_reset.isoformat() if next_reset else None,
        "downloads_folder": str(DOWNLOADS_DIR),
        "account": canonical_account(current_user),
    }

@app.post("/contact")
@app.post("/contact")
def send_contact_message(req: ContactMessage, request: Request):
    name = (req.name or "").strip()[:200]
    email = (req.email or "").strip()[:255].lower()
    message = (req.message or "").strip()

    if not name or not message or not _email_re.match(email):
        raise HTTPException(status_code=400, detail="Please provide a valid name, email, and message.")

    ip = (request.client.host if request.client else "unknown")
    now = time.time()
    window, limit = 600, 3
    with _contact_lock:
        q = _contact_hits[ip]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(status_code=429, detail="Too many messages. Please try again later.")
        q.append(now)

    try:
        _send_contact_email(name, email, message)
        return {"ok": True, "message": "Thanks! Your message has been sent."}
    except Exception as e:
        logger.error(f"Contact email failed: {e}")
        raise HTTPException(status_code=500, detail="Unable to send message right now.")

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

@app.head("/health")
def health_head():
    # return 200 with no body
    return Response(status_code=200)

@app.get("/debug/users")
def debug_users(db: Session = Depends(get_db)):
    if os.getenv("ENVIRONMENT", "development") != "development":
        raise HTTPException(status_code=404, detail="Not found")
    users = db.query(User).all()
    return {"total_users": len(users),
            "users": [{"id": u.id, "username": u.username, "email": (u.email or '').strip().lower(),
                       "created_at": u.created_at.isoformat() if u.created_at else None,
                       "subscription_tier": getattr(u, "subscription_tier", "free"),
                       "is_active": getattr(u, "is_active", True)} for u in users]}

if batch_router:
    try:
        app.include_router(batch_router, tags=["batch"])
    except Exception as e:
        logger.error("Could not include batch routes: %s", e)

if activity_router:
    try:
        app.include_router(activity_router, tags=["activity"])
        logger.info("✅ Activity tracking router loaded")
    except Exception as e:
        logger.warning("Could not include activity routes: %s", e)

try:
    app.include_router(payment_router, tags=["payments"])
except Exception as e:
    logger.error("Could not include payment routes: %s", e)

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

#===============End main Module=============================


