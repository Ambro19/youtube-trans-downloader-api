from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple, Literal
import re, time, socket, mimetypes, logging, jwt, threading, os # pyright: ignore[reportMissingImports]
from collections import defaultdict, deque

from dotenv import load_dotenv, find_dotenv # type: ignore
load_dotenv()
load_dotenv(dotenv_path=find_dotenv(".env.local"), override=True)
load_dotenv(dotenv_path=find_dotenv(".env"),       override=False)

from fastapi import FastAPI, HTTPException, Depends, Request, Query, Response # type: ignore
from fastapi.middleware.cors import CORSMiddleware # type: ignore
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm # pyright: ignore[reportMissingImports]
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse, ORJSONResponse # type: ignore
from fastapi.staticfiles import StaticFiles # type: ignore
from passlib.context import CryptContext # type: ignore
from starlette.middleware.base import BaseHTTPMiddleware # type: ignore
from starlette.types import ASGIApp # type: ignore
from sqlalchemy.orm import Session # pyright: ignore[reportMissingImports]
from sqlalchemy import delete as sqla_delete # type: ignore
from sqlalchemy.exc import OperationalError, IntegrityError # type: ignore
import smtplib, ssl
from email.message import EmailMessage
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired # type: ignore
from pydantic import BaseModel, EmailStr # type: ignore
from email_utils import send_password_reset_email
from auth_utils import get_password_hash, verify_password, validate_password_strength
from subscription_sync import sync_user_subscription_from_stripe, _apply_local_overdue_downgrade_if_possible
from youtube_transcript_api import YouTubeTranscriptApi # type: ignore
from transcript_fetcher import get_transcript_smart as get_transcript_youtube_api

# IMPORTANT: make sure you have these imports available in main.py
from subscription_sync import (
     sync_user_subscription_from_stripe,
     _apply_local_overdue_downgrade_if_possible,  # <-- ensure this exists in your subscription_sync.py
 )

from webhook_handler import handle_stripe_webhook  # if you call it from main.py webhook endpoint


# ---------------------------------------------------
# Safe imports for youtube_transcript_api exceptions
# ---------------------------------------------------
try:
    from youtube_transcript_api import ( # type: ignore
        YouTubeTranscriptApi,
        TranscriptsDisabled,
        NoTranscriptFound,
    )
except Exception:
    class YouTubeTranscriptApi:  # type: ignore
        pass

    class TranscriptsDisabled(Exception):
        pass

    class NoTranscriptFound(Exception):
        pass

try:
    from youtube_transcript_api import CouldNotRetrieveTranscript # pyright: ignore[reportMissingImports]
except Exception:
    class CouldNotRetrieveTranscript(Exception):
        pass

# ----------------------
# JWT Secret (required)
# ----------------------
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY env var is required. "
        'Generate one with:  python -c "import secrets; print(secrets.token_urlsafe(64))"'
    )

RESET_TOKEN_TTL_SECONDS = int(os.getenv("RESET_TOKEN_TTL_SECONDS", "3600"))
serializer = URLSafeTimedSerializer(SECRET_KEY)

# ----------------------------
# Database URL normalization
# ----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg2" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

logger = logging.getLogger("youtube_trans_downloader")
logger.setLevel(logging.INFO)

driver = "sqlite" if DATABASE_URL.startswith("sqlite") else "postgres"
logger.info(f"✅ Config OK — using database driver: {driver}")

from models import (
    User,
    TranscriptDownload,
    Subscription,
    get_db,
    initialize_database,
    engine,
)
from db_migrations import run_startup_migrations
from auth_deps import get_current_user
from webhook_handler import handle_stripe_webhook #fix_existing_premium_users
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

# --------------------------------------------
# Stripe configuration (module-level handle)
# --------------------------------------------
# ===== FIX #1: Stripe configuration with NO_PROXY support =====
stripe = None
try:
    import stripe as _stripe

    if os.getenv("STRIPE_SECRET_KEY"):
        _stripe.api_key = os.getenv("STRIPE_SECRET_KEY").strip()
        stripe = _stripe
        
        # ✅ CRITICAL: Exclude Stripe from proxy (fix for 403 Forbidden errors)
        os.environ.setdefault("NO_PROXY", "")
        current_no_proxy = os.environ.get("NO_PROXY", "")
        stripe_domains = "api.stripe.com,files.stripe.com,checkout.stripe.com"
        
        if stripe_domains not in current_no_proxy:
            os.environ["NO_PROXY"] = f"{current_no_proxy},{stripe_domains}" if current_no_proxy else stripe_domains
            logger.info("✅ Stripe domains excluded from proxy: %s", stripe_domains)
            
except Exception as e:
    logger.warning("⚠️ Stripe initialization issue: %s", e)
    stripe = None

app = FastAPI(
    title="YouTube Content Downloader API",
    version="3.4.3",
    default_response_class=ORJSONResponse,
)

# ---------------------------------------------------------------------------
# Proxy configuration (from ENV) — used for yt-dlp / outbound HTTP(S)
#   PROXY_ENABLED=true/false
#   PROXY_HOST=gate.decodo.com
#   PROXY_PORT=10010
#   PROXY_USERNAME=...
#   PROXY_PASSWORD=...
# ---------------------------------------------------------------------------

# Proxy configuration for YouTube downloads (NOT Stripe)
def _configure_global_proxy_from_env() -> None:
    """Configure process-wide proxy for YouTube downloads ONLY (excludes Stripe)"""
    try:
        enabled = (os.getenv("PROXY_ENABLED", "false") or "").lower() in {
            "1", "true", "yes", "on",
        }
        host = (os.getenv("PROXY_HOST") or "").strip()
        port = (os.getenv("PROXY_PORT") or "").strip()
        user = (os.getenv("PROXY_USERNAME") or "").strip()
        pwd = (os.getenv("PROXY_PASSWORD") or "").strip()

        if not enabled:
            for key in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "YTDLP_PROXY"]:
                os.environ.pop(key, None)
            logger.info("🌐 Proxy: disabled (PROXY_ENABLED not true)")
            return

        if not host or not port:
            logger.warning(
                "🌐 Proxy: PROXY_ENABLED is true but PROXY_HOST/PROXY_PORT are missing."
            )
            return

        from urllib.parse import quote as urlquote

        if user and pwd:
            proxy_url = f"http://{urlquote(user)}:{urlquote(pwd)}@{host}:{port}"
            auth_label = "with-auth"
        else:
            proxy_url = f"http://{host}:{port}"
            auth_label = "no-auth"

        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        os.environ["ALL_PROXY"] = proxy_url
        os.environ["YTDLP_PROXY"] = proxy_url

        # ✅ Ensure NO_PROXY includes Stripe (redundant but safe)
        os.environ.setdefault("NO_PROXY", "api.stripe.com,files.stripe.com,checkout.stripe.com")

        logger.info("🌐 Proxy: enabled (%s) host=%s port=%s", auth_label, host, port)
    except Exception as e:
        logger.warning("🌐 Proxy: configuration failed: %s", e)


# --------------
# Stripe helper
# --------------

# ===== FIX #2: Graceful Stripe customer creation (doesn't fail registration) =====
def ensure_stripe_customer_for_user(user: "User", db: Session) -> None:
    """Ensure user has Stripe customer ID (graceful - doesn't fail if Stripe is down)"""
    if not stripe or not os.getenv("STRIPE_SECRET_KEY"):
        return  # Silent skip if Stripe not configured

    if getattr(user, "stripe_customer_id", None):
        return  # Already linked

    email = (user.email or "").strip().lower()
    username = (user.username or "").strip() or None

    if not email:
        return

    try:
        customer = None

        # Try Customer.search
        try:
            found = stripe.Customer.search(
                query=f"email:'{email}' AND -deleted:'true'",
                limit=1,
            )
            if getattr(found, "data", []):
                customer = found.data[0]
        except Exception as e:
            logger.debug("Customer search failed (non-fatal): %s", e)

        # Fallback to Customer.list
        if not customer:
            try:
                listed = stripe.Customer.list(email=email, limit=1)
                if getattr(listed, "data", []):
                    customer = listed.data[0]
            except Exception as e:
                logger.debug("Customer list failed (non-fatal): %s", e)

        if customer:
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
            logger.info("🔄 Linked existing Stripe customer %s", customer["id"])
            return

        # Create new customer
        created = stripe.Customer.create(
            email=email,
            name=username,
            metadata={"app_user_id": str(user.id)},
            idempotency_key=f"user_backfill_{user.id}",
        )
        user.stripe_customer_id = created["id"]
        db.commit()
        db.refresh(user)
        logger.info("✅ Created Stripe customer %s", created["id"])

    except Exception as e:
        # ✅ FIX: Don't fail registration if Stripe is unavailable
        logger.warning("⚠️ Stripe customer creation skipped (non-fatal): %s", e)
        # User account still created successfully


#---------------------------------------------------------------------------
# Stripe Helper: The one-line call point (with a safe throttle guard)
#---------------------------------------------------------------------------

def should_sync_stripe_now(user: User) -> bool:
    # Only hit Stripe if we haven’t synced recently
    last = getattr(user, "subscription_updated_at", None)
    if not last:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - last >= timedelta(minutes=10)  # adjust as you like


# ---------------------------------------------------------------------------
# Additional Helper: Add this helper (server-side reset enforcement)
# ---------------------------------------------------------------------------

def enforce_billing_cycle_and_tier(db: Session, user_id: int, *, force_stripe_sync: bool) -> None:
    """
    1) If now >= next_reset: reset monthly usage counters and advance next_reset
    2) If Stripe inactive: downgrade to free
    NOTE: Implementation must use your actual models/tables.
    """
    now = datetime.now(timezone.utc)

    sub = db.query(Subscription).filter(Subscription.user_id == user_id).one_or_none()
    if not sub:
        return

    # Optional: if force_stripe_sync, call your existing Stripe sync logic here
    # e.g. sync_user_subscription_from_stripe(db, user_id)

    # --- Handle monthly reset ---
    if sub.next_reset and now >= sub.next_reset:
        # Reset usage counters (whatever table/columns you use)
        usage = db.query(Usage).filter(Usage.user_id == user_id).one_or_none() # pyright: ignore[reportUndefinedVariable]
        if usage:
            usage.clean_transcripts = 0
            usage.unclean_transcripts = 0
            usage.audio_downloads = 0
            usage.video_downloads = 0

        # Advance next_reset (example: +30 days; better: add 1 month with dateutil.relativedelta)
        sub.next_reset = sub.next_reset.replace(year=sub.next_reset.year) + timedelta(days=30)

    # --- Enforce downgrade if not active ---
    if sub.status != "active":
        sub.tier = "free"

    db.commit()

# ---------------------------------------------------------------------------
# Timestamp middleware (optional)
# ---------------------------------------------------------------------------
try:
    from timestamp_patch import EnsureUtcZMiddleware

    app.add_middleware(EnsureUtcZMiddleware)
except Exception as e:
    logger.info("Timestamp middleware not loaded: %s", e)


# ------------------------------
# Security headers middleware
# ------------------------------
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
    "connect-src 'self' https://api.onetechly.com https://youtube-trans-downloader-api.onrender.com https://api.stripe.com; "
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


# -------------------------
# Rate limiting middleware
# -------------------------
class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)
        self.now = time.time
        self.buckets: Dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()
        self.enabled = (os.getenv("RL_ENABLED", "true").lower() in {"1", "true", "yes", "on"})
        self.default_window = int(os.getenv("RL_DEFAULT_WINDOW_SEC", "60"))
        self.default_max = int(os.getenv("RL_DEFAULT_MAX", "120"))
        self.auth_window = int(os.getenv("RL_AUTH_WINDOW_SEC", "60"))
        self.auth_max = int(os.getenv("RL_AUTH_MAX", "10"))
        self.dl_window = int(os.getenv("RL_DL_WINDOW_SEC", "60"))
        self.dl_max = int(os.getenv("RL_DL_MAX", "40"))

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

# -----
# CORS
# ------
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

PUBLIC_ORIGINS = [
    "https://onetechly.com",
    "https://www.onetechly.com",
    "https://api.onetechly.com",
]

DEV_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://192.168.1.185:3000",
]

if ENVIRONMENT != "production":
    allow_origins = PUBLIC_ORIGINS + DEV_ORIGINS + (
        [FRONTEND_URL] if FRONTEND_URL else []
    )
else:
    allow_origins = PUBLIC_ORIGINS + ([FRONTEND_URL] if FRONTEND_URL else [])

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in allow_origins if o],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Content-Disposition", "Content-Type", "Content-Length", "Content-Range",
    ],
)

print("✅ CORS enabled for origins:", allow_origins)

# -------------------
# Download directory
# -------------------

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


# --------------------
# JWT / Auth helpers
# --------------------
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
    return {
        "username": (user.username or "").strip(),
        "email": (user.email or "").strip().lower(),
    }


def is_mobile_request(request: Request) -> bool:
    ua = (request.headers.get("user-agent") or "").lower()
    return any(
        p in ua
        for p in [
            "android", "iphone", "ipad", "ipod", "blackberry", "windows phone",
            "mobile", "webos", "opera mini",
        ]
    )


def get_safe_filename(filename: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", filename)[:120]


def get_mobile_mime_type(path: str, file_type: str) -> str:
    if file_type == "audio" or path.endswith((".mp3", ".m4a", ".aac")):
        return "audio/mpeg"
    if file_type == "video" or path.endswith((".mp4", ".m4v", ".mov")):
        return "video/mp4"
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def check_internet() -> bool:
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


def check_youtube() -> bool:
    try:
        socket.create_connection(("www.youtube.com", 443), timeout=5)
        return True
    except OSError:
        return False


_YT_ID_RE = re.compile(r"(?<![\w-])([A-Za-z0-9_-]{11})(?![\w-])")


def extract_youtube_video_id(text: str) -> str:
    t = (text or "").strip()
    pats = [
        r"(?:youtube\.com/watch\?[^#\s]*[?&]v=)([^&\n?#]{11})",
        r"(?:youtu\.be/)([^&\n?#/]{11})",
        r"(?:youtube\.com/embed/)([^&\n?#/]{11})",
        r"(?:youtube\.com/shorts/)([^&\n?#/]{11})",
    ]
    for p in pats:
        m = re.search(p, t)
        if m:
            cand = m.group(1)
            if _YT_ID_RE.fullmatch(cand):
                return cand
    m = _YT_ID_RE.search(t)
    return m.group(1) if m else ""


def is_mobile_request(request: Request) -> bool:
    ua = (request.headers.get("user-agent") or "").lower()
    return any(
        p in ua
        for p in [
            "android", "iphone", "ipad", "ipod", "blackberry", "windows phone",
            "mobile", "webos", "opera mini",
        ]
    )


def get_safe_filename(filename: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "_", filename)[:120]


def get_mobile_mime_type(path: str, file_type: str) -> str:
    if file_type == "audio" or path.endswith((".mp3", ".m4a", ".aac")):
        return "audio/mpeg"
    if file_type == "video" or path.endswith((".mp4", ".m4v", ".mov")):
        return "video/mp4"
    mime, _ = mimetypes.guess_type(path)
    return mime or "application/octet-stream"


def check_internet() -> bool:
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
        return True
    except OSError:
        return False


def check_youtube() -> bool:
    try:
        socket.create_connection(("www.youtube.com", 443), timeout=5)
        return True
    except OSError:
        return False


_YT_ID_RE = re.compile(r"(?<![\w-])([A-Za-z0-9_-]{11})(?![\w-])")


def extract_youtube_video_id(text: str) -> str:
    t = (text or "").strip()
    pats = [
        r"(?:youtube\.com/watch\?[^#\s]*[?&]v=)([^&\n?#]{11})",
        r"(?:youtu\.be/)([^&\n?#/]{11})",
        r"(?:youtube\.com/embed/)([^&\n?#/]{11})",
        r"(?:youtube\.com/shorts/)([^&\n?#/]{11})",
    ]
    for p in pats:
        m = re.search(p, t)
        if m:
            cand = m.group(1)
            if _YT_ID_RE.fullmatch(cand):
                return cand
    m = _YT_ID_RE.search(t)
    return m.group(1) if m else ""

# ------------
# Plan Limits
# -----------
PLAN_LIMITS = {
    "free": {
        "clean_transcripts": 5,
        "unclean_transcripts": 3,
        "audio_downloads": 2,
        "video_downloads": 1,
    },
    "pro": {
        "clean_transcripts": 100,
        "unclean_transcripts": 50,
        "audio_downloads": 50,
        "video_downloads": 20,
    },
    "premium": {
        "clean_transcripts": float("inf"),
        "unclean_transcripts": float("inf"),
        "audio_downloads": float("inf"),
        "video_downloads": float("inf"),
    },
}


def _now_utc() -> datetime:
    return datetime.utcnow().replace(tzinfo=timezone.utc)


def _to_naive_utc(d: Optional[datetime]) -> Optional[datetime]:
    if not d:
        return None
    if d.tzinfo is None:
        return d
    return d.astimezone(timezone.utc).replace(tzinfo=None)


def _compute_next_reset_date(user: User) -> datetime:
    anchor = (
        _to_naive_utc(getattr(user, "subscription_started_at", None))
        or _to_naive_utc(getattr(user, "updated_at", None))
        or _to_naive_utc(getattr(user, "created_at", None))
        or datetime.utcnow()
    )
    today = datetime.utcnow()
    day = min(anchor.day, 28)
    this_month_anniv = today.replace(day=day, hour=0, minute=0, second=0, microsecond=0)
    if today < this_month_anniv:
        return this_month_anniv
    y, m = (today.year + (today.month // 12), ((today.month % 12) + 1))
    return this_month_anniv.replace(year=y, month=m)


# ===== FIX #3: Proper usage reset (fixes "over by 2" display issue) =====
def _reset_usage_counters(user: User):
    """Reset ALL usage counters to 0 (fixes over-limit display bug)"""
    user.usage_clean_transcripts = 0
    user.usage_unclean_transcripts = 0
    user.usage_audio_downloads = 0
    user.usage_video_downloads = 0
    logger.info("🔁 Reset usage counters for user %s", user.email)


def ensure_monthly_reset_and_tier(db: Session, user: User) -> datetime:
    """Enforce tier downgrade + monthly usage reset"""
    try:
        # Check if user should be downgraded based on subscription status
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
                # ✅ FIX: Reset usage when downgrading to free
                _reset_usage_counters(user)
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
        db.commit()
        db.refresh(user)
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
        db.add(rec)
        db.commit()
        db.refresh(rec)
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
                    stripe_sub_id = getattr(
                        subscription, "stripe_subscription_id", None
                    )
                    if stripe_sub_id:
                        try:
                            stripe.Subscription.delete(stripe_sub_id)
                            logger.info(
                                f"Cancelled Stripe subscription {stripe_sub_id} for user {user.id}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to cancel Stripe subscription: {e}"
                            )
                    subscription.status = "cancelled"
                    subscription.cancelled_at = datetime.utcnow()
                    db.commit()
            except OperationalError as e:
                logger.warning(
                    f"DB issue during subscription deletion: {e}"
                )
        try:
            deleted_count = (
                db.query(Subscription)
                .filter(Subscription.user_id == user.id)
                .delete()
            )
            logger.info(
                f"Deleted {deleted_count} subscription rows for user {user.id}"
            )
            db.commit()
        except Exception as e:
            logger.warning(
                f"Error deleting subscription rows: {e}"
            )
            db.rollback()
        return True
    except Exception as e:
        logger.error(f"Error deleting user subscription: {e}")
        db.rollback()
        return False


def delete_user_data(db: Session, user: User) -> bool:
    try:
        try:
            deleted_count = (
                db.query(TranscriptDownload)
                .filter(TranscriptDownload.user_id == user.id)
                .delete()
            )
            logger.info(
                f"Deleted {deleted_count} download rows for user {user.id}"
            )
            db.commit()
        except Exception as e:
            logger.warning(
                f"Error deleting download rows: {e}"
            )
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
            import requests, json # pyright: ignore[reportMissingModuleSource]

            resp = requests.post(
                "https://api.sendgrid.com/v3/mail/send",
                headers={
                    "Authorization": f"Bearer {sg_key}",
                    "Content-Type": "application/json",
                },
                data=json.dumps(
                    {
                        "personalizations": [{"to": [{"email": CONTACT_TO}]}],
                        "from": {"email": CONTACT_FROM},
                        "subject": subject,
                        "content": [{"type": "text/plain", "value": body}],
                    }
                ),
                timeout=10,
            )
            if resp.status_code not in (200, 202):
                raise RuntimeError(
                    f"SendGrid error {resp.status_code}: {resp.text[:300]}"
                )
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
        with smtplib.SMTP_SSL(
            "smtp.gmail.com", 465, context=context, timeout=15
        ) as server:
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return

    raise RuntimeError(
        "No mail provider configured (SENDGRID_API_KEY or Gmail SMTP creds required)."
    )

# ----------------
# Pydantic models
# ----------------
class UserCreate(BaseModel):
    username: str
    email: str
    password: str


class UserResponse(BaseModel):
    id: int
    username: Optional[str] = None
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


class CancelRequest(BaseModel):
    at_period_end: Optional[bool] = True


class DeleteAccountResponse(BaseModel):
    message: str
    deleted_at: str
    user_email: str


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

# ------------------
# File cleanup loop
# ------------------
def _cleanup_stale_files_loop():
    keep_days = max(1, FILE_RETENTION_DAYS)
    suffixes = {".mp3", ".m4a", ".aac", ".mp4", ".mkv", ".mov", ".txt", ".srt", ".vtt"}
    while True:
        try:
            cutoff = time.time() - keep_days * 86400
            for p in DOWNLOADS_DIR.iterdir():
                try:
                    if not p.is_file():
                        continue
                    if p.suffix.lower() not in suffixes:
                        continue
                    if p.stat().st_mtime < cutoff:
                        p.unlink(missing_ok=True)
                except Exception:
                    continue
        except Exception as e:
            logger.warning(f"stale-files cleanup error: {e}")
        time.sleep(12 * 3600)

# ---------------------
# Stripe key sanitizer
# ---------------------

# Stripe key sanitizer
def _sanitize_stripe_key():
    """Remove whitespace from Stripe key and reinitialize"""
    key = os.getenv("STRIPE_SECRET_KEY")
    if key:
        clean_key = key.strip()
        if clean_key != key:
            os.environ["STRIPE_SECRET_KEY"] = clean_key
            logger.info("✅ Sanitized STRIPE_SECRET_KEY (removed whitespace)")

        try:
            import stripe as _stripe
            _stripe.api_key = clean_key
            global stripe
            stripe = _stripe
            logger.info("✅ Stripe configured with clean API key")
        except Exception as e:
            logger.warning("⚠️ Could not reinitialize Stripe: %s", e)


# -----------------------------
# Cookie hydration for yt-dlp
# -----------------------------
# Cookie hydration
def _hydrate_youtube_cookies_to_tmp() -> None:
    """Normalize cookie env for yt-dlp"""
    import base64
    import shutil

    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    os.environ.setdefault("YTDLP_HOME", "/tmp/yt-dlp")
    os.environ.setdefault("YT_DLP_DIR", "/tmp/yt-dlp")

    b64 = (os.getenv("YT_COOKIES_B64") or "").strip()
    if b64:
        try:
            raw = base64.b64decode(b64)
            raw = raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            target_dir = Path("/tmp/yt-dlp")
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / "cookies.txt"

            with open(target, "wb") as f:
                f.write(raw)

            if target.exists() and target.stat().st_size > 10:
                os.environ["YT_COOKIES_FILE"] = str(target)
                logger.info("✅ Decoded YT_COOKIES_B64 to %s (%d bytes)", target, len(raw))
                return
        except Exception as e:
            logger.error("❌ Failed to decode YT_COOKIES_B64: %s", e)

    fpath = (os.getenv("YT_COOKIES_FILE") or "").strip()
    if fpath and os.path.exists(fpath):
        try:
            if fpath.startswith(("/etc/", "/run/")):
                target_dir = Path("/tmp/yt-dlp")
                target_dir.mkdir(parents=True, exist_ok=True)
                target = target_dir / "cookies.txt"
                shutil.copyfile(fpath, target)
                os.environ["YT_COOKIES_FILE"] = str(target)
                logger.info("✅ Copied YT_COOKIES_FILE to %s", target)
        except Exception as e:
            logger.warning("⚠️ Could not copy YT_COOKIES_FILE: %s", e)

# --------
# Startup
# --------
@app.on_event("startup")
async def on_startup():
    """Application startup tasks"""
    _sanitize_stripe_key()
    _hydrate_youtube_cookies_to_tmp()
    _configure_global_proxy_from_env()

    initialize_database()
    run_startup_migrations(engine)

    threading.Thread(target=_cleanup_stale_files_loop, daemon=True).start()

    logger.info("=" * 60)
    logger.info("🚀 YouTube Content Downloader API Starting")
    logger.info("=" * 60)
    logger.info("📝 Environment: %s", ENVIRONMENT)
    logger.info("📁 Downloads directory: %s", DOWNLOADS_DIR)
    logger.info("🗄️ Database: %s", "PostgreSQL" if "postgres" in DATABASE_URL else "SQLite")

    cookie_file = os.getenv("YT_COOKIES_FILE")
    logger.info("🍪 Cookies configured: %s", bool(cookie_file and os.path.exists(cookie_file)))
    
    proxy_enabled = (os.getenv("PROXY_ENABLED", "false") or "").lower() in {"1", "true", "yes", "on"}
    logger.info("🌐 Proxy enabled: %s", proxy_enabled)
    logger.info("💳 Stripe configured: %s", bool(stripe and os.getenv("STRIPE_SECRET_KEY")))
    logger.info("🎬 yt-dlp available: %s", check_ytdlp_availability())
    logger.info("=" * 60)
    logger.info("✅ Backend started successfully")
    logger.info("=" * 60)

# --------
# Routes
# --------
@app.get("/")
def root():
    return {
        "message": "YouTube Content Downloader API",
        "status": "running",
        "version": "3.4.3",  # Updated version
        "features": ["transcripts", "audio", "video", "mobile", "history", "payments"],
        "downloads_path": str(DOWNLOADS_DIR),
    }

# ===== FIX #4: Graceful registration (doesn't fail if Stripe is down) =====
@app.post("/register")
def register(user: UserCreate, db: Session = Depends(get_db)):
    logger.info(f"🔵 REGISTRATION STARTED - Username: {user.username}, Email: {user.email}")

    username = (user.username or "").strip()
    email = (user.email or "").strip().lower()

    # Validation
    existing_user = db.query(User).filter(User.username == username).first()
    if existing_user:
        logger.warning(f"❌ Username already exists: {username}")
        raise HTTPException(status_code=400, detail="Username already exists.")

    existing_email = db.query(User).filter(User.email == email).first()
    if existing_email:
        logger.warning(f"❌ Email already exists: {email}")
        raise HTTPException(status_code=400, detail="Email already exists.")

    # Create user account (ALWAYS succeeds, even if Stripe fails)
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
        logger.info(f"✅ User account created: {username} (ID: {obj.id})")
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        raise HTTPException(status_code=500, detail="Database error during registration")

    # Try to create Stripe customer (graceful - doesn't fail registration)
    stripe_customer_id = None
    if stripe:
        try:
            ensure_stripe_customer_for_user(obj, db)
            stripe_customer_id = obj.stripe_customer_id
        except Exception as e:
            logger.warning(f"⚠️ Stripe customer creation skipped (non-fatal): {e}")
            # Registration still succeeds!

    logger.info(f"🎉 REGISTRATION COMPLETED: {username}")

    return {
        "message": "User registered successfully.",
        "account": canonical_account(obj),
        "stripe_customer_id": stripe_customer_id,
    }


@app.post("/auth/forgot-password")
def forgot_password(payload: ForgotPasswordIn):
    from models import SessionLocal
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == payload.email).first()
        if user:
            token = serializer.dumps({"email": payload.email})
            reset_link = f"{FRONTEND_URL}/reset?token={token}"
            try:
                send_password_reset_email(payload.email, reset_link)
            except Exception as e:
                logger.exception("Failed to send reset email")
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
        user = db.query(User).filter(User.email == email).first()
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

    # Try to ensure Stripe customer (graceful)
    try:
        ensure_stripe_customer_for_user(user, db)
    except Exception as e:
        logger.warning(f"⚠️ Stripe customer link skipped during login: {e}")

    token = create_access_token(
        {"sub": user.username},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
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

    try:
        ensure_stripe_customer_for_user(user, db)
    except Exception as e:
        logger.warning(f"⚠️ Stripe customer link skipped: {e}")

    token = create_access_token(
        {"sub": user.username},
        timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
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
def change_password(
    req: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not verify_password(req.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    if not req.new_password or len(req.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters")
    
    current_user.hashed_password = get_password_hash(req.new_password)
    try:
        current_user.must_change_password = False
    except Exception:
        pass
    db.commit()
    db.refresh(current_user)
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
            try:
                p.unlink(missing_ok=True)
            except Exception:
                pass
    except Exception:
        pass

    return {
        "message": "Account deleted successfully.",
        "deleted_at": datetime.utcnow().isoformat(),
        "user_email": email,
    }


# ---------------------------
# Stripe webhook idempotency
# ---------------------------
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
    result = await handle_stripe_webhook(request) if hasattr(handle_stripe_webhook, "__call__") else {"status": "ok"}
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
def cancel_subscription(
    req: CancelRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if (current_user.subscription_tier or "free") == "free":
        raise HTTPException(status_code=400, detail="No active subscription to cancel.")

    sub = _latest_subscription(db, current_user.id)
    at_period_end = True if req.at_period_end is None else bool(req.at_period_end)

    stripe_updated = False
    if stripe and sub and hasattr(sub, "stripe_subscription_id"):
        stripe_sub_id = getattr(sub, "stripe_subscription_id", None)
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
            sub.extra_data = ((sub.extra_data or "") + ("\n" if sub.extra_data else "") + note)
        result = {"status": "scheduled_cancellation", "at_period_end": True}
    else:
        if sub:
            sub.status = "cancelled"
            sub.cancelled_at = datetime.utcnow()
        current_user.subscription_tier = "free"
        # ✅ Reset usage when cancelling immediately
        _reset_usage_counters(current_user)
        result = {"status": "cancelled", "at_period_end": False, "tier": "free"}

    try:
        db.commit()
        db.refresh(current_user)
        db.refresh(sub) if sub else None
    except Exception:
        db.rollback()

    result.update({"stripe_updated": stripe_updated})
    return result


class _AuthForm(BaseModel):
    pass

def _touch_now(p: Path):
    try:
        now = time.time()
        os.utime(p, (now, now))
    except Exception as e:
        logger.warning("Could not set mtime: %s", e)

# --------------------
# Download transcript
# --------------------
@app.post("/download_transcript")
@app.post("/download_transcript/")
def download_transcript(
    req: TranscriptRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ensure_monthly_reset_and_tier(db, user)
    start = time.time()
    vid = extract_youtube_video_id(req.youtube_id)
    if not vid or len(vid) != 11:
        raise HTTPException(
            status_code=400, detail="Invalid YouTube video ID."
        )
    if not check_internet():
        raise HTTPException(
            status_code=503, detail="No internet connection available."
        )

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
            "SRT transcript"
            if req.format == "srt"
            else "VTT transcript"
            if req.format == "vtt"
            else "clean transcript"
            if req.clean_transcript
            else "timestamped transcript"
        )
        raise HTTPException(
            status_code=403,
            detail=f"Monthly limit reached for {type_name} ({used}/{limit}).",
        )

    try:
        text = get_transcript_youtube_api(
            vid, clean=req.clean_transcript, fmt=req.format
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Transcript fetch failed for {vid}: {error_msg}")

        if "blocking" in error_msg.lower() or "cloud provider" in error_msg.lower():
            raise HTTPException(
                status_code=503,
                detail=(
                    "YouTube is temporarily limiting transcript access from our servers. "
                    "This video may not have captions available, or please try again later."
                ),
            )

        if "no captions" in error_msg.lower() or "not have captions" in error_msg.lower():
            raise HTTPException(
                status_code=404,
                detail=(
                    "This video does not have captions/transcripts available. "
                    "Please try a different video."
                ),
            )

        raise HTTPException(
            status_code=404,
            detail=(
                "Could not retrieve transcript for this video. "
                "The video may not have captions available, or YouTube may be limiting access."
            ),
        )

    if not text:
        raise HTTPException(
            status_code=404,
            detail="No transcript found for this video.",
        )

    new_usage = increment_user_usage(db, user, usage_key)
    proc = time.time() - start
    rec = create_download_record(
        db=db,
        user=user,
        kind=usage_key,
        youtube_id=vid,
        file_format=file_format,
        file_size=len(text),
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
    }

# ---------------
# Download audio
# ---------------
@app.post("/download_audio/")
def download_audio(
    req: AudioRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ensure_monthly_reset_and_tier(db, user)
    start = time.time()
    vid = extract_youtube_video_id(req.youtube_id)
    if not vid or len(vid) != 11:
        raise HTTPException(
            status_code=400, detail="Invalid YouTube video ID."
        )
    if not check_internet():
        raise HTTPException(
            status_code=503, detail="No internet connection available."
        )
    if not check_ytdlp_availability():
        raise HTTPException(
            status_code=500,
            detail="Audio download service temporarily unavailable.",
        )

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
            vid, req.quality, output_dir=str(DOWNLOADS_DIR)
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Audio download failed for {vid}: {error_msg}")

        if "403" in error_msg or "forbidden" in error_msg.lower():
            raise HTTPException(
                status_code=503,
                detail=(
                    "YouTube is temporarily limiting audio downloads. "
                    "This video may have restrictions. Please try again later or try a different video."
                ),
            )
        elif "available" in error_msg.lower() or "formats" in error_msg.lower():
            raise HTTPException(
                status_code=404,
                detail=(
                    "No audio streams available for this video. "
                    "It may be restricted or unavailable in your region."
                ),
            )
        elif "timeout" in error_msg.lower():
            raise HTTPException(
                status_code=504,
                detail="Download timed out. Please try again.",
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download audio: {error_msg[:200]}",
            )

    if not path or not os.path.exists(path):
        raise HTTPException(
            status_code=404, detail="Failed to download audio."
        )

    downloaded = Path(path)
    fsize = downloaded.stat().st_size
    if fsize < 1000:
        raise HTTPException(
            status_code=500,
            detail="Downloaded audio appears corrupted.",
        )

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

# -----------------
# Download video
# -----------------
@app.post("/download_video/")
def download_video(
    req: VideoRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ensure_monthly_reset_and_tier(db, user)
    start = time.time()
    vid = extract_youtube_video_id(req.youtube_id)
    if not vid or len(vid) != 11:
        raise HTTPException(
            status_code=400, detail="Invalid YouTube video ID."
        )
    if not check_internet():
        raise HTTPException(
            status_code=503, detail="No internet connection available."
        )
    if not check_ytdlp_availability():
        raise HTTPException(
            status_code=500, detail="Video download service unavailable."
        )

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

    path = download_video_with_ytdlp(
        vid, req.quality, output_dir=str(DOWNLOADS_DIR)
    )
    if not path or not os.path.exists(path):
        raise HTTPException(
            status_code=404, detail="Failed to download video."
        )
    downloaded = Path(path)
    fsize = downloaded.stat().st_size
    if fsize < 10_000:
        raise HTTPException(
            status_code=500,
            detail="Downloaded video appears corrupted.",
        )

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

# -----------------------------
# Authenticated file download
# -----------------------------
@app.get("/download-file/{file_type}/{filename}")
async def download_file(
    request: Request,
    file_type: str,
    filename: str,
    auth: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    if file_type not in {"audio", "video"}:
        raise HTTPException(
            status_code=400, detail="Invalid file type"
        )

    user: Optional[User] = None
    if auth:
        try:
            payload = jwt.decode(auth, SECRET_KEY, algorithms=[ALGORITHM])
            username = payload.get("sub")
            if username:
                user = get_user(db, username)
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except Exception:
            raise HTTPException(status_code=401, detail="Invalid token")

    if not user:
        ah = request.headers.get("authorization", "")
        if ah.lower().startswith("bearer "):
            try:
                payload = jwt.decode(
                    ah.split(" ", 1)[1], SECRET_KEY, algorithms=[ALGORITHM]
                )
                username = payload.get("sub")
                if username:
                    user = get_user(db, username)
            except Exception:
                pass

    if not user:
        raise HTTPException(
            status_code=401, detail="Authentication required"
        )

    file_path = (DOWNLOADS_DIR / filename).resolve()
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
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
                    if not chunk:
                        break
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
        return StreamingResponse(
            gen(), media_type=mime, headers=headers
        )

    headers = {
        "Content-Disposition": f'attachment; filename="{safe_name}"',
        "Content-Length": str(size),
        "Accept-Ranges": "bytes",
    }
    return FileResponse(
        path=str(file_path),
        media_type=mime,
        headers=headers,
        filename=safe_name,
    )

# ---------------------------
# History & recent activity
# ---------------------------
@app.get("/user/download-history")
def get_download_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(TranscriptDownload)
        .filter(TranscriptDownload.user_id == current_user.id)
        .order_by(TranscriptDownload.created_at.desc())
        .limit(50)
        .all()
    )
    hist = [
        {
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
        }
        for d in rows
    ]
    return {
        "downloads": hist,
        "total_count": len(hist),
        "account": canonical_account(current_user),
    }

@app.get("/user/recent-activity")
@app.get("/user/recent-activity/")
def get_recent_activity(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(TranscriptDownload)
        .filter(TranscriptDownload.user_id == current_user.id)
        .order_by(TranscriptDownload.created_at.desc())
        .limit(15)
        .all()
    )
    activities = []
    for d in rows:
        action, icon, desc_prefix, category = classify_activity_by_format(
            d.transcript_type or "", d.file_format or "txt"
        )
        description = f"{desc_prefix} for video {d.youtube_id}"
        if d.quality and d.quality != "default":
            description += f" ({d.quality})"
        if d.file_size:
            size_mb = d.file_size / (1024 * 1024)
            description += (
                f" - {size_mb:.1f}MB"
                if size_mb >= 1
                else f" - {d.file_size / 1024:.0f}KB"
            )
        activities.append(
            {
                "id": d.id,
                "action": action,
                "description": description,
                "timestamp": d.created_at.isoformat()
                if d.created_at
                else None,
                "icon": icon,
                "type": d.transcript_type,
                "video_id": d.youtube_id,
                "file_format": d.file_format or "txt",
                "file_size": d.file_size,
                "quality": d.quality,
                "status": getattr(d, "status", "completed"),
                "category": category,
                "processing_time": d.processing_time,
            }
        )
    if not activities:
        activities.append(
            {
                "id": 0,
                "action": "Account created",
                "description": f"Welcome to the app, {current_user.username}!",
                "timestamp": current_user.created_at.isoformat()
                if current_user.created_at
                else None,
                "type": "auth",
                "icon": "🎉",
                "category": "system",
            }
        )
    return {
        "activities": activities,
        "total_count": len(activities),
        "account": canonical_account(current_user),
        "fetched_at": datetime.utcnow().isoformat(),
    }

# ---------------------
# Subscription status
# ---------------------
@app.get("/subscription_status")
@app.get("/subscription_status/")
def subscription_status(
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Local overdue downgrade first
    try:
        _apply_local_overdue_downgrade_if_possible(current_user, db)
    except Exception as e:
        logger.warning(f"Local overdue downgrade check skipped: {e}")

    # Optional Stripe sync
    if request.query_params.get("sync") == "1":
        try:
            sync_user_subscription_from_stripe(current_user, db)
        except Exception as e:
            logger.warning(f"Stripe sync skipped (non-fatal): {e}")

    # Monthly reset AFTER tier enforcement
    next_reset = ensure_monthly_reset_and_tier(db, current_user)

    tier = (getattr(current_user, "subscription_tier", "free") or "free").lower()

    usage = {
        "clean_transcripts": getattr(current_user, "usage_clean_transcripts", 0) or 0,
        "unclean_transcripts": getattr(current_user, "usage_unclean_transcripts", 0) or 0,
        "audio_downloads": getattr(current_user, "usage_audio_downloads", 0) or 0,
        "video_downloads": getattr(current_user, "usage_video_downloads", 0) or 0,
    }

    limits = PLAN_LIMITS.get(tier, PLAN_LIMITS["free"])
    limits_display = {k: ("unlimited" if v == float("inf") else v) for k, v in limits.items()}

    sub_expires_at = getattr(current_user, "subscription_expires_at", None)
    stripe_status = getattr(current_user, "stripe_subscription_status", None)
    sub_updated_at = getattr(current_user, "subscription_updated_at", None)

    def _iso(dt):
        if not dt:
            return None
        if getattr(dt, "tzinfo", None) is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()

    return {
        "tier": tier,
        "status": "active" if tier != "free" else "inactive",
        "usage": usage,
        "limits": limits_display,
        "next_reset": next_reset.isoformat() if next_reset else None,
        "downloads_folder": str(DOWNLOADS_DIR),
        "account": canonical_account(current_user),
        "subscription_expires_at": _iso(sub_expires_at),
        "stripe_subscription_status": (stripe_status.lower() if isinstance(stripe_status, str) else stripe_status),
        "subscription_updated_at": _iso(sub_updated_at),
    }

# -----------------
# Contact form
# -----------------
@app.post("/contact")
def send_contact_message(req: ContactMessage, request: Request):
    name = (req.name or "").strip()[:200]
    email = (req.email or "").strip()[:255].lower()
    message = (req.message or "").strip()

    if not name or not message or not _email_re.match(email):
        raise HTTPException(
            status_code=400,
            detail="Please provide a valid name, email, and message.",
        )

    ip = (request.client.host if request.client else "unknown")
    now = time.time()
    window, limit = 600, 3
    with _contact_lock:
        q = _contact_hits[ip]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            raise HTTPException(
                status_code=429,
                detail="Too many messages. Please try again later.",
            )
        q.append(now)

    try:
        _send_contact_email(name, email, message)
        return {"ok": True, "message": "Thanks! Your message has been sent."}
    except Exception as e:
        logger.error(f"Contact email failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Unable to send message right now.",
        )


# Health endpoints
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
        "proxy_enabled": (os.getenv("PROXY_ENABLED", "false") or "").lower() in {"1", "true", "yes", "on"},
    }

@app.head("/health")
def health_head():
    return Response(status_code=200)

# ---------------------------------------------------------------------------
# Debug users (dev only)
# ---------------------------------------------------------------------------
@app.get("/debug/users")
def debug_users(db: Session = Depends(get_db)):
    if os.getenv("ENVIRONMENT", "development") != "development":
        raise HTTPException(status_code=404, detail="Not found")
    users = db.query(User).all()
    return {
        "total_users": len(users),
        "users": [
            {
                "id": u.id,
                "username": u.username,
                "email": (u.email or "").strip().lower(),
                "created_at": u.created_at.isoformat() if u.created_at else None,
                "subscription_tier": getattr(u, "subscription_tier", "free"),
                "is_active": getattr(u, "is_active", True),
            }
            for u in users
        ],
    }


# Include routers
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


# Frontend SPA catch-all
FRONTEND_BUILD = Path(__file__).resolve().parents[1] / "frontend" / "build"
if FRONTEND_BUILD.exists():
    app.mount("/_spa", StaticFiles(directory=str(FRONTEND_BUILD), html=True), name="spa")

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa_catch_all(full_path: str):
        if full_path.startswith((
            "download_", "user/", "subscription_status", "health",
            "token", "register", "files/", "debug/",
        )):
            raise HTTPException(status_code=404, detail="Not found")
        index_file = FRONTEND_BUILD / "index.html"
        if index_file.exists():
            return HTMLResponse(index_file.read_text(encoding="utf-8"))
        raise HTTPException(status_code=404, detail="Frontend not built")

# ---------------------------------------------------------------------------
# Uvicorn entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn # pyright: ignore[reportMissingImports]

    print(f"Starting server on 0.0.0.0:8000 — files: {DOWNLOADS_DIR}")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

##--------- End of main.py module -------------

