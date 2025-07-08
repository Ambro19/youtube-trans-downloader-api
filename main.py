# main.py (advanced, usage tracked on User model, for youtube-transcript-api only)
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

from database import engine, SessionLocal, get_db  # Make sure this points to your db
from models import User, create_tables  # Use your models.py User model!
#from database import SessionLocal, get_db  # Make sure this points to your db

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

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    return db.query(User).filter(User.email == email).first()

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
        #orm_mode = True
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
             #   lines.append(f"{timestamp} {seg['text'].replace('\n', ' ')}")
                text_clean = seg['text'].replace('\n', ' ')
                lines.append(f"{timestamp} {text_clean}")

            return "\n".join(lines)
    except Exception as e:
        logger.info(f"Transcript API failed: {e}")
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

@app.post("/register", response_model=UserResponse)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    if get_user(db, user_data.username):
        raise HTTPException(status_code=400, detail="Username already registered")
    if get_user_by_email(db, user_data.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        created_at=datetime.utcnow()
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.post("/token", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = get_user(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
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

    # --- Usage logic using models.py advanced fields ---
    usage_key = USAGE_KEYS[request.clean_transcript]
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
        transcript = get_demo_content(clean=request.clean_transcript)

    # Increment usage counter, save
    user.increment_usage(usage_key)
    db.commit()
    logger.info(f"User {user.username} downloaded transcript for {video_id} ({usage_key})")
    return {
        "transcript": transcript,
        "youtube_id": video_id,
        "message": "Transcript downloaded successfully"
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

#===============================

# # Import necessary libraries and modules
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
# import json
# import os
# import logging
# from dotenv import load_dotenv
# import re
# import warnings

# # Suppress bcrypt warning (common in some environments)
# warnings.filterwarnings("ignore", message=".*bcrypt.*")

# # Import database-related functions and models
# from database import get_db, User, Subscription, TranscriptDownload, create_tables

# # Load environment variables from .env file
# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("youtube_trans_downloader.main")

# # Initialize Stripe with secret key from environment variables
# stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
# endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
# DOMAIN = os.getenv("DOMAIN", "https://youtube-trans-downloader-api.onrender.com")

# # Create FastAPI application with metadata
# app = FastAPI(
#     title="YouTubeTransDownloader API",
#     description="API for downloading and processing YouTube video transcripts",
#     version="1.0.0"
# )

# # Determine environment (production or development)
# ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# # Configure CORS (Cross-Origin Resource Sharing) based on environment
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

# # Add CORS middleware to the application
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=allowed_origins,
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
#     allow_headers=["*"],
# )

# # Configure OAuth2 password bearer for token authentication
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # JWT configuration
# SECRET_KEY = os.getenv("SECRET_KEY")
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# # Define subscription limits for different tiers
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

# # Map subscription tiers to Stripe price IDs
# PRICE_ID_MAP = {
#     "pro": os.getenv("PRO_PRICE_ID"),
#     "premium": os.getenv("PREMIUM_PRICE_ID")
# }

# # Startup event - runs when the application starts
# @app.on_event("startup")
# async def startup_event():
#     try:
#         logger.info("🚀 Starting YouTube Transcript Downloader API...")
#         logger.info(f"🌍 Environment: {ENVIRONMENT}")
#         logger.info(f"🔗 Domain: {DOMAIN}")

#         # Check for required environment variables
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

#         # Initialize database tables
#         create_tables()
#         logger.info("✅ Database initialized successfully")
#         logger.info("🎉 Application startup complete!")
#     except Exception as e:
#         logger.error(f"❌ Startup failed: {str(e)}")
#         raise

# # =============================================
# # Pydantic Models for request/response schemas
# # =============================================

# class Token(BaseModel):
#     """Model for JWT token response"""
#     access_token: str
#     token_type: str

# class TokenData(BaseModel):
#     """Model for token data payload"""
#     username: Optional[str] = None

# class UserCreate(BaseModel):
#     """Model for user registration request"""
#     username: str
#     email: str
#     password: str

# class UserResponse(BaseModel):
#     """Model for user data response (without sensitive info)"""
#     id: int
#     username: str
#     email: str
#     created_at: datetime
#     class Config:
#         from_attributes = True

# class TranscriptRequest(BaseModel):
#     """Model for transcript download request"""
#     youtube_id: str
#     clean_transcript: bool = False

# class CreatePaymentIntentRequest(BaseModel):
#     """Model for creating payment intent request"""
#     price_id: str

# class ConfirmPaymentRequest(BaseModel):
#     """Model for confirming payment request"""
#     payment_intent_id: str

# class SubscriptionRequest(BaseModel):
#     """Model for subscription request"""
#     token: Optional[str] = None
#     subscription_tier: str

# class SubscriptionResponse(BaseModel):
#     """Model for subscription status response"""
#     tier: str
#     status: str
#     expiry_date: Optional[str] = None
#     limits: dict
#     usage: Optional[dict] = None
#     remaining: Optional[dict] = None
#     class Config:
#         from_attributes = True

# # =============================================
# # Authentication and User Helper Functions
# # =============================================

# def verify_password(plain_password, hashed_password):
#     """Verify a plain password against a hashed password"""
#     return pwd_context.verify(plain_password, hashed_password)

# def get_password_hash(password):
#     """Generate a password hash"""
#     return pwd_context.hash(password)

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
#     """Create a JWT access token with optional expiration"""
#     to_encode = data.copy()
#     if expires_delta:
#         expire = datetime.utcnow() + expires_delta
#     else:
#         expire = datetime.utcnow() + timedelta(minutes=15)
#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
#     return encoded_jwt

# def get_user(db: Session, username: str):
#     """Get a user by username from the database"""
#     return db.query(User).filter(User.username == username).first()

# def get_user_by_email(db: Session, email: str):
#     """Get a user by email from the database"""
#     return db.query(User).filter(User.email == email).first()

# def authenticate_user(db: Session, username: str, password: str):
#     """Authenticate a user with username and password"""
#     user = get_user(db, username)
#     if not user:
#         return False
#     if not verify_password(password, user.hashed_password):
#         return False
#     return user

# def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
#     """Dependency to get the current authenticated user from JWT token"""
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

# # =============================================
# # Stripe Helper Functions
# # =============================================

# def get_or_create_stripe_customer(user, db: Session):
#     """Get or create a Stripe customer for a user"""
#     try:
#         # Check if user already has a Stripe customer ID
#         if hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
#             try:
#                 # Try to retrieve existing customer
#                 customer = stripe.Customer.retrieve(user.stripe_customer_id)
#                 logger.info(f"✅ Found existing Stripe customer: {customer.id} for user {user.username}")
#                 return customer
#             except stripe.error.InvalidRequestError:
#                 logger.info(f"📝 Stripe customer {user.stripe_customer_id} not found, creating new one")
#                 pass
        
#         # Create new Stripe customer
#         customer = stripe.Customer.create(
#             email=user.email,
#             name=user.username,
#             description=f"YouTube Transcript Downloader user: {user.username}",
#             metadata={
#                 'user_id': str(user.id),
#                 'username': user.username,
#                 'signup_date': user.created_at.isoformat() if user.created_at else None,
#                 'app_name': 'YouTube Transcript Downloader'
#             }
#         )
        
#         # Save the Stripe customer ID to the user record
#         if hasattr(user, 'stripe_customer_id'):
#             user.stripe_customer_id = customer.id
#             db.commit()
#             db.refresh(user)
#             logger.info(f"✅ Stripe customer created and saved: {customer.id}")
#         return customer
#     except Exception as e:
#         logger.error(f"❌ Error creating Stripe customer for user {user.username}: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to create payment customer. Please try again."
#         )

# def check_subscription_limit(user_id: int, transcript_type: str, db: Session):
#     """Check if user has reached their subscription limit for a transcript type"""
#     try:
#         # Get user's subscription
#         subscription = db.query(Subscription).filter(Subscription.user_id == user_id).first()
#         if not subscription:
#             tier = "free"  # Default to free tier if no subscription
#         else:
#             tier = subscription.tier
#             if subscription.expiry_date < datetime.now():
#                 tier = "free"  # Downgrade to free if subscription expired
        
#         # Calculate usage for current month
#         month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
#         try:
#             usage = db.query(TranscriptDownload).filter(
#                 TranscriptDownload.user_id == user_id,
#                 TranscriptDownload.transcript_type == transcript_type,
#                 TranscriptDownload.created_at >= month_start
#             ).count()
#         except Exception as e:
#             # Fallback query if the ORM query fails
#             logger.warning(f"Enhanced query failed, using simple fallback: {e}")
#             result = db.execute(
#                 "SELECT COUNT(*) FROM transcript_downloads WHERE user_id = ? AND transcript_type = ? AND created_at >= ?",
#                 (user_id, transcript_type, month_start.isoformat())
#             )
#             usage = result.scalar() or 0
        
#         # Get limit based on subscription tier
#         limit = SUBSCRIPTION_LIMITS[tier].get(transcript_type, 0)
#         if limit == float('inf'):
#             return True  # No limit for premium tier
#         return usage < limit
#     except Exception as e:
#         logger.error(f"Error checking subscription limit: {e}")
#         return True  # Allow by default if there's an error

# # =============================================
# # Transcript Extraction Functions (Fixed Section)
# # =============================================

# def get_youtube_transcript_corrected(video_id: str, clean: bool = True) -> str:
#     """Main function to get YouTube transcript with multiple fallback methods"""
#     logger.info(f"Attempting transcript extraction for video {video_id}")

#     # 1. First try: YouTubeTranscriptApi (best for manual/auto captions)
#     try:
#         from youtube_transcript_api import YouTubeTranscriptApi
#         # List available transcripts
#         transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
#         # Try preferred languages first, then any available languages
#         preferred_langs = ['en', 'en-US', 'en-GB']
#         all_langs = preferred_langs + [t.language_code for t in transcript_list if t.language_code not in preferred_langs]
        
#         for lang in all_langs:
#             try:
#                 # Try to get transcript for each language
#                 transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
#                 if transcript and len(transcript) > 0:
#                     logger.info(f"SUCCESS with youtube_transcript_api language {lang}: {len(transcript)} segments")
#                     return format_transcript_enhanced(transcript, clean)
#             except Exception as e:
#                 logger.info(f"Language {lang} failed: {e}")
#                 continue
#     except Exception as e:
#         logger.info(f"youtube-transcript-api failed: {e}")

#     # 2. Fallback: yt-dlp (more robust but requires downloading)
#     try:
#         import yt_dlp
#         # Configure yt-dlp options for subtitle extraction
#         ydl_opts = {
#             "skip_download": True,
#             "writesubtitles": True,
#             "writeautomaticsub": True,
#             "subtitleslangs": ["en", "en-US", "en-GB"],
#             "quiet": True,
#             "no_warnings": True,
#         }
        
#         with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#             # Extract video info (without downloading video)
#             info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
#             # Check both manual subtitles and automatic captions
#             for sub_dict in (info.get('subtitles', {}), info.get('automatic_captions', {})):
#                 for lang, entries in sub_dict.items():
#                     for entry in entries:
#                         # Download and parse each subtitle format
#                         transcript_text = extract_subtitle_from_entry(ydl, entry, clean)
#                         if transcript_text and not is_invalid_content(transcript_text):
#                             logger.info(f"✅ Found valid subtitle ({lang}, {entry.get('ext','')}) length: {len(transcript_text)}")
#                             return transcript_text
#     except Exception as ydl_error:
#         logger.info(f"yt-dlp failed: {ydl_error}")

#     # 3. Final fallback: Return demo content if all methods fail
#     logger.warning(f"All extraction methods failed for {video_id}. Returning DEMO content.")
#     return get_demo_content(clean)

# def extract_subtitle_from_entry(url: str, ext: str, clean: bool) -> str:
#     import requests
#     try:
#         resp = requests.get(url, timeout=10)
#         content = resp.text
#         logger.info(f"Downloaded subtitle file, ext={ext}, length={len(content)}")
#         # --- Robust handling of subtitle formats, including json3 ---
#         if ext == "vtt":
#             return parse_vtt(content, clean)
#         elif ext in ("srv1", "srv2", "srv3", "json"):
#             return parse_srv(content, clean)
#         elif ext == "ttml":
#             return parse_ttml(content, clean)
#         elif ext == "srt":
#             return parse_srt(content, clean)
#         elif ext == "json3":
#             return parse_json3(content, clean)
#         elif ext == "m3u8":
#             return parse_m3u8(content, clean)
#         else:
#             # fallback: return text
#             return content if len(content.strip()) > 0 else ""
#     except Exception as e:
#         logger.info(f"Failed to download/parse subtitle file: {e}")
#         return ""


# def parse_json3(content: str, clean: bool) -> str:
#     """
#     Parse YouTube's json3 subtitle format into clean or unclean text.
#     - clean=True: paragraph text only.
#     - clean=False: SRT-style text with timestamps ([MM:SS]).
#     """
#     try:
#         data = json.loads(content)
#         events = data.get('events', [])
#         lines = []
#         for event in events:
#             t_start_ms = event.get('tStartMs')
#             segs = event.get('segs')
#             if segs and isinstance(segs, list):
#                 text = ''.join(seg.get('utf8', '') for seg in segs).replace('\n', ' ').strip()
#                 if not text:
#                     continue
#                 # Format timestamp if needed
#                 if not clean and t_start_ms is not None:
#                     t = int(t_start_ms) // 1000
#                     minutes = t // 60
#                     seconds = t % 60
#                     timestamp = f"[{minutes:02d}:{seconds:02d}]"
#                     lines.append(f"{timestamp} {text}")
#                 elif clean:
#                     lines.append(text)
#         if clean:
#             return ' '.join(lines)
#         else:
#             return '\n'.join(lines)
#     except Exception as e:
#         logger.info(f"Failed to parse json3: {e}")
#         return ""

# def format_transcript_enhanced(transcript_list: list, clean: bool = True) -> str:
#     """Format raw transcript data into clean or timestamped text"""
#     if not transcript_list:
#         raise Exception("Empty transcript data")
    
#     if clean:
#         # Clean format: single paragraph with no timestamps
#         texts = []
#         for item in transcript_list:
#             text = item.get('text', '').strip()
#             if text:
#                 text = clean_transcript_text(text)
#                 if text:
#                     texts.append(text)
#         result = ' '.join(texts)
#         result = ' '.join(result.split())  # Remove extra whitespace
#         result = result.replace(' .', '.').replace(' ,', ',')  # Fix punctuation spacing
        
#         if len(result) < 20:
#             raise Exception("Transcript too short or invalid")
#         logger.info(f"✅ Clean transcript: {len(result)} characters")
#         return result
#     else:
#         # Timestamped format: each line with [MM:SS] prefix
#         lines = []
#         for item in transcript_list:
#             start = item.get('start', 0)
#             text = item.get('text', '').strip()
#             if text:
#                 text = clean_transcript_text(text)
#                 if text:
#                     minutes = int(start // 60)
#                     seconds = int(start % 60)
#                     timestamp = f"[{minutes:02d}:{seconds:02d}]"
#                     lines.append(f"{timestamp} {text}")
        
#         if len(lines) < 5:
#             raise Exception("Transcript has too few valid segments")
#         result = '\n'.join(lines)
#         logger.info(f"✅ Timestamped transcript: {len(lines)} lines")
#         return result

# def clean_transcript_text(text: str) -> str:
#     """Clean and normalize transcript text"""
#     if not text:
#         return ""
#     # Replace HTML entities
#     text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
#     # Remove newlines and HTML tags
#     text = text.replace('\n', ' ').replace('\r', ' ')
#     text = re.sub(r'<[^>]+>', '', text)
#     # Normalize whitespace
#     text = ' '.join(text.split())
#     # Clean up punctuation
#     text = text.strip('.,!?;: ')
#     return text.strip()

# def is_invalid_content(text: str) -> bool:
#     """Check if content is invalid or too short"""
#     if not text or len(text.strip()) < 32:
#         return True
#     # Block HTML content
#     if text.lstrip().lower().startswith("<html"):
#         return True
#     return False

# # =============================================
# # Subtitle Format Parsing Functions
# # =============================================

# def parse_vtt(content: str, clean: bool) -> str:
#     """Parse WebVTT format subtitles"""
#     lines = content.splitlines()
#     # Filter out VTT timestamps and headers
#     filtered = [line for line in lines if line and not re.match(r"^(\d{2}:){1,2}\d{2}\.\d{3} -->", line) and not line.startswith("WEBVTT")]
#     text = "\n".join(filtered)
#     if clean:
#         text = re.sub(r"<[^>]+>", "", text)  # Remove any remaining HTML tags
#     return text.strip()

# def parse_srv(content: str, clean: bool) -> str:
#     # Use this for .srv1, .srv2, .srv3, .json (legacy)
#     try:
#         import xml.etree.ElementTree as ET
#         try:
#             root = ET.fromstring(content)
#             texts = [el.text for el in root.iter() if el.tag == "text" and el.text]
#             text = "\n".join(texts)
#             if clean:
#                 text = re.sub(r"<[^>]+>", "", text)
#             return text.strip()
#         except ET.ParseError:
#             # Not XML, maybe old JSON (very rare)
#             return ""
#     except Exception as e:
#         logger.info(f"Failed to parse srv: {e}")
#         return ""

# def parse_ttml(content: str, clean: bool) -> str:
#     """Parse TTML format subtitles (XML-based)"""
#     import xml.etree.ElementTree as ET
#     try:
#         root = ET.fromstring(content)
#         # Extract all text content
#         texts = [el.text for el in root.iter() if el.text]
#         text = "\n".join(texts)
#         if clean:
#             text = re.sub(r"<[^>]+>", "", text)
#         return text.strip()
#     except Exception as e:
#         logger.info(f"Failed to parse ttml: {e}")
#         return ""

# def parse_srt(content: str, clean: bool) -> str:
#     """Parse SRT format subtitles"""
#     # Minimalist SRT parser - strips sequence numbers and timestamps
#     text_lines = []
#     for line in content.splitlines():
#         if re.match(r"^\d+$", line.strip()):  # Skip sequence numbers
#             continue
#         if re.match(r"^\d{2}:\d{2}:\d{2},\d{3} -->", line.strip()):  # Skip timestamps
#             continue
#         if line.strip():
#             text_lines.append(line.strip())
#     text = "\n".join(text_lines)
#     if clean:
#         text = re.sub(r"<[^>]+>", "", text)
#     return text.strip()

# def parse_m3u8(content: str, clean: bool) -> str:
#     """Placeholder for M3U8 format parsing (not implemented)"""
#     logger.info("M3U8 subtitle parsing not implemented—returning raw content.")
#     return content.strip()

# def parse_json3_subtitle(content: str, clean: bool = True) -> str:
#     """
#     Parse YouTube's json3 subtitle format to clean/timestamped text.
#     """
#     try:
#         data = json.loads(content)
#         if "events" not in data:
#             return ""
#         events = data["events"]
#         lines = []
#         for ev in events:
#             segs = ev.get("segs", [])
#             if not segs:
#                 continue
#             # Join all seg texts for the event
#             text = "".join([seg.get("utf8", "") for seg in segs]).strip()
#             if not text:
#                 continue
#             # Get start time in ms, convert to MM:SS
#             ms = ev.get("tStartMs", 0)
#             minutes = int(ms // 60000)
#             seconds = int((ms % 60000) // 1000)
#             timestamp = f"[{minutes:02d}:{seconds:02d}]"
#             if clean:
#                 lines.append(text)
#             else:
#                 lines.append(f"{timestamp} {text}")
#         # Return as paragraph or timestamped lines
#         if clean:
#             return " ".join(lines)
#         else:
#             return "\n".join(lines)
#     except Exception as e:
#         logger.info(f"Failed to parse json3 subtitle: {e}")
#         return ""


# def get_demo_content(clean: bool = True):
#     """Fallback demo content when transcript extraction fails"""
#     demo = (
#         "This is demo transcript content. "
#         "The real YouTube transcript could not be downloaded."
#     )
#     if clean:
#         return demo
#     else:
#         return "[00:00] " + demo

# # =============================================
# # API Endpoints
# # =============================================

# @app.get("/")
# async def root():
#     """Root endpoint - basic API information"""
#     return {
#         "message": "YouTube Transcript Downloader API", 
#         "status": "running", 
#         "version": "1.0.0"
#     }

# @app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
# def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
#     """Endpoint for user registration"""
#     # Check if username already exists
#     db_user = get_user(db, user_data.username)
#     if db_user:
#         raise HTTPException(status_code=400, detail="Username already registered")
    
#     # Check if email already exists
#     email_exists = get_user_by_email(db, user_data.email)
#     if email_exists:
#         raise HTTPException(status_code=400, detail="Email already registered")
    
#     # Hash password and create new user
#     hashed_password = get_password_hash(user_data.password)
#     new_user = User(
#         username=user_data.username,
#         email=user_data.email,
#         hashed_password=hashed_password,
#         created_at=datetime.now()
#     )
    
#     try:
#         # Save new user to database
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
#     """Endpoint for user login and token generation"""
#     # Authenticate user
#     user = authenticate_user(db, form_data.username, form_data.password)
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect username or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
    
#     # Generate JWT token
#     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token = create_access_token(
#         data={"sub": user.username}, expires_delta=access_token_expires
#     )
#     logger.info(f"User logged in successfully: {form_data.username}")
#     return {"access_token": access_token, "token_type": "bearer"}

# @app.get("/users/me", response_model=UserResponse)
# async def read_users_me(current_user: User = Depends(get_current_user)):
#     """Endpoint to get current user's information"""
#     return current_user

# @app.get("/subscription_status/")
# async def get_subscription_status_ultra_safe(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     try:
#         subscription = db.query(Subscription).filter(
#             Subscription.user_id == current_user.id
#         ).first()
#         if not subscription:
#             tier = "free"
#             status = "inactive"
#             expiry_date = None
#         else:
#             if hasattr(subscription, 'expiry_date') and subscription.expiry_date and subscription.expiry_date < datetime.now():
#                 tier = "free"
#                 status = "expired"
#                 expiry_date = subscription.expiry_date
#             else:
#                 tier = subscription.tier if subscription.tier else "free"
#                 status = "active" if tier != "free" else "inactive"
#                 expiry_date = subscription.expiry_date if hasattr(subscription, 'expiry_date') else None
#         # Usage logic...
#         month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
#         def get_safe_usage(transcript_type):
#             try:
#                 return db.query(TranscriptDownload).filter(
#                     TranscriptDownload.user_id == current_user.id,
#                     TranscriptDownload.transcript_type == transcript_type,
#                     TranscriptDownload.created_at >= month_start
#                 ).count()
#             except Exception:
#                 return 0
#         clean_usage = get_safe_usage("clean")
#         unclean_usage = get_safe_usage("unclean")
#         audio_usage = get_safe_usage("audio")
#         video_usage = get_safe_usage("video")
#         limits = SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS["free"])
#         json_limits = {k: ('unlimited' if v == float('inf') else v) for k, v in limits.items()}
#         return {
#             "tier": tier,
#             "status": status,
#             "usage": {
#                 "clean_transcripts": clean_usage,
#                 "unclean_transcripts": unclean_usage,
#                 "audio_downloads": audio_usage,
#                 "video_downloads": video_usage,
#             },
#             "limits": json_limits,
#             "subscription_id": subscription.payment_id if subscription and hasattr(subscription, 'payment_id') else None,
#             "stripe_customer_id": getattr(current_user, 'stripe_customer_id', None),
#             "current_period_end": expiry_date.isoformat() if expiry_date else None
#         }
#     except Exception as e:
#         logger.error(f"❌ Error getting subscription status: {str(e)}")
#         return {
#             "tier": "free",
#             "status": "inactive", 
#             "usage": {
#                 "clean_transcripts": 0,
#                 "unclean_transcripts": 0,
#                 "audio_downloads": 0,
#                 "video_downloads": 0,
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

# @app.post("/download_transcript/")
# async def download_transcript_corrected(
#     request: TranscriptRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Main endpoint for downloading YouTube transcripts"""
#     video_id = request.youtube_id.strip()
    
#     # Extract video ID from URL if full URL was provided
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
    
#     # Validate video ID format
#     if not video_id or len(video_id) != 11:
#         raise HTTPException(
#             status_code=400,
#             detail="Invalid YouTube video ID. Please provide a valid 11-character video ID or full YouTube URL."
#         )
    
#     logger.info(f"🎯 ENHANCED transcript request for: {video_id}")
#     transcript_type = "clean" if request.clean_transcript else "unclean"
    
#     # Check subscription limits
#     can_download = check_subscription_limit(user.id, transcript_type, db)
#     if not can_download:
#         raise HTTPException(
#             status_code=403,
#             detail=f"You've reached your monthly limit for {transcript_type} transcripts. Please upgrade your plan."
#         )
    
#     try:
#         # Get transcript using the enhanced extraction method
#         transcript_text = get_youtube_transcript_corrected(video_id, clean=request.clean_transcript)
#         if not transcript_text or len(transcript_text.strip()) < 10:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"No transcript content found for video {video_id}."
#             )
        
#         # Record the download in database
#         new_download = TranscriptDownload(
#             user_id=user.id,
#             youtube_id=video_id,
#             transcript_type=transcript_type,
#             created_at=datetime.now()
#         )
#         db.add(new_download)
#         db.commit()
        
#         logger.info(f"🎉 SUCCESS: {user.username} downloaded {len(transcript_text)} chars for {video_id}")
#         return {
#             "transcript": transcript_text,
#             "youtube_id": video_id,
#             "message": "Transcript downloaded successfully"
#         }
#     except HTTPException:
#         raise  # Re-raise HTTP exceptions
#     except Exception as e:
#         db.rollback()
#         logger.error(f"💥 Extraction failed: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to extract transcript for video {video_id}. Error: {str(e)}"
#         )

# @app.get("/health/")
# async def health_check():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "stripe_configured": bool(os.getenv("STRIPE_SECRET_KEY")),
#         "timestamp": datetime.utcnow().isoformat()
#     }

# @app.get("/test_videos")
# async def get_test_videos():
#     """Endpoint with test video IDs for debugging"""
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

# # Main entry point
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)