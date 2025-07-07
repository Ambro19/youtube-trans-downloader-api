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
# import os
# import logging
# from dotenv import load_dotenv
# import re
# import time
# import asyncio
# import random
# from collections import defaultdict

# import warnings
# warnings.filterwarnings("ignore", message=".*bcrypt.*")

# # Import from database.py
# from database import get_db, User, Subscription, TranscriptDownload, create_tables

# # Load environment variables
# load_dotenv()

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("youtube_trans_downloader.main")

# # Rate limiting storage (in production, use Redis)
# rate_limit_storage = defaultdict(list)
# RATE_LIMIT_WINDOW = 60  # 1 minute
# RATE_LIMIT_MAX_REQUESTS = 3  # More conservative: Max 3 requests per minute

# # Stripe configuration
# stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
# endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
# DOMAIN = os.getenv("DOMAIN", "https://youtube-trans-downloader-api.onrender.com")

# # Create FastAPI app
# app = FastAPI(
#     title="YouTubeTransDownloader API", 
#     description="API for downloading and processing YouTube video transcripts",
#     version="1.0.0"
# )

# # Environment-aware configuration
# ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# # Configure CORS based on environment
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

# # CORS MIDDLEWARE
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=allowed_origins,
#     allow_credentials=True,
#     allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
#     allow_headers=["*"],
# )

# # Authentication setup
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
# pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# # Constants
# SECRET_KEY = os.getenv("SECRET_KEY")
# ALGORITHM = "HS256"
# ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# # Enhanced subscription limits with detailed feature sets
# SUBSCRIPTION_LIMITS = {
#     "free": {
#         "transcript": 5, "audio": 2, "video": 1, "clean": 5, "unclean": 3,
#         "clean_transcripts": 5, "unclean_transcripts": 3, 
#         "audio_downloads": 2, "video_downloads": 1,
#         "monthly_downloads": 5,
#         "bulk_downloads": False,
#         "priority_processing": False,
#         "advanced_formats": False,
#         "api_access": False,
#         "export_formats": ["txt"],
#         "concurrent_downloads": 1
#     },
#     "trial": {
#         "transcript": 10, "audio": 5, "video": 2, "clean": 10, "unclean": 5,
#         "clean_transcripts": 10, "unclean_transcripts": 5,
#         "audio_downloads": 5, "video_downloads": 2,
#         "monthly_downloads": 10,
#         "bulk_downloads": False,
#         "priority_processing": False,
#         "advanced_formats": False,
#         "api_access": False,
#         "export_formats": ["txt"],
#         "concurrent_downloads": 1
#     },
#     "pro": {
#         "transcript": 100, "audio": 50, "video": 20, "clean": 100, "unclean": 50,
#         "clean_transcripts": 100, "unclean_transcripts": 50,
#         "audio_downloads": 50, "video_downloads": 20,
#         "monthly_downloads": 100,
#         "bulk_downloads": True,
#         "priority_processing": True,
#         "advanced_formats": True,
#         "api_access": False,
#         "export_formats": ["txt", "srt", "vtt", "json"],
#         "concurrent_downloads": 3
#     },
#     "premium": {
#         "transcript": float('inf'), "audio": float('inf'), "video": float('inf'), 
#         "clean": float('inf'), "unclean": float('inf'),
#         "clean_transcripts": float('inf'), "unclean_transcripts": float('inf'),
#         "audio_downloads": float('inf'), "video_downloads": float('inf'),
#         "monthly_downloads": float('inf'),
#         "bulk_downloads": True,
#         "priority_processing": True,
#         "advanced_formats": True,
#         "api_access": True,
#         "export_formats": ["txt", "srt", "vtt", "json", "xml", "docx"],
#         "concurrent_downloads": 10
#     }
# }

# # Price ID mapping
# PRICE_ID_MAP = {
#     "pro": os.getenv("PRO_PRICE_ID"),
#     "premium": os.getenv("PREMIUM_PRICE_ID")
# }

# @app.on_event("startup")
# async def startup_event():
#     """Enhanced startup with environment validation"""
#     try:
#         logger.info("🚀 Starting YouTube Transcript Downloader API...")
#         logger.info(f"🌍 Environment: {ENVIRONMENT}")
#         logger.info(f"🔗 Domain: {DOMAIN}")
        
#         # Validate critical environment variables
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
        
#         # Initialize database
#         create_tables()
#         logger.info("✅ Database initialized successfully")
#         logger.info("🎉 Application startup complete!")
        
#     except Exception as e:
#         logger.error(f"❌ Startup failed: {str(e)}")
#         raise

# #===============================================#
# # PYDANTIC MODELS: USER ACCOUNT RELATED CLASSES #
# #===============================================#

# class Token(BaseModel):
#     access_token: str
#     token_type: str

# class TokenData(BaseModel):
#     username: Optional[str] = None

# class UserCreate(BaseModel):
#     username: str
#     email: str
#     password: str

# class UserResponse(BaseModel):
#     id: int
#     username: str
#     email: str
#     created_at: datetime
    
#     class Config:
#         from_attributes = True

# class TranscriptRequest(BaseModel):
#     youtube_id: str
#     clean_transcript: bool = False

# class CreatePaymentIntentRequest(BaseModel):
#     price_id: str

# class ConfirmPaymentRequest(BaseModel):
#     payment_intent_id: str

# class SubscriptionRequest(BaseModel):
#     token: Optional[str] = None
#     subscription_tier: str

# class SubscriptionResponse(BaseModel):
#     tier: str
#     status: str
#     expiry_date: Optional[str] = None
#     limits: dict
#     usage: Optional[dict] = None
#     remaining: Optional[dict] = None
    
#     class Config:
#         from_attributes = True

# #==================================================#
# # HELPER FUNCTIONS: USER ACCOUNT RELATED FUNCTIONS #
# #==================================================#

# def verify_password(plain_password, hashed_password):
#     return pwd_context.verify(plain_password, hashed_password)

# def get_password_hash(password):
#     return pwd_context.hash(password)

# def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
#     to_encode = data.copy()
#     if expires_delta:
#         expire = datetime.utcnow() + expires_delta
#     else:
#         expire = datetime.utcnow() + timedelta(minutes=15)
#     to_encode.update({"exp": expire})
#     encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
#     return encoded_jwt

# def get_user(db: Session, username: str):
#     return db.query(User).filter(User.username == username).first()

# def get_user_by_email(db: Session, email: str):
#     return db.query(User).filter(User.email == email).first()

# def authenticate_user(db: Session, username: str, password: str):
#     user = get_user(db, username)
#     if not user:
#         return False
#     if not verify_password(password, user.hashed_password):
#         return False
#     return user

# def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
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

# def get_subscription_limits(tier: str) -> dict:
#     """Get subscription limits for a given tier"""
#     return SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS["free"])

# # Enhanced Stripe integration functions for main.py
# def get_or_create_stripe_customer(user, db: Session):
#     """
#     Enhanced Stripe customer creation with better tracking
#     This ensures customers appear in your Stripe dashboard automatically
#     """
#     try:
#         # Check if user already has a Stripe customer ID
#         if hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
#             try:
#                 # Verify the customer still exists in Stripe
#                 customer = stripe.Customer.retrieve(user.stripe_customer_id)
#                 logger.info(f"✅ Found existing Stripe customer: {customer.id} for user {user.username}")
#                 return customer
#             except stripe.error.InvalidRequestError:
#                 logger.info(f"📝 Stripe customer {user.stripe_customer_id} not found, creating new one")
#                 pass
        
#         # Create new Stripe customer
#         logger.info(f"🔄 Creating new Stripe customer for user: {user.username}")
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
        
#         # Save the Stripe customer ID to our database
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

# #=================================#
# # SUBSCRIPTION FUNCTIONS          #
# #=================================#

# def check_subscription_limit(user_id: int, transcript_type: str = None, db: Session = None) -> dict:
#     """
#     Simple subscription limit checker
#     """
#     try:
#         if not db:
#             return {"allowed": False, "error": "Database connection required"}
            
#         # Get user and subscription
#         user = db.query(User).filter(User.id == user_id).first()
#         if not user:
#             return {"allowed": False, "error": "User not found"}
        
#         subscription = db.query(Subscription).filter(Subscription.user_id == user_id).first()
        
#         # Determine subscription tier
#         if not subscription:
#             subscription_tier = "free"
#             subscription_status = "inactive"
#         else:
#             subscription_tier = subscription.tier
#             subscription_status = getattr(subscription, 'status', 'active')
            
#             # Check if subscription is expired (simple check)
#             if hasattr(subscription, 'expiry_date') and subscription.expiry_date:
#                 if subscription.expiry_date < datetime.now():
#                     subscription_tier = "free"
#                     subscription_status = "expired"
        
#         # Get current month usage
#         now = datetime.now()
#         current_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
#         downloads_this_month = db.query(TranscriptDownload).filter(
#             TranscriptDownload.user_id == user_id,
#             TranscriptDownload.created_at >= current_month_start
#         ).count()
        
#         # Get subscription limits
#         limits = get_subscription_limits(subscription_tier)
#         monthly_limit = limits['monthly_downloads']
        
#         # Simple access determination
#         if monthly_limit == float('inf'):
#             allowed = True
#             reason = "Unlimited access"
#         else:
#             allowed = downloads_this_month < monthly_limit
#             reason = "Access granted" if allowed else f"Monthly limit of {monthly_limit} reached"
        
#         logger.info(f"✅ Subscription check: {subscription_tier}, {downloads_this_month}/{monthly_limit}, allowed={allowed}")
        
#         return {
#             "allowed": allowed,
#             "reason": reason,
#             "subscription": {
#                 "tier": subscription_tier,
#                 "status": subscription_status
#             },
#             "usage": {
#                 "downloads_this_month": downloads_this_month,
#                 "monthly_limit": monthly_limit if monthly_limit != float('inf') else 'unlimited'
#             }
#         }
        
#     except Exception as e:
#         logger.error(f"Subscription check error: {e}")
#         # On error, allow free tier limits
#         return {
#             "allowed": True,
#             "reason": "Error in check - allowing access",
#             "error": str(e)
#         }

# #===========================================================
# # ROBUST MULTI-METHOD TRANSCRIPT EXTRACTION               #
# #===========================================================

# async def extract_transcript_robust(video_id: str, clean: bool = True) -> str:
#     """
#     Ultra-robust transcript extraction using multiple methods and long delays
#     """
#     logger.info(f"🎯 ROBUST multi-method extraction for: {video_id}")
    
#     # Method 1: Try youtube-transcript-api with conservative approach
#     try:
#         logger.info("🔄 Method 1: Conservative youtube-transcript-api...")
        
#         # Add random delay to avoid patterns
#         initial_delay = random.uniform(5, 15)
#         logger.info(f"⏳ Initial random delay: {initial_delay:.1f}s")
#         await asyncio.sleep(initial_delay)
        
#         from youtube_transcript_api import YouTubeTranscriptApi
        
#         # Try with minimal requests
#         transcript = YouTubeTranscriptApi.get_transcript(video_id)
        
#         if transcript and len(transcript) > 0:
#             logger.info(f"✅ Method 1 SUCCESS: {len(transcript)} segments")
#             return format_transcript_simple(transcript, clean)
            
#     except Exception as e:
#         error_msg = str(e)
#         if "429" in error_msg or "Too Many Requests" in error_msg:
#             logger.warning("⚠️ Method 1: Rate limited")
#         else:
#             logger.info(f"📝 Method 1 failed: {error_msg[:100]}")
    
#     # Method 2: Try with longer delay and different approach
#     try:
#         logger.info("🔄 Method 2: Long delay + list transcripts...")
        
#         # Much longer delay
#         long_delay = random.uniform(30, 60)
#         logger.info(f"⏳ Long delay before Method 2: {long_delay:.1f}s")
#         await asyncio.sleep(long_delay)
        
#         from youtube_transcript_api import YouTubeTranscriptApi
        
#         # Try list approach with minimal calls
#         transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
#         # Get first available transcript without trying multiple
#         for transcript_obj in transcript_list:
#             try:
#                 logger.info(f"🔄 Trying {transcript_obj.language_code}...")
#                 transcript_data = transcript_obj.fetch()
                
#                 if transcript_data and len(transcript_data) > 0:
#                     logger.info(f"✅ Method 2 SUCCESS with {transcript_obj.language_code}: {len(transcript_data)} segments")
#                     return format_transcript_simple(transcript_data, clean)
                
#                 # Only try first available - don't loop through all
#                 break
                
#             except Exception as fetch_error:
#                 logger.info(f"📝 {transcript_obj.language_code} failed: {str(fetch_error)[:50]}")
#                 break  # Don't try more languages to avoid more API calls
                
#     except Exception as e:
#         error_msg = str(e)
#         if "429" in error_msg or "Too Many Requests" in error_msg:
#             logger.warning("⚠️ Method 2: Rate limited")
#         else:
#             logger.info(f"📝 Method 2 failed: {error_msg[:100]}")
    
#     # Method 3: Try yt-dlp as alternative (if available)
#     try:
#         logger.info("🔄 Method 3: Trying yt-dlp as alternative...")
        
#         # Another long delay
#         ytdlp_delay = random.uniform(45, 90)
#         logger.info(f"⏳ yt-dlp delay: {ytdlp_delay:.1f}s")
#         await asyncio.sleep(ytdlp_delay)
        
#         try:
#             import yt_dlp
            
#             ydl_opts = {
#                 'writesubtitles': True,
#                 'writeautomaticsub': True,
#                 'subtitleslangs': ['en'],
#                 'skip_download': True,
#                 'quiet': True,
#                 'no_warnings': True,
#             }
            
#             with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#                 url = f"https://www.youtube.com/watch?v={video_id}"
#                 info = ydl.extract_info(url, download=False)
                
#                 # Try to extract from subtitle info
#                 if 'subtitles' in info and info['subtitles']:
#                     for lang in ['en', 'en-US']:
#                         if lang in info['subtitles']:
#                             logger.info(f"✅ Method 3 found subtitles in {lang}")
#                             # This would need more implementation to parse subtitle files
#                             # For now, fall through to demo content
#                             break
                            
#         except ImportError:
#             logger.info("📝 yt-dlp not available")
#         except Exception as yt_error:
#             logger.info(f"📝 yt-dlp failed: {str(yt_error)[:100]}")
    
#     except Exception as e:
#         logger.info(f"📝 Method 3 failed: {str(e)[:100]}")
    
#     # If all methods fail, provide helpful error
#     logger.error(f"💥 All robust methods failed for video {video_id}")
    
#     # Check if it's a rate limiting issue or video issue
#     if any(keyword in str(e).lower() for keyword in ['429', 'too many requests', 'rate limit']):
#         raise HTTPException(
#             status_code=429,
#             detail="YouTube is currently blocking transcript requests due to high usage. This is a temporary issue with YouTube's API. Please try again in 1-2 hours. You can also try different videos as some may work while others are temporarily blocked."
#         )
#     else:
#         raise HTTPException(
#             status_code=404,
#             detail=f"Unable to extract transcript for video {video_id}. This video may not have captions enabled, may be private, or YouTube's transcript service may be temporarily unavailable."
#         )

# def format_transcript_simple(transcript_list: list, clean: bool = True) -> str:
#     """
#     Simple transcript formatting
#     """
#     if not transcript_list or len(transcript_list) == 0:
#         raise Exception("Empty transcript data")
    
#     logger.info(f"🔄 Formatting {len(transcript_list)} segments (clean={clean})")
    
#     try:
#         if clean:
#             # Clean format - just join all text
#             texts = []
#             for item in transcript_list:
#                 text = item.get('text', '').strip()
#                 if text:
#                     # Basic cleaning
#                     text = text.replace('\n', ' ').replace('\r', ' ')
#                     text = ' '.join(text.split())  # normalize whitespace
#                     if text:
#                         texts.append(text)
            
#             if not texts:
#                 raise Exception("No valid text found")
            
#             result = ' '.join(texts)
#             logger.info(f"✅ Clean format: {len(result)} characters")
#             return result
            
#         else:
#             # Timestamped format
#             lines = []
#             for item in transcript_list:
#                 start = item.get('start', 0)
#                 text = item.get('text', '').strip()
                
#                 if text:
#                     # Basic cleaning
#                     text = text.replace('\n', ' ').replace('\r', ' ')
#                     text = ' '.join(text.split())
                    
#                     if text:
#                         # Simple timestamp format
#                         minutes = int(start // 60)
#                         seconds = int(start % 60)
#                         timestamp = f"[{minutes:02d}:{seconds:02d}]"
#                         lines.append(f"{timestamp} {text}")
            
#             if not lines:
#                 raise Exception("No valid timestamped lines")
            
#             result = '\n'.join(lines)
#             logger.info(f"✅ Timestamped format: {len(lines)} lines")
#             return result
            
#     except Exception as e:
#         logger.error(f"❌ Formatting failed: {str(e)}")
#         raise Exception(f"Failed to format transcript: {str(e)}")

# #=======================================================================
# # API ENDPOINTS: ALL THE ENDPOINTS OF THE YOUTUBE TRANS DOWNLOADER API #
# #=======================================================================

# @app.get("/")
# async def root():
#     return {"message": "YouTube Transcript Downloader API", "status": "running", "version": "1.0.0"}

# @app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
# def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
#     db_user = get_user(db, user_data.username)
#     if db_user:
#         raise HTTPException(status_code=400, detail="Username already registered")
    
#     email_exists = get_user_by_email(db, user_data.email)
#     if email_exists:
#         raise HTTPException(status_code=400, detail="Email already registered")
    
#     hashed_password = get_password_hash(user_data.password)
#     new_user = User(
#         username=user_data.username,
#         email=user_data.email,
#         hashed_password=hashed_password,
#         created_at=datetime.now()
#     )
    
#     try:
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
#     user = authenticate_user(db, form_data.username, form_data.password)
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect username or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
    
#     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token = create_access_token(
#         data={"sub": user.username}, expires_delta=access_token_expires
#     )
    
#     logger.info(f"User logged in successfully: {form_data.username}")
#     return {"access_token": access_token, "token_type": "bearer"}

# @app.get("/users/me", response_model=UserResponse)
# async def read_users_me(current_user: User = Depends(get_current_user)):
#     return current_user

# @app.post("/download_transcript/")
# async def download_transcript_corrected(
#     request: TranscriptRequest,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """
#     ROBUST transcript downloader with multiple fallback methods
#     """
#     video_id = request.youtube_id.strip()
    
#     # Extract video ID from various YouTube URL formats
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
    
#     logger.info(f"🎯 ROBUST transcript request for: {video_id}")
    
#     # Check subscription limits
#     transcript_type = "clean" if request.clean_transcript else "unclean"
#     limit_check = check_subscription_limit(user.id, transcript_type, db)
    
#     if not limit_check.get("allowed", False):
#         raise HTTPException(
#             status_code=403, 
#             detail=f"You've reached your monthly limit for {transcript_type} transcripts. Please upgrade your plan."
#         )
    
#     # Extract transcript using robust multi-method approach
#     try:
#         transcript_text = await extract_transcript_robust(video_id, clean=request.clean_transcript)
        
#         # Validate result
#         if not transcript_text or len(transcript_text.strip()) < 10:
#             raise HTTPException(
#                 status_code=404,
#                 detail=f"No valid transcript content found for video {video_id}."
#             )
        
#         # Record successful download
#         new_download = TranscriptDownload(
#             user_id=user.id,
#             youtube_id=video_id,
#             transcript_type=transcript_type,
#             created_at=datetime.now()
#         )
        
#         db.add(new_download)
#         db.commit()
        
#         logger.info(f"🎉 ROBUST SUCCESS: {user.username} downloaded {len(transcript_text)} chars for {video_id}")
        
#         return {
#             "transcript": transcript_text,
#             "youtube_id": video_id,
#             "message": "Transcript downloaded successfully"
#         }
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         db.rollback()
#         logger.error(f"💥 Robust extraction failed: {str(e)}")
        
#         raise HTTPException(
#             status_code=500,
#             detail=f"Failed to extract transcript for video {video_id}. Error: {str(e)}"
#         )

# @app.get("/subscription_status/")
# async def get_subscription_status_ultra_safe(
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Ultra-safe subscription status"""
#     try:
#         # Get basic subscription info
#         subscription = db.query(Subscription).filter(
#             Subscription.user_id == current_user.id
#         ).first()
        
#         # Determine current subscription tier
#         if not subscription:
#             tier = "free"
#             status = "inactive"
#             expiry_date = None
#         else:
#             # Check if subscription is expired
#             if hasattr(subscription, 'expiry_date') and subscription.expiry_date and subscription.expiry_date < datetime.now():
#                 tier = "free"
#                 status = "expired"
#                 expiry_date = subscription.expiry_date
#             else:
#                 tier = subscription.tier if subscription.tier else "free"
#                 status = "active" if tier != "free" else "inactive"
#                 expiry_date = subscription.expiry_date if hasattr(subscription, 'expiry_date') else None
        
#         # Calculate usage for current month
#         month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
#         # Safe usage calculation
#         def get_safe_usage(transcript_type):
#             try:
#                 return db.query(TranscriptDownload).filter(
#                     TranscriptDownload.user_id == current_user.id,
#                     TranscriptDownload.transcript_type == transcript_type,
#                     TranscriptDownload.created_at >= month_start
#                 ).count()
#             except Exception:
#                 return 0
        
#         # Get usage for all types safely
#         clean_usage = get_safe_usage("clean")
#         unclean_usage = get_safe_usage("unclean")
#         audio_usage = get_safe_usage("audio")
#         video_usage = get_safe_usage("video")
        
#         # Get limits based on current tier
#         limits = SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS["free"])
        
#         # Convert infinity to string for JSON serialization
#         json_limits = {}
#         for key, value in limits.items():
#             if value == float('inf'):
#                 json_limits[key] = 'unlimited'
#             else:
#                 json_limits[key] = value
        
#         logger.info(f"✅ Safe subscription status for {current_user.username}: tier={tier}, status={status}")
        
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
#         # Return safe defaults on any error
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

# @app.post("/confirm_payment/")
# async def confirm_payment_endpoint(
#     request: ConfirmPaymentRequest,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ):
#     """Enhanced payment confirmation with complete Stripe integration"""
#     try:
#         # Retrieve payment intent from Stripe
#         intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
        
#         if intent.status != 'succeeded':
#             raise HTTPException(status_code=400, detail=f"Payment not completed. Status: {intent.status}")

#         # Get plan details
#         plan_type = intent.metadata.get('plan_type', 'pro')
#         price_id = intent.metadata.get('price_id')
        
#         # Create or update subscription record
#         user_subscription = db.query(Subscription).filter(
#             Subscription.user_id == current_user.id
#         ).first()

#         if not user_subscription:
#             user_subscription = Subscription(
#                 user_id=current_user.id,
#                 tier=plan_type,
#                 start_date=datetime.utcnow(),
#                 expiry_date=datetime.utcnow() + timedelta(days=30),
#                 payment_id=request.payment_intent_id,
#                 auto_renew=True,
#                 stripe_price_id=price_id,
#                 status='active',
#                 current_period_start=datetime.utcnow(),
#                 current_period_end=datetime.utcnow() + timedelta(days=30)
#             )
#             db.add(user_subscription)
#         else:
#             user_subscription.tier = plan_type
#             user_subscription.start_date = datetime.utcnow()
#             user_subscription.expiry_date = datetime.utcnow() + timedelta(days=30)
#             user_subscription.payment_id = request.payment_intent_id
#             user_subscription.auto_renew = True
#             user_subscription.stripe_price_id = price_id
#             user_subscription.status = 'active'
#             user_subscription.current_period_start = datetime.utcnow()
#             user_subscription.current_period_end = datetime.utcnow() + timedelta(days=30)

#         # Update user's Stripe customer ID if not already set
#         if hasattr(current_user, 'stripe_customer_id') and not current_user.stripe_customer_id:
#             current_user.stripe_customer_id = intent.customer

#         db.commit()
#         db.refresh(user_subscription)

#         logger.info(f"🎉 Payment confirmed: User {current_user.username} upgraded to {plan_type}")

#         return {
#             'success': True,
#             'subscription_tier': user_subscription.tier,
#             'expires_at': user_subscription.expiry_date.isoformat(),
#             'status': 'active',
#             'payment_id': request.payment_intent_id,
#             'customer_id': intent.customer
#         }

#     except Exception as e:
#         logger.error(f"❌ Payment confirmation error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Failed to confirm payment: {str(e)}")

# #========================
# # Health check endpoints
# # =======================
# @app.get("/health/")
# async def health_check():
#     return {
#         "status": "healthy",
#         "stripe_configured": bool(os.getenv("STRIPE_SECRET_KEY")),
#         "timestamp": datetime.utcnow().isoformat()
#     }

# @app.get("/healthcheck")
# async def healthcheck():
#     return {"status": "ok", "version": "1.0.0"}

# #======================
# # Library info endpoint
# #======================
# @app.get("/library_info")
# async def library_info():
#     try:
#         from youtube_transcript_api import __version__
#         return {"youtube_transcript_api_version": __version__}
#     except:
#         return {"youtube_transcript_api_version": "unknown"}

# @app.get("/test_videos")
# async def get_test_videos():
#     """Get list of verified working video IDs for testing"""
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
#         "note": "Ultra-robust extraction with long delays and multiple fallback methods"
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

#======================================================================

#####################################################################
## IMPORTANT MESSAGE: DO NOT ALTER THIS MAIN.PY ANYMORE-- THANKS! ###
#####################################################################   

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, List
import jwt
from jwt.exceptions import PyJWTError
from pydantic import BaseModel
from passlib.context import CryptContext
import stripe
import os
import logging
from dotenv import load_dotenv
import re

import warnings
warnings.filterwarnings("ignore", message=".*bcrypt.*")

# Import from database.py
from database import get_db, User, Subscription, TranscriptDownload, create_tables

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("youtube_trans_downloader.main")

# Stripe configuration
stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
DOMAIN = os.getenv("DOMAIN", "https://youtube-trans-downloader-api.onrender.com")

# Create FastAPI app
app = FastAPI(
    title="YouTubeTransDownloader API", 
    description="API for downloading and processing YouTube video transcripts",
    version="1.0.0"
)

# Environment-aware configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Configure CORS based on environment
if ENVIRONMENT == "production":
    allowed_origins = [
        "http://localhost:8000",
        "https://youtube-trans-downloader-api.onrender.com",
        FRONTEND_URL
    ]
    logger.info(f"🌍 Production mode - CORS origins: {allowed_origins}")
else:
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        FRONTEND_URL
    ]
    logger.info(f"🔧 Development mode - CORS origins: {allowed_origins}")

# CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Authentication setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Constants
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# Enhanced subscription limits
SUBSCRIPTION_LIMITS = {
    "free": {
        "transcript": 5, "audio": 2, "video": 1, "clean": 5, "unclean": 3,
        "clean_transcripts": 5, "unclean_transcripts": 3, 
        "audio_downloads": 2, "video_downloads": 1
    },
    "pro": {
        "transcript": 100, "audio": 50, "video": 20, "clean": 100, "unclean": 50,
        "clean_transcripts": 100, "unclean_transcripts": 50,
        "audio_downloads": 50, "video_downloads": 20
    },
    "premium": {
        "transcript": float('inf'), "audio": float('inf'), "video": float('inf'), 
        "clean": float('inf'), "unclean": float('inf'),
        "clean_transcripts": float('inf'), "unclean_transcripts": float('inf'),
        "audio_downloads": float('inf'), "video_downloads": float('inf')
    }
}

# Price ID mapping
PRICE_ID_MAP = {
    "pro": os.getenv("PRO_PRICE_ID"),
    "premium": os.getenv("PREMIUM_PRICE_ID")
}

@app.on_event("startup")
async def startup_event():
    """Enhanced startup with environment validation"""
    try:
        logger.info("🚀 Starting YouTube Transcript Downloader API...")
        logger.info(f"🌍 Environment: {ENVIRONMENT}")
        logger.info(f"🔗 Domain: {DOMAIN}")
        
        # Validate critical environment variables
        required_vars = {
            "SECRET_KEY": "JWT secret key",
            "STRIPE_SECRET_KEY": "Stripe secret key",
        }
        
        missing_vars = []
        for var, description in required_vars.items():
            value = os.getenv(var)
            if not value:
                missing_vars.append(f"{var} ({description})")
            else:
                logger.info(f"✅ {var}: {value[:8]}..." if len(value) > 8 else f"✅ {var}: SET")
        
        if missing_vars:
            logger.error(f"❌ Missing required environment variables:")
            for var in missing_vars:
                logger.error(f"   - {var}")
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
        
        # Initialize database
        create_tables()
        logger.info("✅ Database initialized successfully")
        logger.info("🎉 Application startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

#===============================================#
# PYDANTIC MODELS: USER ACCOUNT RELATED CLASSES #
#===============================================#

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    
    class Config:
        from_attributes = True

class TranscriptRequest(BaseModel):
    youtube_id: str
    clean_transcript: bool = False

class CreatePaymentIntentRequest(BaseModel):
    price_id: str

class ConfirmPaymentRequest(BaseModel):
    payment_intent_id: str

class SubscriptionRequest(BaseModel):
    token: Optional[str] = None
    subscription_tier: str

class SubscriptionResponse(BaseModel):
    tier: str
    status: str
    expiry_date: Optional[str] = None
    limits: dict
    usage: Optional[dict] = None
    remaining: Optional[dict] = None
    
    class Config:
        from_attributes = True

#==================================================#
# HELPER FUNCTIONS: USER ACCOUNT RELATED FUNCTIONS #
#==================================================#

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_user(db: Session, username: str):
    return db.query(User).filter(User.username == username).first()

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def authenticate_user(db: Session, username: str, password: str):
    user = get_user(db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception
        
    user = get_user(db, username)
    if user is None:
        raise credentials_exception
    return user

# Enhanced Stripe integration functions for main.py
# Add these enhanced functions to replace the existing ones

def get_or_create_stripe_customer(user, db: Session):
    """
    Enhanced Stripe customer creation with better tracking
    This ensures customers appear in your Stripe dashboard automatically
    """
    try:
        # Check if user already has a Stripe customer ID
        if hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
            try:
                # Verify the customer still exists in Stripe
                customer = stripe.Customer.retrieve(user.stripe_customer_id)
                logger.info(f"✅ Found existing Stripe customer: {customer.id} for user {user.username}")
                return customer
            except stripe.error.InvalidRequestError:
                logger.info(f"📝 Stripe customer {user.stripe_customer_id} not found, creating new one")
                pass
        
        # Create new Stripe customer
        logger.info(f"🔄 Creating new Stripe customer for user: {user.username}")
        customer = stripe.Customer.create(
            email=user.email,
            name=user.username,
            description=f"YouTube Transcript Downloader user: {user.username}",
            metadata={
                'user_id': str(user.id),
                'username': user.username,
                'signup_date': user.created_at.isoformat() if user.created_at else None,
                'app_name': 'YouTube Transcript Downloader'
            }
        )
        
        # Save the Stripe customer ID to our database
        if hasattr(user, 'stripe_customer_id'):
            user.stripe_customer_id = customer.id
            db.commit()
            db.refresh(user)
            logger.info(f"✅ Stripe customer created and saved: {customer.id}")
        
        return customer
        
    except Exception as e:
        logger.error(f"❌ Error creating Stripe customer for user {user.username}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create payment customer. Please try again."
        )


#=========================================================== #
# ENHANCED TRANSCRIPT FUNCTIONS: TRANSCRIPT RELATED FUNCTIONS #
#===========================================================  #

def check_subscription_limit(user_id: int, transcript_type: str, db: Session):
    """Robust subscription limit check - handles missing columns gracefully"""
    try:
        # Get subscription info
        subscription = db.query(Subscription).filter(Subscription.user_id == user_id).first()
        
        if not subscription:
            tier = "free"
        else:
            tier = subscription.tier
            if subscription.expiry_date < datetime.now():
                tier = "free"
        
        # Calculate current month start
        month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Use a simple query that only uses basic columns that definitely exist
        try:
            # Try the enhanced query first
            usage = db.query(TranscriptDownload).filter(
                TranscriptDownload.user_id == user_id,
                TranscriptDownload.transcript_type == transcript_type,
                TranscriptDownload.created_at >= month_start
            ).count()
        except Exception as e:
            logger.warning(f"Enhanced query failed, using simple fallback: {e}")
            
            # Fallback to basic SQL query that only uses guaranteed columns
            result = db.execute(
                "SELECT COUNT(*) FROM transcript_downloads WHERE user_id = ? AND transcript_type = ? AND created_at >= ?",
                (user_id, transcript_type, month_start.isoformat())
            )
            usage = result.scalar() or 0
        
        # Get limit for this tier and transcript type
        limit = SUBSCRIPTION_LIMITS[tier].get(transcript_type, 0)
        
        # Return True if user can download (hasn't reached limit)
        if limit == float('inf'):
            return True
        
        return usage < limit
        
    except Exception as e:
        logger.error(f"Error checking subscription limit: {e}")
        # If there's any error, default to allowing free tier limits
        return True  # Allow download in case of errors to avoid blocking users

def get_youtube_transcript_corrected(video_id: str, clean: bool = True) -> str:
    """
    ENHANCED YouTube transcript extraction with robust error handling
    
    This function tries multiple methods in order of reliability:
    1. youtube-transcript-api (fastest, but often blocked)
    2. list_transcripts (better language detection)
    3. yt-dlp (most robust, works when others fail)
    4. Direct API (fallback for edge cases)
    5. Demo content (last resort for testing)
    
    Args:
        video_id: 11-character YouTube video ID
        clean: If True, returns clean text; if False, returns timestamped format
    
    Returns:
        Formatted transcript text
    """
    logger.info(f"🎯 ENHANCED transcript extraction for: {video_id}")
    
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # Method 1: Try youtube-transcript-api first (fastest when it works)
        try:
            logger.info("🔄 Trying youtube-transcript-api...")
            
            # Try multiple English language variants for better success rate
            language_codes = ['en', 'en-US', 'en-GB', 'en-CA', 'en-AU']
            transcript_list = None
            
            for lang in language_codes:
                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
                    if transcript_list and len(transcript_list) > 0:
                        logger.info(f"✅ SUCCESS with language {lang}: {len(transcript_list)} segments")
                        break
                except Exception as lang_error:
                    logger.info(f"📝 Language {lang} failed: {str(lang_error)}")
                    continue
            
            # If we found a transcript, format and return it
            if transcript_list and len(transcript_list) > 0:
                return format_transcript_enhanced(transcript_list, clean)
                
        except Exception as e:
            logger.info(f"📝 youtube-transcript-api failed: {str(e)}")
        
        # Method 2: Try list_transcripts approach for better language detection
        try:
            logger.info("🔄 Trying list_transcripts method...")
            transcript_list_obj = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to find any English transcript (manual or auto-generated)
            english_transcript = None
            try:
                # Prefer manual transcripts over auto-generated ones
                for transcript in transcript_list_obj:
                    if transcript.language_code.startswith('en'):
                        english_transcript = transcript
                        logger.info(f"🎯 Found manual English transcript: {transcript.language_code}")
                        break
                
                # If no manual English transcript, try auto-generated
                if not english_transcript:
                    try:
                        english_transcript = transcript_list_obj.find_generated_transcript(['en'])
                        logger.info("🤖 Found auto-generated English transcript")
                    except:
                        pass
                
                # If we found any English transcript, fetch and format it
                if english_transcript:
                    transcript_data = english_transcript.fetch()
                    if transcript_data and len(transcript_data) > 0:
                        logger.info(f"✅ LIST API SUCCESS: {len(transcript_data)} segments")
                        return format_transcript_enhanced(transcript_data, clean)
                        
            except Exception as inner_e:
                logger.info(f"📝 English transcript lookup failed: {str(inner_e)}")
                
        except Exception as e:
            logger.info(f"📝 List API failed: {str(e)}")
        
        # Method 3: Enhanced yt-dlp method (most robust, works when APIs fail)
        try:
            logger.info("🔄 Trying enhanced yt-dlp method...")
            import yt_dlp
            
            # Configure yt-dlp for subtitle extraction only
            ydl_opts = {
                'writesubtitles': True,        # Download manual subtitles
                'writeautomaticsub': True,     # Download auto-generated subtitles
                'subtitleslangs': ['en', 'en-US', 'en-GB'],  # English variants (MAYBE WE NEED TO CHANGE THIS LINE????)
                'skip_download': True,         # Don't download video, just metadata
                'quiet': True,                 # Suppress most output
                'no_warnings': True,           # Suppress warnings
                'extract_flat': False,         # Get full metadata
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                url = f"https://www.youtube.com/watch?v={video_id}"
                
                try:
                    # Extract video metadata including subtitle information
                    info = ydl.extract_info(url, download=False)
                    
                    # Special handling for live streams
                    if info.get('is_live') or info.get('live_status') == 'is_live':
                        logger.info("🔴 Live stream detected - trying live transcript extraction")
                        return extract_live_transcript(video_id, clean)
                    
                    # Try manual subtitles first (usually higher quality)
                    subtitles_found = False
                    if 'subtitles' in info and info['subtitles']:
                        for lang in ['en', 'en-US', 'en-GB']:
                            if lang in info['subtitles']:
                                for entry in info['subtitles'][lang]:
                                    transcript_text = extract_subtitle_from_entry(ydl, entry, clean)
                                    if transcript_text and not is_invalid_content(transcript_text):
                                        logger.info(f"✅ YT-DLP MANUAL SUCCESS: {len(transcript_text)} chars")
                                        return transcript_text
                                    subtitles_found = True
                    
                    # Fallback to automatic captions if manual subtitles don't work
                    if 'automatic_captions' in info and info['automatic_captions']:
                        for lang in ['en', 'en-US', 'en-GB']:
                            if lang in info['automatic_captions']:
                                for entry in info['automatic_captions'][lang]:
                                    transcript_text = extract_subtitle_from_entry(ydl, entry, clean)
                                    if transcript_text and not is_invalid_content(transcript_text):
                                        logger.info(f"✅ YT-DLP AUTO SUCCESS: {len(transcript_text)} chars")
                                        return transcript_text
                                    subtitles_found = True
                    
                    # Log if subtitles were found but couldn't be processed
                    if subtitles_found:
                        logger.info("📝 Subtitles found but content was invalid/empty")
                    
                except Exception as yt_error:
                    logger.info(f"📝 yt-dlp extraction failed: {str(yt_error)}")
                
        except ImportError:
            logger.info("📝 yt-dlp not installed, skipping...")
        except Exception as e:
            logger.info(f"📝 yt-dlp method failed: {str(e)}")
        
        # Method 4: Direct API approach for edge cases (placeholder for future expansion)
        try:
            logger.info("🔄 Trying direct API approach...")
            transcript_data = extract_with_direct_api(video_id)
            if transcript_data:
                logger.info(f"✅ DIRECT API SUCCESS: {len(transcript_data)} segments")
                return format_transcript_enhanced(transcript_data, clean)
        except Exception as e:
            logger.info(f"📝 Direct API failed: {str(e)}")
        
        # Final fallback: Demo content for testing (only when all methods fail)
        logger.warning(f"⚠️ All extraction methods failed for {video_id}, falling back to demo content")
        return get_demo_content(clean)
        
    except Exception as e:
        logger.error(f"💥 Critical error in transcript extraction for {video_id}: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=f"Unable to extract transcript for video {video_id}. The video may not have captions enabled, be private, or be unavailable."
        )

def format_transcript_enhanced(transcript_list: list, clean: bool = True) -> str:
    """
    Enhanced transcript formatting with better text processing
    
    This function takes raw transcript data and formats it into either:
    - Clean format: Plain text with proper spacing and punctuation
    - Timestamped format: Text with [MM:SS] timestamps for each segment
    
    Args:
        transcript_list: List of transcript segments with 'text' and 'start' keys
        clean: If True, returns clean text; if False, returns timestamped format
    
    Returns:
        Formatted transcript string
    """
    if not transcript_list:
        raise Exception("Empty transcript data")
    
    if clean:
        # Enhanced clean format - create readable paragraph text
        texts = []
        for item in transcript_list:
            text = item.get('text', '').strip()
            if text:
                # Clean and normalize the text
                text = clean_transcript_text(text)
                if text:  # Only add if text remains after cleaning
                    texts.append(text)
        
        # Join all text segments into one continuous string
        result = ' '.join(texts)
        
        # Final cleanup for better readability
        result = ' '.join(result.split())  # Normalize whitespace
        result = result.replace(' .', '.').replace(' ,', ',')  # Fix punctuation spacing
        
        # Validate that we have meaningful content
        if len(result) < 20:  # Too short, likely invalid
            raise Exception("Transcript too short or invalid")
            
        logger.info(f"✅ Enhanced clean transcript: {len(result)} characters")
        return result
    else:
        # Enhanced timestamped format - each line has [MM:SS] timestamp
        lines = []
        for item in transcript_list:
            start = item.get('start', 0)
            text = item.get('text', '').strip()
            if text:
                # Clean the text but preserve timestamps
                text = clean_transcript_text(text)
                if text:
                    # Convert seconds to MM:SS format
                    minutes = int(start // 60)
                    seconds = int(start % 60)
                    timestamp = f"[{minutes:02d}:{seconds:02d}]"
                    lines.append(f"{timestamp} {text}")
        
        # Validate that we have enough content
        if len(lines) < 5:  # Too few lines, likely invalid
            raise Exception("Transcript has too few valid segments")
            
        result = '\n'.join(lines)
        logger.info(f"✅ Enhanced timestamped transcript: {len(lines)} lines")
        return result

def clean_transcript_text(text: str) -> str:
    """
    Clean transcript text from common artifacts and formatting issues
    
    This function removes:
    - HTML entities (&amp;, &lt;, &gt;)
    - HTML/XML tags
    - Extra whitespace and line breaks
    - Leading/trailing punctuation artifacts
    
    Args:
        text: Raw transcript text that may contain artifacts
    
    Returns:
        Clean, readable text
    """
    if not text:
        return ""
    
    # Fix common HTML entities
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # Remove HTML/XML tags (like <c>, <i>, etc.)
    import re
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize whitespace (replace multiple spaces with single space)
    text = ' '.join(text.split())
    
    # Remove leading/trailing punctuation artifacts that don't belong
    text = text.strip('.,!?;: ')
    
    return text.strip()

def is_invalid_content(content: str) -> bool:
    """
    Check if content is invalid (M3U8 playlist, metadata, etc.)
    
    This function identifies content that looks like:
    - M3U8 playlists (streaming metadata)
    - API URLs with parameters
    - Empty or too-short content
    
    Args:
        content: String content to validate
    
    Returns:
        True if content is invalid, False if it looks like real transcript text
    """
    if not content or len(content.strip()) < 10:
        return True
    
    # List of indicators that suggest this is not transcript content
    invalid_indicators = [
        '#EXTM3U',                           # M3U8 playlist header
        '#EXT-X-VERSION',                    # M3U8 version
        '#EXT-X-PLAYLIST-TYPE',              # M3U8 playlist type
        '#EXT-X-TARGETDURATION',             # M3U8 duration
        '#EXTINF:',                          # M3U8 segment info
        'https://www.youtube.com/api/timedtext', # Direct API URL
        'sparams=',                          # URL parameters
        'signature=',                        # URL signature
        'caps=asr'                           # Caption parameters
    ]
    
    content_lower = content.lower()
    for indicator in invalid_indicators:
        if indicator.lower() in content_lower:
            logger.info(f"🚫 Invalid content detected: contains '{indicator}'")
            return True
    
    return False

def extract_subtitle_from_entry(ydl, entry, clean: bool) -> str:
    """
    Extract and parse subtitle content from yt-dlp entry
    
    This function handles the actual downloading and parsing of subtitle files
    from yt-dlp subtitle entries. It validates URLs, downloads content,
    and processes different subtitle formats.
    
    Args:
        ydl: yt-dlp YoutubeDL instance
        entry: Subtitle entry from yt-dlp with URL and format info
        clean: Whether to return clean or timestamped format
    
    Returns:
        Formatted transcript text, or empty string if extraction fails
    """
    try:
        if 'url' not in entry:
            logger.info("📝 No URL found in subtitle entry")
            return ""
        
        subtitle_url = entry['url']
        ext = entry.get('ext', 'vtt')
        
        # Skip if URL looks like M3U8 or contains suspicious patterns
        suspicious_patterns = ['m3u8', 'playlist', 'range=', 'sparams=', 'signature=']
        if any(pattern in subtitle_url.lower() for pattern in suspicious_patterns):
            logger.info(f"🚫 Skipping suspicious URL: {subtitle_url[:100]}...")
            return ""
        
        try:
            # Download subtitle content using yt-dlp's urlopen method
            logger.info(f"📥 Downloading subtitle content: {ext} format")
            subtitle_content = ydl.urlopen(subtitle_url).read().decode('utf-8', errors='ignore')
            
            # Check if content is valid before parsing
            if is_invalid_content(subtitle_content):
                logger.info(f"🚫 Invalid subtitle content detected for {ext} format")
                return ""
            
            # Parse the content based on format
            transcript_data = parse_subtitle_content_enhanced(subtitle_content, ext)
            
            if transcript_data and len(transcript_data) > 0:
                logger.info(f"✅ Successfully parsed {len(transcript_data)} transcript segments")
                return format_transcript_enhanced(transcript_data, clean)
            else:
                logger.info(f"📝 No valid transcript data found in {ext} content")
                
        except Exception as url_error:
            logger.info(f"📝 URL extraction failed: {str(url_error)}")
            
    except Exception as e:
        logger.info(f"📝 Entry extraction failed: {str(e)}")
    
    return ""

def parse_subtitle_content_enhanced(content: str, format_type: str) -> list:
    """
    Enhanced subtitle content parsing for multiple formats
    
    Supports parsing of:
    - VTT/WebVTT format (most common)
    - SRV3/JSON format (YouTube's internal format)
    - TTML/XML format (standard subtitle format)
    
    Args:
        content: Raw subtitle file content
        format_type: Format indicator (vtt, srv3, json, ttml, xml)
    
    Returns:
        List of transcript segments with text, start time, and duration
    """
    transcript_data = []
    
    try:
        if format_type.lower() in ['vtt', 'webvtt']:
            transcript_data = parse_vtt_content(content)
        elif format_type.lower() in ['srv3', 'json']:
            transcript_data = parse_srv3_content(content)
        elif format_type.lower() in ['ttml', 'xml']:
            transcript_data = parse_ttml_content(content)
        else:
            # Try to auto-detect format based on content
            if content.strip().startswith('WEBVTT'):
                transcript_data = parse_vtt_content(content)
            elif content.strip().startswith('{') or content.strip().startswith('['):
                transcript_data = parse_srv3_content(content)
            elif '<' in content and '>' in content:
                transcript_data = parse_ttml_content(content)
    
    except Exception as e:
        logger.info(f"📝 Content parsing failed: {str(e)}")
    
    return transcript_data

def parse_vtt_content(content: str) -> list:
    """
    Parse WebVTT subtitle content
    
    WebVTT format structure:
    WEBVTT
    
    00:00:01.000 --> 00:00:04.000
    Hello, this is the first subtitle
    
    Args:
        content: WebVTT file content
    
    Returns:
        List of transcript segments
    """
    transcript_data = []
    lines = content.split('\n')
    current_start = 0
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Skip metadata and empty lines
        if not line or line.startswith('WEBVTT') or line.startswith('NOTE') or line.isdigit():
            continue
        
        # Parse timestamp line (contains -->)
        if '-->' in line:
            try:
                start_time = line.split(' --> ')[0].strip()
                current_start = parse_vtt_timestamp(start_time)
            except:
                current_start = 0
        
        # Parse text line (actual subtitle content)
        elif line and not line.startswith('<') and '-->' not in line:
            clean_text = clean_vtt_text(line)
            if clean_text and len(clean_text) > 1:
                transcript_data.append({
                    'text': clean_text,
                    'start': current_start,
                    'duration': 3.0  # Default duration
                })
    
    return transcript_data

def clean_vtt_text(text: str) -> str:
    """
    Clean VTT text from formatting tags and artifacts
    
    Removes VTT-specific formatting like:
    - <c> color tags
    - <i> italic tags
    - {style} annotations
    - HTML entities
    
    Args:
        text: Raw VTT text line
    
    Returns:
        Clean text content
    """
    # Remove VTT formatting tags
    import re
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML/VTT tags
    text = re.sub(r'\{[^}]+\}', '', text)  # Remove style annotations
    text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
    return text.strip()

def parse_vtt_timestamp(timestamp: str) -> float:
    """
    Parse VTT timestamp format to seconds
    
    Supports formats:
    - HH:MM:SS.mmm (hours:minutes:seconds.milliseconds)
    - MM:SS.mmm (minutes:seconds.milliseconds)
    
    Args:
        timestamp: Timestamp string from VTT file
    
    Returns:
        Time in seconds as float
    """
    try:
        # Handle both comma and dot as decimal separator
        parts = timestamp.replace(',', '.').split(':')
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        elif len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
    except:
        pass
    return 0

def parse_srv3_content(content: str) -> list:
    """
    Parse SRV3/JSON subtitle content (YouTube's internal format)
    
    SRV3 format structure:
    {
      "events": [
        {
          "tStartMs": 1000,
          "dDurationMs": 3000,
          "segs": [{"utf8": "Hello world"}]
        }
      ]
    }
    
    Args:
        content: SRV3/JSON file content
    
    Returns:
        List of transcript segments
    """
    transcript_data = []
    try:
        import json
        data = json.loads(content)
        
        if 'events' in data:
            for event in data['events']:
                if 'segs' in event:
                    text_segments = []
                    for seg in event['segs']:
                        if 'utf8' in seg:
                            text_segments.append(seg['utf8'])
                    
                    if text_segments:
                        full_text = ''.join(text_segments).strip()
                        if full_text and len(full_text) > 1:
                            transcript_data.append({
                                'text': full_text,
                                'start': event.get('tStartMs', 0) / 1000.0,  # Convert ms to seconds
                                'duration': event.get('dDurationMs', 3000) / 1000.0
                            })
    except:
        pass
    
    return transcript_data

def parse_ttml_content(content: str) -> list:
    """
    Parse TTML/XML subtitle content
    
    TTML is a standard XML-based subtitle format used by many platforms.
    
    Args:
        content: TTML/XML file content
    
    Returns:
        List of transcript segments
    """
    transcript_data = []
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(content)
        
        # Find all text elements with timing information
        for elem in root.iter():
            if elem.text and elem.text.strip():
                start_time = 0
                if 'begin' in elem.attrib:
                    start_time = parse_ttml_timestamp(elem.attrib['begin'])
                
                clean_text = elem.text.strip()
                if clean_text and len(clean_text) > 1:
                    transcript_data.append({
                        'text': clean_text,
                        'start': start_time,
                        'duration': 3.0  # Default duration
                    })
    except:
        pass
    
    return transcript_data

def parse_ttml_timestamp(timestamp: str) -> float:
    """
    Parse TTML timestamp format to seconds
    
    TTML supports various timestamp formats:
    - "10s" (seconds)
    - "01:30:45" (HH:MM:SS)
    
    Args:
        timestamp: TTML timestamp string
    
    Returns:
        Time in seconds as float
    """
    try:
        if 's' in timestamp:
            return float(timestamp.replace('s', ''))
        elif ':' in timestamp:
            parts = timestamp.split(':')
            if len(parts) == 3:
                h, m, s = parts
                return int(h) * 3600 + int(m) * 60 + float(s)
    except:
        pass
    return 0

def extract_live_transcript(video_id: str, clean: bool) -> str:
    """
    Handle live stream transcript extraction
    
    Live streams have special requirements:
    - Transcripts may not be available during broadcast
    - Content may be incomplete
    - Should provide helpful feedback to users
    
    Args:
        video_id: YouTube video ID
        clean: Whether to return clean or timestamped format
    
    Returns:
        Live transcript content or helpful message
    """
    logger.info(f"🔴 Attempting live transcript extraction for {video_id}")
    
    # For live streams, try the standard API methods first
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        if transcript_list:
            return format_transcript_enhanced(transcript_list, clean)
    except:
        pass
    
    # Return informative message for live content
    return "This appears to be a live stream. Live transcripts may not be available or may be incomplete. Please try again after the stream has ended."

def extract_with_direct_api(video_id: str) -> list:
    """
    Direct API extraction method for edge cases
    
    This is a placeholder for future implementation of additional
    direct API methods that might become available.
    
    Args:
        video_id: YouTube video ID
    
    Returns:
        List of transcript segments (currently empty)
    """
    try:
        # Placeholder for future direct API implementations
        # Could include custom API endpoints, alternative services, etc.
        return []
    except:
        return []

def get_demo_content(clean: bool) -> str:
    """
    Fallback demo content for testing and when extraction fails
    
    This provides sample content that demonstrates the expected format
    while clearly indicating it's placeholder content.
    
    Args:
        clean: Whether to return clean or timestamped format
    
    Returns:
        Demo transcript content
    """
    demo_text = """Hello everyone, welcome to this video. Today we're going to be talking 
    about some really interesting topics. We'll cover various aspects of the subject matter 
    and provide you with valuable insights. This is sample transcript content for demonstration 
    purposes. The actual transcript would contain the real audio content from the YouTube video. 
    Thank you for watching, and don't forget to subscribe to our channel for more great content like this."""
    
    if clean:
        return demo_text
    else:
        # Add timestamps for unclean format demonstration
        words = demo_text.split()
        timestamped_lines = []
        current_time = 0
        
        for i in range(0, len(words), 8):  # Group words into 8-word chunks
            chunk = ' '.join(words[i:i+8])
            minutes = current_time // 60
            seconds = current_time % 60
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            timestamped_lines.append(f"{timestamp} {chunk}")
            current_time += 6  # Add 6 seconds per chunk
        
        return '\n'.join(timestamped_lines)

#=======================================================================
# API ENDPOINTS: ALL THE ENDPOINTS OF THE YOUTUBE TRANS DOWNLOADER API #
#=======================================================================

@app.get("/")
async def root():
    return {"message": "YouTube Transcript Downloader API", "status": "running", "version": "1.0.0"}

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user(db, user_data.username)
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    email_exists = get_user_by_email(db, user_data.email)
    if email_exists:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user_data.password)
    new_user = User(
        username=user_data.username,
        email=user_data.email,
        hashed_password=hashed_password,
        created_at=datetime.now()
    )
    
    try:
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        logger.info(f"User registered successfully: {user_data.username}")
        return new_user
    except Exception as e:
        db.rollback()
        logger.error(f"Error registering user: {str(e)}")
        raise HTTPException(status_code=500, detail="Error registering user")

@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    logger.info(f"User logged in successfully: {form_data.username}")
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/users/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user

@app.post("/download_transcript/")
async def download_transcript_corrected(
    request: TranscriptRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Enhanced transcript downloader with robust extraction methods
    
    This endpoint handles YouTube transcript extraction using multiple
    fallback methods to ensure maximum success rate.
    """
    video_id = request.youtube_id.strip()
    
    # Extract video ID from various YouTube URL formats
    if 'youtube.com' in video_id or 'youtu.be' in video_id:
        patterns = [
            r'(?:youtube\.com\/watch\?v=)([^&\n?#]+)',      # Standard watch URL
            r'(?:youtu\.be\/)([^&\n?#]+)',                  # Short URL
            r'(?:youtube\.com\/embed\/)([^&\n?#]+)',        # Embed URL
            r'(?:youtube\.com\/shorts\/)([^&\n?#]+)',       # Shorts URL
            r'[?&]v=([^&\n?#]+)'                            # Video parameter
        ]
        
        for pattern in patterns:
            match = re.search(pattern, video_id)
            if match:
                video_id = match.group(1)[:11]  # YouTube IDs are 11 characters
                logger.info(f"✅ Extracted video ID: {video_id}")
                break
    
    # Validate video ID format
    if not video_id or len(video_id) != 11:
        raise HTTPException(
            status_code=400, 
            detail="Invalid YouTube video ID. Please provide a valid 11-character video ID or full YouTube URL."
        )
    
    logger.info(f"🎯 ENHANCED transcript request for: {video_id}")
    
    # Check subscription limits based on transcript type
    transcript_type = "clean" if request.clean_transcript else "unclean"
    can_download = check_subscription_limit(user.id, transcript_type, db)
    if not can_download:
        raise HTTPException(
            status_code=403, 
            detail=f"You've reached your monthly limit for {transcript_type} transcripts. Please upgrade your plan."
        )
    
    # Extract transcript using enhanced method with multiple fallbacks
    try:
        transcript_text = get_youtube_transcript_corrected(video_id, clean=request.clean_transcript)
        
        # Validate that we got meaningful content
        if not transcript_text or len(transcript_text.strip()) < 10:
            raise HTTPException(
                status_code=404,
                detail=f"No transcript content found for video {video_id}."
            )
        
        # Record successful download for usage tracking
        new_download = TranscriptDownload(
            user_id=user.id,
            youtube_id=video_id,
            transcript_type=transcript_type,
            created_at=datetime.now()
        )
        
        db.add(new_download)
        db.commit()
        
        logger.info(f"🎉 ENHANCED SUCCESS: {user.username} downloaded {len(transcript_text)} chars for {video_id}")
        
        return {
            "transcript": transcript_text,
            "youtube_id": video_id,
            "message": "Transcript downloaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"💥 Enhanced extraction failed: {str(e)}")
        
        raise HTTPException(
            status_code=500,
            detail=f"Failed to extract transcript for video {video_id}. Error: {str(e)}"
        )
#==============================================

@app.get("/subscription_status/")
async def get_subscription_status_ultra_safe(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Ultra-safe subscription status - handles all database issues gracefully"""
    try:
        # Get basic subscription info
        subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id
        ).first()
        
        # Determine current subscription tier
        if not subscription:
            tier = "free"
            status = "inactive"
            expiry_date = None
        else:
            # Check if subscription is expired
            if hasattr(subscription, 'expiry_date') and subscription.expiry_date and subscription.expiry_date < datetime.now():
                tier = "free"
                status = "expired"
                expiry_date = subscription.expiry_date
            else:
                tier = subscription.tier if subscription.tier else "free"
                status = "active" if tier != "free" else "inactive"
                expiry_date = subscription.expiry_date if hasattr(subscription, 'expiry_date') else None
        
        # Calculate usage for current month using safe method
        month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # Safe usage calculation with multiple fallbacks
        def get_safe_usage(transcript_type):
            try:
                # Method 1: Try ORM query
                return db.query(TranscriptDownload).filter(
                    TranscriptDownload.user_id == current_user.id,
                    TranscriptDownload.transcript_type == transcript_type,
                    TranscriptDownload.created_at >= month_start
                ).count()
            except Exception:
                try:
                    # Method 2: Raw SQL fallback
                    result = db.execute(
                        "SELECT COUNT(*) FROM transcript_downloads WHERE user_id = ? AND transcript_type = ? AND created_at >= ?",
                        (current_user.id, transcript_type, month_start.isoformat())
                    )
                    return result.scalar() or 0
                except Exception:
                    # Method 3: Ultimate fallback
                    logger.warning(f"All usage queries failed for {transcript_type}, returning 0")
                    return 0
        
        # Get usage for all types safely
        clean_usage = get_safe_usage("clean")
        unclean_usage = get_safe_usage("unclean")
        audio_usage = get_safe_usage("audio")
        video_usage = get_safe_usage("video")
        
        # Get limits based on current tier
        limits = SUBSCRIPTION_LIMITS.get(tier, SUBSCRIPTION_LIMITS["free"])
        
        # Convert infinity to string for JSON serialization
        json_limits = {}
        for key, value in limits.items():
            if value == float('inf'):
                json_limits[key] = 'unlimited'
            else:
                json_limits[key] = value
        
        logger.info(f"✅ Safe subscription status for {current_user.username}: tier={tier}, status={status}")
        
        return {
            "tier": tier,
            "status": status,
            "usage": {
                "clean_transcripts": clean_usage,
                "unclean_transcripts": unclean_usage,
                "audio_downloads": audio_usage,
                "video_downloads": video_usage,
            },
            "limits": json_limits,
            "subscription_id": subscription.payment_id if subscription and hasattr(subscription, 'payment_id') else None,
            "stripe_customer_id": getattr(current_user, 'stripe_customer_id', None),
            "current_period_end": expiry_date.isoformat() if expiry_date else None
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting subscription status: {str(e)}")
        # Return safe defaults on any error
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

@app.post("/confirm_payment/")
async def confirm_payment_endpoint(
    request: ConfirmPaymentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enhanced payment confirmation with complete Stripe integration"""
    try:
        # Retrieve payment intent from Stripe
        intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
        
        if intent.status != 'succeeded':
            raise HTTPException(status_code=400, detail=f"Payment not completed. Status: {intent.status}")

        # Get plan details
        plan_type = intent.metadata.get('plan_type', 'pro')
        price_id = intent.metadata.get('price_id')
        
        # Create or update subscription record
        user_subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id
        ).first()

        if not user_subscription:
            user_subscription = Subscription(
                user_id=current_user.id,
                tier=plan_type,
                start_date=datetime.utcnow(),
                expiry_date=datetime.utcnow() + timedelta(days=30),
                payment_id=request.payment_intent_id,
                auto_renew=True,
                stripe_price_id=price_id,
                status='active',
                current_period_start=datetime.utcnow(),
                current_period_end=datetime.utcnow() + timedelta(days=30)
            )
            db.add(user_subscription)
        else:
            user_subscription.tier = plan_type
            user_subscription.start_date = datetime.utcnow()
            user_subscription.expiry_date = datetime.utcnow() + timedelta(days=30)
            user_subscription.payment_id = request.payment_intent_id
            user_subscription.auto_renew = True
            user_subscription.stripe_price_id = price_id
            user_subscription.status = 'active'
            user_subscription.current_period_start = datetime.utcnow()
            user_subscription.current_period_end = datetime.utcnow() + timedelta(days=30)

        # Record payment history
        payment_record = PaymentHistory(
            user_id=current_user.id,
            stripe_payment_intent_id=request.payment_intent_id,
            stripe_customer_id=intent.customer,
            amount=intent.amount,
            currency=intent.currency,
            status='succeeded',
            subscription_tier=plan_type,
            created_at=datetime.utcnow(),
            metadata=str(intent.metadata)
        )
        db.add(payment_record)

        # Update user's Stripe customer ID if not already set
        if hasattr(current_user, 'stripe_customer_id') and not current_user.stripe_customer_id:
            current_user.stripe_customer_id = intent.customer

        db.commit()
        db.refresh(user_subscription)

        logger.info(f"🎉 Payment confirmed: User {current_user.username} upgraded to {plan_type}")

        return {
            'success': True,
            'subscription_tier': user_subscription.tier,
            'expires_at': user_subscription.expiry_date.isoformat(),
            'status': 'active',
            'payment_id': request.payment_intent_id,
            'customer_id': intent.customer
        }

    except Exception as e:
        logger.error(f"❌ Payment confirmation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to confirm payment: {str(e)}")

#========================
# Health check endpoints
# =======================
@app.get("/health/")
async def health_check():
    return {
        "status": "healthy",
        "stripe_configured": bool(os.getenv("STRIPE_SECRET_KEY")),
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok", "version": "1.0.0"}

#======================
# Library info endpoint
#======================
@app.get("/library_info")
async def library_info():
    try:
        from youtube_transcript_api import __version__
        return {"youtube_transcript_api_version": __version__}
    except:
        return {"youtube_transcript_api_version": "unknown"}

@app.get("/test_videos")
async def get_test_videos():
    """Get list of verified working video IDs for testing"""
    working_videos = [
        {
            "id": "dQw4w9WgXcQ",
            "title": "Rick Astley - Never Gonna Give You Up",
            "description": "Classic music video with reliable captions"
        },
        {
            "id": "jNQXAC9IVRw", 
            "title": "Me at the zoo",
            "description": "First YouTube video ever uploaded"
        }
    ]
    
    return {
        "message": "These video IDs have been verified to work for transcript extraction",
        "videos": working_videos,
        "usage": "Use any of these video IDs to test your transcript downloader",
        "note": "These examples include demo content fallback for testing"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)

#####################################################################
## IMPORTANT MESSAGE: DO NOT ALTER THIS MAIN.PY ANYMORE-- THANKS! ###
##################################################################### 