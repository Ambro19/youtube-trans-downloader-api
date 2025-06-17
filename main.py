# main.py - CLEAN WORKING VERSION (Based on your old successful approach)

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, List

import jwt
from jwt import PyJWTError

from pydantic import BaseModel
from passlib.context import CryptContext
import stripe
from youtube_transcript_api import YouTubeTranscriptApi
import os
import json
import logging
from dotenv import load_dotenv
import requests
import re

import warnings
warnings.filterwarnings("ignore", message=".*bcrypt.*")

# Import from database.py
from database import get_db, User, Subscription, TranscriptDownload, create_tables

# Load environment variables
load_dotenv()

# Configure logging
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
    logger.info(f"Production mode - CORS origins: {allowed_origins}")
else:
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        FRONTEND_URL
    ]
    logger.info(f"Development mode - CORS origins: {allowed_origins}")

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

# Subscription limits
SUBSCRIPTION_LIMITS = {
    "free": {
        "clean_transcripts": 5, "unclean_transcripts": 3, 
        "audio_downloads": 2, "video_downloads": 1
    },
    "pro": {
        "clean_transcripts": 100, "unclean_transcripts": 50,
        "audio_downloads": 50, "video_downloads": 20
    },
    "premium": {
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
    """Startup event"""
    try:
        logger.info("Starting YouTube Transcript Downloader API...")
        logger.info(f"Environment: {ENVIRONMENT}")
        
        # Initialize database
        create_tables()
        logger.info("Database initialized successfully")
        logger.info("Application startup complete!")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str

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
    youtube_id: str = None
    video_url: str = None
    clean_transcript: bool = False

class CreatePaymentIntentRequest(BaseModel):
    price_id: str

class ConfirmPaymentRequest(BaseModel):
    payment_intent_id: str

class SubscriptionRequest(BaseModel):
    subscription_tier: str

# Helper functions
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
    except Exception:
        raise credentials_exception
        
    user = get_user(db, username)
    if user is None:
        raise credentials_exception
    return user

def check_user_limits(user, action_type: str, db: Session):
    """Check if user has exceeded their limits for the current month"""
    subscription = db.query(Subscription).filter(Subscription.user_id == user.id).first()
    
    if not subscription or subscription.expiry_date < datetime.now():
        tier = "free"
    else:
        tier = subscription.tier
    
    month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    type_mapping = {
        "clean_transcripts": "clean",
        "unclean_transcripts": "unclean",
        "audio_downloads": "audio",
        "video_downloads": "video"
    }
    
    transcript_type = type_mapping.get(action_type, action_type)
    
    current_usage = db.query(TranscriptDownload).filter(
        TranscriptDownload.user_id == user.id,
        TranscriptDownload.transcript_type == transcript_type,
        TranscriptDownload.created_at >= month_start
    ).count()
    
    limit = SUBSCRIPTION_LIMITS[tier].get(action_type, 0)
    
    if limit == float('inf'):
        return True
    
    return current_usage < limit

def get_or_create_stripe_customer(user, db: Session):
    """Get or create a Stripe customer for the user"""
    try:
        stripe_customer_id = getattr(user, 'stripe_customer_id', None)
        
        if stripe_customer_id:
            try:
                customer = stripe.Customer.retrieve(stripe_customer_id)
                return customer
            except stripe.error.InvalidRequestError:
                pass
        
        customer = stripe.Customer.create(
            email=user.email,
            name=user.username,
            metadata={'user_id': str(user.id)}
        )
        
        try:
            if hasattr(user, 'stripe_customer_id'):
                user.stripe_customer_id = customer.id
                db.commit()
        except Exception:
            pass
        
        return customer
        
    except Exception as e:
        logger.error(f"Stripe error creating customer: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create payment customer"
        )

# SIMPLE WORKING TRANSCRIPT PROCESSING (Based on your old successful approach)
def process_youtube_transcript(video_id: str, clean: bool = True) -> str:
    """
    Simple transcript processing - BACK TO YOUR WORKING VERSION
    """
    try:
        logger.info(f"Getting transcript for video: {video_id}")
        
        # Simple direct call to YouTube Transcript API (your original working method)
        transcript_list = YouTubeTranscriptApi.get_transcript(
            video_id,
            languages=['en', 'en-US', 'en-GB']
        )
        
        if not transcript_list:
            raise HTTPException(
                status_code=404,
                detail="No transcript data found for this video."
            )
        
        logger.info(f"Retrieved {len(transcript_list)} transcript segments")
        
        if clean:
            # Clean format - text only
            text_parts = []
            for item in transcript_list:
                if 'text' in item and item['text'].strip():
                    text = item['text'].strip()
                    # Remove sound effect markers like [Music], [Applause], etc.
                    text = text.replace('[Music]', '').replace('[Applause]', '').replace('[Laughter]', '').strip()
                    if text and not (text.startswith('[') and text.endswith(']')):
                        text_parts.append(text)
            
            if not text_parts:
                raise HTTPException(
                    status_code=404,
                    detail="Transcript contains no readable text content."
                )
            
            return ' '.join(text_parts)
        else:
            # Unclean format - with timestamps
            formatted_transcript = []
            for item in transcript_list:
                if 'text' in item and 'start' in item:
                    start_time = float(item['start'])
                    minutes = int(start_time // 60)
                    seconds = int(start_time % 60)
                    timestamp = f"[{minutes:02d}:{seconds:02d}]"
                    text = item['text'].strip()
                    if text:
                        formatted_transcript.append(f"{timestamp} {text}")
            
            if not formatted_transcript:
                raise HTTPException(
                    status_code=404,
                    detail="Transcript contains no content with timestamps."
                )
            
            return '\n'.join(formatted_transcript)
            
    except Exception as e:
        error_msg = str(e).lower()
        logger.error(f"Error getting transcript for {video_id}: {str(e)}")
        
        # Simple error handling based on error message patterns
        if "transcript" in error_msg and "disabled" in error_msg:
            raise HTTPException(
                status_code=404,
                detail="Transcripts are disabled for this video."
            )
        elif "no transcript" in error_msg or "not found" in error_msg or "could not retrieve" in error_msg:
            raise HTTPException(
                status_code=404,
                detail="No captions found for this video. Try a different video with captions enabled."
            )
        elif "video unavailable" in error_msg or "private" in error_msg:
            raise HTTPException(
                status_code=404,
                detail="Video is unavailable, private, or doesn't exist."
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve transcript: {str(e)}"
            )

# API Endpoints
@app.get("/")
async def root():
    return {"message": "YouTube Transcript Downloader API", "status": "running", "version": "1.0.0"}

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user(db, user_data.username)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    email_exists = get_user_by_email(db, user_data.email)
    if email_exists:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error registering user"
        )

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

# MAIN DOWNLOAD ENDPOINT - SIMPLE WORKING VERSION
@app.post("/download_transcript/")
async def download_transcript(
    request: TranscriptRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Simple transcript download - BACK TO YOUR WORKING VERSION"""
    
    try:
        # Get video ID from request
        video_identifier = getattr(request, 'video_url', None) or request.youtube_id
        
        if not video_identifier:
            return {
                "success": False,
                "error_type": "missing_identifier",
                "message": "Please provide a YouTube URL or video ID"
            }
        
        # Simple video ID extraction (your original method)
        video_id = video_identifier.strip()
        
        # If it's a URL, extract the ID
        if 'youtube.com' in video_id or 'youtu.be' in video_id:
            patterns = [
                r'(?:youtube\.com\/watch\?v=)([a-zA-Z0-9_-]{11})',
                r'(?:youtu\.be\/)([a-zA-Z0-9_-]{11})',
                r'(?:youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, video_id)
                if match:
                    video_id = match.group(1)
                    break
        
        # Validate video ID length
        if len(video_id) != 11:
            return {
                "success": False,
                "error_type": "invalid_url",
                "message": "Please enter a valid YouTube URL or 11-character Video ID"
            }
        
        logger.info(f"Processing transcript request for video: {video_id}")
        
        # Check subscription limits
        try:
            transcript_type = "clean_transcripts" if request.clean_transcript else "unclean_transcripts"
            can_download = check_user_limits(user, transcript_type, db)
            if not can_download:
                return {
                    "success": False,
                    "error_type": "subscription_limit",
                    "message": "Monthly limit reached! Please upgrade your plan."
                }
        except Exception as e:
            logger.warning(f"Subscription check error: {e}")
        
        # Use the SIMPLE approach that was working for you
        try:
            transcript_content = process_youtube_transcript(video_id, request.clean_transcript)
            
            if not transcript_content.strip():
                return {
                    "success": False,
                    "error_type": "no_transcript",
                    "message": "Transcript is empty or contains no readable content."
                }
            
        except HTTPException as e:
            return {
                "success": False,
                "error_type": "no_transcript",
                "message": e.detail
            }
        except Exception as e:
            logger.error(f"Transcript extraction failed for {video_id}: {str(e)}")
            return {
                "success": False,
                "error_type": "retrieval_failed",
                "message": "Failed to retrieve transcript. Please try a different video."
            }
        
        # Record successful download in database  
        try:
            transcript_type_db = "clean" if request.clean_transcript else "unclean"
            new_download = TranscriptDownload(
                user_id=user.id,
                youtube_id=video_id,
                transcript_type=transcript_type_db,
                created_at=datetime.now()
            )
            db.add(new_download)
            db.commit()
            logger.info(f"SUCCESS: {user.username} downloaded {transcript_type_db} transcript for {video_id}")
        except Exception as e:
            logger.warning(f"Failed to update usage tracking: {e}")
        
        # Return response compatible with your frontend
        return {
            "success": True,
            "transcript": transcript_content,
            "youtube_id": video_id,
            "message": "Transcript downloaded successfully"
        }
        
    except Exception as e:
        logger.error(f"Unexpected error in download_transcript: {str(e)}")
        return {
            "success": False,
            "error_type": "internal_error",
            "message": "An unexpected error occurred. Please try again."
        }

# Payment endpoints
@app.post("/create_payment_intent/")
async def create_payment_intent_endpoint(
    request: CreatePaymentIntentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a payment intent for subscription upgrade"""
    try:
        valid_price_ids = [
            os.getenv("PRO_PRICE_ID"),
            os.getenv("PREMIUM_PRICE_ID")
        ]
        
        if request.price_id not in valid_price_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid price ID: {request.price_id}"
            )

        price = stripe.Price.retrieve(request.price_id)
        plan_type = 'pro' if request.price_id == os.getenv("PRO_PRICE_ID") else 'premium'
        customer = get_or_create_stripe_customer(current_user, db)
        
        intent = stripe.PaymentIntent.create(
            amount=price.unit_amount,
            currency=price.currency,
            customer=customer.id,
            automatic_payment_methods={
                'enabled': True,
                'allow_redirects': 'never'
            },
            metadata={
                'user_id': str(current_user.id),
                'user_email': current_user.email,
                'price_id': request.price_id,
                'plan_type': plan_type
            }
        )

        return {
            'client_secret': intent.client_secret,
            'payment_intent_id': intent.id,
            'amount': price.unit_amount,
            'currency': price.currency,
            'plan_type': plan_type
        }

    except Exception as e:
        logger.error(f"Payment intent creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create payment intent: {str(e)}"
        )

@app.post("/confirm_payment/")
async def confirm_payment_endpoint(
    request: ConfirmPaymentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Confirm payment and update user subscription"""
    try:
        intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
        
        if intent.status != 'succeeded':
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Payment not completed. Status: {intent.status}"
            )

        user_subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id
        ).first()

        plan_type = intent.metadata.get('plan_type', 'pro')

        if not user_subscription:
            user_subscription = Subscription(
                user_id=current_user.id,
                tier=plan_type,
                start_date=datetime.utcnow(),
                expiry_date=datetime.utcnow() + timedelta(days=30),
                payment_id=request.payment_intent_id,
                auto_renew=True
            )
            db.add(user_subscription)
        else:
            user_subscription.tier = plan_type
            user_subscription.start_date = datetime.utcnow()
            user_subscription.expiry_date = datetime.utcnow() + timedelta(days=30)
            user_subscription.payment_id = request.payment_intent_id
            user_subscription.auto_renew = True

        db.commit()
        db.refresh(user_subscription)

        return {
            'success': True,
            'subscription_tier': user_subscription.tier,
            'expires_at': user_subscription.expiry_date.isoformat(),
            'status': 'active'
        }

    except Exception as e:
        logger.error(f"Payment confirmation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to confirm payment: {str(e)}"
        )

# Subscription status endpoint
@app.get("/subscription_status/")
async def get_subscription_status_enhanced(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get subscription status with detailed usage info"""
    try:
        subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id
        ).first()
        
        if not subscription or subscription.expiry_date < datetime.now():
            tier = "free"
            status = "inactive"
        else:
            tier = subscription.tier
            status = "active"
        
        # Get current month's usage
        month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        clean_usage = db.query(TranscriptDownload).filter(
            TranscriptDownload.user_id == current_user.id,
            TranscriptDownload.transcript_type == "clean",
            TranscriptDownload.created_at >= month_start
        ).count()
        
        unclean_usage = db.query(TranscriptDownload).filter(
            TranscriptDownload.user_id == current_user.id,
            TranscriptDownload.transcript_type == "unclean",
            TranscriptDownload.created_at >= month_start
        ).count()
        
        limits = SUBSCRIPTION_LIMITS[tier]
        
        # Convert infinity to string for JSON serialization
        json_limits = {}
        for key, value in limits.items():
            if value == float('inf'):
                json_limits[key] = 'unlimited'
            else:
                json_limits[key] = value
        
        return {
            "tier": tier,
            "status": status,
            "usage": {
                "clean_transcripts": clean_usage,
                "unclean_transcripts": unclean_usage,
            },
            "limits": json_limits,
            "subscription_id": subscription.payment_id if subscription else None,
            "current_period_end": subscription.expiry_date.isoformat() if subscription and subscription.expiry_date else None
        }
        
    except Exception as e:
        logger.error(f"Error getting subscription status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get subscription status"
        )

# Health check
@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat()
    }

# Test endpoint for working videos
@app.get("/test_working_videos/")
async def test_working_videos():
    """Test endpoint with known working video IDs"""
    working_videos = [
        {
            "id": "k-RjskuqxzU",
            "title": "Your Previously Working Video",
            "description": "This was working in Picture #3 & #4"
        },
        {
            "id": "jNQXAC9IVRw",
            "title": "Me at the zoo",
            "description": "First YouTube video"
        },
        {
            "id": "dQw4w9WgXcQ",
            "title": "Rick Astley - Never Gonna Give You Up",
            "description": "Popular video with captions"
        }
    ]
    
    results = []
    for video in working_videos:
        try:
            # Simple test using our process_youtube_transcript function
            transcript = process_youtube_transcript(video["id"], clean=True)
            results.append({
                "video": video,
                "status": "SUCCESS",
                "transcript_length": len(transcript)
            })
        except Exception as e:
            results.append({
                "video": video,
                "status": "FAILED",
                "error": str(e)
            })
    
    return {
        "test_results": results,
        "recommendation": "Use videos marked as 'SUCCESS' for testing"
    }