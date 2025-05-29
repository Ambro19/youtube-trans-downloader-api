# main.py

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware  # ADD THIS IMPORT
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, List
import jwt
from pydantic import BaseModel
from passlib.context import CryptContext
import stripe
import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi
import os
import json
import logging
from dotenv import load_dotenv

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

# ADD CORS MIDDLEWARE - THIS IS THE KEY FIX!
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React app URLs
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Include OPTIONS
    allow_headers=["*"],  # Allow all headers
)

# Authentication setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Constants - Load from environment variables
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))  # 24 hours

SUBSCRIPTION_LIMITS = {
    "free": {"transcript": 5, "audio": 2, "video": 1, "clean": 5, "unclean": 3},
    "pro": {"transcript": 100, "audio": 50, "video": 20, "clean": 100, "unclean": 50},
    "premium": {"transcript": float('inf'), "audio": float('inf'), "video": float('inf'), "clean": float('inf'), "unclean": float('inf')}
}

# Price ID mapping (use "pro" instead of "basic")
PRICE_ID_MAP = {
    "pro": os.getenv("PRO_PRICE_ID"),
    "premium": os.getenv("PREMIUM_PRICE_ID")
}

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    try:
        logger.info("Initializing application...")
        create_tables()
        logger.info("Application initialized successfully")
    except Exception as e:
        logger.error(f"Error during application startup: {str(e)}")
        raise

# Pydantic models for request/response validation
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
        from_attributes = True  # FIXED: Changed from orm_mode to from_attributes

class TranscriptRequest(BaseModel):
    youtube_id: str
    clean_transcript: bool = False

class PaymentRequest(BaseModel):
    token: str
    subscription_tier: str

class SubscriptionResponse(BaseModel):
    tier: str
    status: str
    expiry_date: Optional[str] = None
    limits: dict
    usage: Optional[dict] = None
    remaining: Optional[dict] = None
    
    class Config:
        from_attributes = True  # FIXED: Changed from orm_mode to from_attributes

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

def get_user_by_id(db: Session, user_id: int):
    return db.query(User).filter(User.id == user_id).first()

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

def check_subscription_limit(user_id: int, transcript_type: str, db: Session):
    # Get user's subscription
    subscription = db.query(Subscription).filter(Subscription.user_id == user_id).first()
    
    if not subscription:
        tier = "free"
    else:
        tier = subscription.tier
        # Check if subscription is expired
        if subscription.expiry_date < datetime.now():
            tier = "free"
    
    # Get current month's usage
    month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    usage = db.query(TranscriptDownload).filter(
        TranscriptDownload.user_id == user_id,
        TranscriptDownload.transcript_type == transcript_type,
        TranscriptDownload.created_at >= month_start
    ).count()
    
    # Check if usage exceeds limit
    limit = SUBSCRIPTION_LIMITS[tier][transcript_type]
    if usage >= limit:
        return False
    return True

def process_youtube_transcript(youtube_id: str, clean: bool):
    try:
        # Get transcript from YouTube API
        transcript_list = YouTubeTranscriptApi.get_transcript(youtube_id)
        
        if clean:
            # Clean version (without timestamps)
            full_text = " ".join([item['text'] for item in transcript_list])
            return full_text
        else:
            # Unclean version (with timestamps)
            formatted_transcript = []
            for item in transcript_list:
                start_time = item['start']
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                text = item['text']
                formatted_transcript.append(f"[{minutes:02d}:{seconds:02d}] {text}")
            
            return "\n".join(formatted_transcript)
    
    except youtube_transcript_api._errors.TranscriptsDisabled:
        logger.warning(f"Transcripts are disabled for video: {youtube_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcripts are disabled for this video"
        )
    except youtube_transcript_api._errors.NoTranscriptFound:
        logger.warning(f"No transcript found for video: {youtube_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No transcript found for this video"
        )
    except Exception as e:
        logger.error(f"Error retrieving transcript for video {youtube_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving transcript: {str(e)}"
        )

# API Endpoints
@app.get("/")
async def root():
    return {"message": "YouTube Transcript Downloader API", "status": "running", "version": "1.0.0"}

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if username exists
    db_user = get_user(db, user_data.username)
    if db_user:
        logger.warning(f"Registration attempt with existing username: {user_data.username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email exists
    email_exists = get_user_by_email(db, user_data.email)
    if email_exists:
        logger.warning(f"Registration attempt with existing email: {user_data.email}")
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
        logger.warning(f"Failed login attempt for user: {form_data.username}")
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
async def download_transcript(
    request: TranscriptRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    transcript_type = "clean" if request.clean_transcript else "unclean"
    
    # Check if user has reached subscription limit
    can_download = check_subscription_limit(user.id, transcript_type, db)
    if not can_download:
        logger.warning(f"User {user.username} reached subscription limit for {transcript_type} transcripts")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You've reached your monthly limit for {transcript_type} transcripts. Please upgrade your subscription."
        )
    
    # Process transcript download
    transcript_text = process_youtube_transcript(
        request.youtube_id, 
        clean=request.clean_transcript
    )
    
    # Record the download
    new_download = TranscriptDownload(
        user_id=user.id,
        youtube_id=request.youtube_id,
        transcript_type=transcript_type,
        created_at=datetime.now()
    )
    
    try:
        db.add(new_download)
        db.commit()
        logger.info(f"User {user.username} downloaded {transcript_type} transcript for video {request.youtube_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Error recording transcript download: {str(e)}")
        # Continue anyway as the transcript was already processed
    
    # Return transcript data
    return {"transcript": transcript_text, "youtube_id": request.youtube_id}

@app.post("/create_subscription/")
async def create_subscription(
    request: PaymentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if request.subscription_tier not in PRICE_ID_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid subscription tier. Must be one of: {', '.join(PRICE_ID_MAP.keys())}"
        )
        
    try:
        # Create Stripe customer & subscription
        customer = stripe.Customer.create(
            source=request.token,
            email=current_user.email,
            name=current_user.username
        )
        
        subscription = stripe.Subscription.create(
            customer=customer.id,
            items=[{"price": PRICE_ID_MAP[request.subscription_tier]}],
            metadata={
                "user_id": str(current_user.id),
                "username": current_user.username,
                "tier": request.subscription_tier
            }
        )
        
        # Check if user already has a subscription and update it
        existing_subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id
        ).first()
        
        if existing_subscription:
            existing_subscription.tier = request.subscription_tier
            existing_subscription.start_date = datetime.now()
            existing_subscription.expiry_date = datetime.now() + timedelta(days=30)
            existing_subscription.payment_id = subscription.id
            existing_subscription.auto_renew = True
        else:
            # Create new subscription
            new_subscription = Subscription(
                user_id=current_user.id,  
                tier=request.subscription_tier,
                start_date=datetime.now(),
                expiry_date=datetime.now() + timedelta(days=30),
                payment_id=subscription.id,
                auto_renew=True
            )
            db.add(new_subscription)
        
        db.commit()
        logger.info(f"User {current_user.username} created {request.subscription_tier} subscription successfully")
        
        return {
            "status": "success", 
            "subscription_id": subscription.id, 
            "tier": request.subscription_tier
        }
    
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error processing subscription: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating subscription: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing subscription"
        )

# Webhook handler for Stripe events
@app.post("/webhook", status_code=200)
async def webhook_received(request: Request, response: Response, db: Session = Depends(get_db)):
    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature")
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError as e:
        # Invalid payload
        logger.warning(f"Invalid webhook payload: {str(e)}")
        response.status_code = 400
        return {"error": str(e)}
    except stripe.error.SignatureVerificationError as e:
        # Invalid signature
        logger.warning(f"Invalid webhook signature: {str(e)}")
        response.status_code = 400
        return {"error": str(e)}
    
    # Handle the event
    event_type = event['type']
    logger.info(f"Received Stripe webhook event: {event_type}")
    
    if event_type == 'invoice.payment_succeeded':
        # Subscription was paid - update expiration date
        invoice = event['data']['object']
        subscription_id = invoice['subscription']
        
        # Find subscription in database
        subscription = db.query(Subscription).filter(
            Subscription.payment_id == subscription_id
        ).first()
        
        if subscription:
            # Extend subscription by 30 days
            subscription.expiry_date = datetime.now() + timedelta(days=30)
            db.commit()
            logger.info(f"Subscription {subscription_id} extended by 30 days")
        else:
            logger.warning(f"Subscription {subscription_id} not found in database")
    
    elif event_type == 'customer.subscription.deleted':
        # Subscription was cancelled
        subscription_data = event['data']['object']
        subscription_id = subscription_data['id']
        
        # Find subscription in database
        subscription = db.query(Subscription).filter(
            Subscription.payment_id == subscription_id
        ).first()
        
        if subscription:
            # Mark subscription as not auto-renewing
            subscription.auto_renew = False
            db.commit()
            logger.info(f"Subscription {subscription_id} marked as not auto-renewing")
        else:
            logger.warning(f"Subscription {subscription_id} not found in database")
    
    return {"success": True}

@app.get("/subscription/status", response_model=SubscriptionResponse)
async def get_subscription_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Get user's subscription
    subscription = db.query(Subscription).filter(
        Subscription.user_id == current_user.id
    ).first()
    
    if not subscription:
        return {
            "tier": "free",
            "status": "active",
            "limits": SUBSCRIPTION_LIMITS["free"],
            "usage": {"clean": 0, "unclean": 0},
            "remaining": {"clean": SUBSCRIPTION_LIMITS["free"]["clean"], "unclean": SUBSCRIPTION_LIMITS["free"]["unclean"]}
        }

    # Check if subscription is expired
    if subscription.expiry_date < datetime.now():
        status = "expired"
        tier = "free"
    else:
        status = "active"
        tier = subscription.tier
    
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
    
    return {
        "tier": tier,
        "status": status,
        "expiry_date": subscription.expiry_date.isoformat() if subscription.expiry_date else None,
        "limits": SUBSCRIPTION_LIMITS[tier],
        "usage": {
            "clean": clean_usage,
            "unclean": unclean_usage
        },
        "remaining": {
            "clean": max(0, SUBSCRIPTION_LIMITS[tier]["clean"] - clean_usage) if SUBSCRIPTION_LIMITS[tier]["clean"] != float('inf') else "unlimited",
            "unclean": max(0, SUBSCRIPTION_LIMITS[tier]["unclean"] - unclean_usage) if SUBSCRIPTION_LIMITS[tier]["unclean"] != float('inf') else "unlimited",
        }
    }

# Add healthcheck endpoint
@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok", "version": "1.0.0"}