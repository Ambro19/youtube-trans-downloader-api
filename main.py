# main.py - Enhanced with Complete Stripe Payment Integration

from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
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
import secrets

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

# CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

#===========================================================================

# Add this to your main.py - Production-ready CORS configuration

# Environment-aware configuration
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# Configure CORS based on environment
if ENVIRONMENT == "production":
    allowed_origins = [
        #"https://your-actual-frontend-domain.com",  # Replace with your actual frontend URL
        
        "http://localhost:8000", #Or "http://localhost:3000"?
       # "https://youtube-transcript-downloader.netlify.app",  # Example
       "https://youtube-trans-downloader-api.onrender.com",
        FRONTEND_URL  # From environment variable
    ]
    logger.info(f"🌍 Production mode - CORS origins: {allowed_origins}")
else:
    allowed_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        FRONTEND_URL
    ]
    logger.info(f"🔧 Development mode - CORS origins: {allowed_origins}")

# CORS MIDDLEWARE with environment awareness
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
#==================

# Authentication setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Constants
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

# Enhanced subscription limits with new action types
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

# # Price ID mapping
# PRICE_ID_MAP = {
#     "pro": os.getenv("PRO_PRICE_ID") or os.getenv("STRIPE_PRO_PRICE_ID"),
#     "premium": os.getenv("PREMIUM_PRICE_ID") or os.getenv("STRIPE_PREMIUM_PRICE_ID")
# }

#=================

# Price ID mapping - UPDATED to use your standardized variable names
PRICE_ID_MAP = {
    "pro": os.getenv("PRO_PRICE_ID"),
    "premium": os.getenv("PREMIUM_PRICE_ID")
}

# Plan pricing in cents
PLAN_PRICING = {
    "pro": 999,  # $9.99
    "premium": 1999  # $19.99
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
                # Log first few characters (for debugging)
                logger.info(f"✅ {var}: {value[:8]}..." if len(value) > 8 else f"✅ {var}: SET")
        
        if missing_vars:
            logger.error(f"❌ Missing required environment variables:")
            for var in missing_vars:
                logger.error(f"   - {var}")
            raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")
        
        # 🔧 UPDATED: Optional variables with correct names
        optional_vars = {
            "PRO_PRICE_ID": "Pro plan price ID",
            "PREMIUM_PRICE_ID": "Premium plan price ID", 
            "STRIPE_WEBHOOK_SECRET": "Webhook verification"
        }
        
        for var, description in optional_vars.items():
            if not os.getenv(var):
                logger.warning(f"⚠️  {var} not set - {description} will not work")
            else:
                logger.info(f"✅ {var}: SET")
        
        # Initialize database
        create_tables()
        logger.info("✅ Database initialized successfully")
        logger.info("🎉 Application startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")
        raise

#============================================
# Enhanced Pydantic models
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

class PaymentRequest(BaseModel):
    token: str
    subscription_tier: str

# NEW: Enhanced payment models
# 🔧 UPDATED: Payment models to match your new implementation
class CreatePaymentIntentRequest(BaseModel):
    price_id: str  # 🔧 SIMPLIFIED: Only price_id needed

class ConfirmPaymentRequest(BaseModel):
    payment_intent_id: str


class PaymentIntentRequest(BaseModel):
    amount: int  # Amount in cents
    currency: str = 'usd'
    payment_method_id: str
    plan_name: str

# class PaymentIntentResponse(BaseModel):
#     client_secret: str
#     token: str

class PaymentIntentResponse(BaseModel):
    client_secret: str
    payment_intent_id: str  # 🔧 UPDATED: Changed from token to payment_intent_id


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

# Helper functions (existing ones kept)
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

# NEW: Enhanced payment helper functions
def get_or_create_stripe_customer(user, db: Session):
    """Get or create a Stripe customer for the user"""
    try:
        # Check if user has stripe_customer_id attribute (from new User model)
        if hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
            try:
                customer = stripe.Customer.retrieve(user.stripe_customer_id)
                return customer
            except stripe.error.InvalidRequestError:
                pass
        
        # Create new customer
        customer = stripe.Customer.create(
            email=user.email,
            name=user.username,
            metadata={'user_id': str(user.id)}
        )
        
        # Save customer ID if user model supports it
        if hasattr(user, 'stripe_customer_id'):
            user.stripe_customer_id = customer.id
            db.commit()
        
        return customer
        
    except Exception as e:
        logger.error(f"Error creating Stripe customer: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create payment customer"
        )

def check_user_limits(user, action_type: str, db: Session):
    """Check if user has exceeded their limits for the current month"""
    # Get subscription tier
    subscription = db.query(Subscription).filter(Subscription.user_id == user.id).first()
    
    if not subscription or subscription.expiry_date < datetime.now():
        tier = "free"
    else:
        tier = subscription.tier
    
    # Get current month's usage count from TranscriptDownload table
    month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    # Map action types to transcript types
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

def increment_usage(user, action_type: str, db: Session):
    """Increment user's usage counter by recording a download"""
    # Map action types to transcript types
    type_mapping = {
        "clean_transcripts": "clean",
        "unclean_transcripts": "unclean", 
        "audio_downloads": "audio",
        "video_downloads": "video"
    }
    
    transcript_type = type_mapping.get(action_type, action_type)
    
    # Record the download in TranscriptDownload table
    new_download = TranscriptDownload(
        user_id=user.id,
        youtube_id="usage_increment",  # Placeholder for usage tracking
        transcript_type=transcript_type,
        created_at=datetime.now()
    )
    
    db.add(new_download)
    db.commit()

def check_subscription_limit(user_id: int, transcript_type: str, db: Session):
    """Original function maintained for backward compatibility"""
    subscription = db.query(Subscription).filter(Subscription.user_id == user_id).first()
    
    if not subscription:
        tier = "free"
    else:
        tier = subscription.tier
        if subscription.expiry_date < datetime.now():
            tier = "free"
    
    month_start = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    usage = db.query(TranscriptDownload).filter(
        TranscriptDownload.user_id == user_id,
        TranscriptDownload.transcript_type == transcript_type,
        TranscriptDownload.created_at >= month_start
    ).count()
    
    limit = SUBSCRIPTION_LIMITS[tier][transcript_type]
    if usage >= limit:
        return False
    return True

def process_youtube_transcript(youtube_id: str, clean: bool):
    """Original transcript processing function"""
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(youtube_id)
        
        if clean:
            full_text = " ".join([item['text'] for item in transcript_list])
            return full_text
        else:
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

# API Endpoints (existing ones kept, new ones added)

@app.get("/")
async def root():
    return {"message": "YouTube Transcript Downloader API", "status": "running", "version": "1.0.0"}

@app.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user(db, user_data.username)
    if db_user:
        logger.warning(f"Registration attempt with existing username: {user_data.username}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
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
    
    can_download = check_subscription_limit(user.id, transcript_type, db)
    if not can_download:
        logger.warning(f"User {user.username} reached subscription limit for {transcript_type} transcripts")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"You've reached your monthly limit for {transcript_type} transcripts. Please upgrade your subscription."
        )
    
    transcript_text = process_youtube_transcript(
        request.youtube_id, 
        clean=request.clean_transcript
    )
    
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
    
    return {"transcript": transcript_text, "youtube_id": request.youtube_id}

# 🔧 UPDATED: Enhanced payment intent endpoint to match payment.py
@app.post("/create_payment_intent/")
async def create_payment_intent_endpoint(
    request: CreatePaymentIntentRequest,  # 🔧 UPDATED: Use new simple model
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a payment intent for subscription upgrade - UPDATED VERSION"""
    try:
        logger.info(f"Creating payment intent for user {current_user.id} with price_id: {request.price_id}")
        
        # Validate price_id using your standardized variable names
        valid_price_ids = [
            os.getenv("PRO_PRICE_ID"),
            os.getenv("PREMIUM_PRICE_ID")
        ]
        
        if request.price_id not in valid_price_ids:
            logger.error(f"Invalid price ID: {request.price_id}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid price ID: {request.price_id}"
            )

        # Get the price from Stripe
        try:
            price = stripe.Price.retrieve(request.price_id)
            logger.info(f"Retrieved price: {price.unit_amount} {price.currency}")
        except stripe.error.InvalidRequestError as e:
            logger.error(f"Invalid Stripe price ID: {request.price_id}, error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid Stripe price ID: {request.price_id}"
            )
        
        # Determine plan type using your standardized variable names
        plan_type = 'pro' if request.price_id == os.getenv("PRO_PRICE_ID") else 'premium'
        logger.info(f"Plan type: {plan_type}")
        
        # Get or create Stripe customer
        customer = get_or_create_stripe_customer(current_user, db)
        logger.info(f"Stripe customer: {customer.id}")
        
        # 🔧 FIXED: Create PaymentIntent with proper configuration
        intent = stripe.PaymentIntent.create(
            amount=price.unit_amount,  # Amount in cents
            currency=price.currency,
            customer=customer.id,
            automatic_payment_methods={
                'enabled': True,
                'allow_redirects': 'never'  # 🔧 THIS FIXES THE STRIPE REDIRECT ERROR!
            },
            metadata={
                'user_id': str(current_user.id),
                'user_email': current_user.email,
                'price_id': request.price_id,
                'plan_type': plan_type
            }
        )

        logger.info(f"✅ Payment intent created successfully: {intent.id}")

        return {
            'client_secret': intent.client_secret,
            'payment_intent_id': intent.id,
            'amount': price.unit_amount,
            'currency': price.currency,
            'plan_type': plan_type
        }

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stripe error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Payment intent creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create payment intent: {str(e)}"
        )

#===================== Newly Added ==============================

# 🔧 NEW: Add confirm payment endpoint to main.py
@app.post("/confirm_payment/")
async def confirm_payment_endpoint(
    request: ConfirmPaymentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Confirm payment and update user subscription"""
    try:
        logger.info(f"Confirming payment for user {current_user.id} with payment_intent: {request.payment_intent_id}")
        
        # Retrieve the PaymentIntent from Stripe
        intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
        
        if intent.status != 'succeeded':
            logger.error(f"Payment not completed. Status: {intent.status}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Payment not completed. Status: {intent.status}"
            )

        # Update user subscription in database
        user_subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id
        ).first()

        plan_type = intent.metadata.get('plan_type', 'pro')

        if not user_subscription:
            # Create new subscription record
            user_subscription = Subscription(
                user_id=current_user.id,
                tier=plan_type,
                status='active',
                stripe_payment_intent_id=request.payment_intent_id,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(days=30)
            )
            db.add(user_subscription)
        else:
            # Update existing subscription
            user_subscription.tier = plan_type
            user_subscription.status = 'active'
            user_subscription.stripe_payment_intent_id = request.payment_intent_id
            user_subscription.expires_at = datetime.utcnow() + timedelta(days=30)

        db.commit()
        db.refresh(user_subscription)

        logger.info(f"✅ User {current_user.id} subscription updated to {plan_type}")

        return {
            'success': True,
            'subscription_tier': user_subscription.tier,
            'expires_at': user_subscription.expires_at.isoformat(),
            'status': 'active'
        }

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error during confirmation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Stripe error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Payment confirmation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to confirm payment: {str(e)}"
        )

#==============================================================
# ENHANCED: Updated create_subscription endpoint
@app.post("/create_subscription/")
async def create_subscription_enhanced(
    request: SubscriptionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enhanced subscription creation with proper Stripe integration"""
    if request.subscription_tier not in PRICE_ID_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid subscription tier. Must be one of: {', '.join(PRICE_ID_MAP.keys())}"
        )
        
    try:
        # Get or create Stripe customer
        customer = get_or_create_stripe_customer(current_user, db)
        
        # Create Stripe subscription
        subscription = stripe.Subscription.create(
            customer=customer.id,
            items=[{
                'price': PRICE_ID_MAP[request.subscription_tier],
            }],
            metadata={
                'user_id': str(current_user.id),
                'username': current_user.username,
                'plan_name': request.subscription_tier
            }
        )
        
        # Update or create subscription in database
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
            "subscription_id": subscription.id,
            "status": subscription.status,
            "current_period_end": subscription.current_period_end,
            "tier": request.subscription_tier
        }
    
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error during subscription creation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create subscription"
        )
    except Exception as e:
        db.rollback()
        logger.error(f"Subscription creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process subscription"
        )

# NEW: Cancel subscription endpoint
@app.post("/cancel_subscription/")
async def cancel_subscription(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Cancel user's current subscription"""
    try:
        # Get user's subscription
        subscription = db.query(Subscription).filter(
            Subscription.user_id == current_user.id
        ).first()
        
        if not subscription or not subscription.payment_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No active subscription found"
            )
        
        # Cancel subscription in Stripe
        stripe_subscription = stripe.Subscription.modify(
            subscription.payment_id,
            cancel_at_period_end=True
        )
        
        # Update database
        subscription.auto_renew = False
        db.commit()
        
        logger.info(f"User {current_user.username} cancelled subscription {subscription.payment_id}")
        
        return {
            "message": "Subscription cancelled successfully",
            "will_expire_at": stripe_subscription.current_period_end
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error during cancellation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel subscription"
        )
    except Exception as e:
        logger.error(f"Subscription cancellation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cancel subscription"
        )

# ENHANCED: Updated subscription status endpoint
@app.get("/subscription_status/")
async def get_subscription_status_enhanced(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Enhanced subscription status with detailed usage info"""
    try:
        # Get user's subscription
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
        
        audio_usage = db.query(TranscriptDownload).filter(
            TranscriptDownload.user_id == current_user.id,
            TranscriptDownload.transcript_type == "audio",
            TranscriptDownload.created_at >= month_start
        ).count()
        
        video_usage = db.query(TranscriptDownload).filter(
            TranscriptDownload.user_id == current_user.id,
            TranscriptDownload.transcript_type == "video",
            TranscriptDownload.created_at >= month_start
        ).count()
        
        # Get limits based on tier
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
                "audio_downloads": audio_usage,
                "video_downloads": video_usage
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

# EXISTING: Keep original subscription status endpoint for backward compatibility
@app.get("/subscription/status", response_model=SubscriptionResponse)
async def get_subscription_status(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Original subscription status endpoint (maintained for compatibility)"""
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

    if subscription.expiry_date < datetime.now():
        status = "expired"
        tier = "free"
    else:
        status = "active"
        tier = subscription.tier
    
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

# EXISTING: Webhook handler (enhanced)
@app.post("/webhook", status_code=200)
async def webhook_received(request: Request, response: Response, db: Session = Depends(get_db)):
    """Enhanced webhook handler"""
    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature")
    
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, endpoint_secret
        )
    except ValueError as e:
        logger.warning(f"Invalid webhook payload: {str(e)}")
        response.status_code = 400
        return {"error": str(e)}
    except stripe.error.SignatureVerificationError as e:
        logger.warning(f"Invalid webhook signature: {str(e)}")
        response.status_code = 400
        return {"error": str(e)}
    
    event_type = event['type']
    logger.info(f"Received Stripe webhook event: {event_type}")
    
    if event_type == 'invoice.payment_succeeded':
        invoice = event['data']['object']
        subscription_id = invoice['subscription']
        
        subscription = db.query(Subscription).filter(
            Subscription.payment_id == subscription_id
        ).first()
        
        if subscription:
            subscription.expiry_date = datetime.now() + timedelta(days=30)
            db.commit()
            logger.info(f"Subscription {subscription_id} extended by 30 days")
    
    elif event_type == 'customer.subscription.deleted':
        subscription_data = event['data']['object']
        subscription_id = subscription_data['id']
        
        subscription = db.query(Subscription).filter(
            Subscription.payment_id == subscription_id
        ).first()
        
        if subscription:
            subscription.auto_renew = False
            db.commit()
            logger.info(f"Subscription {subscription_id} marked as not auto-renewing")
    
    return {"success": True}

# NEW: Alternative webhook endpoint for new payment system
@app.post("/stripe_webhook/")
async def stripe_webhook_enhanced(request: Request, db: Session = Depends(get_db)):
    """Enhanced webhook endpoint for new payment system"""
    try:
        payload = await request.body()
        sig_header = request.headers.get("Stripe-Signature")
        endpoint_secret = os.getenv("STRIPE_WEBHOOK_SECRET")
        
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, endpoint_secret
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid payload")
        except stripe.error.SignatureVerificationError:
            raise HTTPException(status_code=400, detail="Invalid signature")
        
        # Handle the event
        if event['type'] == 'invoice.payment_succeeded':
            subscription_id = event['data']['object']['subscription']
            # Update user subscription status in database
            subscription = db.query(Subscription).filter(
                Subscription.payment_id == subscription_id
            ).first()
            if subscription:
                subscription.expiry_date = datetime.now() + timedelta(days=30)
                db.commit()
            
        elif event['type'] == 'invoice.payment_failed':
            subscription_id = event['data']['object']['subscription']
            # Handle failed payment
            logger.warning(f"Payment failed for subscription {subscription_id}")
            
        elif event['type'] == 'customer.subscription.deleted':
            subscription_id = event['data']['object']['id']
            # Downgrade user to free tier
            subscription = db.query(Subscription).filter(
                Subscription.payment_id == subscription_id
            ).first()
            if subscription:
                subscription.auto_renew = False
                db.commit()
        
        return {"status": "success"}
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Webhook processing failed: {str(e)}"
        )

# NEW: Health check endpoint
@app.get("/health/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "stripe_configured": bool(os.getenv("STRIPE_SECRET_KEY")),
        "timestamp": datetime.utcnow().isoformat()
    }

# EXISTING: Healthcheck endpoint (maintained for compatibility)
@app.get("/healthcheck")
async def healthcheck():
    return {"status": "ok", "version": "1.0.0"}