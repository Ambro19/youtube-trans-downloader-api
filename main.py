from fastapi import FastAPI, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional
import jwt
from pydantic import BaseModel
from passlib.context import CryptContext
import stripe
import youtube_transcript_api
from youtube_transcript_api import YouTubeTranscriptApi

# Import from database.py
from database import get_db, User, Subscription, TranscriptDownload, create_tables

# Create FastAPI app
app = FastAPI(title="YouTubeTransDownloader API")

# Authentication setup
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Constants
SECRET_KEY = "your-secure-secret-key-change-this-in-production"  # Store in environment variables in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours

# Subscription tiers and limits
SUBSCRIPTION_LIMITS = {
    "free": {"unclean": 5, "clean": 0},
    "basic": {"unclean": 30, "clean": 10},
    "premium": {"unclean": float('inf'), "clean": 50}
}

# Stripe setup (payment processing)
stripe.api_key = "your-stripe-secret-key"  # Store in environment variables

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    create_tables()

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
        orm_mode = True

class TranscriptRequest(BaseModel):
    youtube_id: str
    clean_transcript: bool = False

class PaymentRequest(BaseModel):
    token: str
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
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transcripts are disabled for this video"
        )
    except youtube_transcript_api._errors.NoTranscriptFound:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No transcript found for this video"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving transcript: {str(e)}"
        )

# API Endpoints
@app.post("/register", response_model=UserResponse)
def register_user(user_data: UserCreate, db: Session = Depends(get_db)):
    db_user = get_user(db, user_data.username)
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    # Check if email exists
    email_exists = db.query(User).filter(User.email == user_data.email).first()
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
    
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user

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
    db.add(new_download)
    db.commit()
    
    # Return transcript data
    return {"transcript": transcript_text, "youtube_id": request.youtube_id}

@app.post("/create_subscription/")
async def create_subscription(
    request: PaymentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        # Create Stripe customer & subscription
        customer = stripe.Customer.create(
            source=request.token,
            email=current_user.email
        )
        
        # Map your tiers to Stripe price IDs
        price_id_map = {
            "basic": "price_basic_id_from_stripe",  # Replace with actual Stripe price IDs
            "premium": "price_premium_id_from_stripe"  # Replace with actual Stripe price IDs
        }
        
        subscription = stripe.Subscription.create(
            customer=customer.id,
            items=[{"price": price_id_map[request.subscription_tier]}]
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
        
        return {"status": "success", "subscription_id": subscription.id, "tier": request.subscription_tier}
    
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/subscription/status")
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
            "limits": SUBSCRIPTION_LIMITS["free"]
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

# Run the application with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)