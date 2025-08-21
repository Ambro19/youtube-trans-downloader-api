# backend/models.py
from datetime import datetime
import os
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

# --- SQLAlchemy base/engine/session -----------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

# --- Models ------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)

    # subscription + usage fields expected by main.py
    subscription_tier = Column(String(50), default="free")   # free | pro | premium
    stripe_customer_id = Column(String(255), nullable=True)

    usage_clean_transcripts = Column(Integer, default=0)
    usage_unclean_transcripts = Column(Integer, default=0)
    usage_audio_downloads = Column(Integer, default=0)
    usage_video_downloads = Column(Integer, default=0)
    usage_reset_date = Column(DateTime, default=datetime.utcnow)

    subscriptions = relationship("Subscription", back_populates="user")
    downloads = relationship("TranscriptDownload", back_populates="user")

    def __repr__(self):
        return f"<User id={self.id} username={self.username!r} tier={self.subscription_tier!r}>"

class Subscription(Base):
    __tablename__ = "subscriptions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    tier = Column(String(50), nullable=False, default="free")
    status = Column(String(50), default="active")

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)

    stripe_subscription_id = Column(String(255), nullable=True)
    stripe_payment_intent_id = Column(String(255), nullable=True)
    stripe_customer_id = Column(String(255), nullable=True)

    price_paid = Column(Float, nullable=True)
    currency = Column(String(10), default="usd")
    extra_data = Column(Text, nullable=True)  # JSON string

    user = relationship("User", back_populates="subscriptions")

class TranscriptDownload(Base):
    __tablename__ = "transcript_downloads"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    youtube_id = Column(String(20), index=True, nullable=False)

    transcript_type = Column(String(50), nullable=False)  # clean_transcripts | unclean_transcripts | audio_downloads | video_downloads
    quality = Column(String(20), nullable=True)
    file_format = Column(String(10), nullable=True)
    file_size = Column(Integer, nullable=True)

    processing_time = Column(Float, nullable=True)
    status = Column(String(20), default="completed")
    error_message = Column(Text, nullable=True)

    language = Column(String(10), default="en")
    video_title = Column(String(500), nullable=True)
    video_uploader = Column(String(255), nullable=True)
    video_duration = Column(Integer, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="downloads")

# --- Helpers used by main.py -------------------------------------------------
def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def initialize_database() -> bool:
    try:
        Base.metadata.create_all(bind=engine)
        return True
    except Exception:
        return False

def create_download_record_safe(
    db: Session, user_id: int, download_type: str, youtube_id: str, **kw
) -> Optional[TranscriptDownload]:
    try:
        rec = TranscriptDownload(
            user_id=user_id,
            youtube_id=youtube_id,
            transcript_type=download_type,
            quality=kw.get("quality", "default"),
            file_format=kw.get("file_format", "txt"),
            file_size=kw.get("file_size", 0),
            processing_time=kw.get("processing_time", 0.0),
            video_title=kw.get("video_title"),
            video_uploader=kw.get("video_uploader"),
            video_duration=kw.get("video_duration"),
            status="completed",
            created_at=datetime.utcnow(),
        )
        db.add(rec); db.commit(); db.refresh(rec)
        return rec
    except Exception:
        db.rollback()
        return None

__all__ = [
    "User", "Subscription", "TranscriptDownload",
    "engine", "SessionLocal", "Base",
    "get_db", "initialize_database", "create_download_record_safe",
]





# # backend/models.py
# from datetime import datetime
# import os
# from typing import Optional

# from sqlalchemy import (
#     create_engine, Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
# )
# from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship

# # --- SQLAlchemy base/engine/session -----------------------------------------
# DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")
# engine = create_engine(
#     DATABASE_URL,
#     connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
#     echo=False,
# )
# SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
# Base = declarative_base()

# # --- Models ------------------------------------------------------------------
# class User(Base):
#     __tablename__ = "users"
#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String(100), unique=True, index=True, nullable=False)
#     email = Column(String(255), unique=True, index=True, nullable=False)
#     hashed_password = Column(String(255), nullable=False)

#     created_at = Column(DateTime, default=datetime.utcnow)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

#     is_active = Column(Boolean, default=True)
#     is_verified = Column(Boolean, default=False)

#     # subscription + usage fields expected by main.py
#     subscription_tier = Column(String(50), default="free")   # free | pro | premium
#     stripe_customer_id = Column(String(255), nullable=True)

#     usage_clean_transcripts = Column(Integer, default=0)
#     usage_unclean_transcripts = Column(Integer, default=0)
#     usage_audio_downloads = Column(Integer, default=0)
#     usage_video_downloads = Column(Integer, default=0)
#     usage_reset_date = Column(DateTime, default=datetime.utcnow)

#     subscriptions = relationship("Subscription", back_populates="user")
#     downloads = relationship("TranscriptDownload", back_populates="user")

#     def __repr__(self):
#         return f"<User id={self.id} username={self.username!r} tier={self.subscription_tier!r}>"

# class Subscription(Base):
#     __tablename__ = "subscriptions"
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
#     tier = Column(String(50), nullable=False, default="free")
#     status = Column(String(50), default="active")

#     created_at = Column(DateTime, default=datetime.utcnow)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
#     expires_at = Column(DateTime, nullable=True)
#     cancelled_at = Column(DateTime, nullable=True)

#     stripe_subscription_id = Column(String(255), nullable=True)
#     stripe_payment_intent_id = Column(String(255), nullable=True)
#     stripe_customer_id = Column(String(255), nullable=True)

#     price_paid = Column(Float, nullable=True)
#     currency = Column(String(10), default="usd")
#     extra_data = Column(Text, nullable=True)  # JSON string

#     user = relationship("User", back_populates="subscriptions")

# class TranscriptDownload(Base):
#     __tablename__ = "transcript_downloads"
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
#     youtube_id = Column(String(20), index=True, nullable=False)

#     transcript_type = Column(String(50), nullable=False)  # clean_transcripts | unclean_transcripts | audio_downloads | video_downloads
#     quality = Column(String(20), nullable=True)
#     file_format = Column(String(10), nullable=True)
#     file_size = Column(Integer, nullable=True)

#     processing_time = Column(Float, nullable=True)
#     status = Column(String(20), default="completed")
#     error_message = Column(Text, nullable=True)

#     language = Column(String(10), default="en")
#     video_title = Column(String(500), nullable=True)
#     video_uploader = Column(String(255), nullable=True)
#     video_duration = Column(Integer, nullable=True)

#     created_at = Column(DateTime, default=datetime.utcnow, index=True)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

#     user = relationship("User", back_populates="downloads")

# # --- Helpers used by main.py -------------------------------------------------
# def get_db() -> Session:
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# def initialize_database() -> bool:
#     try:
#         Base.metadata.create_all(bind=engine)
#         return True
#     except Exception:
#         return False

# def create_download_record_safe(
#     db: Session, user_id: int, download_type: str, youtube_id: str, **kw
# ) -> Optional[TranscriptDownload]:
#     try:
#         rec = TranscriptDownload(
#             user_id=user_id,
#             youtube_id=youtube_id,
#             transcript_type=download_type,
#             quality=kw.get("quality", "default"),
#             file_format=kw.get("file_format", "txt"),
#             file_size=kw.get("file_size", 0),
#             processing_time=kw.get("processing_time", 0.0),
#             video_title=kw.get("video_title"),
#             video_uploader=kw.get("video_uploader"),
#             video_duration=kw.get("video_duration"),
#             status="completed",
#             created_at=datetime.utcnow(),
#         )
#         db.add(rec); db.commit(); db.refresh(rec)
#         return rec
#     except Exception:
#         db.rollback()
#         return None

# __all__ = [
#     "User", "Subscription", "TranscriptDownload",
#     "engine", "SessionLocal", "Base",
#     "get_db", "initialize_database", "create_download_record_safe",
# ]


#================= check main.py file ... ===================

# # payment.py — resilient Stripe price resolution + server-side confirm
# # Production-safe: blocks redirect payment methods (no return_url needed),
# # but still supplies a return_url just in case your dashboard enables them.

# import os
# import logging
# from typing import Optional, Tuple, Dict

# import stripe
# from fastapi import APIRouter, Depends, HTTPException
# from pydantic import BaseModel

# logger = logging.getLogger("payment")

# # Keep paths exactly as your frontend expects (no prefix)
# router = APIRouter(tags=["payments"])

# # === Stripe init =============================================================

# STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
# if STRIPE_SECRET_KEY:
#     stripe.api_key = STRIPE_SECRET_KEY
#     logger.info("✅ Stripe configured successfully")
# else:
#     logger.error("❌ STRIPE_SECRET_KEY missing. Billing will run in demo mode.")

# def stripe_mode() -> str:
#     k = STRIPE_SECRET_KEY or ""
#     if k.startswith("sk_live_"):
#         return "live"
#     if k.startswith("sk_test_"):
#         return "test"
#     return "demo"

# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# # === Config / names / lookup keys ===========================================

# ENV_PRO_PRICE_ID = os.getenv("STRIPE_PRO_PRICE_ID")
# ENV_PREMIUM_PRICE_ID = os.getenv("STRIPE_PREMIUM_PRICE_ID")

# PRO_LOOKUP_KEY = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
# PREMIUM_LOOKUP_KEY = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")

# PRO_PRODUCT_NAME = os.getenv("STRIPE_PRO_PRODUCT_NAME", "Pro Plan")
# PREMIUM_PRODUCT_NAME = os.getenv("STRIPE_PREMIUM_PRODUCT_NAME", "Premium Plan")

# # Cache so we don’t hit Stripe on every request
# _PRICE_CACHE: Dict[str, Optional[str]] = {"pro": None, "premium": None}

# # === Request bodies ==========================================================

# class CreateIntentBody(BaseModel):
#     plan_type: Optional[str] = None  # "pro" | "premium"
#     price_id: Optional[str] = None

# class ConfirmBody(BaseModel):
#     payment_intent_id: str

# # === Helpers =================================================================

# def _safe_retrieve_price(price_id: str) -> Optional[str]:
#     try:
#         p = stripe.Price.retrieve(price_id)
#         return p["id"]
#     except Exception as e:
#         logger.error(f"Stripe couldn't find price '{price_id}': {e}")
#         return None

# def _find_by_lookup_key(lookup_key: str) -> Optional[str]:
#     try:
#         prices = stripe.Price.list(active=True, lookup_keys=[lookup_key], limit=1)
#         if prices.data:
#             return prices.data[0]["id"]
#     except Exception as e:
#         logger.warning(f"Lookup by key '{lookup_key}' failed: {e}")
#     return None

# def _find_by_product_name(name: str) -> Optional[str]:
#     try:
#         prods = stripe.Product.list(active=True, limit=100, expand=["data.default_price"])
#         name_lc = name.strip().lower()
#         for p in prods.auto_paging_iter():
#             if str(p["name"]).strip().lower() == name_lc:
#                 dp = p.get("default_price")
#                 if isinstance(dp, dict):
#                     return dp.get("id")
#                 if isinstance(dp, str):
#                     return dp
#     except Exception as e:
#         logger.warning(f"Product lookup by name '{name}' failed: {e}")
#     return None

# def resolve_price_id(plan_type: str, preferred: Optional[str]) -> Tuple[Optional[str], str]:
#     plan = (plan_type or "").strip().lower()
#     if plan not in ("pro", "premium"):
#         raise HTTPException(status_code=400, detail="Invalid plan_type. Use 'pro' or 'premium'.")

#     cached = _PRICE_CACHE.get(plan)
#     if cached:
#         return cached, "cache"

#     if preferred:
#         ok = _safe_retrieve_price(preferred)
#         if ok:
#             _PRICE_CACHE[plan] = ok
#             return ok, "client"

#     env_id = ENV_PRO_PRICE_ID if plan == "pro" else ENV_PREMIUM_PRICE_ID
#     if env_id:
#         ok = _safe_retrieve_price(env_id)
#         if ok:
#             _PRICE_CACHE[plan] = ok
#             return ok, "env"

#     lk = PRO_LOOKUP_KEY if plan == "pro" else PREMIUM_LOOKUP_KEY
#     ok = _find_by_lookup_key(lk)
#     if ok:
#         _PRICE_CACHE[plan] = ok
#         return ok, "lookup_key"

#     prod_name = PRO_PRODUCT_NAME if plan == "pro" else PREMIUM_PRODUCT_NAME
#     ok = _find_by_product_name(prod_name)
#     if ok:
#         _PRICE_CACHE[plan] = ok
#         return ok, "product_name"

#     return None, "unavailable"

# def _demo_mode() -> bool:
#     return stripe_mode() == "demo" or not STRIPE_SECRET_KEY

# # === Public endpoints ========================================================

# @router.get("/billing/config")
# def billing_config():
#     if _demo_mode():
#         return {
#             "mode": stripe_mode(),
#             "is_demo": True,
#             "prices": {"pro": None, "premium": None},
#             "source": "demo",
#         }

#     pro_id, pro_src = resolve_price_id("pro", None)
#     prem_id, prem_src = resolve_price_id("premium", None)

#     is_demo = not (pro_id and prem_id)
#     return {
#         "mode": stripe_mode(),
#         "is_demo": is_demo,
#         "prices": {"pro": pro_id, "premium": prem_id},
#         "source": {"pro": pro_src, "premium": prem_src},
#     }

# def _require_stripe_ready():
#     if _demo_mode():
#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 "Stripe is not fully configured (missing STRIPE_SECRET_KEY or prices). "
#                 "Set up prices or enable demo upgrade endpoint."
#             ),
#         )

# @router.post("/create_payment_intent/")
# def create_payment_intent(body: CreateIntentBody, user=Depends(lambda: None)):
#     """
#     Accepts either:
#       - plan_type: "pro" | "premium"
#       - price_id: preferred price id (optional)
#     """
#     _require_stripe_ready()

#     plan = (body.plan_type or "").strip().lower()
#     if not plan:
#         plan = "pro" if "pro" in (body.price_id or "").lower() else "premium"

#     price_id, src = resolve_price_id(plan, body.price_id)
#     if not price_id:
#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 "Unable to resolve a valid Stripe price for this plan. "
#                 "Ensure your Stripe products exist (Pro Plan / Premium Plan) "
#                 "or set STRIPE_PRO_PRICE_ID / STRIPE_PREMIUM_PRICE_ID."
#             ),
#         )

#     logger.info(f"🔥 Creating PaymentIntent for plan={plan} using price_id={price_id} [source={src}]")

#     price = stripe.Price.retrieve(price_id)
#     if not price.get("unit_amount") or not price.get("currency"):
#         raise HTTPException(status_code=400, detail="Stripe price is missing amount/currency.")

#     # IMPORTANT: Disallow redirects so no return_url is required
#     intent = stripe.PaymentIntent.create(
#         amount=price["unit_amount"],
#         currency=price["currency"],
#         automatic_payment_methods={"enabled": True, "allow_redirects": "never"},
#         metadata={"plan": plan},
#         # Optional: make test/QA easier to spot
#         description=f"YouTube Content Downloader – {plan.capitalize()} Plan",
#     )

#     return {"payment_intent_id": intent["id"]}

# @router.post("/confirm_payment/")
# def confirm_payment(body: ConfirmBody):
#     """
#     Server-side confirm with a universal test payment method.
#     In live mode you'd confirm with the PaymentMethod collected by Stripe Elements.
#     """
#     _require_stripe_ready()

#     try:
#         pi = stripe.PaymentIntent.confirm(
#             body.payment_intent_id,
#             payment_method="pm_card_visa",  # test payment method (no 3DS)
#             return_url=f"{FRONTEND_URL}/subscription?paid=1",
#         )
#         if pi["status"] != "succeeded":
#             # If your dashboard forces next_action, you’ll see it here instead of an error.
#             return {"status": pi["status"], "client_secret": pi.get("client_secret")}
#         return {"status": "succeeded", "client_secret": pi.get("client_secret")}
#     except stripe.error.StripeError as e:
#         logger.error(f"Stripe confirm error: {e}")
#         raise HTTPException(status_code=400, detail=str(e))

# # Back-compat: if you renamed the user model, expose it as `User` so
# # `from models import User` keeps working.
# if 'User' not in globals():
#     for _name in ('AppUser', 'Account', 'Users', 'AuthUser'):
#         if _name in globals():
#             User = globals()[_name]  # alias
#             break


# # models.py - COMPLETE DATABASE MODELS with Subscription and TranscriptDownload
# # 🔥 FIXES:
# # - ✅ Added missing Subscription model
# # - ✅ Added missing TranscriptDownload model for history tracking
# # - ✅ Updated User model with subscription fields
# # - ✅ Complete database schema for all functionality

# from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, ForeignKey
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.orm import sessionmaker, Session, relationship
# from sqlalchemy import create_engine
# from datetime import datetime
# import logging
# import os

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Create declarative base
# Base = declarative_base()

# # =============================================================================
# # USER MODEL
# # =============================================================================

# class User(Base):
#     __tablename__ = "users"
    
#     # Primary fields
#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String(100), unique=True, index=True, nullable=False)
#     email = Column(String(255), unique=True, index=True, nullable=False)
#     hashed_password = Column(String(255), nullable=False)
    
#     # Timestamps
#     created_at = Column(DateTime, default=datetime.utcnow)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
#     # Account status
#     is_active = Column(Boolean, default=True)
#     is_verified = Column(Boolean, default=False)
    
#     # 🔥 Subscription fields
#     subscription_tier = Column(String(50), default='free')  # 'free', 'pro', 'premium'
#     stripe_customer_id = Column(String(255), nullable=True)
    
#     # 🔥 Usage tracking fields (monthly)
#     usage_clean_transcripts = Column(Integer, default=0)
#     usage_unclean_transcripts = Column(Integer, default=0)
#     usage_audio_downloads = Column(Integer, default=0)
#     usage_video_downloads = Column(Integer, default=0)
#     usage_reset_date = Column(DateTime, default=datetime.utcnow)
    
#     # Additional fields
#     full_name = Column(String(255), nullable=True)
#     avatar_url = Column(String(500), nullable=True)
    
#     # 🔥 Relationships
#     subscriptions = relationship("Subscription", back_populates="user")
#     downloads = relationship("TranscriptDownload", back_populates="user")
    
#     def __repr__(self):
#         return f"<User(id={self.id}, username='{self.username}', email='{self.email}', tier='{self.subscription_tier}')>"

# # =============================================================================
# # SUBSCRIPTION MODEL
# # =============================================================================

# class Subscription(Base):
#     __tablename__ = "subscriptions"
    
#     # Primary fields
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
#     # Subscription details
#     tier = Column(String(50), nullable=False)  # 'free', 'pro', 'premium'
#     status = Column(String(50), default='active')  # 'active', 'cancelled', 'expired', 'pending'
    
#     # Timestamps
#     created_at = Column(DateTime, default=datetime.utcnow)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
#     expires_at = Column(DateTime, nullable=True)
#     cancelled_at = Column(DateTime, nullable=True)
    
#     # Payment details
#     stripe_subscription_id = Column(String(255), nullable=True)
#     stripe_payment_intent_id = Column(String(255), nullable=True)
#     stripe_customer_id = Column(String(255), nullable=True)
    
#     # Pricing
#     price_paid = Column(Float, nullable=True)
#     currency = Column(String(10), default='usd')
    
#     # Additional data (renamed from metadata to avoid SQLAlchemy reserved word)
#     extra_data = Column(Text, nullable=True)  # JSON string for additional data
    
#     # 🔥 Relationships
#     user = relationship("User", back_populates="subscriptions")
    
#     def __repr__(self):
#         return f"<Subscription(id={self.id}, user_id={self.user_id}, tier='{self.tier}', status='{self.status}')>"

# # =============================================================================
# # TRANSCRIPT DOWNLOAD MODEL (for history tracking)
# # =============================================================================

# class TranscriptDownload(Base):
#     __tablename__ = "transcript_downloads"
    
#     # Primary fields
#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    
#     # Download details
#     youtube_id = Column(String(20), nullable=False, index=True)  # YouTube video ID
#     transcript_type = Column(String(50), nullable=False)  # 'clean_transcripts', 'unclean_transcripts', 'audio_downloads', 'video_downloads'
    
#     # File details
#     quality = Column(String(20), nullable=True)  # 'high', 'medium', 'low' for audio; '1080p', '720p', etc. for video
#     file_format = Column(String(10), nullable=True)  # 'txt', 'srt', 'vtt', 'mp3', 'mp4'
#     file_size = Column(Integer, nullable=True)  # Size in bytes
    
#     # Processing details
#     processing_time = Column(Float, nullable=True)  # Time taken in seconds
#     status = Column(String(20), default='completed')  # 'completed', 'failed', 'processing'
#     error_message = Column(Text, nullable=True)
    
#     # Metadata
#     language = Column(String(10), default='en')
#     video_title = Column(String(500), nullable=True)
#     video_uploader = Column(String(255), nullable=True)
#     video_duration = Column(Integer, nullable=True)  # Duration in seconds
    
#     # Timestamps
#     created_at = Column(DateTime, default=datetime.utcnow, index=True)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
#     # 🔥 Relationships
#     user = relationship("User", back_populates="downloads")
    
#     def __repr__(self):
#         return f"<TranscriptDownload(id={self.id}, user_id={self.user_id}, youtube_id='{self.youtube_id}', type='{self.transcript_type}')>"

# # =============================================================================
# # DATABASE CONFIGURATION
# # =============================================================================

# # Database URL - using SQLite for development
# DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")

# # Create engine
# engine = create_engine(
#     DATABASE_URL,
#     connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
#     echo=False  # Set to True for SQL query logging
# )

# # Create session factory
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# # =============================================================================
# # DATABASE UTILITY FUNCTIONS
# # =============================================================================

# def get_db() -> Session:
#     """
#     Dependency to get database session
#     """
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# def initialize_database():
#     """
#     Initialize database and create all tables
#     """
#     try:
#         logger.info("📊 Initializing database...")
        
#         # Create all tables
#         Base.metadata.create_all(bind=engine)
        
#         logger.info("✅ All database tables created successfully")
#         logger.info("✅ Database initialization completed")
        
#         # Log table information
#         tables = Base.metadata.tables.keys()
#         logger.info(f"📋 Created tables: {', '.join(tables)}")
        
#         return True
        
#     except Exception as e:
#         logger.error(f"❌ Database initialization failed: {e}")
#         return False

# def create_download_record_safe(db: Session, user_id: int, download_type: str, youtube_id: str, **kwargs):
#     """
#     Safely create a download record with error handling
#     """
#     try:
#         download_record = TranscriptDownload(
#             user_id=user_id,
#             youtube_id=youtube_id,
#             transcript_type=download_type,
#             quality=kwargs.get('quality', 'default'),
#             file_format=kwargs.get('file_format', 'txt'),
#             file_size=kwargs.get('file_size', 0),
#             processing_time=kwargs.get('processing_time', 0),
#             video_title=kwargs.get('video_title', None),
#             video_uploader=kwargs.get('video_uploader', None),
#             video_duration=kwargs.get('video_duration', None),
#             status='completed',
#             created_at=datetime.utcnow()
#         )
        
#         db.add(download_record)
#         db.commit()
#         db.refresh(download_record)
        
#         logger.info(f"✅ Download record created: {download_type} for video {youtube_id} by user {user_id}")
#         return download_record
        
#     except Exception as e:
#         logger.error(f"❌ Error creating download record: {e}")
#         db.rollback()
#         return None

# def get_user_stats(db: Session, user_id: int) -> dict:
#     """
#     Get user statistics for downloads
#     """
#     try:
#         user = db.query(User).filter(User.id == user_id).first()
#         if not user:
#             return {}
        
#         # Get download counts
#         total_downloads = db.query(TranscriptDownload).filter(TranscriptDownload.user_id == user_id).count()
        
#         # Get downloads by type
#         transcript_downloads = db.query(TranscriptDownload).filter(
#             TranscriptDownload.user_id == user_id,
#             TranscriptDownload.transcript_type.in_(['clean_transcripts', 'unclean_transcripts'])
#         ).count()
        
#         audio_downloads = db.query(TranscriptDownload).filter(
#             TranscriptDownload.user_id == user_id,
#             TranscriptDownload.transcript_type == 'audio_downloads'
#         ).count()
        
#         video_downloads = db.query(TranscriptDownload).filter(
#             TranscriptDownload.user_id == user_id,
#             TranscriptDownload.transcript_type == 'video_downloads'
#         ).count()
        
#         return {
#             'total_downloads': total_downloads,
#             'transcript_downloads': transcript_downloads,
#             'audio_downloads': audio_downloads,
#             'video_downloads': video_downloads,
#             'subscription_tier': user.subscription_tier,
#             'member_since': user.created_at,
#             'usage': {
#                 'clean_transcripts': user.usage_clean_transcripts or 0,
#                 'unclean_transcripts': user.usage_unclean_transcripts or 0,
#                 'audio_downloads': user.usage_audio_downloads or 0,
#                 'video_downloads': user.usage_video_downloads or 0
#             }
#         }
        
#     except Exception as e:
#         logger.error(f"❌ Error getting user stats: {e}")
#         return {}

# # =============================================================================
# # DATABASE MIGRATION HELPERS
# # =============================================================================

# def add_missing_columns():
#     """
#     Add missing columns to existing tables (for migrations)
#     """
#     try:
#         from sqlalchemy import text
        
#         with engine.connect() as conn:
#             # Check if subscription_tier exists in users table
#             try:
#                 result = conn.execute(text("PRAGMA table_info(users)"))
#                 columns = [row[1] for row in result.fetchall()]
                
#                 if 'subscription_tier' not in columns:
#                     logger.info("Adding missing subscription columns to users table...")
#                     conn.execute(text("ALTER TABLE users ADD COLUMN subscription_tier VARCHAR(50) DEFAULT 'free'"))
#                     conn.execute(text("ALTER TABLE users ADD COLUMN stripe_customer_id VARCHAR(255)"))
#                     conn.execute(text("ALTER TABLE users ADD COLUMN usage_clean_transcripts INTEGER DEFAULT 0"))
#                     conn.execute(text("ALTER TABLE users ADD COLUMN usage_unclean_transcripts INTEGER DEFAULT 0"))
#                     conn.execute(text("ALTER TABLE users ADD COLUMN usage_audio_downloads INTEGER DEFAULT 0"))
#                     conn.execute(text("ALTER TABLE users ADD COLUMN usage_video_downloads INTEGER DEFAULT 0"))
#                     conn.execute(text("ALTER TABLE users ADD COLUMN usage_reset_date DATETIME"))
#                     conn.commit()
#                     logger.info("✅ Missing columns added successfully")
                    
#             except Exception as e:
#                 logger.warning(f"Column migration warning: {e}")
                
#     except Exception as e:
#         logger.error(f"❌ Migration failed: {e}")

# # =============================================================================
# # INITIALIZE ON IMPORT
# # =============================================================================

# if __name__ == "__main__":
#     # Initialize database when run directly
#     initialize_database()
#     add_missing_columns()
#     logger.info("🔥 Database models initialized successfully")
# else:
#     # Add missing columns when imported
#     try:
#         add_missing_columns()
#     except Exception as e:
#         logger.warning(f"Migration warning on import: {e}")

# # Export commonly used items
# __all__ = [
#     'User', 
#     'Subscription', 
#     'TranscriptDownload', 
#     'get_db', 
#     'initialize_database', 
#     'create_download_record_safe',
#     'get_user_stats',
#     'engine',
#     'SessionLocal',
#     'Base'
# ]
