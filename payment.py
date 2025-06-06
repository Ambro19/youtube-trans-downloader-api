# payment.py - Complete Stripe Backend Integration

import stripe
import os
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import logging

# Configure Stripe
stripe.api_key = os.getenv('STRIPE_SECRET_KEY', 'sk_test_...')  # Replace with your Stripe secret key

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class PaymentIntentRequest(BaseModel):
    amount: int  # Amount in cents
    currency: str = 'usd'
    payment_method_id: str
    plan_name: str

class SubscriptionRequest(BaseModel):
    token: Optional[str] = None
    subscription_tier: str

class PaymentIntentResponse(BaseModel):
    client_secret: str
    token: str

class SubscriptionResponse(BaseModel):
    subscription_id: str
    status: str
    current_period_end: int
    tier: str

# Database models (add these to your existing models)
"""
Add these fields to your User model:

class User(Base):
    # ... existing fields ...
    subscription_tier = Column(String, default='free')
    subscription_status = Column(String, default='inactive')
    subscription_id = Column(String, nullable=True)
    subscription_current_period_end = Column(DateTime, nullable=True)
    stripe_customer_id = Column(String, nullable=True)
    
    # Usage tracking for current month
    usage_clean_transcripts = Column(Integer, default=0)
    usage_unclean_transcripts = Column(Integer, default=0)
    usage_audio_downloads = Column(Integer, default=0)
    usage_video_downloads = Column(Integer, default=0)
    usage_reset_date = Column(DateTime, default=datetime.utcnow)
"""

# Plan configurations
SUBSCRIPTION_PLANS = {
    'pro': {
        'price_id': os.getenv('STRIPE_PRO_PRICE_ID', 'price_pro_monthly'),
        'amount': 999,  # $9.99 in cents
        'limits': {
            'clean_transcripts': 100,
            'unclean_transcripts': 50,
            'audio_downloads': 50,
            'video_downloads': 20
        }
    },
    'premium': {
        'price_id': os.getenv('STRIPE_PREMIUM_PRICE_ID', 'price_premium_monthly'),
        'amount': 1999,  # $19.99 in cents
        'limits': {
            'clean_transcripts': float('inf'),
            'unclean_transcripts': float('inf'),
            'audio_downloads': float('inf'),
            'video_downloads': float('inf')
        }
    }
}

def get_or_create_stripe_customer(user, db: Session):
    """Get or create a Stripe customer for the user"""
    try:
        if user.stripe_customer_id:
            # Verify customer exists in Stripe
            try:
                customer = stripe.Customer.retrieve(user.stripe_customer_id)
                return customer
            except stripe.error.InvalidRequestError:
                # Customer doesn't exist, create new one
                pass
        
        # Create new customer
        customer = stripe.Customer.create(
            email=user.email,
            name=user.full_name if hasattr(user, 'full_name') else user.email,
            metadata={'user_id': str(user.id)}
        )
        
        # Save customer ID to database
        user.stripe_customer_id = customer.id
        db.commit()
        
        return customer
        
    except Exception as e:
        logger.error(f"Error creating Stripe customer: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create payment customer"
        )

async def create_payment_intent(
    request: PaymentIntentRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_database)
):
    """Create a payment intent for subscription upgrade"""
    try:
        # Validate plan
        if request.plan_name not in SUBSCRIPTION_PLANS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid subscription plan"
            )
        
        plan = SUBSCRIPTION_PLANS[request.plan_name]
        
        # Validate amount
        if request.amount != plan['amount']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid payment amount"
            )
        
        # Get or create Stripe customer
        customer = get_or_create_stripe_customer(current_user, db)
        
        # Create payment intent
        payment_intent = stripe.PaymentIntent.create(
            amount=request.amount,
            currency=request.currency,
            customer=customer.id,
            payment_method=request.payment_method_id,
            confirmation_method='manual',
            confirm=True,
            metadata={
                'user_id': str(current_user.id),
                'plan_name': request.plan_name,
                'subscription_upgrade': 'true'
            }
        )
        
        # Generate a simple token for verification
        import secrets
        token = secrets.token_urlsafe(32)
        
        return PaymentIntentResponse(
            client_secret=payment_intent.client_secret,
            token=token
        )
        
    except stripe.error.CardError as e:
        logger.error(f"Card error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Card error: {e.user_message}"
        )
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Payment processing error"
        )
    except Exception as e:
        logger.error(f"Payment intent creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create payment intent"
        )

async def create_subscription(
    request: SubscriptionRequest,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_database)
):
    """Create or update user subscription after successful payment"""
    try:
        # Validate plan
        if request.subscription_tier not in SUBSCRIPTION_PLANS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid subscription tier"
            )
        
        plan = SUBSCRIPTION_PLANS[request.subscription_tier]
        
        # Get or create Stripe customer
        customer = get_or_create_stripe_customer(current_user, db)
        
        # Create Stripe subscription
        subscription = stripe.Subscription.create(
            customer=customer.id,
            items=[{
                'price': plan['price_id'],
            }],
            metadata={
                'user_id': str(current_user.id),
                'plan_name': request.subscription_tier
            }
        )
        
        # Update user subscription in database
        current_user.subscription_tier = request.subscription_tier
        current_user.subscription_status = 'active'
        current_user.subscription_id = subscription.id
        current_user.subscription_current_period_end = datetime.fromtimestamp(
            subscription.current_period_end
        )
        
        # Reset usage counters for new billing period
        reset_user_usage(current_user)
        
        db.commit()
        
        logger.info(f"User {current_user.id} upgraded to {request.subscription_tier}")
        
        return SubscriptionResponse(
            subscription_id=subscription.id,
            status=subscription.status,
            current_period_end=subscription.current_period_end,
            tier=request.subscription_tier
        )
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error during subscription creation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create subscription"
        )
    except Exception as e:
        logger.error(f"Subscription creation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process subscription"
        )

async def cancel_subscription(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_database)
):
    """Cancel user's current subscription"""
    try:
        if not current_user.subscription_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No active subscription found"
            )
        
        # Cancel subscription in Stripe
        subscription = stripe.Subscription.modify(
            current_user.subscription_id,
            cancel_at_period_end=True
        )
        
        # Update database - don't immediately downgrade, let it expire
        current_user.subscription_status = 'cancelling'
        db.commit()
        
        logger.info(f"User {current_user.id} cancelled subscription {current_user.subscription_id}")
        
        return {
            "message": "Subscription cancelled successfully",
            "will_expire_at": subscription.current_period_end
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

def reset_user_usage(user):
    """Reset user's monthly usage counters"""
    user.usage_clean_transcripts = 0
    user.usage_unclean_transcripts = 0
    user.usage_audio_downloads = 0
    user.usage_video_downloads = 0
    user.usage_reset_date = datetime.utcnow()

def check_user_limits(user, action_type: str):
    """Check if user has exceeded their limits for the current month"""
    # Reset usage if it's a new month
    if user.usage_reset_date and user.usage_reset_date.month != datetime.utcnow().month:
        reset_user_usage(user)
    
    plan_limits = SUBSCRIPTION_PLANS.get(user.subscription_tier, {}).get('limits', {})
    
    # Free tier limits
    if user.subscription_tier == 'free':
        free_limits = {
            'clean_transcripts': 5,
            'unclean_transcripts': 3,
            'audio_downloads': 2,
            'video_downloads': 1
        }
        plan_limits = free_limits
    
    current_usage = getattr(user, f'usage_{action_type}', 0)
    limit = plan_limits.get(action_type, 0)
    
    if limit == float('inf'):  # Unlimited
        return True
    
    return current_usage < limit

def increment_usage(user, action_type: str, db: Session):
    """Increment user's usage counter for the given action"""
    current_usage = getattr(user, f'usage_{action_type}', 0)
    setattr(user, f'usage_{action_type}', current_usage + 1)
    db.commit()

async def get_subscription_status(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_database)
):
    """Get current user's subscription status and usage"""
    try:
        # Check if subscription is expired
        if (current_user.subscription_current_period_end and 
            current_user.subscription_current_period_end < datetime.utcnow()):
            # Subscription expired, downgrade to free
            current_user.subscription_tier = 'free'
            current_user.subscription_status = 'inactive'
            current_user.subscription_id = None
            current_user.subscription_current_period_end = None
            db.commit()
        
        # Get current limits based on tier
        if current_user.subscription_tier == 'free':
            limits = {
                'clean_transcripts': 5,
                'unclean_transcripts': 3,
                'audio_downloads': 2,
                'video_downloads': 1
            }
        else:
            limits = SUBSCRIPTION_PLANS.get(current_user.subscription_tier, {}).get('limits', {})
        
        # Convert infinity to string for JSON serialization
        for key, value in limits.items():
            if value == float('inf'):
                limits[key] = 'unlimited'
        
        return {
            "tier": current_user.subscription_tier,
            "status": current_user.subscription_status,
            "usage": {
                "clean_transcripts": current_user.usage_clean_transcripts or 0,
                "unclean_transcripts": current_user.usage_unclean_transcripts or 0,
                "audio_downloads": current_user.usage_audio_downloads or 0,
                "video_downloads": current_user.usage_video_downloads or 0
            },
            "limits": limits,
            "subscription_id": current_user.subscription_id,
            "current_period_end": current_user.subscription_current_period_end.isoformat() if current_user.subscription_current_period_end else None
        }
        
    except Exception as e:
        logger.error(f"Error getting subscription status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get subscription status"
        )
#==================================================================
# # payment.py
# import stripe
# from fastapi import BackgroundTasks, Depends, HTTPException
# from sqlalchemy.orm import Session
# from datetime import datetime, timedelta
# from pydantic import BaseModel
# import os
# import logging
# from dotenv import load_dotenv

# # Import from database.py
# from database import get_db, User, Subscription

# # Load environment variables
# load_dotenv()

# # Configure logging
# logger = logging.getLogger("youtube_trans_downloader.payment")

# # Load Stripe API key
# stripe.api_key = os.getenv("STRIPE_SECRET_KEY") #STRIPE_SECRET_KEY

# # Price ID mapping from environment variables
# PRICE_ID_MAP = {
#     "basic": os.getenv("BASIC_PRICE_ID"),
#     "premium": os.getenv("PREMIUM_PRICE_ID")
# }

# class PaymentRequest(BaseModel):
#     token: str
#     subscription_tier: str
#     user_id: int

# async def process_subscription(
#     request: PaymentRequest,
#     db: Session = Depends(get_db)
# ):
#     try:
#         # Validate subscription tier
#         if request.subscription_tier not in PRICE_ID_MAP:
#             raise HTTPException(
#                 status_code=400, 
#                 detail=f"Invalid subscription tier. Must be one of: {', '.join(PRICE_ID_MAP.keys())}"
#             )
            
#         # Get user from database
#         user = db.query(User).filter(User.id == request.user_id).first()
#         if not user:
#             logger.error(f"User not found: {request.user_id}")
#             raise HTTPException(status_code=404, detail="User not found")
            
#         # Create Stripe customer & subscription
#         customer = stripe.Customer.create(
#             source=request.token,
#             email=user.email,
#             name=user.username,
#             metadata={"user_id": str(request.user_id)}
#         )
       
#         subscription = stripe.Subscription.create(
#             customer=customer.id,
#             items=[{"price": PRICE_ID_MAP[request.subscription_tier]}],
#             metadata={
#                 "user_id": str(request.user_id),
#                 "username": user.username,
#                 "tier": request.subscription_tier
#             }
#         )
       
#         # Check if user already has a subscription and update it
#         existing_subscription = db.query(Subscription).filter(
#             Subscription.user_id == request.user_id
#         ).first()
        
#         if existing_subscription:
#             existing_subscription.tier = request.subscription_tier
#             existing_subscription.start_date = datetime.now()
#             existing_subscription.expiry_date = datetime.now() + timedelta(days=30)
#             existing_subscription.payment_id = subscription.id
#             existing_subscription.auto_renew = True
#             logger.info(f"Updated subscription for user {user.username} to {request.subscription_tier}")
#         else:
#             # Create new subscription
#             new_subscription = Subscription(
#                 user_id=request.user_id,
#                 tier=request.subscription_tier,
#                 start_date=datetime.now(),
#                 expiry_date=datetime.now() + timedelta(days=30),
#                 payment_id=subscription.id,
#                 auto_renew=True
#             )
#             db.add(new_subscription)
#             logger.info(f"Created new {request.subscription_tier} subscription for user {user.username}")
        
#         db.commit()
        
#         return {
#             "status": "success", 
#             "subscription_id": subscription.id, 
#             "tier": request.subscription_tier
#         }
    
#     except stripe.error.StripeError as e:
#         logger.error(f"Stripe error processing subscription: {str(e)}")
#         raise HTTPException(status_code=400, detail=str(e))
#     except Exception as e:
#         logger.error(f"Error processing subscription: {str(e)}")
#         db.rollback()
#         raise HTTPException(status_code=500, detail="Error processing subscription")