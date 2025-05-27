# payment.py
import stripe
from fastapi import BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from pydantic import BaseModel
import os
import logging
from dotenv import load_dotenv

# Import from database.py
from database import get_db, User, Subscription

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger("youtube_trans_downloader.payment")

# Load Stripe API key
stripe.api_key = os.getenv("STRIPE_SECRET_KEY") #STRIPE_SECRET_KEY

# Price ID mapping from environment variables
PRICE_ID_MAP = {
    "basic": os.getenv("BASIC_PRICE_ID"),
    "premium": os.getenv("PREMIUM_PRICE_ID")
}

class PaymentRequest(BaseModel):
    token: str
    subscription_tier: str
    user_id: int

async def process_subscription(
    request: PaymentRequest,
    db: Session = Depends(get_db)
):
    try:
        # Validate subscription tier
        if request.subscription_tier not in PRICE_ID_MAP:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid subscription tier. Must be one of: {', '.join(PRICE_ID_MAP.keys())}"
            )
            
        # Get user from database
        user = db.query(User).filter(User.id == request.user_id).first()
        if not user:
            logger.error(f"User not found: {request.user_id}")
            raise HTTPException(status_code=404, detail="User not found")
            
        # Create Stripe customer & subscription
        customer = stripe.Customer.create(
            source=request.token,
            email=user.email,
            name=user.username,
            metadata={"user_id": str(request.user_id)}
        )
       
        subscription = stripe.Subscription.create(
            customer=customer.id,
            items=[{"price": PRICE_ID_MAP[request.subscription_tier]}],
            metadata={
                "user_id": str(request.user_id),
                "username": user.username,
                "tier": request.subscription_tier
            }
        )
       
        # Check if user already has a subscription and update it
        existing_subscription = db.query(Subscription).filter(
            Subscription.user_id == request.user_id
        ).first()
        
        if existing_subscription:
            existing_subscription.tier = request.subscription_tier
            existing_subscription.start_date = datetime.now()
            existing_subscription.expiry_date = datetime.now() + timedelta(days=30)
            existing_subscription.payment_id = subscription.id
            existing_subscription.auto_renew = True
            logger.info(f"Updated subscription for user {user.username} to {request.subscription_tier}")
        else:
            # Create new subscription
            new_subscription = Subscription(
                user_id=request.user_id,
                tier=request.subscription_tier,
                start_date=datetime.now(),
                expiry_date=datetime.now() + timedelta(days=30),
                payment_id=subscription.id,
                auto_renew=True
            )
            db.add(new_subscription)
            logger.info(f"Created new {request.subscription_tier} subscription for user {user.username}")
        
        db.commit()
        
        return {
            "status": "success", 
            "subscription_id": subscription.id, 
            "tier": request.subscription_tier
        }
    
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error processing subscription: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing subscription: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Error processing subscription")