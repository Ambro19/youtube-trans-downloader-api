# payment.py - WORKING VERSION that matches your current setup

import stripe
import os
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends, status, APIRouter
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from dotenv import load_dotenv

# Import your existing dependencies
from database import get_db
from models import User, Subscription
from auth import get_current_user

# Load environment variables
load_dotenv()

# Configure Stripe
stripe_secret = os.getenv('STRIPE_SECRET_KEY')
if not stripe_secret:
    raise ValueError("STRIPE_SECRET_KEY environment variable is required")

stripe.api_key = stripe_secret

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# 🔧 FIXED: Simple request model that matches frontend
class CreatePaymentIntentRequest(BaseModel):
    price_id: str

class ConfirmPaymentRequest(BaseModel):
    payment_intent_id: str

# Validate environment variables
def validate_environment():
    required_vars = ['STRIPE_SECRET_KEY', 'STRIPE_PRO_PRICE_ID', 'STRIPE_PREMIUM_PRICE_ID']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing Stripe environment variables: {', '.join(missing_vars)}")
        return False
    
    logger.info("✅ All required Stripe environment variables are set")
    return True

validate_environment()

def get_or_create_stripe_customer(user, db: Session):
    """Get or create a Stripe customer for the user"""
    try:
        if hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
            try:
                customer = stripe.Customer.retrieve(user.stripe_customer_id)
                return customer
            except stripe.error.InvalidRequestError:
                pass
        
        # Create new customer
        customer = stripe.Customer.create(
            email=user.email,
            name=getattr(user, 'full_name', user.email),
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

# 🔧 ALSO UPDATE the endpoint signature to use the new model:

@router.post("/create_payment_intent/")
async def create_payment_intent(
    request: CreatePaymentIntentRequest,  # 🔧 Use the new simple model
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Create a PaymentIntent for subscription upgrade - WORKING VERSION
    """
    try:
        logger.info(f"Creating payment intent for user {current_user.id} with price_id: {request.price_id}")
        
        # Validate price_id
        valid_price_ids = [
            os.getenv("STRIPE_PRO_PRICE_ID"),
            os.getenv("STRIPE_PREMIUM_PRICE_ID")
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
        
        # Determine plan type
        plan_type = 'pro' if request.price_id == os.getenv("STRIPE_PRO_PRICE_ID") else 'premium'
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

@router.post("/confirm_payment/")
async def confirm_payment(
    request: ConfirmPaymentRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Confirm payment and update user subscription
    """
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


