# payment.py - WORKING Stripe Backend Integration (FIXED VERSION)

import stripe
import os
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends, status, APIRouter, Request
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from dotenv import load_dotenv
import json

# Import your existing dependencies
from database import get_db
from models import User, Subscription
from auth import get_current_user

# Load environment variables
load_dotenv()

# Configure Stripe - SECURE: No hardcoded keys
stripe_secret = os.getenv('STRIPE_SECRET_KEY')
if not stripe_secret:
    raise ValueError("STRIPE_SECRET_KEY environment variable is required")

stripe.api_key = stripe_secret

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# 🔧 SIMPLIFIED: Pydantic models for request/response
class CreatePaymentIntentRequest(BaseModel):
    price_id: str

class PaymentConfirmRequest(BaseModel):
    payment_intent_id: str

# Validate required environment variables
def validate_environment():
    """Validate that all required environment variables are set"""
    required_vars = [
        'STRIPE_SECRET_KEY',
        'STRIPE_PRO_PRICE_ID', 
        'STRIPE_PREMIUM_PRICE_ID'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Missing Stripe environment variables: {', '.join(missing_vars)}")
        return False
    
    logger.info("✅ All required Stripe environment variables are set")
    return True

# Validate on import
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

@router.post("/create_payment_intent/")
async def create_payment_intent(
    request: CreatePaymentIntentRequest,
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

        # Get the price from Stripe to determine amount
        try:
            price = stripe.Price.retrieve(request.price_id)
        except stripe.error.InvalidRequestError as e:
            logger.error(f"Invalid Stripe price ID: {request.price_id}, error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid Stripe price ID: {request.price_id}"
            )
        
        # Determine plan type
        plan_type = 'pro' if request.price_id == os.getenv("STRIPE_PRO_PRICE_ID") else 'premium'
        
        # Get or create Stripe customer
        customer = get_or_create_stripe_customer(current_user, db)
        
        # 🔧 FIXED: Create PaymentIntent with proper configuration
        intent = stripe.PaymentIntent.create(
            amount=price.unit_amount,  # Amount in cents
            currency=price.currency,
            customer=customer.id,
            automatic_payment_methods={
                'enabled': True,
                'allow_redirects': 'never'  # 🔧 THIS FIXES THE STRIPE ERROR!
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
            'currency': price.currency
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
    request: PaymentConfirmRequest,
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
                expires_at=datetime.utcnow() + timedelta(days=30)  # 30 days from now
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

#==============================================================================
# # payment.py - Complete Stripe Backend Integration (FIXED VERSION)

# import stripe
# import os
# from datetime import datetime, timedelta
# from fastapi import HTTPException, Depends, status, APIRouter
# from sqlalchemy.orm import Session
# from pydantic import BaseModel
# from typing import Optional, Dict, Any
# import logging
# from dotenv import load_dotenv

# # Import your existing dependencies
# from database import get_db
# from models import User, Subscription
# from auth import get_current_user

# # Load environment variables
# load_dotenv()

# # Configure Stripe - SECURE: No hardcoded keys
# stripe_secret = os.getenv('STRIPE_SECRET_KEY')
# if not stripe_secret:
#     raise ValueError("STRIPE_SECRET_KEY environment variable is required")

# stripe.api_key = stripe_secret

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Create router
# router = APIRouter()

# # Pydantic models for request/response
# class PaymentIntentRequest(BaseModel):
#     price_id: str  # Changed to price_id for simplified flow

# class PaymentIntentResponse(BaseModel):
#     client_secret: str
#     payment_intent_id: str

# class SubscriptionRequest(BaseModel):
#     token: Optional[str] = None
#     subscription_tier: str

# class SubscriptionResponse(BaseModel):
#     subscription_id: str
#     status: str
#     current_period_end: int
#     tier: str

# # Plan configurations - SECURE: Get from environment
# SUBSCRIPTION_PLANS = {
#     'pro': {
#         'price_id': os.getenv('STRIPE_PRO_PRICE_ID'),
#         'amount': 999,  # $9.99 in cents
#         'limits': {
#             'clean_transcripts': 100,
#             'unclean_transcripts': 50,
#             'audio_downloads': 50,
#             'video_downloads': 20
#         }
#     },
#     'premium': {
#         'price_id': os.getenv('STRIPE_PREMIUM_PRICE_ID'),
#         'amount': 1999,  # $19.99 in cents
#         'limits': {
#             'clean_transcripts': float('inf'),
#             'unclean_transcripts': float('inf'),
#             'audio_downloads': float('inf'),
#             'video_downloads': float('inf')
#         }
#     }
# }

# # Validate required environment variables
# def validate_environment():
#     """Validate that all required environment variables are set"""
#     required_vars = [
#         'STRIPE_SECRET_KEY',
#         'STRIPE_PRO_PRICE_ID', 
#         'STRIPE_PREMIUM_PRICE_ID'
#     ]
    
#     missing_vars = []
#     for var in required_vars:
#         if not os.getenv(var):
#             missing_vars.append(var)
    
#     if missing_vars:
#         raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
#     logger.info("✅ All required Stripe environment variables are set")

# # Validate on import
# validate_environment()

# def get_or_create_stripe_customer(user, db: Session):
#     """Get or create a Stripe customer for the user"""
#     try:
#         if hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
#             try:
#                 customer = stripe.Customer.retrieve(user.stripe_customer_id)
#                 return customer
#             except stripe.error.InvalidRequestError:
#                 pass
        
#         # Create new customer
#         customer = stripe.Customer.create(
#             email=user.email,
#             name=getattr(user, 'full_name', user.email),
#             metadata={'user_id': str(user.id)}
#         )
        
#         # Save customer ID if user model supports it
#         if hasattr(user, 'stripe_customer_id'):
#             user.stripe_customer_id = customer.id
#             db.commit()
        
#         return customer
        
#     except Exception as e:
#         logger.error(f"Error creating Stripe customer: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to create payment customer"
#         )

# @router.post("/create_payment_intent/")
# async def create_payment_intent(
#     request: PaymentIntentRequest,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ) -> PaymentIntentResponse:
#     """
#     Create a PaymentIntent for subscription upgrade - FIXED VERSION
#     """
#     try:
#         # Validate price_id
#         valid_price_ids = [
#             os.getenv("STRIPE_PRO_PRICE_ID"),
#             os.getenv("STRIPE_PREMIUM_PRICE_ID")
#         ]
        
#         if request.price_id not in valid_price_ids:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Invalid price ID"
#             )

#         # Get the price from Stripe to determine amount
#         price = stripe.Price.retrieve(request.price_id)
        
#         # Determine plan type
#         plan_type = 'pro' if request.price_id == os.getenv("STRIPE_PRO_PRICE_ID") else 'premium'
        
#         # Get or create Stripe customer
#         customer = get_or_create_stripe_customer(current_user, db)
        
#         # Create PaymentIntent with FIXED configuration
#         intent = stripe.PaymentIntent.create(
#             amount=price.unit_amount,  # Amount in cents
#             currency=price.currency,
#             customer=customer.id,
#             automatic_payment_methods={
#                 'enabled': True,
#                 'allow_redirects': 'never'  # 🔧 THIS FIXES THE STRIPE ERROR!
#             },
#             metadata={
#                 'user_id': str(current_user.id),
#                 'user_email': current_user.email,
#                 'price_id': request.price_id,
#                 'plan_type': plan_type
#             }
#         )

#         logger.info(f"Payment intent created for user {current_user.id}: {intent.id}")

#         return PaymentIntentResponse(
#             client_secret=intent.client_secret,
#             payment_intent_id=intent.id
#         )

#     except stripe.error.StripeError as e:
#         logger.error(f"Stripe error: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"Stripe error: {str(e)}"
#         )
#     except Exception as e:
#         logger.error(f"Payment intent creation error: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to create payment intent"
#         )

# @router.post("/confirm_payment/")
# async def confirm_payment(
#     payment_intent_id: str,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ) -> Dict[str, Any]:
#     """
#     Confirm payment and update user subscription
#     """
#     try:
#         # Retrieve the PaymentIntent from Stripe
#         intent = stripe.PaymentIntent.retrieve(payment_intent_id)
        
#         if intent.status != 'succeeded':
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Payment not completed"
#             )

#         # Update user subscription in database
#         user_subscription = db.query(Subscription).filter(
#             Subscription.user_id == current_user.id
#         ).first()

#         plan_type = intent.metadata.get('plan_type', 'pro')

#         if not user_subscription:
#             # Create new subscription record
#             user_subscription = Subscription(
#                 user_id=current_user.id,
#                 tier=plan_type,
#                 status='active',
#                 stripe_payment_intent_id=payment_intent_id,
#                 created_at=datetime.utcnow(),
#                 expires_at=datetime.utcnow() + timedelta(days=30)  # 30 days from now
#             )
#             db.add(user_subscription)
#         else:
#             # Update existing subscription
#             user_subscription.tier = plan_type
#             user_subscription.status = 'active'
#             user_subscription.stripe_payment_intent_id = payment_intent_id
#             user_subscription.expires_at = datetime.utcnow() + timedelta(days=30)

#         db.commit()
#         db.refresh(user_subscription)

#         logger.info(f"User {current_user.id} subscription updated to {plan_type}")

#         return {
#             'success': True,
#             'subscription_tier': user_subscription.tier,
#             'expires_at': user_subscription.expires_at.isoformat()
#         }

#     except stripe.error.StripeError as e:
#         logger.error(f"Stripe error: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"Stripe error: {str(e)}"
#         )
#     except Exception as e:
#         logger.error(f"Payment confirmation error: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to confirm payment"
#         )

# # Legacy function - keep for backward compatibility
# async def create_subscription(
#     request: SubscriptionRequest,
#     current_user,
#     db: Session
# ):
#     """Create or update user subscription after successful payment"""
#     try:
#         # Validate plan
#         if request.subscription_tier not in SUBSCRIPTION_PLANS:
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail="Invalid subscription tier"
#             )
        
#         plan = SUBSCRIPTION_PLANS[request.subscription_tier]
        
#         # Validate price_id exists
#         if not plan['price_id']:
#             raise HTTPException(
#                 status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 detail=f"Price ID not configured for {request.subscription_tier} plan"
#             )
        
#         # Get or create Stripe customer
#         customer = get_or_create_stripe_customer(current_user, db)
        
#         # Create Stripe subscription
#         subscription = stripe.Subscription.create(
#             customer=customer.id,
#             items=[{
#                 'price': plan['price_id'],
#             }],
#             metadata={
#                 'user_id': str(current_user.id),
#                 'plan_name': request.subscription_tier
#             }
#         )
        
#         # Update user subscription in database
#         # (Database update logic here - depends on your User model structure)
        
#         logger.info(f"User {current_user.id} upgraded to {request.subscription_tier}")
        
#         return SubscriptionResponse(
#             subscription_id=subscription.id,
#             status=subscription.status,
#             current_period_end=subscription.current_period_end,
#             tier=request.subscription_tier
#         )
        
#     except stripe.error.StripeError as e:
#         logger.error(f"Stripe error during subscription creation: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to create subscription"
#         )
#     except Exception as e:
#         logger.error(f"Subscription creation error: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to process subscription"
#         )
