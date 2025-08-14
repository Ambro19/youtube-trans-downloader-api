# payment.py - FULLY FIXED VERSION with Environment Variable Price IDs

import stripe
import os
from datetime import datetime, timedelta
from fastapi import HTTPException, Depends, status, APIRouter
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging
from dotenv import load_dotenv
import jwt
from jwt.exceptions import PyJWTError
from fastapi.security import OAuth2PasswordBearer

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

# Auth configuration (moved from separate auth module)
SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
ALGORITHM = "HS256"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# 🔧 FIXED: Simple request model that matches frontend
class CreatePaymentIntentRequest(BaseModel):
    price_id: str

class ConfirmPaymentRequest(BaseModel):
    payment_intent_id: str

# 🔧 NEW: Get price IDs from environment with validation
def get_stripe_price_ids():
    """Get Stripe price IDs from environment variables"""
    pro_price_id = os.getenv('STRIPE_PRO_PRICE_ID')
    premium_price_id = os.getenv('STRIPE_PREMIUM_PRICE_ID')
    
    if not pro_price_id or not premium_price_id:
        logger.error("❌ Stripe price IDs not found in environment variables")
        logger.error("Please run setup_stripe_prices.py to create price IDs")
        logger.error("Or add STRIPE_PRO_PRICE_ID and STRIPE_PREMIUM_PRICE_ID to your .env file")
        
        # Fallback to hardcoded values for development (will likely fail)
        pro_price_id = "price_1RTZOEGqJlA2TvtBA1ftMuBs"
        premium_price_id = "price_1RTZPkGqJlA2TvtBAr7cLIYh"
        logger.warning(f"Using fallback price IDs: Pro={pro_price_id}, Premium={premium_price_id}")
    
    return pro_price_id, premium_price_id

# Validate environment variables
def validate_environment():
    required_vars = ['STRIPE_SECRET_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.warning(f"Missing Stripe environment variables: {', '.join(missing_vars)}")
        return False
    
    # Check for price IDs
    pro_price_id, premium_price_id = get_stripe_price_ids()
    if not os.getenv('STRIPE_PRO_PRICE_ID') or not os.getenv('STRIPE_PREMIUM_PRICE_ID'):
        logger.warning("⚠️ Stripe price IDs not found in environment. Please run setup_stripe_prices.py")
        return False
    
    logger.info("✅ All required Stripe environment variables are set")
    return True

validate_environment()

# 🔧 FIXED: Database dependency (no external imports)
def get_db():
    """Get database session"""
    from models import SessionLocal
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user_by_username(db: Session, username: str):
    """Get user by username"""
    from models import User
    return db.query(User).filter(User.username == username).first()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    """Get current authenticated user"""
    from models import User
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, 
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except PyJWTError:
        raise credentials_exception
    
    user = get_user_by_username(db, username)
    if user is None:
        raise credentials_exception
    return user

def get_or_create_stripe_customer(user, db: Session):
    """Get or create a Stripe customer for the user"""
    try:
        if hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
            try:
                customer = stripe.Customer.retrieve(user.stripe_customer_id)
                return customer
            except stripe.error.InvalidRequestError:
                pass
        
        # Create new customer with REAL email address
        real_email = user.email
        if user.email == "LovePets@example.com":
            real_email = "lovepets@gmail.com"
        elif user.email == "OneTechly@example.com":
            real_email = "onetechtly@gmail.com"
        
        customer = stripe.Customer.create(
            email=real_email,
            name=getattr(user, 'full_name', user.username),
            metadata={'user_id': str(user.id), 'username': user.username}
        )
        
        # Save customer ID if user model supports it
        if hasattr(user, 'stripe_customer_id'):
            user.stripe_customer_id = customer.id
            db.commit()
        
        logger.info(f"✅ Created Stripe customer for {user.username} with email {real_email}")
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
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    🔧 FIXED: Create a PaymentIntent for subscription upgrade - WORKING VERSION
    """
    try:
        logger.info(f"🔥 Creating payment intent for user {current_user.username} with price_id: {request.price_id}")
        
        # 🔧 FIXED: Get price IDs from environment variables
        pro_price_id, premium_price_id = get_stripe_price_ids()
        
        # Validate price_id
        valid_price_ids = [pro_price_id, premium_price_id]
        
        if request.price_id not in valid_price_ids:
            logger.error(f"Invalid price ID: {request.price_id}")
            logger.error(f"Valid price IDs: {valid_price_ids}")
            logger.error("Please run setup_stripe_prices.py to create valid price IDs")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid price ID: {request.price_id}. Please contact support."
            )

        # Get the price from Stripe
        try:
            price = stripe.Price.retrieve(request.price_id)
            logger.info(f"✅ Retrieved price: {price.unit_amount} {price.currency}")
        except stripe.error.InvalidRequestError as e:
            logger.error(f"Invalid Stripe price ID: {request.price_id}, error: {str(e)}")
            logger.error("This means the price ID doesn't exist in your Stripe account")
            logger.error("Please run setup_stripe_prices.py to create the required price IDs")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid Stripe price ID: {request.price_id}. Please run setup to create price IDs."
            )
        
        # Determine plan type
        plan_type = 'pro' if request.price_id == pro_price_id else 'premium'
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
                'user_email': customer.email,  # Use the real email from customer
                'username': current_user.username,
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
            'plan_type': plan_type,
            'customer_email': customer.email
        }

    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Payment system error: {str(e)}"
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
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    🔧 FIXED: Confirm payment and update user subscription
    """
    try:
        from models import Subscription
        
        logger.info(f"Confirming payment for user {current_user.username} with payment_intent: {request.payment_intent_id}")
        
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

        # 🔧 FIXED: Also update user's subscription_tier attribute
        current_user.subscription_tier = plan_type

        db.commit()
        db.refresh(user_subscription)
        db.refresh(current_user)

        logger.info(f"✅ User {current_user.username} subscription updated to {plan_type}")

        return {
            'success': True,
            'subscription_tier': user_subscription.tier,
            'expires_at': user_subscription.expires_at.isoformat(),
            'status': 'active',
            'user_tier_updated': current_user.subscription_tier
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
# # 2. payment.py - COMPLETELY FIXED VERSION with Demo Mode Support

# import stripe
# import os
# from datetime import datetime, timedelta
# from fastapi import HTTPException, Depends, status, APIRouter
# from sqlalchemy.orm import Session
# from pydantic import BaseModel
# from typing import Optional, Dict, Any
# import logging
# from dotenv import load_dotenv
# import jwt
# from jwt.exceptions import PyJWTError
# from fastapi.security import OAuth2PasswordBearer

# # Load environment variables
# load_dotenv()

# # Configure Stripe
# stripe_secret = os.getenv('STRIPE_SECRET_KEY')
# if stripe_secret:
#     stripe.api_key = stripe_secret
#     print("✅ Stripe configured successfully")
# else:
#     print("⚠️ Stripe not configured - Demo mode will be used")

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Create router
# router = APIRouter()

# # Auth configuration (moved from separate auth module)
# SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
# ALGORITHM = "HS256"
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# # 🔧 FIXED: Simple request model that matches frontend
# class CreatePaymentIntentRequest(BaseModel):
#     price_id: str

# class ConfirmPaymentRequest(BaseModel):
#     payment_intent_id: str

# # 🔧 FIXED: Database dependency (no external imports)
# def get_db():
#     """Get database session"""
#     from models import SessionLocal
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# def get_user_by_username(db: Session, username: str):
#     """Get user by username"""
#     from models import User
#     return db.query(User).filter(User.username == username).first()

# def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
#     """Get current authenticated user"""
#     from models import User
    
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED, 
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")
#         if username is None:
#             raise credentials_exception
#     except PyJWTError:
#         raise credentials_exception
    
#     user = get_user_by_username(db, username)
#     if user is None:
#         raise credentials_exception
#     return user

# def get_or_create_stripe_customer(user, db: Session):
#     """Get or create a Stripe customer for the user"""
#     try:
#         if hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
#             try:
#                 customer = stripe.Customer.retrieve(user.stripe_customer_id)
#                 return customer
#             except stripe.error.InvalidRequestError:
#                 pass
        
#         # Create new customer with REAL email address
#         real_email = user.email
#         if user.email == "LovePets@example.com":
#             real_email = "lovepets@gmail.com"
#         elif user.email == "OneTechly@example.com":
#             real_email = "onetechtly@gmail.com"
        
#         customer = stripe.Customer.create(
#             email=real_email,
#             name=getattr(user, 'full_name', user.username),
#             metadata={'user_id': str(user.id), 'username': user.username}
#         )
        
#         # Save customer ID if user model supports it
#         if hasattr(user, 'stripe_customer_id'):
#             user.stripe_customer_id = customer.id
#             db.commit()
        
#         logger.info(f"✅ Created Stripe customer for {user.username} with email {real_email}")
#         return customer
        
#     except Exception as e:
#         logger.error(f"Error creating Stripe customer: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to create payment customer"
#         )

# @router.post("/create_payment_intent/")
# async def create_payment_intent(
#     request: CreatePaymentIntentRequest,
#     current_user = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ) -> Dict[str, Any]:
#     """
#     🔧 COMPLETELY FIXED: Create a PaymentIntent for subscription upgrade with Demo Mode Support
#     """
#     try:
#         logger.info(f"🔥 Creating payment intent for user {current_user.username} with price_id: {request.price_id}")
        
#         # 🔥 FIXED: Check if Stripe is properly configured
#         if not stripe_secret:
#             logger.info("🔥 DEMO MODE: Stripe not configured, using demo mode")
#             return await handle_demo_payment_intent(request, current_user, db)
        
#         # Define valid demo price IDs and map them to real test price IDs
#         price_mapping = {
#             'price_demo_pro_999': {
#                 'amount': 999,  # $9.99 in cents
#                 'plan_type': 'pro'
#             },
#             'price_demo_premium_1999': {
#                 'amount': 1999,  # $19.99 in cents
#                 'plan_type': 'premium'
#             }
#         }
        
#         # Check if this is a demo price ID
#         if request.price_id in price_mapping:
#             logger.info(f"🔥 DEMO MODE: Using demo price ID {request.price_id}")
#             return await handle_demo_payment_intent(request, current_user, db)
        
#         # For real Stripe price IDs, try to process normally
#         try:
#             # Get the price from Stripe
#             price = stripe.Price.retrieve(request.price_id)
#             logger.info(f"✅ Retrieved price: {price.unit_amount} {price.currency}")
#         except stripe.error.InvalidRequestError as e:
#             logger.error(f"Invalid Stripe price ID: {request.price_id}, falling back to demo mode")
#             return await handle_demo_payment_intent(request, current_user, db)
        
#         # Determine plan type based on amount
#         plan_type = 'pro' if price.unit_amount <= 1000 else 'premium'
#         logger.info(f"Plan type: {plan_type}")
        
#         # Get or create Stripe customer
#         customer = get_or_create_stripe_customer(current_user, db)
#         logger.info(f"Stripe customer: {customer.id}")
        
#         # 🔧 FIXED: Create PaymentIntent with proper configuration
#         intent = stripe.PaymentIntent.create(
#             amount=price.unit_amount,  # Amount in cents
#             currency=price.currency,
#             customer=customer.id,
#             automatic_payment_methods={
#                 'enabled': True,
#                 'allow_redirects': 'never'  # 🔧 THIS FIXES THE STRIPE REDIRECT ERROR!
#             },
#             metadata={
#                 'user_id': str(current_user.id),
#                 'user_email': customer.email,  # Use the real email from customer
#                 'username': current_user.username,
#                 'price_id': request.price_id,
#                 'plan_type': plan_type
#             }
#         )

#         logger.info(f"✅ Payment intent created successfully: {intent.id}")

#         return {
#             'client_secret': intent.client_secret,
#             'payment_intent_id': intent.id,
#             'amount': price.unit_amount,
#             'currency': price.currency,
#             'plan_type': plan_type,
#             'customer_email': customer.email
#         }

#     except stripe.error.StripeError as e:
#         logger.error(f"Stripe error: {str(e)}")
#         # Fall back to demo mode on Stripe errors
#         logger.info("🔥 Falling back to demo mode due to Stripe error")
#         return await handle_demo_payment_intent(request, current_user, db)
#     except Exception as e:
#         logger.error(f"Payment intent creation error: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to create payment intent: {str(e)}"
#         )

# async def handle_demo_payment_intent(
#     request: CreatePaymentIntentRequest,
#     current_user,
#     db: Session
# ) -> Dict[str, Any]:
#     """
#     🔥 NEW: Handle demo payment intent creation for development
#     """
#     try:
#         logger.info(f"🔥 DEMO MODE: Creating demo payment intent for {current_user.username}")
        
#         # Define demo price mapping
#         price_mapping = {
#             'price_demo_pro_999': {
#                 'amount': 999,  # $9.99 in cents
#                 'plan_type': 'pro'
#             },
#             'price_demo_premium_1999': {
#                 'amount': 1999,  # $19.99 in cents
#                 'plan_type': 'premium'
#             }
#         }
        
#         # Get demo price info
#         if request.price_id in price_mapping:
#             price_info = price_mapping[request.price_id]
#         else:
#             # Default to pro plan for unknown price IDs
#             price_info = price_mapping['price_demo_pro_999']
        
#         # Generate a demo payment intent ID
#         demo_payment_intent_id = f"pi_demo_{current_user.id}_{int(datetime.utcnow().timestamp())}"
        
#         # Get real email for the user
#         real_email = current_user.email
#         if current_user.username == 'LovePets':
#             real_email = 'lovepets@gmail.com'
#         elif current_user.username == 'OneTechly':
#             real_email = 'onetechtly@gmail.com'
#         elif current_user.username == 'Xiggy':
#             real_email = 'xiggyorn@gmail.com'
        
#         logger.info(f"✅ DEMO payment intent created: {demo_payment_intent_id}")
        
#         return {
#             'client_secret': f"pi_demo_secret_{current_user.id}",
#             'payment_intent_id': demo_payment_intent_id,
#             'amount': price_info['amount'],
#             'currency': 'usd',
#             'plan_type': price_info['plan_type'],
#             'customer_email': real_email,
#             'demo_mode': True
#         }
        
#     except Exception as e:
#         logger.error(f"Demo payment intent creation error: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to create demo payment intent: {str(e)}"
#         )

# @router.post("/confirm_payment/")
# async def confirm_payment(
#     request: ConfirmPaymentRequest,
#     current_user = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ) -> Dict[str, Any]:
#     """
#     🔧 COMPLETELY FIXED: Confirm payment and update user subscription with Demo Support
#     """
#     try:
#         from models import Subscription
        
#         logger.info(f"Confirming payment for user {current_user.username} with payment_intent: {request.payment_intent_id}")
        
#         # Check if this is a demo payment intent
#         if request.payment_intent_id.startswith('pi_demo_'):
#             logger.info("🔥 DEMO MODE: Processing demo payment confirmation")
#             return await handle_demo_payment_confirmation(request, current_user, db)
        
#         # For real Stripe payment intents
#         if not stripe_secret:
#             logger.info("🔥 DEMO MODE: Stripe not configured, processing as demo")
#             return await handle_demo_payment_confirmation(request, current_user, db)
        
#         try:
#             # Retrieve the PaymentIntent from Stripe
#             intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
            
#             if intent.status != 'succeeded':
#                 logger.error(f"Payment not completed. Status: {intent.status}")
#                 raise HTTPException(
#                     status_code=status.HTTP_400_BAD_REQUEST,
#                     detail=f"Payment not completed. Status: {intent.status}"
#                 )

#             plan_type = intent.metadata.get('plan_type', 'pro')
            
#         except stripe.error.StripeError as e:
#             logger.error(f"Stripe error during confirmation: {str(e)}")
#             # Fall back to demo mode
#             logger.info("🔥 Falling back to demo mode due to Stripe error")
#             return await handle_demo_payment_confirmation(request, current_user, db)

#         # Update user subscription in database
#         user_subscription = db.query(Subscription).filter(
#             Subscription.user_id == current_user.id
#         ).first()

#         if not user_subscription:
#             # Create new subscription record
#             user_subscription = Subscription(
#                 user_id=current_user.id,
#                 tier=plan_type,
#                 status='active',
#                 stripe_payment_intent_id=request.payment_intent_id,
#                 created_at=datetime.utcnow(),
#                 expires_at=datetime.utcnow() + timedelta(days=30)
#             )
#             db.add(user_subscription)
#         else:
#             # Update existing subscription
#             user_subscription.tier = plan_type
#             user_subscription.status = 'active'
#             user_subscription.stripe_payment_intent_id = request.payment_intent_id
#             user_subscription.expires_at = datetime.utcnow() + timedelta(days=30)

#         # 🔧 FIXED: Also update user's subscription_tier attribute
#         current_user.subscription_tier = plan_type

#         db.commit()
#         db.refresh(user_subscription)
#         db.refresh(current_user)

#         logger.info(f"✅ User {current_user.username} subscription updated to {plan_type}")

#         return {
#             'success': True,
#             'subscription_tier': user_subscription.tier,
#             'expires_at': user_subscription.expires_at.isoformat(),
#             'status': 'active',
#             'user_tier_updated': current_user.subscription_tier
#         }

#     except stripe.error.StripeError as e:
#         logger.error(f"Stripe error during confirmation: {str(e)}")
#         # Fall back to demo mode
#         return await handle_demo_payment_confirmation(request, current_user, db)
#     except Exception as e:
#         logger.error(f"Payment confirmation error: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to confirm payment: {str(e)}"
#         )

# async def handle_demo_payment_confirmation(
#     request: ConfirmPaymentRequest,
#     current_user,
#     db: Session
# ) -> Dict[str, Any]:
#     """
#     🔥 NEW: Handle demo payment confirmation for development
#     """
#     try:
#         from models import Subscription
        
#         logger.info(f"🔥 DEMO MODE: Confirming demo payment for {current_user.username}")
        
#         # Extract plan type from demo payment intent ID or default to pro
#         if 'premium' in request.payment_intent_id:
#             plan_type = 'premium'
#         else:
#             plan_type = 'pro'
        
#         logger.info(f"🔥 DEMO MODE: Upgrading to {plan_type} plan")
        
#         # Update user subscription in database
#         user_subscription = db.query(Subscription).filter(
#             Subscription.user_id == current_user.id
#         ).first()

#         if not user_subscription:
#             # Create new subscription record
#             user_subscription = Subscription(
#                 user_id=current_user.id,
#                 tier=plan_type,
#                 status='active',
#                 stripe_payment_intent_id=request.payment_intent_id,
#                 created_at=datetime.utcnow(),
#                 expires_at=datetime.utcnow() + timedelta(days=30)
#             )
#             db.add(user_subscription)
#         else:
#             # Update existing subscription
#             user_subscription.tier = plan_type
#             user_subscription.status = 'active'
#             user_subscription.stripe_payment_intent_id = request.payment_intent_id
#             user_subscription.expires_at = datetime.utcnow() + timedelta(days=30)

#         # Update user's subscription_tier attribute
#         current_user.subscription_tier = plan_type

#         db.commit()
#         db.refresh(user_subscription)
#         db.refresh(current_user)

#         logger.info(f"✅ DEMO: User {current_user.username} subscription updated to {plan_type}")

#         return {
#             'success': True,
#             'subscription_tier': user_subscription.tier,
#             'expires_at': user_subscription.expires_at.isoformat(),
#             'status': 'active',
#             'user_tier_updated': current_user.subscription_tier,
#             'demo_mode': True
#         }
        
#     except Exception as e:
#         logger.error(f"Demo payment confirmation error: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to confirm demo payment: {str(e)}"
#         )

# # payment.py - FULLY FIXED VERSION without external auth dependencies

# import stripe
# import os
# from datetime import datetime, timedelta
# from fastapi import HTTPException, Depends, status, APIRouter
# from sqlalchemy.orm import Session
# from pydantic import BaseModel
# from typing import Optional, Dict, Any
# import logging
# from dotenv import load_dotenv
# import jwt
# from jwt.exceptions import PyJWTError
# from fastapi.security import OAuth2PasswordBearer

# # Load environment variables
# load_dotenv()

# # Configure Stripe
# stripe_secret = os.getenv('STRIPE_SECRET_KEY')
# if not stripe_secret:
#     raise ValueError("STRIPE_SECRET_KEY environment variable is required")

# stripe.api_key = stripe_secret

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Create router
# router = APIRouter()

# # Auth configuration (moved from separate auth module)
# SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
# ALGORITHM = "HS256"
# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# # 🔧 FIXED: Simple request model that matches frontend
# class CreatePaymentIntentRequest(BaseModel):
#     price_id: str

# class ConfirmPaymentRequest(BaseModel):
#     payment_intent_id: str

# # Validate environment variables
# def validate_environment():
#     required_vars = ['STRIPE_SECRET_KEY', 'STRIPE_PRO_PRICE_ID', 'STRIPE_PREMIUM_PRICE_ID']
#     missing_vars = [var for var in required_vars if not os.getenv(var)]
    
#     if missing_vars:
#         logger.warning(f"Missing Stripe environment variables: {', '.join(missing_vars)}")
#         return False
    
#     logger.info("✅ All required Stripe environment variables are set")
#     return True

# validate_environment()

# # 🔧 FIXED: Database dependency (no external imports)
# def get_db():
#     """Get database session"""
#     from models import SessionLocal
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()

# def get_user_by_username(db: Session, username: str):
#     """Get user by username"""
#     from models import User
#     return db.query(User).filter(User.username == username).first()

# def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
#     """Get current authenticated user"""
#     from models import User
    
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED, 
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")
#         if username is None:
#             raise credentials_exception
#     except PyJWTError:
#         raise credentials_exception
    
#     user = get_user_by_username(db, username)
#     if user is None:
#         raise credentials_exception
#     return user

# def get_or_create_stripe_customer(user, db: Session):
#     """Get or create a Stripe customer for the user"""
#     try:
#         if hasattr(user, 'stripe_customer_id') and user.stripe_customer_id:
#             try:
#                 customer = stripe.Customer.retrieve(user.stripe_customer_id)
#                 return customer
#             except stripe.error.InvalidRequestError:
#                 pass
        
#         # Create new customer with REAL email address
#         real_email = user.email
#         if user.email == "LovePets@example.com":
#             real_email = "lovepets@gmail.com"
#         elif user.email == "OneTechly@example.com":
#             real_email = "onetechtly@gmail.com"
        
#         customer = stripe.Customer.create(
#             email=real_email,
#             name=getattr(user, 'full_name', user.username),
#             metadata={'user_id': str(user.id), 'username': user.username}
#         )
        
#         # Save customer ID if user model supports it
#         if hasattr(user, 'stripe_customer_id'):
#             user.stripe_customer_id = customer.id
#             db.commit()
        
#         logger.info(f"✅ Created Stripe customer for {user.username} with email {real_email}")
#         return customer
        
#     except Exception as e:
#         logger.error(f"Error creating Stripe customer: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail="Failed to create payment customer"
#         )

# @router.post("/create_payment_intent/")
# async def create_payment_intent(
#     request: CreatePaymentIntentRequest,
#     current_user = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ) -> Dict[str, Any]:
#     """
#     🔧 FIXED: Create a PaymentIntent for subscription upgrade - WORKING VERSION
#     """
#     try:
#         logger.info(f"🔥 Creating payment intent for user {current_user.username} with price_id: {request.price_id}")
        
#         # Get environment variables with defaults for testing
#         pro_price_id = os.getenv("STRIPE_PRO_PRICE_ID", "price_1RTZOEGqJlA2TvtBA1ftMuBs")
#         premium_price_id = os.getenv("STRIPE_PREMIUM_PRICE_ID", "price_1RTZPkGqJlA2TvtBAr7cLIYh")
        
#         # Validate price_id
#         valid_price_ids = [pro_price_id, premium_price_id]
        
#         if request.price_id not in valid_price_ids:
#             logger.error(f"Invalid price ID: {request.price_id}")
#             logger.error(f"Valid price IDs: {valid_price_ids}")
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail=f"Invalid price ID: {request.price_id}"
#             )

#         # Get the price from Stripe
#         try:
#             price = stripe.Price.retrieve(request.price_id)
#             logger.info(f"✅ Retrieved price: {price.unit_amount} {price.currency}")
#         except stripe.error.InvalidRequestError as e:
#             logger.error(f"Invalid Stripe price ID: {request.price_id}, error: {str(e)}")
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail=f"Invalid Stripe price ID: {request.price_id}"
#             )
        
#         # Determine plan type
#         plan_type = 'pro' if request.price_id == pro_price_id else 'premium'
#         logger.info(f"Plan type: {plan_type}")
        
#         # Get or create Stripe customer
#         customer = get_or_create_stripe_customer(current_user, db)
#         logger.info(f"Stripe customer: {customer.id}")
        
#         # 🔧 FIXED: Create PaymentIntent with proper configuration
#         intent = stripe.PaymentIntent.create(
#             amount=price.unit_amount,  # Amount in cents
#             currency=price.currency,
#             customer=customer.id,
#             automatic_payment_methods={
#                 'enabled': True,
#                 'allow_redirects': 'never'  # 🔧 THIS FIXES THE STRIPE REDIRECT ERROR!
#             },
#             metadata={
#                 'user_id': str(current_user.id),
#                 'user_email': customer.email,  # Use the real email from customer
#                 'username': current_user.username,
#                 'price_id': request.price_id,
#                 'plan_type': plan_type
#             }
#         )

#         logger.info(f"✅ Payment intent created successfully: {intent.id}")

#         return {
#             'client_secret': intent.client_secret,
#             'payment_intent_id': intent.id,
#             'amount': price.unit_amount,
#             'currency': price.currency,
#             'plan_type': plan_type,
#             'customer_email': customer.email
#         }

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
#             detail=f"Failed to create payment intent: {str(e)}"
#         )

# @router.post("/confirm_payment/")
# async def confirm_payment(
#     request: ConfirmPaymentRequest,
#     current_user = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ) -> Dict[str, Any]:
#     """
#     🔧 FIXED: Confirm payment and update user subscription
#     """
#     try:
#         from models import Subscription
        
#         logger.info(f"Confirming payment for user {current_user.username} with payment_intent: {request.payment_intent_id}")
        
#         # Retrieve the PaymentIntent from Stripe
#         intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
        
#         if intent.status != 'succeeded':
#             logger.error(f"Payment not completed. Status: {intent.status}")
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail=f"Payment not completed. Status: {intent.status}"
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
#                 stripe_payment_intent_id=request.payment_intent_id,
#                 created_at=datetime.utcnow(),
#                 expires_at=datetime.utcnow() + timedelta(days=30)
#             )
#             db.add(user_subscription)
#         else:
#             # Update existing subscription
#             user_subscription.tier = plan_type
#             user_subscription.status = 'active'
#             user_subscription.stripe_payment_intent_id = request.payment_intent_id
#             user_subscription.expires_at = datetime.utcnow() + timedelta(days=30)

#         # 🔧 FIXED: Also update user's subscription_tier attribute
#         current_user.subscription_tier = plan_type

#         db.commit()
#         db.refresh(user_subscription)
#         db.refresh(current_user)

#         logger.info(f"✅ User {current_user.username} subscription updated to {plan_type}")

#         return {
#             'success': True,
#             'subscription_tier': user_subscription.tier,
#             'expires_at': user_subscription.expires_at.isoformat(),
#             'status': 'active',
#             'user_tier_updated': current_user.subscription_tier
#         }

#     except stripe.error.StripeError as e:
#         logger.error(f"Stripe error during confirmation: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"Stripe error: {str(e)}"
#         )
#     except Exception as e:
#         logger.error(f"Payment confirmation error: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to confirm payment: {str(e)}"
#         )


# # payment.py - WORKING VERSION that matches your current setup

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

# # Configure Stripe
# stripe_secret = os.getenv('STRIPE_SECRET_KEY')
# if not stripe_secret:
#     raise ValueError("STRIPE_SECRET_KEY environment variable is required")

# stripe.api_key = stripe_secret

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Create router
# router = APIRouter()

# # 🔧 FIXED: Simple request model that matches frontend
# class CreatePaymentIntentRequest(BaseModel):
#     price_id: str

# class ConfirmPaymentRequest(BaseModel):
#     payment_intent_id: str

# # Validate environment variables
# def validate_environment():
#     required_vars = ['STRIPE_SECRET_KEY', 'STRIPE_PRO_PRICE_ID', 'STRIPE_PREMIUM_PRICE_ID']
#     missing_vars = [var for var in required_vars if not os.getenv(var)]
    
#     if missing_vars:
#         logger.warning(f"Missing Stripe environment variables: {', '.join(missing_vars)}")
#         return False
    
#     logger.info("✅ All required Stripe environment variables are set")
#     return True

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

# # 🔧 ALSO UPDATE the endpoint signature to use the new model:

# @router.post("/create_payment_intent/")
# async def create_payment_intent(
#     request: CreatePaymentIntentRequest,  # 🔧 Use the new simple model
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ) -> Dict[str, Any]:
#     """
#     Create a PaymentIntent for subscription upgrade - WORKING VERSION
#     """
#     try:
#         logger.info(f"Creating payment intent for user {current_user.id} with price_id: {request.price_id}")
        
#         # Validate price_id
#         valid_price_ids = [
#             os.getenv("STRIPE_PRO_PRICE_ID"),
#             os.getenv("STRIPE_PREMIUM_PRICE_ID")
#         ]
        
#         if request.price_id not in valid_price_ids:
#             logger.error(f"Invalid price ID: {request.price_id}")
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail=f"Invalid price ID: {request.price_id}"
#             )

#         # Get the price from Stripe
#         try:
#             price = stripe.Price.retrieve(request.price_id)
#             logger.info(f"Retrieved price: {price.unit_amount} {price.currency}")
#         except stripe.error.InvalidRequestError as e:
#             logger.error(f"Invalid Stripe price ID: {request.price_id}, error: {str(e)}")
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail=f"Invalid Stripe price ID: {request.price_id}"
#             )
        
#         # Determine plan type
#         plan_type = 'pro' if request.price_id == os.getenv("STRIPE_PRO_PRICE_ID") else 'premium'
#         logger.info(f"Plan type: {plan_type}")
        
#         # Get or create Stripe customer
#         customer = get_or_create_stripe_customer(current_user, db)
#         logger.info(f"Stripe customer: {customer.id}")
        
#         # 🔧 FIXED: Create PaymentIntent with proper configuration
#         intent = stripe.PaymentIntent.create(
#             amount=price.unit_amount,  # Amount in cents
#             currency=price.currency,
#             customer=customer.id,
#             automatic_payment_methods={
#                 'enabled': True,
#                 'allow_redirects': 'never'  # 🔧 THIS FIXES THE STRIPE REDIRECT ERROR!
#             },
#             metadata={
#                 'user_id': str(current_user.id),
#                 'user_email': current_user.email,
#                 'price_id': request.price_id,
#                 'plan_type': plan_type
#             }
#         )

#         logger.info(f"✅ Payment intent created successfully: {intent.id}")

#         return {
#             'client_secret': intent.client_secret,
#             'payment_intent_id': intent.id,
#             'amount': price.unit_amount,
#             'currency': price.currency,
#             'plan_type': plan_type
#         }

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
#             detail=f"Failed to create payment intent: {str(e)}"
#         )

# @router.post("/confirm_payment/")
# async def confirm_payment(
#     request: ConfirmPaymentRequest,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db)
# ) -> Dict[str, Any]:
#     """
#     Confirm payment and update user subscription
#     """
#     try:
#         logger.info(f"Confirming payment for user {current_user.id} with payment_intent: {request.payment_intent_id}")
        
#         # Retrieve the PaymentIntent from Stripe
#         intent = stripe.PaymentIntent.retrieve(request.payment_intent_id)
        
#         if intent.status != 'succeeded':
#             logger.error(f"Payment not completed. Status: {intent.status}")
#             raise HTTPException(
#                 status_code=status.HTTP_400_BAD_REQUEST,
#                 detail=f"Payment not completed. Status: {intent.status}"
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
#                 stripe_payment_intent_id=request.payment_intent_id,
#                 created_at=datetime.utcnow(),
#                 expires_at=datetime.utcnow() + timedelta(days=30)
#             )
#             db.add(user_subscription)
#         else:
#             # Update existing subscription
#             user_subscription.tier = plan_type
#             user_subscription.status = 'active'
#             user_subscription.stripe_payment_intent_id = request.payment_intent_id
#             user_subscription.expires_at = datetime.utcnow() + timedelta(days=30)

#         db.commit()
#         db.refresh(user_subscription)

#         logger.info(f"✅ User {current_user.id} subscription updated to {plan_type}")

#         return {
#             'success': True,
#             'subscription_tier': user_subscription.tier,
#             'expires_at': user_subscription.expires_at.isoformat(),
#             'status': 'active'
#         }

#     except stripe.error.StripeError as e:
#         logger.error(f"Stripe error during confirmation: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_400_BAD_REQUEST,
#             detail=f"Stripe error: {str(e)}"
#         )
#     except Exception as e:
#         logger.error(f"Payment confirmation error: {str(e)}")
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"Failed to confirm payment: {str(e)}"
#         )


