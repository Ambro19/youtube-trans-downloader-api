# """
# Stripe webhook handler for automatic subscription upgrades.
# Integrates with existing YouTube Content Downloader API.
# """

# import os
# import json
# import logging
# from datetime import datetime
# from fastapi import Request, HTTPException
# from sqlalchemy.orm import Session

# # Import existing modules
# from models import User, Subscription, SessionLocal

# # Configure logging  
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Import stripe (already configured in main.py)
# try:
#     import stripe
#     stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
#     STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
# except ImportError:
#     stripe = None
#     STRIPE_WEBHOOK_SECRET = None

# def get_db_session():
#     """Get database session for webhook processing"""
#     db = SessionLocal()
#     return db

# async def handle_stripe_webhook(request: Request):
#     """
#     Main webhook handler for Stripe events.
#     Automatically upgrades users when they pay.
#     """
#     if not stripe or not STRIPE_WEBHOOK_SECRET:
#         raise HTTPException(status_code=503, detail="Stripe not configured")
        
#     try:
#         payload = await request.body()
#         sig_header = request.headers.get('stripe-signature')
        
#         # Verify webhook signature
#         try:
#             event = stripe.Webhook.construct_event(
#                 payload, sig_header, STRIPE_WEBHOOK_SECRET
#             )
#         except ValueError as e:
#             logger.error(f"Invalid payload: {e}")
#             raise HTTPException(status_code=400, detail="Invalid payload")
#         except stripe.error.SignatureVerificationError as e:
#             logger.error(f"Invalid signature: {e}")
#             raise HTTPException(status_code=400, detail="Invalid signature")

#         logger.info(f"Received Stripe webhook: {event['type']}")
        
#         # Handle different event types
#         if event['type'] == 'checkout.session.completed':
#             await handle_checkout_completed(event['data']['object'])
#         elif event['type'] == 'customer.subscription.created':
#             await handle_subscription_created(event['data']['object'])
#         elif event['type'] == 'customer.subscription.updated':
#             await handle_subscription_updated(event['data']['object'])
#         elif event['type'] == 'customer.subscription.deleted':
#             await handle_subscription_cancelled(event['data']['object'])
#         elif event['type'] == 'invoice.payment_succeeded':
#             await handle_payment_succeeded(event['data']['object'])
        
#         return {"status": "success", "processed": event['type']}
        
#     except Exception as e:
#         logger.error(f"Webhook error: {str(e)}")
#         raise HTTPException(status_code=500, detail=f"Webhook error: {str(e)}")

# async def handle_checkout_completed(session):
#     """Handle completed checkout session - upgrade user immediately"""
#     logger.info(f"Processing checkout completion for session: {session['id']}")
    
#     db = get_db_session()
#     try:
#         customer_email = session.get('customer_details', {}).get('email')
#         customer_id = session.get('customer')
        
#         if not customer_email:
#             logger.warning("No customer email in checkout session")
#             return
            
#         # Find user by email
#         user = db.query(User).filter(User.email == customer_email).first()
#         if not user:
#             logger.warning(f"User not found for email: {customer_email}")
#             return
            
#         # Update stripe customer ID if not set
#         if not user.stripe_customer_id and customer_id:
#             user.stripe_customer_id = customer_id
            
#         # Determine tier from the checkout session
#         subscription_tier = determine_tier_from_session(session)
#         if subscription_tier and subscription_tier != "free":
#             old_tier = user.subscription_tier
#             user.subscription_tier = subscription_tier
#             user.updated_at = datetime.utcnow()
            
#             logger.info(f"✅ UPGRADED user {user.email} from {old_tier} to {subscription_tier}")
            
#         db.commit()
        
#     except Exception as e:
#         logger.error(f"Error handling checkout completion: {e}")
#         db.rollback()
#     finally:
#         db.close()

# async def handle_subscription_created(subscription):
#     """Handle new subscription creation"""
#     logger.info(f"Processing subscription creation: {subscription['id']}")
    
#     db = get_db_session()
#     try:
#         customer_id = subscription['customer']
        
#         # Find user by stripe customer ID
#         user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
#         if not user:
#             logger.warning(f"User not found for customer: {customer_id}")
#             return
            
#         # Determine tier from subscription
#         tier = determine_tier_from_subscription(subscription)
        
#         # Update user
#         old_tier = user.subscription_tier
#         user.subscription_tier = tier
#         user.stripe_subscription_id = subscription['id']
#         user.updated_at = datetime.utcnow()
        
#         # Create subscription record
#         sub_record = Subscription(
#             user_id=user.id,
#             tier=tier,
#             status=subscription['status'],
#             stripe_subscription_id=subscription['id'],
#             stripe_customer_id=customer_id,
#             created_at=datetime.utcnow(),
#             updated_at=datetime.utcnow()
#         )
        
#         # Set price if available
#         if subscription.get('items') and subscription['items']['data']:
#             price_data = subscription['items']['data'][0]['price']
#             if price_data.get('unit_amount'):
#                 sub_record.price_paid = price_data['unit_amount'] / 100  # Convert from cents
#                 sub_record.currency = price_data.get('currency', 'usd')
        
#         db.add(sub_record)
#         db.commit()
        
#         logger.info(f"✅ UPGRADED user {user.email} from {old_tier} to {tier}")
        
#     except Exception as e:
#         logger.error(f"Error handling subscription creation: {e}")
#         db.rollback()
#     finally:
#         db.close()

# async def handle_subscription_updated(subscription):
#     """Handle subscription updates"""
#     logger.info(f"Processing subscription update: {subscription['id']}")
    
#     db = get_db_session()
#     try:
#         # Find user by subscription ID or customer ID
#         user = db.query(User).filter(User.stripe_subscription_id == subscription['id']).first()
#         if not user:
#             # Try by customer ID
#             user = db.query(User).filter(User.stripe_customer_id == subscription['customer']).first()
        
#         if not user:
#             logger.warning(f"User not found for subscription: {subscription['id']}")
#             return
            
#         # Update tier based on current subscription
#         tier = determine_tier_from_subscription(subscription)
#         old_tier = user.subscription_tier
#         user.subscription_tier = tier
#         user.stripe_subscription_id = subscription['id']
#         user.updated_at = datetime.utcnow()
        
#         # Update subscription record
#         sub_record = db.query(Subscription).filter(
#             Subscription.stripe_subscription_id == subscription['id']
#         ).first()
        
#         if sub_record:
#             sub_record.tier = tier
#             sub_record.status = subscription['status']
#             sub_record.updated_at = datetime.utcnow()
            
#         db.commit()
        
#         logger.info(f"✅ UPDATED user {user.email} from {old_tier} to {tier}")
        
#     except Exception as e:
#         logger.error(f"Error handling subscription update: {e}")
#         db.rollback()
#     finally:
#         db.close()

# async def handle_subscription_cancelled(subscription):
#     """Handle subscription cancellation"""
#     logger.info(f"Processing subscription cancellation: {subscription['id']}")
    
#     db = get_db_session()
#     try:
#         # Find user by subscription ID
#         user = db.query(User).filter(User.stripe_subscription_id == subscription['id']).first()
#         if not user:
#             logger.warning(f"User not found for subscription: {subscription['id']}")
#             return
            
#         # Downgrade to free
#         old_tier = user.subscription_tier
#         user.subscription_tier = "free"
#         user.stripe_subscription_id = None
#         user.updated_at = datetime.utcnow()
        
#         # Update subscription record
#         sub_record = db.query(Subscription).filter(
#             Subscription.stripe_subscription_id == subscription['id']
#         ).first()
        
#         if sub_record:
#             sub_record.status = "cancelled"
#             sub_record.cancelled_at = datetime.utcnow()
#             sub_record.updated_at = datetime.utcnow()
            
#         db.commit()
        
#         logger.info(f"⬇️ DOWNGRADED user {user.email} from {old_tier} to free")
        
#     except Exception as e:
#         logger.error(f"Error handling subscription cancellation: {e}")
#         db.rollback()
#     finally:
#         db.close()

# async def handle_payment_succeeded(invoice):
#     """Handle successful payment - ensure user stays upgraded"""
#     logger.info(f"Processing successful payment: {invoice['id']}")
    
#     # Get subscription from invoice
#     if invoice.get('subscription'):
#         try:
#             subscription = stripe.Subscription.retrieve(invoice['subscription'])
#             await handle_subscription_updated(subscription)
#         except Exception as e:
#             logger.error(f"Error processing payment success: {e}")

# def determine_tier_from_session(session):
#     """Determine subscription tier from checkout session"""
#     # Check metadata first
#     metadata = session.get('metadata', {})
#     if metadata.get('tier'):
#         return metadata['tier']
    
#     # Check line items for amount
#     if session.get('amount_total'):
#         amount = session['amount_total'] / 100  # Convert from cents
#         if amount >= 19.99:
#             return 'premium'
#         elif amount >= 9.99:
#             return 'pro'
    
#     return 'free'

# def determine_tier_from_subscription(subscription):
#     """Determine tier from subscription data"""
#     if not subscription.get('items') or not subscription['items']['data']:
#         return 'free'
        
#     price_id = subscription['items']['data'][0]['price']['id']
#     amount = subscription['items']['data'][0]['price']['unit_amount']
    
#     if amount:
#         amount_dollars = amount / 100
#         if amount_dollars >= 19.99:
#             return 'premium'
#         elif amount_dollars >= 9.99:
#             return 'pro'
    
#     return 'free'

# def fix_existing_premium_users():
#     """
#     One-time function to fix users who paid but weren't upgraded.
#     Checks Stripe for active subscriptions and upgrades users accordingly.
#     """
#     if not stripe:
#         logger.error("Stripe not configured")
#         return {"error": "Stripe not configured"}
    
#     db = SessionLocal()
#     fixed_count = 0
    
#     try:
#         # Find users with stripe_customer_id but still on free tier
#         users_to_check = db.query(User).filter(
#             User.stripe_customer_id.isnot(None),
#             User.subscription_tier == "free"
#         ).all()
        
#         logger.info(f"Checking {len(users_to_check)} users who might need fixing...")
        
#         for user in users_to_check:
#             try:
#                 # Check their Stripe subscriptions
#                 subscriptions = stripe.Subscription.list(
#                     customer=user.stripe_customer_id, 
#                     status='active',
#                     limit=10
#                 )
                
#                 if subscriptions.data:
#                     # They have active subscriptions, upgrade them
#                     sub = subscriptions.data[0]
#                     tier = determine_tier_from_subscription(sub)
                    
#                     if tier != "free":
#                         old_tier = user.subscription_tier
#                         user.subscription_tier = tier
#                         user.stripe_subscription_id = sub.id
#                         user.updated_at = datetime.utcnow()
                        
#                         # Create subscription record if not exists
#                         existing_sub = db.query(Subscription).filter(
#                             Subscription.stripe_subscription_id == sub.id
#                         ).first()
                        
#                         if not existing_sub:
#                             sub_record = Subscription(
#                                 user_id=user.id,
#                                 tier=tier,
#                                 status=sub.status,
#                                 stripe_subscription_id=sub.id,
#                                 stripe_customer_id=user.stripe_customer_id,
#                                 created_at=datetime.utcnow(),
#                                 updated_at=datetime.utcnow()
#                             )
#                             if sub.items.data:
#                                 price_data = sub.items.data[0].price
#                                 if price_data.unit_amount:
#                                     sub_record.price_paid = price_data.unit_amount / 100
#                                     sub_record.currency = price_data.currency
#                             db.add(sub_record)
                        
#                         fixed_count += 1
#                         logger.info(f"✅ FIXED user {user.email}: {old_tier} → {tier}")
                    
#             except Exception as e:
#                 logger.error(f"Error checking user {user.email}: {e}")
#                 continue
                
#         db.commit()
#         logger.info(f"Fixed {fixed_count} users total")
        
#         return {
#             "status": "success",
#             "fixed_count": fixed_count,
#             "checked_count": len(users_to_check)
#         }
        
#     except Exception as e:
#         logger.error(f"Error in fix function: {e}")
#         db.rollback()
#         return {"error": str(e)}
#     finally:
#         db.close()

####################################################################################################
##########################################################################################

# backend/webhook_handler.py
import os
import json
import logging
from datetime import datetime
from typing import Optional

from fastapi import Request, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models import User, Subscription, get_db

logger = logging.getLogger("webhook_handler")

# Stripe setup
stripe = None
try:
    import stripe as _stripe
    if os.getenv("STRIPE_SECRET_KEY"):
        _stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        stripe = _stripe
except Exception:
    stripe = None

STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

# Database setup for webhook processing
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_downloader.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_webhook_db():
    """Get database session for webhook processing"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def find_user_by_customer_id(db: Session, customer_id: str) -> Optional[User]:
    """Find user by Stripe customer ID"""
    try:
        return db.query(User).filter(User.stripe_customer_id == customer_id).first()
    except Exception as e:
        logger.error(f"Error finding user by customer ID {customer_id}: {e}")
        return None

def find_user_by_email(db: Session, email: str) -> Optional[User]:
    """Find user by email as fallback"""
    try:
        return db.query(User).filter(User.email == email.lower().strip()).first()
    except Exception as e:
        logger.error(f"Error finding user by email {email}: {e}")
        return None

def update_user_subscription(db: Session, user: User, tier: str, stripe_subscription_id: str = None) -> bool:
    """Update user subscription tier and create/update subscription record"""
    try:
        # Update user tier
        old_tier = getattr(user, 'subscription_tier', 'free')
        user.subscription_tier = tier
        
        # Create or update subscription record
        existing_sub = db.query(Subscription).filter(
            Subscription.user_id == user.id,
            Subscription.stripe_subscription_id == stripe_subscription_id
        ).first() if stripe_subscription_id else None
        
        if existing_sub:
            existing_sub.status = 'active'
            existing_sub.tier = tier
            existing_sub.updated_at = datetime.utcnow()
        else:
            # Create new subscription record
            new_sub = Subscription(
                user_id=user.id,
                tier=tier,
                status='active',
                stripe_subscription_id=stripe_subscription_id,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            db.add(new_sub)
        
        db.commit()
        db.refresh(user)
        
        logger.info(f"✅ Updated user {user.username} ({user.email}) from {old_tier} to {tier}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update user {user.id} subscription: {e}")
        db.rollback()
        return False

def process_subscription_created(db: Session, subscription_data: dict) -> bool:
    """Process subscription.created event"""
    try:
        customer_id = subscription_data.get('customer')
        subscription_id = subscription_data.get('id')
        status = subscription_data.get('status')
        
        if not customer_id or not subscription_id:
            logger.warning("Missing customer_id or subscription_id in subscription.created event")
            return False
            
        # Find user
        user = find_user_by_customer_id(db, customer_id)
        if not user:
            logger.warning(f"User not found for customer_id: {customer_id}")
            return False
            
        # Determine tier based on subscription items
        tier = 'pro'  # Default to pro
        items = subscription_data.get('items', {}).get('data', [])
        if items:
            price_id = items[0].get('price', {}).get('id', '')
            # Check if it's premium based on price or lookup key
            price_lookup = items[0].get('price', {}).get('lookup_key', '')
            if 'premium' in price_lookup.lower() or 'premium' in price_id.lower():
                tier = 'premium'
                
        logger.info(f"Processing subscription.created: user={user.username}, tier={tier}, status={status}")
        
        if status in ['active', 'trialing']:
            return update_user_subscription(db, user, tier, subscription_id)
        else:
            logger.info(f"Subscription status {status} - not activating user yet")
            return True
            
    except Exception as e:
        logger.error(f"Error processing subscription.created: {e}")
        return False

def process_invoice_payment_succeeded(db: Session, invoice_data: dict) -> bool:
    """Process invoice.payment_succeeded event - more reliable for mobile payments"""
    try:
        customer_id = invoice_data.get('customer')
        subscription_id = invoice_data.get('subscription')
        
        if not customer_id:
            logger.warning("Missing customer_id in invoice.payment_succeeded event")
            return False
            
        # Find user
        user = find_user_by_customer_id(db, customer_id)
        if not user:
            # Try to find by email as fallback
            customer_email = invoice_data.get('customer_email')
            if customer_email:
                user = find_user_by_email(db, customer_email)
                if user:
                    # Update user's stripe customer ID
                    user.stripe_customer_id = customer_id
                    db.commit()
                    logger.info(f"Updated user {user.username} with customer_id {customer_id}")
                
        if not user:
            logger.warning(f"User not found for customer_id: {customer_id}, email: {invoice_data.get('customer_email')}")
            return False
            
        # Get subscription details from Stripe
        if stripe and subscription_id:
            try:
                subscription = stripe.Subscription.retrieve(subscription_id)
                tier = 'pro'  # Default
                
                # Determine tier from subscription items
                if subscription.items and subscription.items.data:
                    price = subscription.items.data[0].price
                    if price.lookup_key and 'premium' in price.lookup_key.lower():
                        tier = 'premium'
                    elif price.id and 'premium' in price.id.lower():
                        tier = 'premium'
                        
                logger.info(f"Processing invoice.payment_succeeded: user={user.username}, tier={tier}")
                return update_user_subscription(db, user, tier, subscription_id)
                
            except Exception as e:
                logger.error(f"Error retrieving subscription {subscription_id}: {e}")
                # Fall back to pro tier
                return update_user_subscription(db, user, 'pro', subscription_id)
        else:
            # No subscription ID, treat as one-time payment for pro
            logger.info(f"Processing one-time payment for user={user.username}")
            return update_user_subscription(db, user, 'pro')
            
    except Exception as e:
        logger.error(f"Error processing invoice.payment_succeeded: {e}")
        return False

def process_subscription_updated(db: Session, subscription_data: dict) -> bool:
    """Process subscription.updated event"""
    try:
        customer_id = subscription_data.get('customer')
        subscription_id = subscription_data.get('id')
        status = subscription_data.get('status')
        
        user = find_user_by_customer_id(db, customer_id)
        if not user:
            logger.warning(f"User not found for customer_id: {customer_id}")
            return False
            
        logger.info(f"Processing subscription.updated: user={user.username}, status={status}")
        
        # Update subscription status
        if status == 'canceled':
            return update_user_subscription(db, user, 'free')
        elif status in ['active', 'trialing']:
            # Determine current tier
            tier = 'pro'
            items = subscription_data.get('items', {}).get('data', [])
            if items:
                price_lookup = items[0].get('price', {}).get('lookup_key', '')
                if 'premium' in price_lookup.lower():
                    tier = 'premium'
            return update_user_subscription(db, user, tier, subscription_id)
        else:
            logger.info(f"Subscription status {status} - no action taken")
            return True
            
    except Exception as e:
        logger.error(f"Error processing subscription.updated: {e}")
        return False

async def handle_stripe_webhook(request: Request):
    """Main webhook handler for Stripe events"""
    if not stripe:
        raise HTTPException(status_code=503, detail="Stripe not configured")
        
    try:
        # Get raw payload and signature
        payload = await request.body()
        sig_header = request.headers.get('stripe-signature')
        
        if not sig_header:
            logger.warning("Missing Stripe signature header")
            raise HTTPException(status_code=400, detail="Missing signature")
            
        # Verify webhook signature
        if STRIPE_WEBHOOK_SECRET:
            try:
                event = stripe.Webhook.construct_event(
                    payload, sig_header, STRIPE_WEBHOOK_SECRET
                )
            except ValueError:
                logger.error("Invalid payload")
                raise HTTPException(status_code=400, detail="Invalid payload")
            except stripe.error.SignatureVerificationError:
                logger.error("Invalid signature")
                raise HTTPException(status_code=400, detail="Invalid signature")
        else:
            # No webhook secret configured - parse JSON directly (development only)
            try:
                event = json.loads(payload.decode('utf-8'))
            except json.JSONDecodeError:
                logger.error("Invalid JSON payload")
                raise HTTPException(status_code=400, detail="Invalid JSON")
                
        # Process the event
        event_type = event.get('type')
        event_data = event.get('data', {}).get('object', {})
        
        logger.info(f"🔔 Received webhook: {event_type}")
        
        # Get database session
        db = next(get_webhook_db())
        success = False
        
        try:
            if event_type == 'invoice.payment_succeeded':
                # Most reliable for mobile payments
                success = process_invoice_payment_succeeded(db, event_data)
            elif event_type == 'customer.subscription.created':
                success = process_subscription_created(db, event_data)
            elif event_type == 'customer.subscription.updated':
                success = process_subscription_updated(db, event_data)
            elif event_type == 'customer.subscription.deleted':
                customer_id = event_data.get('customer')
                user = find_user_by_customer_id(db, customer_id)
                if user:
                    success = update_user_subscription(db, user, 'free')
            else:
                logger.info(f"Unhandled event type: {event_type}")
                success = True  # Don't fail for unhandled events
                
        finally:
            db.close()
            
        if success:
            logger.info(f"✅ Successfully processed {event_type}")
            return {"status": "success", "event_type": event_type}
        else:
            logger.error(f"❌ Failed to process {event_type}")
            raise HTTPException(status_code=500, detail="Webhook processing failed")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Webhook error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

def fix_existing_premium_users():
    """One-time function to fix users who paid but weren't upgraded"""
    if not stripe:
        return {"error": "Stripe not configured"}
        
    try:
        db = next(get_webhook_db())
        results = []
        
        try:
            # Get all customers from Stripe who have active subscriptions
            customers = stripe.Customer.list(limit=100)
            
            for customer in customers.data:
                try:
                    # Get active subscriptions for this customer
                    subscriptions = stripe.Subscription.list(
                        customer=customer.id,
                        status='active',
                        limit=10
                    )
                    
                    if subscriptions.data:
                        # Find user in our database
                        user = find_user_by_customer_id(db, customer.id)
                        if not user and customer.email:
                            user = find_user_by_email(db, customer.email)
                            if user:
                                user.stripe_customer_id = customer.id
                                db.commit()
                                
                        if user:
                            current_tier = getattr(user, 'subscription_tier', 'free')
                            
                            # Determine correct tier from subscription
                            sub = subscriptions.data[0]  # Get first active subscription
                            correct_tier = 'pro'  # Default
                            
                            if sub.items and sub.items.data:
                                price = sub.items.data[0].price
                                if price.lookup_key and 'premium' in price.lookup_key.lower():
                                    correct_tier = 'premium'
                                elif price.id and 'premium' in price.id.lower():
                                    correct_tier = 'premium'
                                    
                            # Update if needed
                            if current_tier != correct_tier:
                                if update_user_subscription(db, user, correct_tier, sub.id):
                                    results.append({
                                        "user": user.username,
                                        "email": user.email,
                                        "old_tier": current_tier,
                                        "new_tier": correct_tier,
                                        "action": "upgraded"
                                    })
                                    
                except Exception as e:
                    logger.error(f"Error processing customer {customer.id}: {e}")
                    continue
                    
        finally:
            db.close()
            
        return {"results": results, "total_fixed": len(results)}
        
    except Exception as e:
        logger.error(f"Error in fix_existing_premium_users: {e}")
        return {"error": str(e)}