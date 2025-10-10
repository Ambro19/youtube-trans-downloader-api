"""
Stripe webhook handler for automatic subscription upgrades.
Integrates with existing YouTube Content Downloader API.
"""
import os
import json
import logging
from datetime import datetime
from fastapi import Request, HTTPException
from sqlalchemy.orm import Session

# Import existing modules
from models import User, Subscription, SessionLocal

# Configure logging  
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import stripe (already configured in main.py)
try:
    import stripe
    stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
    STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
except ImportError:
    stripe = None
    STRIPE_WEBHOOK_SECRET = None

def get_db_session():
    """Get database session for webhook processing"""
    db = SessionLocal()
    return db

async def handle_stripe_webhook(request: Request):
    """
    Main webhook handler for Stripe events.
    Uses pre-verified event and updates user subscription tier/status.
    """
    if not stripe or not STRIPE_WEBHOOK_SECRET:
        raise HTTPException(status_code=503, detail="Stripe not configured")

    try:
        # ✅ Use already verified event (set in main.py)
        event = request.state.verified_event

        logger.info(f"✅ Stripe webhook received: {event['type']}")

        event_type = event['type']
        obj = event['data']['object']

        if event_type == 'checkout.session.completed':
            await handle_checkout_completed(obj)
        elif event_type == 'customer.subscription.created':
            await handle_subscription_created(obj)
        elif event_type == 'customer.subscription.updated':
            await handle_subscription_updated(obj)
        elif event_type == 'customer.subscription.deleted':
            await handle_subscription_cancelled(obj)
        elif event_type == 'invoice.payment_succeeded':
            await handle_payment_succeeded(obj)

        return {"status": "success", "processed": event_type}

    except Exception as e:
        logger.error(f"❌ Webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Webhook error: {str(e)}")


async def handle_checkout_completed(session):
    """Handle completed checkout session - upgrade user immediately"""
    logger.info(f"Processing checkout completion for session: {session['id']}")
    
    db = get_db_session()
    try:
        customer_email = session.get('customer_details', {}).get('email')
        customer_id = session.get('customer')
        
        if not customer_email:
            logger.warning("No customer email in checkout session")
            return
            
        # Find user by email
        user = db.query(User).filter(User.email == customer_email).first()
        if not user:
            logger.warning(f"User not found for email: {customer_email}")
            return
            
        # Update stripe customer ID if not set
        if not user.stripe_customer_id and customer_id:
            user.stripe_customer_id = customer_id
            
        # Determine tier from the checkout session
        subscription_tier = determine_tier_from_session(session)
        if subscription_tier and subscription_tier != "free":
            old_tier = user.subscription_tier
            user.subscription_tier = subscription_tier
            user.updated_at = datetime.utcnow()
            
            logger.info(f"✅ UPGRADED user {user.email} from {old_tier} to {subscription_tier}")
            
        db.commit()
        
    except Exception as e:
        logger.error(f"Error handling checkout completion: {e}")
        db.rollback()
    finally:
        db.close()

async def handle_subscription_created(subscription):
    """Handle new subscription creation"""
    logger.info(f"Processing subscription creation: {subscription['id']}")
    
    db = get_db_session()
    try:
        customer_id = subscription['customer']
        
        # Find user by stripe customer ID
        user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
        if not user:
            logger.warning(f"User not found for customer: {customer_id}")
            return
            
        # Determine tier from subscription
        tier = determine_tier_from_subscription(subscription)
        
        # Update user
        old_tier = user.subscription_tier
        user.subscription_tier = tier
        user.stripe_subscription_id = subscription['id']
        user.updated_at = datetime.utcnow()
        
        # Create subscription record
        sub_record = Subscription(
            user_id=user.id,
            tier=tier,
            status=subscription['status'],
            stripe_subscription_id=subscription['id'],
            stripe_customer_id=customer_id,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        # Set price if available
        if subscription.get('items') and subscription['items']['data']:
            price_data = subscription['items']['data'][0]['price']
            if price_data.get('unit_amount'):
                sub_record.price_paid = price_data['unit_amount'] / 100  # Convert from cents
                sub_record.currency = price_data.get('currency', 'usd')
        
        db.add(sub_record)
        db.commit()
        
        logger.info(f"✅ UPGRADED user {user.email} from {old_tier} to {tier}")
        
    except Exception as e:
        logger.error(f"Error handling subscription creation: {e}")
        db.rollback()
    finally:
        db.close()

async def handle_subscription_updated(subscription):
    """Handle subscription updates"""
    logger.info(f"Processing subscription update: {subscription['id']}")
    
    db = get_db_session()
    try:
        # Find user by subscription ID or customer ID
        user = db.query(User).filter(User.stripe_subscription_id == subscription['id']).first()
        if not user:
            # Try by customer ID
            user = db.query(User).filter(User.stripe_customer_id == subscription['customer']).first()
        
        if not user:
            logger.warning(f"User not found for subscription: {subscription['id']}")
            return
            
        # Update tier based on current subscription
        tier = determine_tier_from_subscription(subscription)
        old_tier = user.subscription_tier
        user.subscription_tier = tier
        user.stripe_subscription_id = subscription['id']
        user.updated_at = datetime.utcnow()
        
        # Update subscription record
        sub_record = db.query(Subscription).filter(
            Subscription.stripe_subscription_id == subscription['id']
        ).first()
        
        if sub_record:
            sub_record.tier = tier
            sub_record.status = subscription['status']
            sub_record.updated_at = datetime.utcnow()
            
        db.commit()
        
        logger.info(f"✅ UPDATED user {user.email} from {old_tier} to {tier}")
        
    except Exception as e:
        logger.error(f"Error handling subscription update: {e}")
        db.rollback()
    finally:
        db.close()

async def handle_subscription_cancelled(subscription):
    """Handle subscription cancellation"""
    logger.info(f"Processing subscription cancellation: {subscription['id']}")
    
    db = get_db_session()
    try:
        # Find user by subscription ID
        user = db.query(User).filter(User.stripe_subscription_id == subscription['id']).first()
        if not user:
            logger.warning(f"User not found for subscription: {subscription['id']}")
            return
            
        # Downgrade to free
        old_tier = user.subscription_tier
        user.subscription_tier = "free"
        user.stripe_subscription_id = None
        user.updated_at = datetime.utcnow()
        
        # Update subscription record
        sub_record = db.query(Subscription).filter(
            Subscription.stripe_subscription_id == subscription['id']
        ).first()
        
        if sub_record:
            sub_record.status = "cancelled"
            sub_record.cancelled_at = datetime.utcnow()
            sub_record.updated_at = datetime.utcnow()
            
        db.commit()
        
        logger.info(f"⬇️ DOWNGRADED user {user.email} from {old_tier} to free")
        
    except Exception as e:
        logger.error(f"Error handling subscription cancellation: {e}")
        db.rollback()
    finally:
        db.close()

async def handle_payment_succeeded(invoice):
    """Handle successful payment - ensure user stays upgraded"""
    logger.info(f"Processing successful payment: {invoice['id']}")
    
    # Get subscription from invoice
    if invoice.get('subscription'):
        try:
            subscription = stripe.Subscription.retrieve(invoice['subscription'])
            await handle_subscription_updated(subscription)
        except Exception as e:
            logger.error(f"Error processing payment success: {e}")

def determine_tier_from_session(session):
    """Determine subscription tier from checkout session"""
    # Check metadata first
    metadata = session.get('metadata', {})
    if metadata.get('tier'):
        return metadata['tier']
    
    # Check line items for amount
    if session.get('amount_total'):
        amount = session['amount_total'] / 100  # Convert from cents
        if amount >= 19.99:
            return 'premium'
        elif amount >= 9.99:
            return 'pro'
    
    return 'free'

def determine_tier_from_subscription(subscription):
    """Determine tier from subscription data"""
    if not subscription.get('items') or not subscription['items']['data']:
        return 'free'
        
    price_id = subscription['items']['data'][0]['price']['id']
    amount = subscription['items']['data'][0]['price']['unit_amount']
    
    if amount:
        amount_dollars = amount / 100
        if amount_dollars >= 19.99:
            return 'premium'
        elif amount_dollars >= 9.99:
            return 'pro'
    
    return 'free'

def fix_existing_premium_users():
    """
    One-time function to fix users who paid but weren't upgraded.
    Checks Stripe for active subscriptions and upgrades users accordingly.
    """
    if not stripe:
        logger.error("Stripe not configured")
        return {"error": "Stripe not configured"}
    
    db = SessionLocal()
    fixed_count = 0
    
    try:
        # Find users with stripe_customer_id but still on free tier
        users_to_check = db.query(User).filter(
            User.stripe_customer_id.isnot(None),
            User.subscription_tier == "free"
        ).all()
        
        logger.info(f"Checking {len(users_to_check)} users who might need fixing...")
        
        for user in users_to_check:
            try:
                # Check their Stripe subscriptions
                subscriptions = stripe.Subscription.list(
                    customer=user.stripe_customer_id, 
                    status='active',
                    limit=10
                )
                
                if subscriptions.data:
                    # They have active subscriptions, upgrade them
                    sub = subscriptions.data[0]
                    tier = determine_tier_from_subscription(sub)
                    
                    if tier != "free":
                        old_tier = user.subscription_tier
                        user.subscription_tier = tier
                        user.stripe_subscription_id = sub.id
                        user.updated_at = datetime.utcnow()
                        
                        # Create subscription record if not exists
                        existing_sub = db.query(Subscription).filter(
                            Subscription.stripe_subscription_id == sub.id
                        ).first()
                        
                        if not existing_sub:
                            sub_record = Subscription(
                                user_id=user.id,
                                tier=tier,
                                status=sub.status,
                                stripe_subscription_id=sub.id,
                                stripe_customer_id=user.stripe_customer_id,
                                created_at=datetime.utcnow(),
                                updated_at=datetime.utcnow()
                            )
                            if sub.items.data:
                                price_data = sub.items.data[0].price
                                if price_data.unit_amount:
                                    sub_record.price_paid = price_data.unit_amount / 100
                                    sub_record.currency = price_data.currency
                            db.add(sub_record)
                        
                        fixed_count += 1
                        logger.info(f"✅ FIXED user {user.email}: {old_tier} → {tier}")
                    
            except Exception as e:
                logger.error(f"Error checking user {user.email}: {e}")
                continue
                
        db.commit()
        logger.info(f"Fixed {fixed_count} users total")
        
        return {
            "status": "success",
            "fixed_count": fixed_count,
            "checked_count": len(users_to_check)
        }
        
    except Exception as e:
        logger.error(f"Error in fix function: {e}")
        db.rollback()
        return {"error": str(e)}
    finally:
        db.close()

