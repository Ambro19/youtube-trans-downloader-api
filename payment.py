# backend/payment.py
import os
import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.orm import Session
from pydantic import BaseModel

# ✅ shared auth dependency (no circular import)
from auth_deps import get_current_user
from models import User, get_db

router = APIRouter(prefix="/billing")
logger = logging.getLogger("payment")

# ---- Stripe init (robust / optional) ---------------------------------------
#-----To fix Stripe everywhere ------
stripe = None
try:
    import stripe as _stripe  # type: ignore
    key = (os.getenv("STRIPE_SECRET_KEY") or "").strip()   # ← strip here
    if key:
        _stripe.api_key = key
        stripe = _stripe
except Exception:
    stripe = None

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
PRO_LOOKUP   = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
PREM_LOOKUP  = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")


# ---- Models for small payloads ---------------------------------------------
class CheckoutPayload(BaseModel):
    # allow either plan/tier OR a price_lookup_key; frontend may send both
    plan: Optional[str] = None
    tier: Optional[str] = None
    price_lookup_key: Optional[str] = None


# ---- Helpers ----------------------------------------------------------------
def _get_price_id(lookup_key: str) -> Optional[str]:
    if not stripe:
        return None
    try:
        lst = stripe.Price.list(active=True, lookup_keys=[lookup_key], limit=1)
        if lst.data:
            return lst.data[0].id
    except Exception as e:
        logger.warning("Stripe price lookup failed for %s: %s", lookup_key, e)
    return None

def _get_or_create_customer(user: User, db: Session) -> str:
    """
    Get or create a Stripe customer for the user.
    Handles stale customer IDs and ensures a valid customer is returned.
    """
    if not stripe:
        raise HTTPException(status_code=503, detail="Stripe is not configured")
    
    # Get current stored customer ID
    stored_customer_id = (user.stripe_customer_id or "").strip() or None
    valid_customer_id = None
    
    # Step 1: Verify existing customer ID is still valid
    if stored_customer_id:
        try:
            customer = stripe.Customer.retrieve(stored_customer_id)
            if customer and not customer.get('deleted', False):
                valid_customer_id = stored_customer_id
                logger.info(f"Using existing valid customer ID: {valid_customer_id}")
            else:
                logger.warning(f"Stored customer ID {stored_customer_id} is deleted")
        except stripe.error.InvalidRequestError as e:
            if "No such customer" in str(e):
                logger.warning(f"Stored customer ID {stored_customer_id} does not exist in Stripe")
            else:
                logger.warning(f"Error verifying customer {stored_customer_id}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error verifying customer {stored_customer_id}: {e}")
    
    # Step 2: If no valid customer ID, try to find by email
    user_email = (user.email or "").strip().lower()
    if not valid_customer_id and user_email:
        try:
            customers = stripe.Customer.list(email=user_email, limit=1)
            if customers.data:
                valid_customer_id = customers.data[0].id
                logger.info(f"Found existing customer by email: {valid_customer_id}")
        except Exception as e:
            logger.warning(f"Error searching for customer by email {user_email}: {e}")
    
    # Step 3: Create new customer if none found
    if not valid_customer_id:
        try:
            customer_data = {
                "email": user_email,
                "name": (user.username or user.email or "User").strip(),
            }
            # Add metadata for tracking
            customer_data["metadata"] = {
                "user_id": str(user.id),
                "created_by": "youtube_downloader_app"
            }
            
            customer = stripe.Customer.create(**customer_data)
            valid_customer_id = customer.id
            logger.info(f"Created new customer: {valid_customer_id}")
        except Exception as e:
            msg = getattr(e, "user_message", None) or str(e)
            logger.error(f"Failed to create Stripe customer: {e}")
            raise HTTPException(status_code=500, detail=f"Could not create Stripe customer: {msg}")
    
    # Step 4: Update user record if customer ID changed
    if valid_customer_id != stored_customer_id:
        try:
            user.stripe_customer_id = valid_customer_id
            db.commit()
            db.refresh(user)
            logger.info(f"Updated user {user.id} with customer ID: {valid_customer_id}")
        except Exception as e:
            logger.error(f"Failed to update user with customer ID: {e}")
            db.rollback()
            # Don't fail the request if DB update fails, we can still proceed with checkout
    
    return valid_customer_id


# ---- Public endpoints -------------------------------------------------------
@router.get("/config")
def billing_config():
    """
    Expose which prices the server found (safe to show in UI).
    """
    if not stripe:
        return {
            "mode": "test",
            "is_demo": True,
            "pro_price_id": None,
            "premium_price_id": None,
        }
    pro = _get_price_id(PRO_LOOKUP)
    prem = _get_price_id(PREM_LOOKUP)
    return {
        "mode": "live" if (stripe.api_key or "").startswith("sk_live_") else "test",
        "is_demo": False,
        "pro_price_id": pro,
        "premium_price_id": prem,
        "pro_lookup_key": PRO_LOOKUP,
        "premium_lookup_key": PREM_LOOKUP,
    }


@router.post("/create_checkout_session")
def create_checkout_session(
    payload: CheckoutPayload,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not stripe:
        raise HTTPException(status_code=503, detail="Payments are not configured on this server.")

    # Resolve lookup key from payload (plan/tier OR explicit lookup_key)
    plan_or_tier = (payload.plan or payload.tier or "").strip().lower()
    lookup_key = payload.price_lookup_key
    if not lookup_key:
        if plan_or_tier == "pro":
            lookup_key = PRO_LOOKUP
        elif plan_or_tier == "premium":
            lookup_key = PREM_LOOKUP

    if not lookup_key:
        raise HTTPException(status_code=400, detail="Missing plan/price_lookup_key")

    price_id = _get_price_id(lookup_key)
    if not price_id:
        raise HTTPException(status_code=400, detail=f"Unknown price for lookup key: {lookup_key}")

    # Get or create customer (this handles all the stale ID logic)
    try:
        customer_id = _get_or_create_customer(user, db)
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Unexpected error getting customer: {e}")
        raise HTTPException(status_code=500, detail="Error setting up customer account")

    # Build session params - simplified and working
    try:
        session_params = {
            "mode": "subscription",
            "customer": customer_id,
            "line_items": [{"price": price_id, "quantity": 1}],
            "success_url": f"{FRONTEND_URL}/subscription?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
            "cancel_url": f"{FRONTEND_URL}/subscription",

            # --- make UI predictable ---
            "payment_method_types": ["card"],   # card only
            "phone_number_collection": {"enabled": False},  # removes phone row
            "billing_address_collection": "auto",
            "allow_promotion_codes": True,
            "automatic_tax": {"enabled": False},
        }

        session = stripe.checkout.Session.create(**session_params)
        logger.info(f"Created checkout session {session.id} for customer {customer_id}")
        return {"url": session.url}
        
    except Exception as e:
        logger.error(f"Failed to create Checkout session: {e}", exc_info=True)
        msg = getattr(e, "user_message", None) or str(e)
        raise HTTPException(status_code=500, detail=f"Failed to create checkout session: {msg}")

@router.post("/create_portal_session")
def create_portal_session(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not stripe:
        raise HTTPException(status_code=503, detail="Billing portal is not configured on this server.")
    
    # Ensure we have a valid customer ID
    try:
        customer_id = _get_or_create_customer(user, db)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting customer for portal: {e}")
        raise HTTPException(status_code=500, detail="Error accessing customer account")

    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f"{FRONTEND_URL}/subscription",
        )
        logger.info(f"Created portal session for customer {customer_id}")
        return {"url": session.url}
        
    except Exception as e:
        logger.error(f"Failed to create portal session: {e}")
        msg = getattr(e, "user_message", None) or str(e)
        raise HTTPException(status_code=500, detail=f"Could not open billing portal: {msg}")