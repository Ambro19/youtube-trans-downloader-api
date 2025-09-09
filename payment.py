# # backend/payment.py
# import os
# import logging
# from typing import Optional

# from fastapi import APIRouter, HTTPException, Depends
# from sqlalchemy.orm import Session
# from pydantic import BaseModel

# # ✅ shared auth dependency (no circular import)
# from auth_deps import get_current_user
# from models import User, get_db

# router = APIRouter(prefix="/billing")
# logger = logging.getLogger("payment")

# # ---- Stripe init (robust / optional) ---------------------------------------
# stripe = None
# try:
#     import stripe as _stripe  # type: ignore
#     if os.getenv("STRIPE_SECRET_KEY"):
#         _stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
#         stripe = _stripe
# except Exception:
#     stripe = None

# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
# PRO_LOOKUP   = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
# PREM_LOOKUP  = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")


# # ---- Models for small payloads ---------------------------------------------
# class CheckoutPayload(BaseModel):
#     # allow either plan/tier OR a price_lookup_key; frontend may send both
#     plan: Optional[str] = None
#     tier: Optional[str] = None
#     price_lookup_key: Optional[str] = None


# # ---- Helpers ----------------------------------------------------------------
# def _get_price_id(lookup_key: str) -> Optional[str]:
#     if not stripe:
#         return None
#     try:
#         lst = stripe.Price.list(active=True, lookup_keys=[lookup_key], limit=1)
#         if lst.data:
#             return lst.data[0].id
#     except Exception as e:
#         logger.warning("Stripe price lookup failed for %s: %s", lookup_key, e)
#     return None


# # ---- Public endpoints -------------------------------------------------------
# @router.get("/config")
# def billing_config():
#     """
#     Expose which prices the server found (safe to show in UI).
#     """
#     if not stripe:
#         return {
#             "mode": "test",
#             "is_demo": True,
#             "pro_price_id": None,
#             "premium_price_id": None,
#         }
#     pro = _get_price_id(PRO_LOOKUP)
#     prem = _get_price_id(PREM_LOOKUP)
#     return {
#         "mode": "live" if (stripe.api_key or "").startswith("sk_live_") else "test",
#         "is_demo": False,
#         "pro_price_id": pro,
#         "premium_price_id": prem,
#         "pro_lookup_key": PRO_LOOKUP,
#         "premium_lookup_key": PREM_LOOKUP,
#     }


# @router.post("/create_checkout_session")
# def create_checkout_session(
#     payload: CheckoutPayload,
#     user: User = Depends(get_current_user),
#     db: Session = Depends(get_db),
# ):
#     if not stripe:
#         raise HTTPException(status_code=503, detail="Payments are not configured on this server.")

#     # Resolve lookup key from payload (plan/tier OR explicit lookup_key)
#     plan_or_tier = (payload.plan or payload.tier or "").strip().lower()
#     lookup_key = payload.price_lookup_key
#     if not lookup_key:
#         if plan_or_tier == "pro":
#             lookup_key = PRO_LOOKUP
#         elif plan_or_tier == "premium":
#             lookup_key = PREM_LOOKUP

#     if not lookup_key:
#         raise HTTPException(status_code=400, detail="Missing plan/price_lookup_key")

#     price_id = _get_price_id(lookup_key)
#     if not price_id:
#         raise HTTPException(status_code=400, detail=f"Unknown price for lookup key: {lookup_key}")

#     # Ensure we have or create a Stripe customer (store on the user)
#     cust_id = (user.stripe_customer_id or "").strip() or None

#     # Verify existing ID still valid
#     if cust_id:
#         try:
#             stripe.Customer.retrieve(cust_id)
#         except Exception:
#             cust_id = None

#     # Try find by email if missing
#     if not cust_id:
#         try:
#             found = stripe.Customer.list(email=(user.email or "").strip().lower(), limit=1)
#             if found.data:
#                 cust_id = found.data[0].id
#         except Exception as e:
#             logger.warning("Stripe customer search failed: %s", e)

#     # Create as last resort
#     if not cust_id:
#         try:
#             cust = stripe.Customer.create(
#                 email=(user.email or "").strip().lower(),
#                 name=(user.username or user.email or "User").strip(),
#             )
#             cust_id = cust.id
#         except Exception as e:
#             msg = getattr(e, "user_message", None) or str(e)
#             raise HTTPException(status_code=500, detail=f"Could not create Stripe customer: {msg}")

#     # Persist customer id
#     try:
#         if user.stripe_customer_id != cust_id:
#             user.stripe_customer_id = cust_id
#             db.commit()
#     except Exception:
#         db.rollback()

#     # Build session params – IMPORTANT: send ONLY ONE of {customer, customer_email}
#     params = {
#         "mode": "subscription",
#         "line_items": [{"price": price_id, "quantity": 1}],
#         "success_url": f"{FRONTEND_URL}/subscription?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
#         "cancel_url": f"{FRONTEND_URL}/subscription?checkout=cancel",
#         "allow_promotion_codes": True,
#         "billing_address_collection": "auto",
#         # Avoid test-mode tax config errors:
#         "automatic_tax": {"enabled": False},
#     }

#     if cust_id:
#         params["customer"] = cust_id
#     else:
#         # (fallback only; with cust_id present we must NOT pass customer_email)
#         params["customer_email"] = (user.email or "").strip().lower()

#     try:
#         session = stripe.checkout.Session.create(**params)
#         return {"url": session.url}
#     except Exception as e:
#         logger.error("Failed to create Checkout session: %s", e, exc_info=True)
#         msg = getattr(e, "user_message", None) or str(e)
#         raise HTTPException(status_code=500, detail=f"Failed to create checkout session: {msg}")


# @router.post("/create_portal_session")
# def create_portal_session(
#     user: User = Depends(get_current_user),
# ):
#     if not stripe:
#         raise HTTPException(status_code=503, detail="Billing portal is not configured on this server.")
#     if not user.stripe_customer_id:
#         raise HTTPException(status_code=400, detail="No Stripe customer found for this account.")

#     try:
#         sess = stripe.billing_portal.Session.create(
#             customer=user.stripe_customer_id,
#             return_url=f"{FRONTEND_URL}/subscription",
#         )
#         return {"url": sess.url}
#     except Exception as e:
#         msg = getattr(e, "user_message", None) or str(e)
#         raise HTTPException(status_code=500, detail=f"Could not open billing portal: {msg}")

###############################################

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
stripe = None
try:
    import stripe as _stripe  # type: ignore
    if os.getenv("STRIPE_SECRET_KEY"):
        _stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
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

    # Build session params with consistent payment methods
    try:
        session_params = {
            "mode": "subscription",
            "customer": customer_id,  # Always use customer ID, never email
            "line_items": [{"price": price_id, "quantity": 1}],
            "success_url": f"{FRONTEND_URL}/subscription?checkout=success&session_id={{CHECKOUT_SESSION_ID}}",
            "cancel_url": f"{FRONTEND_URL}/subscription?checkout=cancel",
            "allow_promotion_codes": True,
            "billing_address_collection": "auto",
            "automatic_tax": {"enabled": False},  # Avoid test-mode tax config errors
            # Force consistent card-only design by completely disabling Link and other payment methods
            "payment_method_types": ["card"],  # Only allow cards
            "automatic_payment_methods": {"enabled": False},  # Disable automatic payment method detection
            "payment_method_options": {
                "card": {
                    "setup_future_usage": "off_session"  # Disable Link for cards
                }
            },
            # Additional Link-specific disabling
            "payment_method_configuration": None,  # Don't use any saved payment method configurations
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