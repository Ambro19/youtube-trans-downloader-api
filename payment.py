# backend/payment.py
import os, logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel

# shared auth + DB
from auth_deps import get_current_user
from models import User, get_db  # get_db so we can persist stripe_customer_id

router = APIRouter(prefix="/billing")
logger = logging.getLogger("payment")

# --- Stripe (optional) -------------------------------------------------------
stripe = None
try:
    import stripe as _stripe  # type: ignore
    if os.getenv("STRIPE_SECRET_KEY"):
        _stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        stripe = _stripe
except Exception:
    stripe = None

FRONTEND_URL_ENV = os.getenv("FRONTEND_URL")  # may be None; we’ll fall back to Origin
PRO_LOOKUP  = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
PREM_LOOKUP = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")


# --- helpers -----------------------------------------------------------------
def _frontend_base(request: Request) -> str:
    """
    Pick the correct frontend base for redirects:
    1) explicit env FRONTEND_URL
    2) request Origin header (works on LAN like 192.168.x.x)
    3) fallback to http://localhost:3000
    """
    return (FRONTEND_URL_ENV
            or request.headers.get("origin")
            or "http://localhost:3000").rstrip("/")


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


def _ensure_customer_for_user(user: User, db_session, email_fallback: Optional[str] = None) -> Optional[str]:
    """
    Create or find a Stripe Customer for this user and store user.stripe_customer_id.
    We ALWAYS return a customer id so we can pass ONLY `customer` to Checkout.
    """
    if not stripe:
        return None

    # Already have one?
    if user.stripe_customer_id:
        return user.stripe_customer_id

    email = user.email or email_fallback
    name = user.username or (email or "User")

    try:
        # Look up by email first
        if email:
            existing = stripe.Customer.list(email=email, limit=1)
            if existing.data:
                cust_id = existing.data[0].id
            else:
                created = stripe.Customer.create(email=email, name=name)
                cust_id = created.id
        else:
            created = stripe.Customer.create(name=name)
            cust_id = created.id

        # Persist to DB for next time
        user.stripe_customer_id = cust_id
        try:
            db_session.add(user)
            db_session.commit()
        except Exception:
            db_session.rollback()
        return cust_id
    except Exception as e:
        logger.error("Could not ensure Stripe customer for user %s: %s", user.id, e)
        return None


# --- Schemas -----------------------------------------------------------------
class CheckoutBody(BaseModel):
    # Accept either { plan: 'pro'|'premium' } OR { price_lookup_key: 'pro_monthly'|... }
    plan: Optional[str] = None
    price_lookup_key: Optional[str] = None


# --- Routes ------------------------------------------------------------------
@router.get("/config")
def billing_config() -> Dict[str, Any]:
    """
    Expose price availability (optional). Safe for client.
    """
    if not stripe:
        return {"mode": "disabled", "is_demo": True, "pro_price_id": None, "premium_price_id": None}

    return {
        "mode": "test" if (stripe.api_key or "").startswith("sk_test_") else "live",
        "is_demo": (os.getenv("ENVIRONMENT", "development") != "production"),
        "pro_price_id": _get_price_id(PRO_LOOKUP),
        "premium_price_id": _get_price_id(PREM_LOOKUP),
    }


@router.post("/create_checkout_session")
def create_checkout_session(
    body: CheckoutBody,
    request: Request,
    current_user: User = Depends(get_current_user),
    db_session = Depends(get_db),
):
    if not stripe:
        raise HTTPException(status_code=503, detail="Payments are not configured on the server.")

    # pick lookup key
    if body.price_lookup_key:
        lookup_key = body.price_lookup_key
    elif (body.plan or "").lower() in ("pro", "premium"):
        lookup_key = PRO_LOOKUP if body.plan.lower() == "pro" else PREM_LOOKUP
    else:
        raise HTTPException(status_code=400, detail="Missing plan or price_lookup_key.")

    price_id = _get_price_id(lookup_key)
    if not price_id:
        raise HTTPException(status_code=400, detail=f"Unknown plan/price: {lookup_key}")

    # ensure a Customer and ALWAYS pass only `customer` (never customer_email)
    customer_id = _ensure_customer_for_user(current_user, db_session)
    if not customer_id:
        raise HTTPException(status_code=500, detail="Could not create/find Stripe customer.")

    base = _frontend_base(request)
    success_url = f"{base}/subscription?success=1&session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url  = f"{base}/subscription?canceled=1"

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,               # ✅ only this, no customer_email
            line_items=[{"price": price_id, "quantity": 1}],
            allow_promotion_codes=True,
            # A few niceties — feel free to tweak
            billing_address_collection="auto",
            subscription_data={"trial_period_days": None},
            success_url=success_url,
            cancel_url=cancel_url,
        )
        return {"url": session.url}
    except stripe.error.StripeError as e:
        logger.error("Failed to create Checkout session: %s", getattr(e, "user_message", str(e)))
        raise HTTPException(status_code=500, detail=getattr(e, "user_message", "Stripe error"))
    except Exception as e:
        logger.exception("Unexpected error creating Checkout session")
        raise HTTPException(status_code=500, detail="Failed to create checkout session")


@router.post("/create_portal_session")
def create_portal_session(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    if not stripe:
        raise HTTPException(status_code=503, detail="Payments are not configured on the server.")

    base = _frontend_base(request)

    # Prefer stored customer id; otherwise try to find by email (no DB update here)
    cust_id = current_user.stripe_customer_id
    if not cust_id and current_user.email:
        try:
            found = stripe.Customer.list(email=current_user.email, limit=1)
            if found.data:
                cust_id = found.data[0].id
        except Exception:
            cust_id = None

    if not cust_id:
        raise HTTPException(status_code=400, detail="No Stripe customer found for this account.")

    try:
        portal = stripe.billing_portal.Session.create(customer=cust_id, return_url=f"{base}/subscription")
        return {"url": portal.url}
    except Exception as e:
        logger.error("Failed to create billing portal session: %s", e)
        raise HTTPException(status_code=500, detail="Could not open billing portal")

###############################################################################
###############################################################################

# # backend/payment.py
# import os, logging
# from typing import Optional, List

# from fastapi import APIRouter, HTTPException, Depends, Request
# from pydantic import BaseModel

# from models import User, get_db, Session
# from auth_deps import get_current_user

# router = APIRouter(prefix="/billing")
# logger = logging.getLogger("payment")

# # Stripe (optional)
# stripe = None
# try:
#     import stripe as _stripe  # type: ignore
#     if os.getenv("STRIPE_SECRET_KEY"):
#         _stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
#         stripe = _stripe
# except Exception:
#     stripe = None

# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
# PRO_LOOKUP    = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
# PREM_LOOKUP   = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")

# def _pm_types() -> List[str]:
#     """
#     Explicitly list allowed payment methods so Stripe doesn't show Link.
#     You can override with STRIPE_PM_TYPES=card,cashapp,klarna if you want.
#     """
#     raw = os.getenv("STRIPE_PM_TYPES", "").strip()
#     if raw:
#         return [x.strip() for x in raw.split(",") if x.strip()]
#     # default layout you liked
#     return ["card", "cashapp", "klarna"]

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

# def _get_or_create_customer(db: Session, user: User) -> Optional[str]:
#     if not stripe:
#         return None
#     if user.stripe_customer_id:
#         return user.stripe_customer_id
#     try:
#         found = stripe.Customer.list(email=user.email, limit=1)
#         if found and found.data:
#             cid = found.data[0].id
#         else:
#             created = stripe.Customer.create(email=user.email, name=user.username)
#             cid = created.id
#         user.stripe_customer_id = cid
#         try:
#             db.add(user)
#             db.commit()
#         except Exception:
#             db.rollback()
#         return cid
#     except Exception as e:
#         logger.warning("Stripe get/create customer failed: %s", e)
#         return None

# class CheckoutBody(BaseModel):
#     plan: Optional[str] = None                 # 'pro' | 'premium' (optional)
#     price_lookup_key: Optional[str] = None     # preferred

# @router.get("/config")
# def get_config():
#     """
#     Small helper for the frontend to see price IDs exist and mode.
#     """
#     if not stripe:
#         return {
#             "mode": "disabled",
#             "is_demo": True,
#             "pro_price_id": None,
#             "premium_price_id": None,
#         }
#     return {
#         "mode": "test" if stripe.api_key and "test" in stripe.api_key else "live",
#         "is_demo": False,
#         "pro_price_id": _get_price_id(PRO_LOOKUP),
#         "premium_price_id": _get_price_id(PREM_LOOKUP),
#     }

# @router.post("/create_checkout_session")
# def create_checkout_session(
#     body: CheckoutBody,
#     request: Request,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db),
# ):
#     if not stripe:
#         raise HTTPException(status_code=503, detail="Stripe is not configured")

#     # Resolve lookup key
#     if body.price_lookup_key:
#         lookup = body.price_lookup_key
#     else:
#         plan = (body.plan or "").lower().strip()
#         if plan not in ("pro", "premium"):
#             raise HTTPException(status_code=400, detail="Invalid plan")
#         lookup = PRO_LOOKUP if plan == "pro" else PREM_LOOKUP

#     price_id = _get_price_id(lookup)
#     if not price_id:
#         raise HTTPException(status_code=400, detail="Price not found")

#     # Customer handling (avoid passing both customer and customer_email)
#     customer_id = _get_or_create_customer(db, current_user)

#     base_params = {
#         "mode": "subscription",
#         "line_items": [{"price": price_id, "quantity": 1}],
#         "success_url": f"{FRONTEND_URL}/subscription?upgraded=1&session_id={{CHECKOUT_SESSION_ID}}",
#         "cancel_url": f"{FRONTEND_URL}/subscription?canceled=1",
#         "allow_promotion_codes": True,
#         # Disable automatic_payment_methods so Link doesn't appear
#         "automatic_payment_methods": {"enabled": False},
#         # Keep taxes off by default in dev to avoid origin errors
#         "automatic_tax": {"enabled": False},
#         "subscription_data": {
#             "metadata": {
#                 "app_user_id": str(current_user.id),
#                 "app_plan_lookup": lookup,
#             }
#         },
#     }

#     # Identity
#     if customer_id:
#         base_params["customer"] = customer_id
#     else:
#         base_params["customer_email"] = current_user.email

#     # Prefer your "card/cashapp/klarna" layout; if Stripe rejects, fall back to card only
#     pm_types = _pm_types()
#     try:
#         session = stripe.checkout.Session.create(
#             **base_params, payment_method_types=pm_types
#         )
#     except Exception as e:
#         logger.warning("Retrying checkout with card-only due to: %s", e)
#         session = stripe.checkout.Session.create(
#             **base_params, payment_method_types=["card"]
#         )

#     return {"url": session.url}

# @router.post("/create_portal_session")
# def create_portal_session(
#     request: Request,
#     current_user: User = Depends(get_current_user),
#     db: Session = Depends(get_db),
# ):
#     if not stripe:
#         raise HTTPException(status_code=503, detail="Stripe is not configured")
#     customer_id = _get_or_create_customer(db, current_user)
#     if not customer_id:
#         raise HTTPException(status_code=400, detail="Stripe customer not found")
#     try:
#         portal = stripe.billing_portal.Session.create(
#             customer=customer_id,
#             return_url=f"{FRONTEND_URL}/subscription",
#         )
#         return {"url": portal.url}
#     except Exception as e:
#         logger.error("Failed to create portal session: %s", e)
#         raise HTTPException(status_code=500, detail="Failed to open billing portal")

##### THE NEWEST FILE FAILED TO RUN ##########################

# # backend/payment.py
# from __future__ import annotations

# import os
# from typing import Optional

# import stripe
# from fastapi import APIRouter, Depends, HTTPException, Request
# from pydantic import BaseModel

# from sqlalchemy.orm import Session

# from models import SessionLocal, User, initialize_database
# from auth import get_current_user  # whatever you already use to get the user

# # --- Config -------------------------------------------------------------------

# STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
# if not STRIPE_SECRET_KEY:
#     raise RuntimeError("STRIPE_SECRET_KEY missing")

# stripe.api_key = STRIPE_SECRET_KEY

# FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000").rstrip("/")

# PRO_LOOKUP = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
# PREMIUM_LOOKUP = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")

# router = APIRouter(prefix="/billing", tags=["billing"])


# # --- Helpers ------------------------------------------------------------------

# def get_db() -> Session:
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


# def _success_url() -> str:
#     # Always bring users back to the app’s Subscription page
#     return f"{FRONTEND_URL}/subscription?checkout=success&session_id={{CHECKOUT_SESSION_ID}}"


# def _cancel_url() -> str:
#     # Always bring users back to the app’s Subscription page (not /login)
#     return f"{FRONTEND_URL}/subscription?checkout=cancel"


# def _get_price_from_lookup(lookup_key: str) -> stripe.Price:
#     prices = stripe.Price.list(active=True, lookup_keys=[lookup_key], limit=1)
#     if not prices.data:
#         raise HTTPException(status_code=400, detail=f"Stripe price not found for {lookup_key!r}")
#     return prices.data[0]


# def _safe_retrieve_customer(customer_id: str) -> Optional[str]:
#     """
#     Returns customer_id if it exists at Stripe; otherwise returns None.
#     """
#     if not customer_id:
#         return None
#     try:
#         stripe.Customer.retrieve(customer_id)
#         return customer_id
#     except stripe.error.InvalidRequestError as e:
#         # Most common when the test customer was deleted (resource_missing)
#         if getattr(e, "code", "") == "resource_missing":
#             return None
#         raise


# def _resolve_or_create_customer(db: Session, user: User) -> Optional[str]:
#     """
#     1) If user.stripe_customer_id exists and is valid → use it
#     2) Else, try to find by email
#     3) Else, create a new customer
#     Updates DB when a new/changed id is determined.
#     """
#     cust_id = _safe_retrieve_customer(user.stripe_customer_id or "")
#     if cust_id:
#         return cust_id

#     # Try by email
#     result = stripe.Customer.list(email=user.email, limit=1)
#     if result.data:
#         cust_id = result.data[0].id
#     else:
#         created = stripe.Customer.create(
#             email=user.email,
#             metadata={"app_user_id": user.id, "username": user.username},
#         )
#         cust_id = created.id

#     # Persist for later
#     user.stripe_customer_id = cust_id
#     db.add(user)
#     db.commit()
#     return cust_id


# # --- Schemas ------------------------------------------------------------------

# class CheckoutPayload(BaseModel):
#     # Accept either price_lookup_key *or* plan ('pro'|'premium') for convenience
#     price_lookup_key: Optional[str] = None
#     plan: Optional[str] = None


# # --- Routes -------------------------------------------------------------------

# @router.get("/config")
# def billing_config():
#     """Frontend uses this to show mode/prices; lookup keys are preferred."""
#     pro = _get_price_from_lookup(PRO_LOOKUP)
#     premium = _get_price_from_lookup(PREMIUM_LOOKUP)
#     mode = "live" if stripe.api_key.startswith("sk_live_") else "test"
#     return {
#         "mode": mode,
#         "is_demo": mode == "test",
#         "pro_price_id": pro.id,
#         "premium_price_id": premium.id,
#         "pro_lookup_key": PRO_LOOKUP,
#         "premium_lookup_key": PREMIUM_LOOKUP,
#     }


# @router.post("/create_checkout_session")
# def create_checkout_session(
#     body: CheckoutPayload,
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user),
# ):
#     """
#     Create a Subscription Checkout session with:
#       - Link DISABLED (so you get the “many options” layout you prefer)
#       - Robust handling of stale stripe_customer_id on your users
#       - Correct success/cancel URLs so Back/Cancel takes users to /subscription
#     """
#     # Resolve desired lookup key
#     if body.price_lookup_key:
#         lookup = body.price_lookup_key.lower().strip()
#     elif body.plan:
#         p = body.plan.lower().strip()
#         if p not in ("pro", "premium"):
#             raise HTTPException(status_code=400, detail="Invalid plan")
#         lookup = PRO_LOOKUP if p == "pro" else PREMIUM_LOOKUP
#     else:
#         raise HTTPException(status_code=400, detail="Missing plan/price")

#     price = _get_price_from_lookup(lookup)

#     # Make sure we do NOT pass both customer and customer_email
#     # (this previously caused 400s). Also verify customer actually exists.
#     cust_id = _safe_retrieve_customer(current_user.stripe_customer_id or "")

#     params = dict(
#         mode="subscription",
#         line_items=[{"price": price.id, "quantity": 1}],
#         success_url=_success_url(),
#         cancel_url=_cancel_url(),
#         allow_promotion_codes=True,
#         # 👇 Turn OFF automatic payment methods (disables Link)
#         automatic_payment_methods={"enabled": False},
#         # 👇 Force the “card form” layout; Stripe can still show wallets where applicable
#         payment_method_types=["card"],
#         payment_method_collection="always",
#         client_reference_id=str(current_user.id),
#         metadata={"plan_lookup": lookup, "app_user_id": str(current_user.id)},
#         subscription_data={
#             "metadata": {"plan_lookup": lookup, "app_user_id": str(current_user.id)}
#         },
#         customer=None,
#         customer_email=None,
#     )

#     if cust_id:
#         params["customer"] = cust_id
#     else:
#         # Let Stripe create the customer at checkout
#         params["customer_email"] = current_user.email

#     try:
#         session = stripe.checkout.Session.create(**params)
#     except stripe.error.InvalidRequestError as e:
#         # Handle the common "No such customer" after test data was cleaned in Stripe
#         if getattr(e, "code", "") == "resource_missing" and getattr(e, "param", "") == "customer":
#             # Wipe the bad id and retry once with customer_email (card-only UI)
#             current_user.stripe_customer_id = None
#             db.add(current_user)
#             db.commit()
#             params["customer"] = None
#             params["customer_email"] = current_user.email
#             session = stripe.checkout.Session.create(**params)
#         else:
#             raise

#     return {"url": session.url}


# @router.post("/create_portal_session")
# def create_portal_session(
#     db: Session = Depends(get_db),
#     current_user: User = Depends(get_current_user),
# ):
#     """
#     Customer Portal for managing/upgrading/canceling.
#     Always returns to /subscription on your FRONTEND_URL.
#     """
#     # Ensure we have a valid customer
#     cust_id = _resolve_or_create_customer(db, current_user)

#     portal = stripe.billing_portal.Session.create(
#         customer=cust_id,
#         return_url=_success_url().split("?")[0],  # clean /subscription
#     )
#     return {"url": portal.url}
