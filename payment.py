# backend/payment.py
# Subscription-first Stripe integration using Checkout + Webhooks
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import os, logging, jwt, stripe

from models import User, get_db

router = APIRouter(prefix="/billing", tags=["billing"])
log = logging.getLogger("payment")

# --- ENV
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")  # set this!

# Prefer lookup keys; fall back to explicit price IDs if provided.
PRO_LOOKUP = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
PREMIUM_LOOKUP = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")
PRO_PRICE_ID = os.getenv("STRIPE_PRO_PRICE_ID")
PREMIUM_PRICE_ID = os.getenv("STRIPE_PREMIUM_PRICE_ID")

SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
    log.info("✅ Stripe configured (subscriptions)")
else:
    log.error("❌ STRIPE_SECRET_KEY missing. Billing will run in demo mode.")

# ---------- helpers

def _auth_user(request: Request, db: Session) -> User:
    auth = request.headers.get("authorization") or ""
    if not auth.startswith("Bearer "):
        raise HTTPException(401, "Missing bearer token")
    token = auth.split()[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
    except Exception:
        raise HTTPException(401, "Invalid token")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(401, "User not found")
    return user

def _resolve_price_id(tier: str) -> str:
    if tier not in ("pro", "premium"):
        raise HTTPException(400, "Unknown tier")
    # explicit ID wins
    if tier == "pro" and PRO_PRICE_ID:
        return PRO_PRICE_ID
    if tier == "premium" and PREMIUM_PRICE_ID:
        return PREMIUM_PRICE_ID
    # lookup key
    lookup = PRO_LOOKUP if tier == "pro" else PREMIUM_LOOKUP
    try:
        res = stripe.Price.list(active=True, lookup_keys=[lookup], limit=1)
        if res.data:
            return res.data[0].id
    except Exception as e:
        log.error(f"Stripe lookup failed: {e}")
    raise HTTPException(500, f"Stripe price for '{tier}' not found")

def _get_or_create_customer(db: Session, user: User) -> str:
    cust_id = getattr(user, "stripe_customer_id", None)
    if cust_id:
        try:
            stripe.Customer.retrieve(cust_id)
            return cust_id
        except Exception:
            pass
    customer = stripe.Customer.create(
        email=(user.email or "").strip().lower(),
        name=user.username or user.email,
        metadata={"app_user_id": str(user.id)}
    )
    setattr(user, "stripe_customer_id", customer.id)
    db.add(user); db.commit(); db.refresh(user)
    return customer.id

# ---------- endpoints

@router.get("/config")
def get_config():
    """Frontend can use this for display; not security-sensitive."""
    pro = None; premium = None
    try:
        pro = PRO_PRICE_ID or stripe.Price.list(active=True, lookup_keys=[PRO_LOOKUP], limit=1).data[0].id
    except Exception: pass
    try:
        premium = PREMIUM_PRICE_ID or stripe.Price.list(active=True, lookup_keys=[PREMIUM_LOOKUP], limit=1).data[0].id
    except Exception: pass
    return {
        "publishableKey": STRIPE_PUBLISHABLE_KEY,
        "pro_price_id": pro,
        "premium_price_id": premium
    }

@router.post("/create_checkout_session")
def create_checkout_session(payload: dict, request: Request, db: Session = Depends(get_db)):
    """
    Body: { "tier": "pro" | "premium" }
    Returns: { url } for redirect to Stripe Checkout (subscription mode).
    """
    user = _auth_user(request, db)
    tier = (payload or {}).get("tier", "pro")
    price_id = _resolve_price_id(tier)
    customer_id = _get_or_create_customer(db, user)

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{FRONTEND_URL}/subscription?status=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{FRONTEND_URL}/subscription?status=cancel",
            allow_promotion_codes=True,
            client_reference_id=str(user.id),
            subscription_data={"metadata": {"tier": tier, "app_user_id": str(user.id)}},
            metadata={"tier": tier, "app_user_id": str(user.id)},
        )
        return {"url": session.url}
    except Exception as e:
        log.exception("Failed to create Checkout session")
        raise HTTPException(500, str(e))

@router.post("/create_portal_session")
def create_portal_session(request: Request, db: Session = Depends(get_db)):
    """Return a customer billing portal URL."""
    user = _auth_user(request, db)
    if not getattr(user, "stripe_customer_id", None):
        raise HTTPException(400, "No Stripe customer on file")
    try:
        portal = stripe.billing_portal.Session.create(
            customer=user.stripe_customer_id,
            return_url=f"{FRONTEND_URL}/subscription"
        )
        return {"url": portal.url}
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/cancel_subscription")
def cancel_subscription(request: Request, db: Session = Depends(get_db)):
    """Immediate cancel (no proration logic here)."""
    user = _auth_user(request, db)
    sub_id = getattr(user, "stripe_subscription_id", None)
    if not sub_id:
        raise HTTPException(400, "No subscription to cancel")
    try:
        stripe.Subscription.delete(sub_id)
        setattr(user, "subscription_tier", "free")
        db.add(user); db.commit(); db.refresh(user)
        return {"status": "canceled"}
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/webhook")
async def webhook(request: Request, db: Session = Depends(get_db)):
    if not WEBHOOK_SECRET:
        # Accept but warn in logs so dev flow works without CLI
        log.warning("STRIPE_WEBHOOK_SECRET missing; skipping signature verification")
        try:
            event = (await request.json())
        except Exception as e:
            raise HTTPException(400, f"Invalid payload: {e}")
    else:
        payload = await request.body()
        sig = request.headers.get("stripe-signature")
        try:
            event = stripe.Webhook.construct_event(payload, sig, WEBHOOK_SECRET)
        except Exception as e:
            raise HTTPException(400, f"Invalid signature: {e}")

    etype = event["type"]
    data = event["data"]["object"]

    try:
        # 1) Checkout finished -> mark user tier & save subscription id
        if etype == "checkout.session.completed" and data.get("mode") == "subscription":
            customer_id = data.get("customer")
            sub_id = data.get("subscription")
            meta = data.get("metadata") or {}
            tier = meta.get("tier") or "pro"

            user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
            if not user:
                # fallback: client_reference_id or metadata user id
                ref = data.get("client_reference_id") or meta.get("app_user_id")
                if ref:
                    user = db.get(User, int(ref))

            if user:
                setattr(user, "stripe_subscription_id", sub_id)
                setattr(user, "subscription_tier", tier)
                db.add(user); db.commit()

        # 2) Ongoing updates (renewal, pause, cancel, etc.)
        elif etype in ("customer.subscription.updated", "customer.subscription.created"):
            sub = data
            customer_id = sub["customer"]
            status = sub["status"]
            price_id = None
            try:
                price_id = sub["items"]["data"][0]["price"]["id"]
            except Exception:
                pass
            # infer tier from price id
            tier = "premium" if price_id and (
                price_id == PREMIUM_PRICE_ID
            ) else "pro"

            user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
            if user:
                setattr(user, "stripe_subscription_id", sub["id"])
                setattr(user, "subscription_tier", tier if status in ("active", "trialing") else "free")
                db.add(user); db.commit()

        elif etype in ("customer.subscription.deleted",):
            sub = data
            customer_id = sub["customer"]
            user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
            if user:
                setattr(user, "subscription_tier", "free")
                db.add(user); db.commit()

    except Exception as e:
        log.exception(f"Webhook handling failed for {etype}")
        # Always 200 to avoid Stripe retries storm if your DB hiccups; log loudly.
        return JSONResponse({"ok": False, "error": str(e)})

    return JSONResponse({"ok": True})


#=================== The latest GOOD payment.py File ==============

# # payment.py — drop-in
# from fastapi import APIRouter, Depends, HTTPException, status, Request
# from fastapi.security import OAuth2PasswordBearer
# from sqlalchemy.orm import Session
# from datetime import datetime, timedelta
# from typing import Optional, Dict, Any
# import os
# import jwt
# import stripe
# import logging

# # ====== setup ======
# logger = logging.getLogger("payment")
# logger.setLevel(logging.INFO)

# router = APIRouter(prefix="", tags=["payments"])

# # --- env
# SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
# ALGORITHM = os.getenv("ALGORITHM", "HS256")

# STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
# STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")

# # Prefer explicit price IDs but support lookup keys
# PRO_PRICE_ID = os.getenv("STRIPE_PRO_PRICE_ID")
# PREMIUM_PRICE_ID = os.getenv("STRIPE_PREMIUM_PRICE_ID")
# PRO_LOOKUP_KEY = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
# PREMIUM_LOOKUP_KEY = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")

# if STRIPE_SECRET_KEY:
#     stripe.api_key = STRIPE_SECRET_KEY
#     logger.info("✅ Stripe configured successfully")
# else:
#     logger.error("❌ STRIPE_SECRET_KEY missing. Billing will run in demo mode.")

# oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# # --- models & DB
# from models import get_db, User  # your existing
# # If you have a Subscription model you can import it too:
# try:
#     from models import Subscription
# except Exception:
#     Subscription = None

# # ====== auth ======
# def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
#     credentials_exception = HTTPException(
#         status_code=status.HTTP_401_UNAUTHORIZED,
#         detail="Could not validate credentials",
#         headers={"WWW-Authenticate": "Bearer"},
#     )
#     try:
#         payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
#         username: str = payload.get("sub")
#         if not username:
#             raise credentials_exception
#     except jwt.PyJWTError:
#         raise credentials_exception

#     user = db.query(User).filter(User.username == username).first()
#     if not user:
#         raise credentials_exception
#     return user


# # ====== helpers ======
# def _resolve_price_id(plan: str) -> str:
#     """Return a *Price ID* for plan ('pro'|'premium'), using explicit IDs or lookup keys."""
#     plan = (plan or "").lower()
#     if plan not in ("pro", "premium"):
#         raise HTTPException(400, "Unknown plan")

#     # 1) explicit price id env
#     if plan == "pro" and PRO_PRICE_ID:
#         return PRO_PRICE_ID
#     if plan == "premium" and PREMIUM_PRICE_ID:
#         return PREMIUM_PRICE_ID

#     # 2) lookup keys
#     lookup = PRO_LOOKUP_KEY if plan == "pro" else PREMIUM_LOOKUP_KEY
#     try:
#         plist = stripe.Price.list(active=True, lookup_keys=[lookup], limit=1)
#         if plist and plist.data:
#             return plist.data[0]["id"]
#     except Exception as e:
#         logger.error(f"Stripe price lookup failed for {plan}: {e}")

#     raise HTTPException(500, f"Could not resolve Stripe price for plan '{plan}'")


# def _ensure_customer_for_user(user: User, db: Session) -> str:
#     """Find or create a Stripe Customer for this user; persist id on the user if possible."""
#     # if your User model already has a stripe_customer_id field, use it
#     existing_id = getattr(user, "stripe_customer_id", None)
#     if existing_id:
#         try:
#             # verify it still exists (best effort)
#             stripe.Customer.retrieve(existing_id)
#             return existing_id
#         except Exception:
#             pass

#     # Search by email (idempotent enough for test/dev)
#     cust_id = None
#     try:
#         # Customer.search requires it be enabled; fallback to list filter
#         candidates = stripe.Customer.list(email=(user.email or "").strip().lower(), limit=1)
#         if candidates and candidates.data:
#             cust_id = candidates.data[0]["id"]
#     except Exception:
#         pass

#     if not cust_id:
#         created = stripe.Customer.create(
#             email=(user.email or "").strip().lower(),
#             name=user.username or "",
#             metadata={"app_user_id": str(getattr(user, "id", ""))}
#         )
#         cust_id = created["id"]

#     # persist if model has the field
#     try:
#         if hasattr(user, "stripe_customer_id"):
#             setattr(user, "stripe_customer_id", cust_id)
#             db.add(user)
#             db.commit()
#     except Exception as e:
#         logger.warning(f"Could not persist stripe_customer_id: {e}")
#         db.rollback()

#     return cust_id


# def _mark_user_as_subscribed(user: User, plan: str, db: Session,
#                              customer_id: Optional[str], payment_intent_id: Optional[str]):
#     """Persist local subscription state."""
#     changed = False
#     if hasattr(user, "subscription_tier"):
#         if getattr(user, "subscription_tier", "free") != plan:
#             setattr(user, "subscription_tier", plan)
#             changed = True

#     # optional fields – save if you have them
#     if hasattr(user, "subscription_expires_at"):
#         setattr(user, "subscription_expires_at", datetime.utcnow() + timedelta(days=30))
#         changed = True
#     if hasattr(user, "stripe_customer_id") and customer_id:
#         setattr(user, "stripe_customer_id", customer_id)
#         changed = True

#     try:
#         if changed:
#             db.add(user)
#             db.commit()
#             db.refresh(user)
#     except Exception as e:
#         logger.error(f"Failed to update user subscription_tier: {e}")
#         db.rollback()

#     # Optionally store a row in a Subscription table if you have one
#     if Subscription:
#         try:
#             sub = db.query(Subscription).filter(Subscription.user_id == user.id).first()
#             if not sub:
#                 sub = Subscription(user_id=user.id)
#             if hasattr(sub, "tier"):
#                 sub.tier = plan
#             if hasattr(sub, "status"):
#                 sub.status = "active"
#             if hasattr(sub, "stripe_customer_id") and customer_id:
#                 sub.stripe_customer_id = customer_id
#             if hasattr(sub, "stripe_payment_intent_id") and payment_intent_id:
#                 sub.stripe_payment_intent_id = payment_intent_id
#             if hasattr(sub, "current_period_end"):
#                 sub.current_period_end = datetime.utcnow() + timedelta(days=30)
#             db.add(sub)
#             db.commit()
#         except Exception as e:
#             logger.warning(f"Could not upsert Subscription row: {e}")
#             db.rollback()


# # ====== endpoints ======
# @router.get("/billing/config")
# def billing_config():
#     """Frontend loads keys & resolved price IDs from here."""
#     payload: Dict[str, Any] = {
#         "publishable_key": STRIPE_PUBLISHABLE_KEY or "",
#         "mode": "test" if "test" in (STRIPE_SECRET_KEY or "") else "live",
#     }
#     try:
#         payload["pro_price_id"] = _resolve_price_id("pro")
#     except Exception:
#         payload["pro_price_id"] = None
#     try:
#         payload["premium_price_id"] = _resolve_price_id("premium")
#     except Exception:
#         payload["premium_price_id"] = None
#     return payload


# @router.post("/create_payment_intent/")
# def create_payment_intent(data: Dict[str, Any],
#                           user: User = Depends(get_current_user),
#                           db: Session = Depends(get_db)):
#     """
#     body: { "plan": "pro" | "premium" }
#     """
#     if not STRIPE_SECRET_KEY:
#         raise HTTPException(500, "Stripe is not configured")

#     plan = (data or {}).get("plan", "pro").lower()
#     price_id = _resolve_price_id(plan)

#     # Fetch the Price to get amount/currency (safe + future proof)
#     price = stripe.Price.retrieve(price_id)
#     if not (price and price["unit_amount"] and price["currency"]):
#         raise HTTPException(400, "Invalid Stripe Price")

#     # Ensure Customer in Stripe
#     customer_id = _ensure_customer_for_user(user, db)

#     # Create PI **tied to the customer**
#     intent = stripe.PaymentIntent.create(
#         amount=price["unit_amount"],
#         currency=price["currency"],
#         customer=customer_id,
#         automatic_payment_methods={
#             "enabled": True,"allow_redirects": "never", 
#         },
#         description=f"YouTube Content Downloader — {plan.capitalize()} Plan",
#         metadata={
#             "app_user_id": str(user.id),
#             "plan": plan,
#             "price_id": price_id
#         },
#     )
#     return {
#         "payment_intent_id": intent["id"],
#         "client_secret": intent["client_secret"],
#         "amount": intent["amount"],
#         "currency": intent["currency"],
#         "plan": plan
#     }


# @router.post("/confirm_payment/")
# def confirm_payment(data: Dict[str, Any],
#                     user: User = Depends(get_current_user),
#                     db: Session = Depends(get_db)):
#     """
#     body: { "payment_intent_id": "...", "plan": "pro" | "premium" }
#     NOTE: For local sandbox we confirm server-side. In production, use Stripe.js.
#     """
#     if not STRIPE_SECRET_KEY:
#         raise HTTPException(500, "Stripe is not configured")

#     payment_intent_id = (data or {}).get("payment_intent_id")
#     plan = (data or {}).get("plan", "pro").lower()
#     if not payment_intent_id:
#         raise HTTPException(400, "payment_intent_id is required")

#     try:
#         # For sandbox we can pass a test PM; in production, confirm client-side.
#         confirmed = stripe.PaymentIntent.confirm(
#             payment_intent_id,
#             payment_method="pm_card_visa"  # TEST ONLY
#         )
#     except Exception as e:
#         logger.error(f"Stripe confirm failed: {e}")
#         raise HTTPException(400, "Payment confirmation failed")

#     status_pi = confirmed["status"]
#     if status_pi != "succeeded":
#         # You can expand supported statuses if you need 3DS, etc.
#         raise HTTPException(400, f"Payment not completed (status={status_pi})")

#     # Link Customer & mark user as subscribed locally
#     customer_id = confirmed.get("customer")
#     _mark_user_as_subscribed(user, plan, db, customer_id, payment_intent_id)

#     return {
#         "status": "active",
#         "plan": plan,
#         "stripe": {
#             "payment_intent": confirmed["id"],
#             "customer": customer_id
#         }
#     }

#========================================

