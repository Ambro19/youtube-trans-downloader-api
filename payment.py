# backend/payment.py
# Subscription-first Stripe integration using Checkout + Webhooks
# - /billing/config now matches the frontend shape: { mode, is_demo, prices:{pro,premium} }
# - Robust Checkout session creation (accepts explicit price_id or resolves by lookup key)
# - Billing portal: graceful 400 with clear guidance if not configured; auto-heals stale customers
# - Safer webhook -> updates user's tier and subscription id

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

MODE = "live" if STRIPE_SECRET_KEY.startswith("sk_live_") else "test"
IS_DEMO = False  # flip if you ship demo stubs

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
    log.info("✅ Stripe configured (subscriptions, mode=%s)", MODE)
else:
    log.error("❌ STRIPE_SECRET_KEY missing. Billing will run in demo mode.")
    IS_DEMO = True


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


def _lookup_price_id(lookup_key: str) -> str:
    """Resolve a price by lookup key; raise HTTP 500 if not found."""
    try:
        res = stripe.Price.list(active=True, lookup_keys=[lookup_key], limit=1)
        if res.data:
            return res.data[0].id
    except Exception as e:
        log.error("Stripe price lookup failed for %s: %s", lookup_key, e)
    raise HTTPException(500, f"Stripe price for lookup '{lookup_key}' not found")


def _resolve_price_id(tier: str, explicit: str | None = None) -> str:
    if tier not in ("pro", "premium"):
        raise HTTPException(400, "Unknown tier")
    if explicit:  # frontend may send explicit price_id
        return explicit
    if tier == "pro":
        return PRO_PRICE_ID or _lookup_price_id(PRO_LOOKUP)
    return PREMIUM_PRICE_ID or _lookup_price_id(PREMIUM_LOOKUP)


def _get_or_create_customer(db: Session, user: User) -> str:
    """Fetch existing Stripe customer; if missing/invalid, create a new one and store it."""
    cust_id = getattr(user, "stripe_customer_id", None)
    if cust_id:
        try:
            stripe.Customer.retrieve(cust_id)
            return cust_id
        except Exception as e:
            log.warning("Stale stripe_customer_id on user %s: %s; creating new", user.id, e)
            setattr(user, "stripe_customer_id", None)
            db.add(user)
            db.commit()
            db.refresh(user)
    customer = stripe.Customer.create(
        email=(user.email or "").strip().lower(),
        name=user.username or user.email,
        metadata={"app_user_id": str(user.id)},
    )
    setattr(user, "stripe_customer_id", customer.id)
    db.add(user)
    db.commit()
    db.refresh(user)
    return customer.id


def _prices_payload():
    """Return dict(prices) in the shape the frontend expects."""
    pro = None
    premium = None
    try:
        pro = PRO_PRICE_ID or _lookup_price_id(PRO_LOOKUP)
    except Exception:
        pass
    try:
        premium = PREMIUM_PRICE_ID or _lookup_price_id(PREMIUM_LOOKUP)
    except Exception:
        pass
    return {"pro": pro, "premium": premium}


# ---------- endpoints

@router.get("/config")
def get_config():
    """
    Frontend display/config endpoint. Matches the UI expectation:
    {
      "mode": "test"|"live",
      "is_demo": false,
      "prices": { "pro": "...", "premium": "..." },
      "source": { "publishableKey": "pk_..." }
    }
    """
    prices = _prices_payload()
    return {
        "mode": MODE,
        "is_demo": IS_DEMO,
        "prices": prices,
        "source": {"publishableKey": STRIPE_PUBLISHABLE_KEY},
    }


@router.post("/create_checkout_session")
def create_checkout_session(payload: dict, request: Request, db: Session = Depends(get_db)):
    """
    Body: { "tier": "pro" | "premium", "price_id": optional }
    Returns: { url } for redirect to Stripe Checkout (subscription mode).
    """
    if IS_DEMO or not STRIPE_SECRET_KEY:
        raise HTTPException(503, "Stripe is not configured on the server.")

    user = _auth_user(request, db)
    tier = (payload or {}).get("tier", "pro")
    explicit_price = (payload or {}).get("price_id")
    price_id = _resolve_price_id(tier, explicit_price)
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
            subscription_data={
                "metadata": {"tier": tier, "app_user_id": str(user.id)}
            },
            metadata={"tier": tier, "app_user_id": str(user.id)},
        )
        return {"url": session.url}
    except Exception as e:
        log.exception("Failed to create Checkout session")
        raise HTTPException(500, str(e))


@router.post("/create_portal_session")
def create_portal_session(request: Request, db: Session = Depends(get_db)):
    """
    Return a customer billing portal URL.
    - If user has no (or stale) customer, we create one to avoid 'No such customer'.
    - If the portal isn't configured, return 400 with a clear message.
    """
    if IS_DEMO or not STRIPE_SECRET_KEY:
        raise HTTPException(503, "Stripe is not configured on the server.")

    user = _auth_user(request, db)
    cust_id = _get_or_create_customer(db, user)

    try:
        portal = stripe.billing_portal.Session.create(
            customer=cust_id,
            return_url=f"{FRONTEND_URL}/subscription",
        )
        return {"url": portal.url}
    except stripe.error.InvalidRequestError as e:
        # Most common case in Test mode when portal default config not saved
        msg = (
            "Billing portal is not configured yet. "
            "In Stripe Dashboard, go to Settings → Billing → Customer portal and click Save to create "
            "a default configuration (Test or Live)."
        )
        log.warning("Stripe portal error: %s", e)
        raise HTTPException(400, msg)
    except Exception as e:
        log.exception("Create portal session failed")
        raise HTTPException(500, str(e))


@router.post("/cancel_subscription")
def cancel_subscription(request: Request, db: Session = Depends(get_db)):
    """
    Cancel at period end if possible; otherwise immediate delete fallback.
    """
    if IS_DEMO or not STRIPE_SECRET_KEY:
        raise HTTPException(503, "Stripe is not configured on the server.")

    user = _auth_user(request, db)
    sub_id = getattr(user, "stripe_subscription_id", None)
    if not sub_id:
        raise HTTPException(400, "No subscription to cancel")

    try:
        try:
            stripe.Subscription.modify(sub_id, cancel_at_period_end=True)
        except Exception:
            # older or invalid -> hard delete
            stripe.Subscription.delete(sub_id)
        setattr(user, "subscription_tier", "free")
        db.add(user)
        db.commit()
        db.refresh(user)
        return {"status": "canceled"}
    except Exception as e:
        log.exception("Cancel subscription failed")
        raise HTTPException(500, str(e))


@router.post("/webhook")
async def webhook(request: Request, db: Session = Depends(get_db)):
    """
    Handles:
      - checkout.session.completed (mode=subscription)
      - customer.subscription.{created,updated,deleted}
    """
    if not STRIPE_SECRET_KEY:
        return JSONResponse({"ok": True, "skipped": "stripe not configured"})

    # Verify signature if provided
    if WEBHOOK_SECRET:
        payload_bytes = await request.body()
        sig = request.headers.get("stripe-signature")
        try:
            event = stripe.Webhook.construct_event(payload_bytes, sig, WEBHOOK_SECRET)
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Invalid signature: {e}"}, status_code=400)
    else:
        try:
            event = await request.json()
        except Exception as e:
            return JSONResponse({"ok": False, "error": f"Invalid payload: {e}"}, status_code=400)

    etype = event.get("type")
    data = event.get("data", {}).get("object", {})

    try:
        if etype == "checkout.session.completed" and data.get("mode") == "subscription":
            customer_id = data.get("customer")
            sub_id = data.get("subscription")
            meta = data.get("metadata") or {}
            tier = meta.get("tier") or "pro"

            user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
            if not user:
                ref = data.get("client_reference_id") or meta.get("app_user_id")
                if ref:
                    user = db.get(User, int(ref))

            if user:
                setattr(user, "stripe_subscription_id", sub_id)
                setattr(user, "subscription_tier", tier)
                db.add(user)
                db.commit()

        elif etype in ("customer.subscription.updated", "customer.subscription.created"):
            sub = data
            customer_id = sub.get("customer")
            status = sub.get("status")
            price_id = None
            try:
                price_id = sub["items"]["data"][0]["price"]["id"]
            except Exception:
                pass

            # Infer tier from price id
            premium_id = PREMIUM_PRICE_ID or (_lookup_price_id(PREMIUM_LOOKUP) if STRIPE_SECRET_KEY else None)
            tier = "premium" if premium_id and price_id == premium_id else "pro"

            user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
            if user:
                setattr(user, "stripe_subscription_id", sub.get("id"))
                setattr(user, "subscription_tier", tier if status in ("active", "trialing") else "free")
                db.add(user)
                db.commit()

        elif etype == "customer.subscription.deleted":
            sub = data
            customer_id = sub.get("customer")
            user = db.query(User).filter(User.stripe_customer_id == customer_id).first()
            if user:
                setattr(user, "subscription_tier", "free")
                db.add(user)
                db.commit()

    except Exception as e:
        log.exception("Webhook handling failed for %s", etype)
        # Return 200 to avoid Stripe retry storms; error is logged.
        return JSONResponse({"ok": False, "error": str(e)})

    return JSONResponse({"ok": True})
