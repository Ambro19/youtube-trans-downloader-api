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
WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

# Prefer lookup keys; fall back to explicit price IDs if provided.
PRO_LOOKUP = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
PREMIUM_LOOKUP = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")
PRO_PRICE_ID = os.getenv("STRIPE_PRO_PRICE_ID")
PREMIUM_PRICE_ID = os.getenv("STRIPE_PREMIUM_PRICE_ID")

# Optional final fallback by Product name (nice for screenshots/config)
PRO_PRODUCT_NAME = os.getenv("STRIPE_PRO_PRODUCT_NAME", "Pro Plan")
PREMIUM_PRODUCT_NAME = os.getenv("STRIPE_PREMIUM_PRODUCT_NAME", "Premium Plan")

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


def _price_by_product_name(name: str) -> str | None:
    """Find the first active recurring price for a product with this name."""
    try:
        prods = stripe.Product.list(active=True, limit=100)
        for p in prods.auto_paging_iter():
            if (p.name or "").strip().lower() == (name or "").strip().lower():
                prices = stripe.Price.list(product=p.id, active=True, limit=10)
                for pr in prices.data:
                    if pr.get("type") == "recurring":
                        return pr.id
                # If nothing recurring, still return the first active
                if prices.data:
                    return prices.data[0].id
    except Exception as e:
        log.warning(f"Product-name lookup failed for {name!r}: {e}")
    return None


def _resolve_price_id(tier: str) -> str:
    if tier not in ("pro", "premium"):
        raise HTTPException(400, "Unknown tier")

    # 1) explicit env price ID
    if tier == "pro" and PRO_PRICE_ID:
        return PRO_PRICE_ID
    if tier == "premium" and PREMIUM_PRICE_ID:
        return PREMIUM_PRICE_ID

    # 2) lookup keys
    lookup = PRO_LOOKUP if tier == "pro" else PREMIUM_LOOKUP
    try:
        res = stripe.Price.list(active=True, lookup_keys=[lookup], limit=1)
        if res.data:
            return res.data[0].id
    except Exception as e:
        log.warning(f"Price lookup by key failed for {lookup!r}: {e}")

    # 3) product name fallback
    name = PRO_PRODUCT_NAME if tier == "pro" else PREMIUM_PRODUCT_NAME
    price_id = _price_by_product_name(name)
    if price_id:
        return price_id

    raise HTTPException(500, f"Stripe price for '{tier}' not found")


def _get_or_create_customer(db: Session, user: User) -> str:
    cust_id = getattr(user, "stripe_customer_id", None)
    if cust_id:
        try:
            stripe.Customer.retrieve(cust_id)
            return cust_id
        except Exception:
            log.warning("Stale Stripe customer id on user; recreating.")
            setattr(user, "stripe_customer_id", None)
            db.add(user); db.commit(); db.refresh(user)

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
    mode = "live" if (STRIPE_SECRET_KEY and STRIPE_SECRET_KEY.startswith("sk_live")) else "test"
    pro = None
    premium = None
    try:
        pro = PRO_PRICE_ID or stripe.Price.list(active=True, lookup_keys=[PRO_LOOKUP], limit=1).data[0].id
    except Exception:
        pro = _price_by_product_name(PRO_PRODUCT_NAME)
    try:
        premium = PREMIUM_PRICE_ID or stripe.Price.list(active=True, lookup_keys=[PREMIUM_LOOKUP], limit=1).data[0].id
    except Exception:
        premium = _price_by_product_name(PREMIUM_PRODUCT_NAME)

    return {
        "mode": mode,
        "is_demo": not bool(STRIPE_SECRET_KEY),
        "publishableKey": STRIPE_PUBLISHABLE_KEY,
        "prices": {"pro": pro, "premium": premium},
        "source": {"lookup": {"pro": PRO_LOOKUP, "premium": PREMIUM_LOOKUP}},
    }

@router.post("/create_checkout_session")
async def create_checkout_session(payload: dict, request: Request, db: Session = Depends(get_db)):
    """
    Body: { "price_lookup_key": "pro_monthly" | "premium_monthly" }
    Returns: { url } for redirect to Stripe Checkout (subscription mode).
    """
    user = _auth_user(request, db)
    
    try:
        lookup = payload.get("price_lookup_key")
        if not lookup:
            raise HTTPException(status_code=400, detail="Missing price_lookup_key")
        
        # Resolve Stripe Price by lookup key (recommended)
        prices = stripe.Price.list(lookup_keys=[lookup], expand=["data.product"], limit=1)
        if not prices.data:
            raise HTTPException(status_code=400, detail="Unknown price lookup key")
        
        price_id = prices.data[0].id
        customer_id = _get_or_create_customer(db, user)
        
        # Determine tier from lookup key for metadata
        tier = "premium" if "premium" in lookup.lower() else "pro"
        
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{FRONTEND_URL}/subscription?status=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{FRONTEND_URL}/subscription?status=cancelled",
            allow_promotion_codes=True,
            billing_address_collection="auto",
            # 🚫 Keep express wallets minimal to avoid the Link OTP wall.
            # Restrict methods to show the card/payment list like Picture #2.
            payment_method_types=["card", "cashapp", "klarna", "amazon_pay"],
            client_reference_id=str(user.id),
            subscription_data={"metadata": {"tier": tier, "app_user_id": str(user.id)}},
            metadata={"tier": tier, "app_user_id": str(user.id)},
        )
        return {"url": session.url}
    except stripe.error.StripeError as e:
        log.exception("Stripe error creating Checkout session")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Failed to create Checkout session")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/create_portal_session")
def create_portal_session(request: Request, db: Session = Depends(get_db)):
    """Return a customer billing portal URL. Heals stale customer ids."""
    user = _auth_user(request, db)
    cust_id = getattr(user, "stripe_customer_id", None)

    # (Re)create the customer if it's missing or invalid
    if not cust_id:
        cust_id = _get_or_create_customer(db, user)
    else:
        try:
            stripe.Customer.retrieve(cust_id)
        except Exception:
            cust_id = _get_or_create_customer(db, user)

    # Now open the portal
    try:
        portal = stripe.billing_portal.Session.create(
            customer=cust_id,
            return_url=f"{FRONTEND_URL}/subscription"
        )
        return {"url": portal.url}
    except Exception as e:
        # Forward the helpful Stripe message to the UI (matches your screenshot)
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
            tier = "premium" if price_id and price_id == (PREMIUM_PRICE_ID or _price_by_product_name(PREMIUM_PRODUCT_NAME)) else "pro"

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
        return JSONResponse({"ok": False, "error": str(e)})

    return JSONResponse({"ok": True})

######################################################################
###################################################################

@router.post("/create_checkout_session")
async def create_checkout_session(payload: dict, request: Request, db: Session = Depends(get_db)):
    """
    Body: { "price_lookup_key": "pro_monthly" | "premium_monthly" }
    Returns: { url } for redirect to Stripe Checkout (subscription mode).
    """
    user = _auth_user(request, db)
    
    try:
        lookup = payload.get("price_lookup_key")
        if not lookup:
            raise HTTPException(status_code=400, detail="Missing price_lookup_key")
        
        # Resolve Stripe Price by lookup key (recommended)
        prices = stripe.Price.list(lookup_keys=[lookup], expand=["data.product"], limit=1)
        if not prices.data:
            raise HTTPException(status_code=400, detail="Unknown price lookup key")
        
        price_id = prices.data[0].id
        customer_id = _get_or_create_customer(db, user)
        
        # Determine tier from lookup key for metadata
        tier = "premium" if "premium" in lookup.lower() else "pro"
        
        session = stripe.checkout.Session.create(
            mode="subscription",
            customer=customer_id,
            line_items=[{"price": price_id, "quantity": 1}],
            success_url=f"{FRONTEND_URL}/subscription?status=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{FRONTEND_URL}/subscription?status=cancelled",
            allow_promotion_codes=True,
            billing_address_collection="auto",
            # 🚫 Keep express wallets minimal to avoid the Link OTP wall.
            # Restrict methods to show the card/payment list like Picture #2.
            payment_method_types=["card", "cashapp", "klarna", "amazon_pay"],
            client_reference_id=str(user.id),
            subscription_data={"metadata": {"tier": tier, "app_user_id": str(user.id)}},
            metadata={"tier": tier, "app_user_id": str(user.id)},
        )
        return {"url": session.url}
    except stripe.error.StripeError as e:
        log.exception("Stripe error creating Checkout session")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("Failed to create Checkout session")
        raise HTTPException(status_code=500, detail=str(e))