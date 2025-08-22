# payment.py — drop-in
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import os
import jwt
import stripe
import logging

# ====== setup ======
logger = logging.getLogger("payment")
logger.setLevel(logging.INFO)

router = APIRouter(prefix="", tags=["payments"])

# --- env
SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
ALGORITHM = os.getenv("ALGORITHM", "HS256")

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")

# Prefer explicit price IDs but support lookup keys
PRO_PRICE_ID = os.getenv("STRIPE_PRO_PRICE_ID")
PREMIUM_PRICE_ID = os.getenv("STRIPE_PREMIUM_PRICE_ID")
PRO_LOOKUP_KEY = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
PREMIUM_LOOKUP_KEY = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
    logger.info("✅ Stripe configured successfully")
else:
    logger.error("❌ STRIPE_SECRET_KEY missing. Billing will run in demo mode.")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# --- models & DB
from models import get_db, User  # your existing
# If you have a Subscription model you can import it too:
try:
    from models import Subscription
except Exception:
    Subscription = None

# ====== auth ======
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise credentials_exception
    return user


# ====== helpers ======
def _resolve_price_id(plan: str) -> str:
    """Return a *Price ID* for plan ('pro'|'premium'), using explicit IDs or lookup keys."""
    plan = (plan or "").lower()
    if plan not in ("pro", "premium"):
        raise HTTPException(400, "Unknown plan")

    # 1) explicit price id env
    if plan == "pro" and PRO_PRICE_ID:
        return PRO_PRICE_ID
    if plan == "premium" and PREMIUM_PRICE_ID:
        return PREMIUM_PRICE_ID

    # 2) lookup keys
    lookup = PRO_LOOKUP_KEY if plan == "pro" else PREMIUM_LOOKUP_KEY
    try:
        plist = stripe.Price.list(active=True, lookup_keys=[lookup], limit=1)
        if plist and plist.data:
            return plist.data[0]["id"]
    except Exception as e:
        logger.error(f"Stripe price lookup failed for {plan}: {e}")

    raise HTTPException(500, f"Could not resolve Stripe price for plan '{plan}'")


def _ensure_customer_for_user(user: User, db: Session) -> str:
    """Find or create a Stripe Customer for this user; persist id on the user if possible."""
    # if your User model already has a stripe_customer_id field, use it
    existing_id = getattr(user, "stripe_customer_id", None)
    if existing_id:
        try:
            # verify it still exists (best effort)
            stripe.Customer.retrieve(existing_id)
            return existing_id
        except Exception:
            pass

    # Search by email (idempotent enough for test/dev)
    cust_id = None
    try:
        # Customer.search requires it be enabled; fallback to list filter
        candidates = stripe.Customer.list(email=(user.email or "").strip().lower(), limit=1)
        if candidates and candidates.data:
            cust_id = candidates.data[0]["id"]
    except Exception:
        pass

    if not cust_id:
        created = stripe.Customer.create(
            email=(user.email or "").strip().lower(),
            name=user.username or "",
            metadata={"app_user_id": str(getattr(user, "id", ""))}
        )
        cust_id = created["id"]

    # persist if model has the field
    try:
        if hasattr(user, "stripe_customer_id"):
            setattr(user, "stripe_customer_id", cust_id)
            db.add(user)
            db.commit()
    except Exception as e:
        logger.warning(f"Could not persist stripe_customer_id: {e}")
        db.rollback()

    return cust_id


def _mark_user_as_subscribed(user: User, plan: str, db: Session,
                             customer_id: Optional[str], payment_intent_id: Optional[str]):
    """Persist local subscription state."""
    changed = False
    if hasattr(user, "subscription_tier"):
        if getattr(user, "subscription_tier", "free") != plan:
            setattr(user, "subscription_tier", plan)
            changed = True

    # optional fields – save if you have them
    if hasattr(user, "subscription_expires_at"):
        setattr(user, "subscription_expires_at", datetime.utcnow() + timedelta(days=30))
        changed = True
    if hasattr(user, "stripe_customer_id") and customer_id:
        setattr(user, "stripe_customer_id", customer_id)
        changed = True

    try:
        if changed:
            db.add(user)
            db.commit()
            db.refresh(user)
    except Exception as e:
        logger.error(f"Failed to update user subscription_tier: {e}")
        db.rollback()

    # Optionally store a row in a Subscription table if you have one
    if Subscription:
        try:
            sub = db.query(Subscription).filter(Subscription.user_id == user.id).first()
            if not sub:
                sub = Subscription(user_id=user.id)
            if hasattr(sub, "tier"):
                sub.tier = plan
            if hasattr(sub, "status"):
                sub.status = "active"
            if hasattr(sub, "stripe_customer_id") and customer_id:
                sub.stripe_customer_id = customer_id
            if hasattr(sub, "stripe_payment_intent_id") and payment_intent_id:
                sub.stripe_payment_intent_id = payment_intent_id
            if hasattr(sub, "current_period_end"):
                sub.current_period_end = datetime.utcnow() + timedelta(days=30)
            db.add(sub)
            db.commit()
        except Exception as e:
            logger.warning(f"Could not upsert Subscription row: {e}")
            db.rollback()


# ====== endpoints ======
@router.get("/billing/config")
def billing_config():
    """Frontend loads keys & resolved price IDs from here."""
    payload: Dict[str, Any] = {
        "publishable_key": STRIPE_PUBLISHABLE_KEY or "",
        "mode": "test" if "test" in (STRIPE_SECRET_KEY or "") else "live",
    }
    try:
        payload["pro_price_id"] = _resolve_price_id("pro")
    except Exception:
        payload["pro_price_id"] = None
    try:
        payload["premium_price_id"] = _resolve_price_id("premium")
    except Exception:
        payload["premium_price_id"] = None
    return payload


@router.post("/create_payment_intent/")
def create_payment_intent(data: Dict[str, Any],
                          user: User = Depends(get_current_user),
                          db: Session = Depends(get_db)):
    """
    body: { "plan": "pro" | "premium" }
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(500, "Stripe is not configured")

    plan = (data or {}).get("plan", "pro").lower()
    price_id = _resolve_price_id(plan)

    # Fetch the Price to get amount/currency (safe + future proof)
    price = stripe.Price.retrieve(price_id)
    if not (price and price["unit_amount"] and price["currency"]):
        raise HTTPException(400, "Invalid Stripe Price")

    # Ensure Customer in Stripe
    customer_id = _ensure_customer_for_user(user, db)

    # Create PI **tied to the customer**
    intent = stripe.PaymentIntent.create(
        amount=price["unit_amount"],
        currency=price["currency"],
        customer=customer_id,
        automatic_payment_methods={"enabled": True},
        description=f"YouTube Content Downloader — {plan.capitalize()} Plan",
        metadata={
            "app_user_id": str(user.id),
            "plan": plan,
            "price_id": price_id
        },
    )
    return {
        "payment_intent_id": intent["id"],
        "client_secret": intent["client_secret"],
        "amount": intent["amount"],
        "currency": intent["currency"],
        "plan": plan
    }


@router.post("/confirm_payment/")
def confirm_payment(data: Dict[str, Any],
                    user: User = Depends(get_current_user),
                    db: Session = Depends(get_db)):
    """
    body: { "payment_intent_id": "...", "plan": "pro" | "premium" }
    NOTE: For local sandbox we confirm server-side. In production, use Stripe.js.
    """
    if not STRIPE_SECRET_KEY:
        raise HTTPException(500, "Stripe is not configured")

    payment_intent_id = (data or {}).get("payment_intent_id")
    plan = (data or {}).get("plan", "pro").lower()
    if not payment_intent_id:
        raise HTTPException(400, "payment_intent_id is required")

    try:
        # For sandbox we can pass a test PM; in production, confirm client-side.
        confirmed = stripe.PaymentIntent.confirm(
            payment_intent_id,
            payment_method="pm_card_visa"  # TEST ONLY
        )
    except Exception as e:
        logger.error(f"Stripe confirm failed: {e}")
        raise HTTPException(400, "Payment confirmation failed")

    status_pi = confirmed["status"]
    if status_pi != "succeeded":
        # You can expand supported statuses if you need 3DS, etc.
        raise HTTPException(400, f"Payment not completed (status={status_pi})")

    # Link Customer & mark user as subscribed locally
    customer_id = confirmed.get("customer")
    _mark_user_as_subscribed(user, plan, db, customer_id, payment_intent_id)

    return {
        "status": "active",
        "plan": plan,
        "stripe": {
            "payment_intent": confirmed["id"],
            "customer": customer_id
        }
    }


#========================================

# # payment.py — lazy Stripe init + server-side confirm (no redirects), robust price resolution

# import os
# import logging
# from typing import Optional, Tuple, Dict

# import stripe
# from fastapi import APIRouter, Depends, HTTPException
# from pydantic import BaseModel

# logger = logging.getLogger("payment")
# router = APIRouter(tags=["payments"])

# # --- Lazy env helpers --------------------------------------------------------

# def _env(k: str, default: str = "") -> str:
#     return os.getenv(k, default)

# _STRIPE_LAST_KEY = None
# _STRIPE_INIT_LOGGED = False

# def _ensure_stripe():
#     """
#     Lazily read STRIPE_SECRET_KEY and configure stripe.api_key.
#     Safe across uvicorn reload workers on Windows.
#     """
#     global _STRIPE_LAST_KEY, _STRIPE_INIT_LOGGED
#     key = _env("STRIPE_SECRET_KEY", "")
#     if key != _STRIPE_LAST_KEY:
#         _STRIPE_LAST_KEY = key
#         if key:
#             stripe.api_key = key
#             if not _STRIPE_INIT_LOGGED:
#                 logger.info("✅ Stripe configured successfully")
#                 _STRIPE_INIT_LOGGED = True
#         else:
#             logger.error("❌ STRIPE_SECRET_KEY missing. Billing will run in demo mode.")

# def _stripe_mode() -> str:
#     k = _env("STRIPE_SECRET_KEY", "")
#     if k.startswith("sk_live_"): return "live"
#     if k.startswith("sk_test_"): return "test"
#     return "demo"

# FRONTEND_URL = _env("FRONTEND_URL", "http://localhost:3000")

# # --- Price resolution config -------------------------------------------------

# ENV_PRO_PRICE_ID = _env("STRIPE_PRO_PRICE_ID") or None
# ENV_PREMIUM_PRICE_ID = _env("STRIPE_PREMIUM_PRICE_ID") or None

# PRO_LOOKUP_KEY = _env("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
# PREMIUM_LOOKUP_KEY = _env("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")

# PRO_PRODUCT_NAME = _env("STRIPE_PRO_PRODUCT_NAME", "Pro Plan")
# PREMIUM_PRODUCT_NAME = _env("STRIPE_PREMIUM_PRODUCT_NAME", "Premium Plan")

# _PRICE_CACHE: Dict[str, Optional[str]] = {"pro": None, "premium": None}

# # --- Schemas ----------------------------------------------------------------

# class CreateIntentBody(BaseModel):
#     plan_type: Optional[str] = None  # "pro" | "premium"
#     price_id: Optional[str] = None

# class ConfirmBody(BaseModel):
#     payment_intent_id: str

# # --- Helpers ----------------------------------------------------------------

# def _safe_retrieve_price(price_id: str) -> Optional[str]:
#     try:
#         _ensure_stripe()
#         p = stripe.Price.retrieve(price_id)
#         return p["id"]
#     except Exception as e:
#         logger.error(f"Stripe couldn't find price '{price_id}': {e}")
#         return None

# def _find_by_lookup_key(lookup_key: str) -> Optional[str]:
#     try:
#         _ensure_stripe()
#         prices = stripe.Price.list(active=True, lookup_keys=[lookup_key], limit=1)
#         if prices.data:
#             return prices.data[0]["id"]
#     except Exception as e:
#         logger.warning(f"Lookup by key '{lookup_key}' failed: {e}")
#     return None

# def _find_by_product_name(name: str) -> Optional[str]:
#     try:
#         _ensure_stripe()
#         prods = stripe.Product.list(active=True, limit=100, expand=["data.default_price"])
#         name_lc = name.strip().lower()
#         for p in prods.auto_paging_iter():
#             if str(p["name"]).strip().lower() == name_lc:
#                 dp = p.get("default_price")
#                 if isinstance(dp, dict):
#                     return dp.get("id")
#                 if isinstance(dp, str):
#                     return dp
#     except Exception as e:
#         logger.warning(f"Product lookup by name '{name}' failed: {e}")
#     return None

# def resolve_price_id(plan_type: str, preferred: Optional[str]) -> Tuple[Optional[str], str]:
#     plan = (plan_type or "").strip().lower()
#     if plan not in ("pro", "premium"):
#         raise HTTPException(status_code=400, detail="Invalid plan_type. Use 'pro' or 'premium'.")

#     cached = _PRICE_CACHE.get(plan)
#     if cached:
#         return cached, "cache"

#     if preferred:
#         ok = _safe_retrieve_price(preferred)
#         if ok:
#             _PRICE_CACHE[plan] = ok
#             return ok, "client"

#     env_id = ENV_PRO_PRICE_ID if plan == "pro" else ENV_PREMIUM_PRICE_ID
#     if env_id:
#         ok = _safe_retrieve_price(env_id)
#         if ok:
#             _PRICE_CACHE[plan] = ok
#             return ok, "env"

#     lk = PRO_LOOKUP_KEY if plan == "pro" else PREMIUM_LOOKUP_KEY
#     ok = _find_by_lookup_key(lk)
#     if ok:
#         _PRICE_CACHE[plan] = ok
#         return ok, "lookup_key"

#     prod_name = PRO_PRODUCT_NAME if plan == "pro" else PREMIUM_PRODUCT_NAME
#     ok = _find_by_product_name(prod_name)
#     if ok:
#         _PRICE_CACHE[plan] = ok
#         return ok, "product_name"

#     return None, "unavailable"

# def _demo_mode() -> bool:
#     return _stripe_mode() == "demo" or not _env("STRIPE_SECRET_KEY", "")

# def _require_stripe_ready():
#     _ensure_stripe()
#     if _demo_mode():
#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 "Stripe is not fully configured (missing STRIPE_SECRET_KEY or prices). "
#                 "Set up prices or enable demo upgrade endpoint."
#             ),
#         )

# # --- Public endpoints --------------------------------------------------------

# @router.get("/billing/config")
# def billing_config():
#     _ensure_stripe()

#     if _demo_mode():
#         return {
#             "mode": _stripe_mode(),
#             "is_demo": True,
#             "prices": {"pro": None, "premium": None},
#             "source": "demo",
#         }

#     pro_id, pro_src = resolve_price_id("pro", None)
#     prem_id, prem_src = resolve_price_id("premium", None)

#     is_demo = not (pro_id and prem_id)
#     return {
#         "mode": _stripe_mode(),
#         "is_demo": is_demo,
#         "prices": {"pro": pro_id, "premium": prem_id},
#         "source": {"pro": pro_src, "premium": prem_src},
#     }

# @router.post("/create_payment_intent/")
# def create_payment_intent(body: CreateIntentBody, user=Depends(lambda: None)):
#     _require_stripe_ready()

#     plan = (body.plan_type or "").strip().lower()
#     if not plan:
#         plan = "pro" if "pro" in (body.price_id or "").lower() else "premium"

#     price_id, src = resolve_price_id(plan, body.price_id)
#     if not price_id:
#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 "Unable to resolve a valid Stripe price for this plan. "
#                 "Ensure your Stripe products exist (Pro Plan / Premium Plan) "
#                 "or set STRIPE_PRO_PRICE_ID / STRIPE_PREMIUM_PRICE_ID."
#             ),
#         )

#     price = stripe.Price.retrieve(price_id)
#     if not price.get("unit_amount") or not price.get("currency"):
#         raise HTTPException(status_code=400, detail="Stripe price is missing amount/currency.")

#     intent = stripe.PaymentIntent.create(
#         amount=price["unit_amount"],
#         currency=price["currency"],
#         automatic_payment_methods={"enabled": True, "allow_redirects": "never"},
#         metadata={"plan": plan},
#         description=f"YouTube Content Downloader – {plan.capitalize()} Plan",
#     )

#     return {"payment_intent_id": intent["id"]}

# @router.post("/confirm_payment/")
# def confirm_payment(body: ConfirmBody):
#     _require_stripe_ready()

#     try:
#         pi = stripe.PaymentIntent.confirm(
#             body.payment_intent_id,
#             payment_method="pm_card_visa",  # universal test method (no 3DS)
#             return_url=f"{FRONTEND_URL}/subscription?paid=1",
#         )
#         if pi["status"] != "succeeded":
#             return {"status": pi["status"], "client_secret": pi.get("client_secret")}
#         return {"status": "succeeded", "client_secret": pi.get("client_secret")}
#     except stripe.error.StripeError as e:
#         logger.error(f"Stripe confirm error: {e}")
#         raise HTTPException(status_code=400, detail=str(e))


#================================================================
# # payment.py — resilient Stripe price resolution + simple billing config
# # Works in both dev/test and prod, and survives wrong/blank price IDs coming from the client.

# import os
# import logging
# from typing import Optional, Tuple, Dict

# import stripe
# from fastapi import APIRouter, Depends, HTTPException, status
# from pydantic import BaseModel

# logger = logging.getLogger("payment")
# router = APIRouter(tags=["billing"])
# router = APIRouter(tags=["payments"])  # no prefix so existing paths remain the same

# # === Stripe init =============================================================

# STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
# if STRIPE_SECRET_KEY:
#     stripe.api_key = STRIPE_SECRET_KEY
#     logger.info("✅ Stripe configured successfully")
# else:
#     logger.error("❌ STRIPE_SECRET_KEY missing. Billing will run in demo mode.")

# def stripe_mode() -> str:
#     k = STRIPE_SECRET_KEY or ""
#     if k.startswith("sk_live_"):
#         return "live"
#     if k.startswith("sk_test_"):
#         return "test"
#     return "demo"

# # === Config / names / lookup keys ===========================================

# # Option 1: direct price IDs (strongest)
# ENV_PRO_PRICE_ID = os.getenv("STRIPE_PRO_PRICE_ID")
# ENV_PREMIUM_PRICE_ID = os.getenv("STRIPE_PREMIUM_PRICE_ID")

# # Option 2: lookup keys on Price objects (if you used them when creating prices)
# PRO_LOOKUP_KEY = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
# PREMIUM_LOOKUP_KEY = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")

# # Option 3: product names (fallback – matches your screenshots)
# PRO_PRODUCT_NAME = os.getenv("STRIPE_PRO_PRODUCT_NAME", "Pro Plan")
# PREMIUM_PRODUCT_NAME = os.getenv("STRIPE_PREMIUM_PRODUCT_NAME", "Premium Plan")

# # Cache so we don’t hit Stripe on every request
# _PRICE_CACHE: Dict[str, Optional[str]] = {"pro": None, "premium": None}

# # === Request bodies ==========================================================

# class CreateIntentBody(BaseModel):
#     # You can send either one. Backend will resolve safely.
#     plan_type: Optional[str] = None  # "pro" | "premium"
#     price_id: Optional[str] = None


# class ConfirmBody(BaseModel):
#     payment_intent_id: str


# # === Helpers =================================================================

# def _safe_retrieve_price(price_id: str) -> Optional[str]:
#     """Return price_id if it exists in this account/mode, else None."""
#     try:
#         p = stripe.Price.retrieve(price_id)
#         return p["id"]
#     except Exception as e:
#         # Stripe returns 404 for missing price in this account/mode
#         logger.error(f"Stripe couldn't find price '{price_id}': {e}")
#         return None


# def _find_by_lookup_key(lookup_key: str) -> Optional[str]:
#     """Find a price by its lookup_key if you use them."""
#     try:
#         # API supports listing by lookup_keys (array)
#         prices = stripe.Price.list(active=True, lookup_keys=[lookup_key], limit=1)
#         if prices.data:
#             return prices.data[0]["id"]
#     except Exception as e:
#         logger.warning(f"Lookup by key '{lookup_key}' failed: {e}")
#     return None


# def _find_by_product_name(name: str) -> Optional[str]:
#     """
#     Fallback: find a product by name and return its default price.
#     We expand default_price so we don't need a second call.
#     """
#     try:
#         # Pull a reasonable page and match by name (case-insensitive)
#         prods = stripe.Product.list(active=True, limit=100, expand=["data.default_price"])
#         name_lc = name.strip().lower()
#         for p in prods.auto_paging_iter():
#             if str(p["name"]).strip().lower() == name_lc:
#                 dp = p.get("default_price")
#                 if isinstance(dp, dict):
#                     return dp.get("id")
#                 if isinstance(dp, str):
#                     return dp
#     except Exception as e:
#         logger.warning(f"Product lookup by name '{name}' failed: {e}")
#     return None


# def resolve_price_id(plan_type: str, preferred: Optional[str]) -> Tuple[Optional[str], str]:
#     """
#     Resolve a usable Stripe price ID for a plan.
#     Returns (price_id, source) where source explains how it was found.
#     """
#     plan = (plan_type or "").strip().lower()
#     if plan not in ("pro", "premium"):
#         raise HTTPException(status_code=400, detail="Invalid plan_type. Use 'pro' or 'premium'.")

#     # 0) Cache
#     cached = _PRICE_CACHE.get(plan)
#     if cached:
#         return cached, "cache"

#     # 1) Take and verify the preferred (client-provided) price_id if present
#     if preferred:
#         ok = _safe_retrieve_price(preferred)
#         if ok:
#             _PRICE_CACHE[plan] = ok
#             return ok, "client"

#     # 2) Env price ID
#     env_id = ENV_PRO_PRICE_ID if plan == "pro" else ENV_PREMIUM_PRICE_ID
#     if env_id:
#         ok = _safe_retrieve_price(env_id)
#         if ok:
#             _PRICE_CACHE[plan] = ok
#             return ok, "env"

#     # 3) Lookup key, if you use them
#     lk = PRO_LOOKUP_KEY if plan == "pro" else PREMIUM_LOOKUP_KEY
#     ok = _find_by_lookup_key(lk)
#     if ok:
#         _PRICE_CACHE[plan] = ok
#         return ok, "lookup_key"

#     # 4) Product name fallback (matches your Stripe screenshots)
#     prod_name = PRO_PRODUCT_NAME if plan == "pro" else PREMIUM_PRODUCT_NAME
#     ok = _find_by_product_name(prod_name)
#     if ok:
#         _PRICE_CACHE[plan] = ok
#         return ok, "product_name"

#     # Nothing worked → demo mode / misconfiguration
#     return None, "unavailable"


# def _demo_mode() -> bool:
#     return stripe_mode() == "demo" or not STRIPE_SECRET_KEY


# # === Public endpoints ========================================================

# @router.get("/billing/config")
# def billing_config():
#     """
#     Frontend can call this to know if real Stripe is available and which IDs to use.
#     Never exposes your secret; only returns safe info.
#     """
#     if _demo_mode():
#         return {
#             "mode": stripe_mode(),
#             "is_demo": True,
#             "prices": {"pro": None, "premium": None},
#             "source": "demo",
#         }

#     pro_id, pro_src = resolve_price_id("pro", None)
#     prem_id, prem_src = resolve_price_id("premium", None)

#     is_demo = not (pro_id and prem_id)
#     return {
#         "mode": stripe_mode(),
#         "is_demo": is_demo,
#         "prices": {"pro": pro_id, "premium": prem_id},
#         "source": {"pro": pro_src, "premium": prem_src},
#     }


# def _require_stripe_ready():
#     if _demo_mode():
#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 "Stripe is not fully configured (missing STRIPE_SECRET_KEY or prices). "
#                 "Set up prices or enable demo upgrade endpoint."
#             ),
#         )


# # NOTE: keep same paths as your current frontend expects
# @router.post("/create_payment_intent/")
# def create_payment_intent(
#     body: CreateIntentBody,
#     user=Depends(lambda: None),  # keep signature compatible even if your auth dep differs
# ):
#     """
#     Accepts either:
#       - plan_type: "pro" | "premium"  (recommended)
#       - price_id: any price; backend validates and will auto-fix if it belongs to this account
#     Returns: { payment_intent_id }
#     """
#     _require_stripe_ready()

#     # Defensive: derive a plan when only a price_id came in (best effort)
#     plan = (body.plan_type or "").strip().lower()
#     if not plan:
#         # crude inference by amount nicknames if needed — but better to send plan_type from frontend
#         plan = "pro" if "pro" in (body.price_id or "").lower() else "premium"

#     price_id, src = resolve_price_id(plan, body.price_id)
#     if not price_id:
#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 "Unable to resolve a valid Stripe price for this plan. "
#                 "Ensure your Stripe products exist (Pro Plan / Premium Plan) "
#                 "or set STRIPE_PRO_PRICE_ID / STRIPE_PREMIUM_PRICE_ID."
#             ),
#         )

#     logger.info(f"🔥 Creating PaymentIntent for plan={plan} using price_id={price_id} [source={src}]")

#     # Retrieve the price so we get exact amount/currency
#     price = stripe.Price.retrieve(price_id)
#     if not price.get("unit_amount") or not price.get("currency"):
#         raise HTTPException(status_code=400, detail="Stripe price is missing amount/currency.")

#     # Simple one-off PaymentIntent (your UI simulates card with test PM)
#     intent = stripe.PaymentIntent.create(
#         amount=price["unit_amount"],
#         currency=price["currency"],
#         automatic_payment_methods={"enabled": True},
#         metadata={
#             "plan": plan,
#             # Add anything you want to identify the user — avoid PII if possible
#             # "user_id": getattr(user, "id", None),
#             # "email": getattr(user, "email", None),
#         },
#     )

#     return {"payment_intent_id": intent["id"]}


# @router.post("/confirm_payment/")
# def confirm_payment(body: ConfirmBody):
#     """
#     Confirms a PaymentIntent using the universal test payment method.
#     If you later wire Stripe Elements, you’ll pass a real payment_method here.
#     """
#     _require_stripe_ready()

#     try:
#         pi = stripe.PaymentIntent.confirm(
#             body.payment_intent_id,
#             payment_method="pm_card_visa",  # test PM
#         )
#         if pi["status"] != "succeeded":
#             # In test it should usually go to 'succeeded' immediately.
#             return {"status": pi["status"], "client_secret": pi.get("client_secret")}
#         return {"status": "succeeded", "client_secret": pi.get("client_secret")}
#     except stripe.error.StripeError as e:
#         logger.error(f"Stripe confirm error: {e}")
#         raise HTTPException(status_code=400, detail=str(e))

#=========================================================================

