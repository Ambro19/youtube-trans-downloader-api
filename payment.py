# payment.py — resilient Stripe price resolution + simple billing config
# Works in both dev/test and prod, and survives wrong/blank price IDs coming from the client.

import os
import logging
from typing import Optional, Tuple, Dict

import stripe
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

logger = logging.getLogger("payment")
router = APIRouter(tags=["billing"])
router = APIRouter(tags=["payments"])  # no prefix so existing paths remain the same

# === Stripe init =============================================================

STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
    logger.info("✅ Stripe configured successfully")
else:
    logger.error("❌ STRIPE_SECRET_KEY missing. Billing will run in demo mode.")

def stripe_mode() -> str:
    k = STRIPE_SECRET_KEY or ""
    if k.startswith("sk_live_"):
        return "live"
    if k.startswith("sk_test_"):
        return "test"
    return "demo"

# === Config / names / lookup keys ===========================================

# Option 1: direct price IDs (strongest)
ENV_PRO_PRICE_ID = os.getenv("STRIPE_PRO_PRICE_ID")
ENV_PREMIUM_PRICE_ID = os.getenv("STRIPE_PREMIUM_PRICE_ID")

# Option 2: lookup keys on Price objects (if you used them when creating prices)
PRO_LOOKUP_KEY = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
PREMIUM_LOOKUP_KEY = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")

# Option 3: product names (fallback – matches your screenshots)
PRO_PRODUCT_NAME = os.getenv("STRIPE_PRO_PRODUCT_NAME", "Pro Plan")
PREMIUM_PRODUCT_NAME = os.getenv("STRIPE_PREMIUM_PRODUCT_NAME", "Premium Plan")

# Cache so we don’t hit Stripe on every request
_PRICE_CACHE: Dict[str, Optional[str]] = {"pro": None, "premium": None}

# === Request bodies ==========================================================

class CreateIntentBody(BaseModel):
    # You can send either one. Backend will resolve safely.
    plan_type: Optional[str] = None  # "pro" | "premium"
    price_id: Optional[str] = None


class ConfirmBody(BaseModel):
    payment_intent_id: str


# === Helpers =================================================================

def _safe_retrieve_price(price_id: str) -> Optional[str]:
    """Return price_id if it exists in this account/mode, else None."""
    try:
        p = stripe.Price.retrieve(price_id)
        return p["id"]
    except Exception as e:
        # Stripe returns 404 for missing price in this account/mode
        logger.error(f"Stripe couldn't find price '{price_id}': {e}")
        return None


def _find_by_lookup_key(lookup_key: str) -> Optional[str]:
    """Find a price by its lookup_key if you use them."""
    try:
        # API supports listing by lookup_keys (array)
        prices = stripe.Price.list(active=True, lookup_keys=[lookup_key], limit=1)
        if prices.data:
            return prices.data[0]["id"]
    except Exception as e:
        logger.warning(f"Lookup by key '{lookup_key}' failed: {e}")
    return None


def _find_by_product_name(name: str) -> Optional[str]:
    """
    Fallback: find a product by name and return its default price.
    We expand default_price so we don't need a second call.
    """
    try:
        # Pull a reasonable page and match by name (case-insensitive)
        prods = stripe.Product.list(active=True, limit=100, expand=["data.default_price"])
        name_lc = name.strip().lower()
        for p in prods.auto_paging_iter():
            if str(p["name"]).strip().lower() == name_lc:
                dp = p.get("default_price")
                if isinstance(dp, dict):
                    return dp.get("id")
                if isinstance(dp, str):
                    return dp
    except Exception as e:
        logger.warning(f"Product lookup by name '{name}' failed: {e}")
    return None


def resolve_price_id(plan_type: str, preferred: Optional[str]) -> Tuple[Optional[str], str]:
    """
    Resolve a usable Stripe price ID for a plan.
    Returns (price_id, source) where source explains how it was found.
    """
    plan = (plan_type or "").strip().lower()
    if plan not in ("pro", "premium"):
        raise HTTPException(status_code=400, detail="Invalid plan_type. Use 'pro' or 'premium'.")

    # 0) Cache
    cached = _PRICE_CACHE.get(plan)
    if cached:
        return cached, "cache"

    # 1) Take and verify the preferred (client-provided) price_id if present
    if preferred:
        ok = _safe_retrieve_price(preferred)
        if ok:
            _PRICE_CACHE[plan] = ok
            return ok, "client"

    # 2) Env price ID
    env_id = ENV_PRO_PRICE_ID if plan == "pro" else ENV_PREMIUM_PRICE_ID
    if env_id:
        ok = _safe_retrieve_price(env_id)
        if ok:
            _PRICE_CACHE[plan] = ok
            return ok, "env"

    # 3) Lookup key, if you use them
    lk = PRO_LOOKUP_KEY if plan == "pro" else PREMIUM_LOOKUP_KEY
    ok = _find_by_lookup_key(lk)
    if ok:
        _PRICE_CACHE[plan] = ok
        return ok, "lookup_key"

    # 4) Product name fallback (matches your Stripe screenshots)
    prod_name = PRO_PRODUCT_NAME if plan == "pro" else PREMIUM_PRODUCT_NAME
    ok = _find_by_product_name(prod_name)
    if ok:
        _PRICE_CACHE[plan] = ok
        return ok, "product_name"

    # Nothing worked → demo mode / misconfiguration
    return None, "unavailable"


def _demo_mode() -> bool:
    return stripe_mode() == "demo" or not STRIPE_SECRET_KEY


# === Public endpoints ========================================================

@router.get("/billing/config")
def billing_config():
    """
    Frontend can call this to know if real Stripe is available and which IDs to use.
    Never exposes your secret; only returns safe info.
    """
    if _demo_mode():
        return {
            "mode": stripe_mode(),
            "is_demo": True,
            "prices": {"pro": None, "premium": None},
            "source": "demo",
        }

    pro_id, pro_src = resolve_price_id("pro", None)
    prem_id, prem_src = resolve_price_id("premium", None)

    is_demo = not (pro_id and prem_id)
    return {
        "mode": stripe_mode(),
        "is_demo": is_demo,
        "prices": {"pro": pro_id, "premium": prem_id},
        "source": {"pro": pro_src, "premium": prem_src},
    }


def _require_stripe_ready():
    if _demo_mode():
        raise HTTPException(
            status_code=400,
            detail=(
                "Stripe is not fully configured (missing STRIPE_SECRET_KEY or prices). "
                "Set up prices or enable demo upgrade endpoint."
            ),
        )


# NOTE: keep same paths as your current frontend expects
@router.post("/create_payment_intent/")
def create_payment_intent(
    body: CreateIntentBody,
    user=Depends(lambda: None),  # keep signature compatible even if your auth dep differs
):
    """
    Accepts either:
      - plan_type: "pro" | "premium"  (recommended)
      - price_id: any price; backend validates and will auto-fix if it belongs to this account
    Returns: { payment_intent_id }
    """
    _require_stripe_ready()

    # Defensive: derive a plan when only a price_id came in (best effort)
    plan = (body.plan_type or "").strip().lower()
    if not plan:
        # crude inference by amount nicknames if needed — but better to send plan_type from frontend
        plan = "pro" if "pro" in (body.price_id or "").lower() else "premium"

    price_id, src = resolve_price_id(plan, body.price_id)
    if not price_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unable to resolve a valid Stripe price for this plan. "
                "Ensure your Stripe products exist (Pro Plan / Premium Plan) "
                "or set STRIPE_PRO_PRICE_ID / STRIPE_PREMIUM_PRICE_ID."
            ),
        )

    logger.info(f"🔥 Creating PaymentIntent for plan={plan} using price_id={price_id} [source={src}]")

    # Retrieve the price so we get exact amount/currency
    price = stripe.Price.retrieve(price_id)
    if not price.get("unit_amount") or not price.get("currency"):
        raise HTTPException(status_code=400, detail="Stripe price is missing amount/currency.")

    # Simple one-off PaymentIntent (your UI simulates card with test PM)
    intent = stripe.PaymentIntent.create(
        amount=price["unit_amount"],
        currency=price["currency"],
        automatic_payment_methods={"enabled": True},
        metadata={
            "plan": plan,
            # Add anything you want to identify the user — avoid PII if possible
            # "user_id": getattr(user, "id", None),
            # "email": getattr(user, "email", None),
        },
    )

    return {"payment_intent_id": intent["id"]}


@router.post("/confirm_payment/")
def confirm_payment(body: ConfirmBody):
    """
    Confirms a PaymentIntent using the universal test payment method.
    If you later wire Stripe Elements, you’ll pass a real payment_method here.
    """
    _require_stripe_ready()

    try:
        pi = stripe.PaymentIntent.confirm(
            body.payment_intent_id,
            payment_method="pm_card_visa",  # test PM
        )
        if pi["status"] != "succeeded":
            # In test it should usually go to 'succeeded' immediately.
            return {"status": pi["status"], "client_secret": pi.get("client_secret")}
        return {"status": "succeeded", "client_secret": pi.get("client_secret")}
    except stripe.error.StripeError as e:
        logger.error(f"Stripe confirm error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

#=========================================================================

