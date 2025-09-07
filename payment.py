# backend/payment.py
import os
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

# ✅ shared auth dependency (avoids circular import with main.py)
from auth_deps import get_current_user
from models import User

router = APIRouter(prefix="/billing", tags=["billing"])
logger = logging.getLogger("payment")

# --- Stripe (enabled only if a secret key is present) ---
STRIPE_SECRET = os.getenv("STRIPE_SECRET_KEY", "").strip()
stripe = None
try:
    import stripe as _stripe  # type: ignore
    if STRIPE_SECRET:
        _stripe.api_key = STRIPE_SECRET
        stripe = _stripe
except Exception as e:
    logger.warning("Stripe import/init failed: %s", e)
    stripe = None

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
PRO_LOOKUP    = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
PREM_LOOKUP   = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")

class CheckoutPayload(BaseModel):
    # new safer payload — we map plan/tier to lookup keys
    plan: Optional[str] = None
    tier: Optional[str] = None
    price_lookup_key: Optional[str] = None  # backward-compat

@router.get("/config")
def config():
    """
    Returns minimal config for the UI and resolves price IDs via lookup keys.
    """
    mode = "live" if (os.getenv("STRIPE_SECRET_KEY", "").startswith("sk_live_")) else "test"
    if not stripe:
        return {
            "mode": mode,
            "is_demo": True,
            "pro_price_id": None,
            "premium_price_id": None
        }

    def _get_price_id(lookup_key: str) -> Optional[str]:
        try:
            # Resilient: use lookup_keys[] filter
            lst = stripe.Price.list(active=True, lookup_keys=[lookup_key], limit=1)
            if lst.data:
                return lst.data[0].id
        except Exception as e:
            logger.warning("Stripe price lookup failed for %s: %s", lookup_key, e)
        return None

    return {
        "mode": mode,
        "is_demo": False,
        "pro_price_id": _get_price_id(PRO_LOOKUP),
        "premium_price_id": _get_price_id(PREM_LOOKUP),
    }

def _resolve_lookup_key(payload: CheckoutPayload) -> str:
    if payload.price_lookup_key:
        return payload.price_lookup_key
    plan = (payload.plan or payload.tier or "").lower().strip()
    if plan == "pro":
        return PRO_LOOKUP
    if plan == "premium":
        return PREM_LOOKUP
    raise HTTPException(status_code=400, detail="Missing plan (expected 'pro' or 'premium').")

def _price_id_from_lookup(lookup_key: str) -> str:
    if not stripe:
        # In dev without Stripe, simulate a session error with message
        raise HTTPException(status_code=503, detail="Stripe is not configured on the server.")
    try:
        prices = stripe.Price.list(active=True, lookup_keys=[lookup_key], limit=1)
        if not prices.data:
            raise HTTPException(status_code=404, detail=f"No Stripe price for lookup key '{lookup_key}'.")
        return prices.data[0].id
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Stripe price list failed")
        raise HTTPException(status_code=502, detail=f"Stripe error: {e}")

def _get_price_id(lookup_key: str) -> Optional[str]:
    if not stripe:
        logger.info("Stripe is not configured; cannot resolve price for %s", lookup_key)
        return None
    try:
        lst = stripe.Price.list(active=True, lookup_keys=[lookup_key], limit=1)
        if lst.data:
            return lst.data[0].id
    except Exception as e:
        logger.warning("Stripe price lookup failed for %s: %s", lookup_key, e)
    return None

@router.post("/create_checkout_session")
async def create_checkout_session(payload: CheckoutPayload, user: User = Depends(get_current_user)):
    """
    Create a Stripe Checkout Session for subscriptions.
    Accepts { plan: 'pro'|'premium' } (or legacy { price_lookup_key }).
    """
    try:
      lookup_key = _resolve_lookup_key(payload)
      price_id   = _price_id_from_lookup(lookup_key)

      success_url = f"{FRONTEND_URL}/subscription?status=success"
      cancel_url  = f"{FRONTEND_URL}/subscription?status=cancel"

      # Find or create a customer by email (simple approach for test mode)
      customer_id = None
      if stripe and getattr(user, "email", None):
          try:
              # search API may not be enabled in all accounts; using list as fallback
              existing = stripe.Customer.list(email=user.email, limit=1)
              if existing.data:
                  customer_id = existing.data[0].id
              else:
                  cust = stripe.Customer.create(email=user.email, name=(user.username or ""))
                  customer_id = cust.id
          except Exception as e:
              logger.warning("Could not ensure Stripe customer for %s: %s", user.email, e)

      session = stripe.checkout.Session.create(
          mode="subscription",
          line_items=[{"price": price_id, "quantity": 1}],
          success_url=success_url,
          cancel_url=cancel_url,
          customer=customer_id,
          allow_promotion_codes=True,
          automatic_tax={"enabled": True},
      )

      return {"id": session.id, "url": session.url}
    except HTTPException:
      raise
    except Exception as e:
      logger.error("Failed to create Checkout session: %s", e, exc_info=True)
      raise HTTPException(status_code=500, detail="Failed to create checkout session")

@router.post("/create_portal_session")
async def create_portal_session(user: User = Depends(get_current_user)):
    if not stripe:
        raise HTTPException(status_code=503, detail="Stripe is not configured on the server.")
    try:
        # Same customer lookup as above
        customer_id = None
        if getattr(user, "email", None):
            existing = stripe.Customer.list(email=user.email, limit=1)
            if existing.data:
                customer_id = existing.data[0].id
            else:
                cust = stripe.Customer.create(email=user.email, name=(user.username or ""))
                customer_id = cust.id

        portal = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f"{FRONTEND_URL}/subscription"
        )
        return {"url": portal.url}
    except Exception as e:
        logger.error("Failed to create portal session: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to open billing portal")

@router.post("/cancel_subscription")
async def cancel_subscription(user: User = Depends(get_current_user)):
    # delegate to your existing main.py handler if you prefer;
    # leaving a thin stub lets the front-end always POST /billing/cancel_subscription.
    from main import cancel_subscription as _cancel  # local import to avoid circular at import-time
    return _cancel.__wrapped__(  # type: ignore[attr-defined]
        req=type("R", (), {"at_period_end": True})(),  # mimic Pydantic model default
        current_user=user
    )
