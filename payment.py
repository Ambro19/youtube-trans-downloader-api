# backend/payment.py
import os
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

# ✅ shared auth dependency (no circular import)
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

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000").rstrip("/")
PRO_LOOKUP   = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
PREM_LOOKUP  = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")
ENABLE_TAX   = os.getenv("STRIPE_ENABLE_TAX", "false").strip().lower() == "true"


class CheckoutPayload(BaseModel):
    # accept either plan or tier for compatibility
    plan: Optional[str] = None
    tier: Optional[str] = None


def _get_price_id(lookup_key: str) -> Optional[str]:
    if not stripe:
        logger.info("Stripe not configured; cannot resolve price for %s", lookup_key)
        return None
    try:
        lst = stripe.Price.list(active=True, lookup_keys=[lookup_key], limit=1)
        if lst.data:
            return lst.data[0].id
    except Exception as e:
        logger.warning("Stripe price lookup failed for %s: %s", lookup_key, e)
    return None


@router.get("/config")
def billing_config():
    """
    Frontend uses this to know if Stripe is test/live and to get price ids (if resolvable).
    """
    if not stripe:
        return {
            "mode": "disabled",
            "is_demo": True,
            "pro_price_id": None,
            "premium_price_id": None,
        }
    try:
        mode = "live" if STRIPE_SECRET.startswith("sk_live_") else "test"
        pro = _get_price_id(PRO_LOOKUP)
        premium = _get_price_id(PREM_LOOKUP)
        return {
            "mode": mode,
            "is_demo": False,
            "pro_price_id": pro,
            "premium_price_id": premium,
        }
    except Exception as e:
        logger.warning("billing_config error: %s", e)
        return {"mode": "unknown", "is_demo": True, "pro_price_id": None, "premium_price_id": None}


@router.post("/create_checkout_session")
def create_checkout_session(payload: CheckoutPayload, user: User = Depends(get_current_user)):
    """
    Create Stripe Checkout for Pro/Premium subscriptions.
    Disables automatic tax by default; can be enabled with STRIPE_ENABLE_TAX=true.
    """
    if not stripe:
        raise HTTPException(status_code=503, detail="Billing is not configured")

    tier = (payload.tier or payload.plan or "").strip().lower()
    if tier not in {"pro", "premium"}:
        raise HTTPException(status_code=400, detail="Invalid plan. Choose 'pro' or 'premium'.")

    lookup = PRO_LOOKUP if tier == "pro" else PREM_LOOKUP
    price_id = _get_price_id(lookup)
    if not price_id:
        raise HTTPException(status_code=400, detail="Price not found for requested plan")

    base_params = {
        "mode": "subscription",
        "line_items": [{"price": price_id, "quantity": 1}],
        "success_url": f"{FRONTEND_URL}/subscription?status=success",
        "cancel_url": f"{FRONTEND_URL}/subscription?status=cancelled",
        "allow_promotion_codes": True,
        "client_reference_id": f"user-{user.id}",
        # prefer letting Stripe dedupe by email; you can also pass 'customer' if you store IDs
        "customer_email": (user.email or None),
    }

    def try_create(with_tax: bool):
        params = dict(base_params)
        if with_tax:
            params["automatic_tax"] = {"enabled": True}
        return stripe.checkout.Session.create(**params)

    try:
        if ENABLE_TAX:
            try:
                session = try_create(with_tax=True)
            except Exception as e:
                # Common in test mode if origin address not configured
                logger.warning("Checkout with automatic tax failed; retrying without tax: %s", e)
                session = try_create(with_tax=False)
        else:
            session = try_create(with_tax=False)

        return {"url": session.url}
    except Exception as e:
        logger.error("Failed to create Checkout session: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create checkout session")


@router.post("/create_portal_session")
def create_portal_session(user: User = Depends(get_current_user)):
    """
    Open the Stripe Customer Portal if available. (Optional)
    """
    if not stripe:
        raise HTTPException(status_code=503, detail="Billing is not configured")
    try:
        # Find or create customer by email
        cust = None
        if user.email:
            res = stripe.Customer.list(email=user.email, limit=1)
            cust = res.data[0] if res.data else None
        if not cust and user.email:
            cust = stripe.Customer.create(email=user.email, name=user.username or user.email)

        if not cust:
            raise HTTPException(status_code=400, detail="No customer email on file")

        session = stripe.billing_portal.Session.create(
            customer=cust.id,
            return_url=f"{FRONTEND_URL}/subscription",
        )
        return {"url": session.url}
    except Exception as e:
        logger.error("Failed to create billing portal session: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to open billing portal")
