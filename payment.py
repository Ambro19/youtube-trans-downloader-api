# backend/payment.py
import os
import logging
from typing import Optional, Dict, Any

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request

# IMPORTANT: import after FastAPI imports so the name is available.
# main.py defines get_current_user before it imports this router,
# so this does NOT create a problematic circular reference.
from main import get_current_user  # <-- fixes NameError

logger = logging.getLogger("payment")
router = APIRouter(prefix="/billing", tags=["billing"])

# --- Env ---
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.getenv("STRIPE_PUBLISHABLE_KEY", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000").rstrip("/")

# Prefer lookup keys so you can rotate prices safely
PRO_LOOKUP_KEY = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
PREMIUM_LOOKUP_KEY = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")

if not STRIPE_SECRET_KEY:
    logger.warning("STRIPE_SECRET_KEY missing; billing endpoints will fail.")

stripe.api_key = STRIPE_SECRET_KEY


def _get_price_by_lookup_key(lookup_key: str) -> stripe.Price:
    """Fetch a single active price by lookup key."""
    try:
        prices = stripe.Price.list(active=True, lookup_keys=[lookup_key], limit=1)
        if not prices.data:
            raise HTTPException(status_code=400, detail=f"Unknown lookup key: {lookup_key}")
        return prices.data[0]
    except stripe.error.StripeError as e:
        logger.exception("Stripe error while fetching price for lookup key %s", lookup_key)
        raise HTTPException(status_code=502, detail=str(e)) from e


def _lookup_key_from_plan(plan: Optional[str]) -> Optional[str]:
    if not plan:
        return None
    p = plan.strip().lower()
    if p == "pro":
        return PRO_LOOKUP_KEY
    if p == "premium":
        return PREMIUM_LOOKUP_KEY
    return None


@router.get("/config")
def get_billing_config() -> Dict[str, Any]:
    """Expose publishable key and the current price IDs resolved from lookup keys."""
    pro_price_id = None
    premium_price_id = None
    try:
        pro_price_id = _get_price_by_lookup_key(PRO_LOOKUP_KEY).id
    except Exception:
        pass
    try:
        premium_price_id = _get_price_by_lookup_key(PREMIUM_LOOKUP_KEY).id
    except Exception:
        pass

    return {
        "publishableKey": STRIPE_PUBLISHABLE_KEY,
        "prices": {
            "pro": {"lookup_key": PRO_LOOKUP_KEY, "price_id": pro_price_id},
            "premium": {"lookup_key": PREMIUM_LOOKUP_KEY, "price_id": premium_price_id},
        },
    }


@router.post("/create_checkout_session")
async def create_checkout_session(
    payload: Dict[str, Any],
    request: Request,
    user=Depends(get_current_user),  # <-- secured; now properly imported
):
    """
    Body can be either:
      { "price_lookup_key": "pro_monthly" }
      OR
      { "plan": "pro" | "premium" }
    """
    # Accept both shapes
    lookup_key = payload.get("price_lookup_key") or _lookup_key_from_plan(payload.get("plan"))
    if not lookup_key:
        logger.error("create_checkout_session: Missing price lookup key/plan")
        raise HTTPException(status_code=400, detail="Missing price_lookup_key")

    price = _get_price_by_lookup_key(lookup_key)

    # Build URLs so "Back" returns to the subscription page (not Login)
    success_url = f"{FRONTEND_URL}/subscription?session_id={{CHECKOUT_SESSION_ID}}"
    cancel_url = f"{FRONTEND_URL}/subscription?canceled=true"

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price.id, "quantity": 1}],
            success_url=success_url,
            cancel_url=cancel_url,
            # Attach email if we have it (helps customer linking)
            customer_email=getattr(user, "email", None) or getattr(user, "username", None),
            metadata={
                "app_user": getattr(user, "username", None) or getattr(user, "email", None) or "unknown",
                "lookup_key": lookup_key,
            },
            allow_promotion_codes=True,
            automatic_tax={"enabled": True},
        )
        return {"id": session.id, "url": session.url}
    except stripe.error.StripeError as e:
        logger.exception("Failed to create Checkout session")
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.post("/create_portal_session")
async def create_portal_session(
    payload: Dict[str, Any],
    user=Depends(get_current_user),
):
    """
    Expects { "customer_id": "cus_..." } from your stored customer mapping,
    or you can look it up by email if you maintain that mapping.
    """
    customer_id = payload.get("customer_id")
    if not customer_id:
        raise HTTPException(status_code=400, detail="Missing customer_id")

    try:
        session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=f"{FRONTEND_URL}/subscription",
        )
        return {"url": session.url}
    except stripe.error.StripeError as e:
        logger.exception("Failed to create billing portal session")
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.post("/cancel_subscription")
async def cancel_subscription(
    payload: Dict[str, Any],
    user=Depends(get_current_user),
):
    subscription_id = payload.get("subscription_id")
    if not subscription_id:
        raise HTTPException(status_code=400, detail="Missing subscription_id")

    try:
        sub = stripe.Subscription.update(subscription_id, cancel_at_period_end=True)
        return {"status": sub.status, "cancel_at_period_end": sub.cancel_at_period_end}
    except stripe.error.StripeError as e:
        logger.exception("Failed to cancel subscription")
        raise HTTPException(status_code=502, detail=str(e)) from e
