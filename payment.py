# backend/payment.py
import os, logging
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel

# ✅ shared auth dependency (no circular import)
from auth_deps import get_current_user
from models import User

router = APIRouter(prefix="/billing", tags=["billing"])
logger = logging.getLogger("payment")

# ---------- Optional Stripe ----------
stripe = None
try:
    import stripe as _stripe  # type: ignore
    if os.getenv("STRIPE_SECRET_KEY"):
        _stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        stripe = _stripe
except Exception:
    stripe = None

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000").rstrip("/")
PRO_LOOKUP  = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
PREM_LOOKUP = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")

class CheckoutIn(BaseModel):
    # accept either key name
    plan: Optional[str] = None
    tier: Optional[str] = None

def _get_price_id(lookup_key: str) -> Optional[str]:
    if not stripe:
        return None
    try:
        # resilient: filter by lookup_keys
        lst = stripe.Price.list(active=True, lookup_keys=[lookup_key], limit=1)
        if lst.data:
            return lst.data[0].id
    except Exception as e:
        logger.warning("Stripe price lookup failed for %s: %s", lookup_key, e)
    return None

def _ensure_customer(email: str) -> Optional[str]:
    if not stripe or not email:
        return None
    try:
        existing = stripe.Customer.list(email=email, limit=1)
        if existing.data:
            return existing.data[0].id
        created = stripe.Customer.create(email=email)
        return created.id
    except Exception as e:
        logger.warning("Stripe ensure customer failed for %s: %s", email, e)
        return None

@router.get("/config")
def billing_config() -> Dict[str, Any]:
    """Expose which mode we're in + price ids (if available)."""
    mode = "live" if (os.getenv("STRIPE_SECRET_KEY","").startswith("sk_live")) else "test"
    pro_price = _get_price_id(PRO_LOOKUP) if stripe else None
    prem_price = _get_price_id(PREM_LOOKUP) if stripe else None
    return {
        "mode": mode,
        "is_demo": not bool(stripe),
        "pro_price_id": pro_price,
        "premium_price_id": prem_price,
        "pro_lookup_key": PRO_LOOKUP,
        "premium_lookup_key": PREM_LOOKUP,
    }

@router.post("/create_checkout_session")
def create_checkout_session(payload: CheckoutIn, user: User = Depends(get_current_user)):
    """
    Creates a Stripe Checkout Session for a subscription.
    - Disables Link so the page shows **payment options** like card/Klarna/Cash App (pic #4 style).
    - Sets cancel/success to /subscription.
    """
    if not stripe:
        raise HTTPException(status_code=503, detail="Stripe is not configured on this server")

    tier = (payload.plan or payload.tier or "").strip().lower()
    if tier not in {"pro", "premium"}:
        raise HTTPException(status_code=400, detail="Invalid plan; expected 'pro' or 'premium'.")

    lookup = PRO_LOOKUP if tier == "pro" else PREM_LOOKUP
    price_id = _get_price_id(lookup)
    if not price_id:
        raise HTTPException(status_code=400, detail="Missing or invalid Stripe price for plan")

    customer_id = _ensure_customer((user.email or "").strip().lower())

    # ⚙️ Show the multi-method UI (card + optional methods). Stripe will only display ones you’ve enabled.
    payment_method_types = ["card", "cashapp", "klarna"]  # add "amazon_pay" if you’ve enabled it in Dashboard

    try:
        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            customer=customer_id,                     # associates with the user
            customer_email=(user.email or None),      # helps autofill email
            
            # ⬇️ You’ll land back on our app (no login redirect surprises)
            success_url=f"{FRONTEND_URL}/subscription?status=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{FRONTEND_URL}/subscription?status=cancel",

            # Better UX
            allow_promotion_codes=True,
            billing_address_collection="auto",

            # ✅ Key bit: disable Link so you get the full “Payment methods” layout
            payment_method_types=payment_method_types,

            # Prevent the “origin address for automatic tax” error in test mode
            automatic_tax={"enabled": False},
        )
        return {"url": session.url}
    except Exception as e:
        logger.error("Failed to create Checkout session: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create checkout session")

@router.post("/create_portal_session")
def create_portal_session(user: User = Depends(get_current_user)):
    """Stripe Customer Portal for manage/cancel."""
    if not stripe:
        raise HTTPException(status_code=503, detail="Stripe is not configured on this server")
    cid = _ensure_customer((user.email or "").strip().lower())
    try:
        portal = stripe.billing_portal.Session.create(
            customer=cid,
            return_url=f"{FRONTEND_URL}/subscription",
        )
        return {"url": portal.url}
    except Exception as e:
        logger.error("Failed to create Billing Portal session: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to open billing portal")
