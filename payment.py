# payment.py  — robust checkout creator
import os
import stripe
from fastapi import APIRouter, Depends, HTTPException
from typing import Optional

router = APIRouter(prefix="/billing", tags=["billing"])

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

# If you keep lookup keys in env, set them (defaults below are safe)
PRO_LOOKUP_FALLBACK = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
PREMIUM_LOOKUP_FALLBACK = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")

@router.post("/create_checkout_session")
async def create_checkout_session(payload: dict, user=Depends(get_current_user)):
    """
    Accepts any of:
      { "price_lookup_key": "pro_monthly" }
      { "price_id": "price_xxx" }
      { "plan": "pro" | "premium" }
    """
    try:
        lookup: Optional[str] = payload.get("price_lookup_key")
        price_id: Optional[str] = payload.get("price_id")
        plan: Optional[str] = payload.get("plan")

        # derive from plan if needed
        if not lookup and not price_id and plan:
            plan = str(plan).lower()
            if plan == "pro":
                lookup = PRO_LOOKUP_FALLBACK
            elif plan == "premium":
                lookup = PREMIUM_LOOKUP_FALLBACK

        # still nothing? fail clearly
        if not lookup and not price_id:
            raise HTTPException(status_code=400, detail="Missing price_lookup_key or price_id")

        if not price_id and lookup:
            prices = stripe.Price.list(
                lookup_keys=[lookup], expand=["data.product"], active=True, limit=1
            )
            if not prices.data:
                raise HTTPException(status_code=400, detail=f"Unknown price lookup key: {lookup}")
            price_id = prices.data[0].id

        session = stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": price_id, "quantity": 1}],
            # land users back on Subscription page, not Login:
            success_url=f"{FRONTEND_URL}/subscription?status=success&session_id={{CHECKOUT_SESSION_ID}}",
            cancel_url=f"{FRONTEND_URL}/subscription?status=cancelled",
            customer_email=getattr(user, "email", None),
            allow_promotion_codes=True,
            billing_address_collection="auto",

            # keep Checkout looking like your Picture #2
            payment_method_types=["card", "cashapp", "klarna", "amazon_pay"],

            metadata={"app_user_id": str(getattr(user, "id", ""))},
        )
        return {"url": session.url}

    except HTTPException:
        # let FastAPI return the exact code/message
        raise
    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Checkout error: {e}")


