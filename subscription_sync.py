# subscription_sync.py
import os
import logging
from typing import Optional
from sqlalchemy.orm import Session

from models import User

log = logging.getLogger("payment")  # keep same channel used in your logs

# ---- Stripe init ------------------------------------------------------------
stripe = None
try:
    import stripe as _stripe  # type: ignore
    if os.getenv("STRIPE_SECRET_KEY"):
        _stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        stripe = _stripe
except Exception:
    stripe = None

PRO_LOOKUP  = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly").lower()
PREM_LOOKUP = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly").lower()

ACTIVE_STATUSES = {"active", "trialing", "past_due"}  # consider these “entitled”


def _tier_from_price(price: dict) -> Optional[str]:
    """
    Decide Pro/Premium from a Stripe Price object.
    We rely on Price.lookup_key (no deep product expansion required).
    Fallbacks: price.metadata.tier or nickname strings.
    """
    if not price:
        return None

    lk = (price.get("lookup_key") or "").lower()
    if lk == PRO_LOOKUP:
        return "pro"
    if lk == PREM_LOOKUP:
        return "premium"

    meta_tier = (price.get("metadata") or {}).get("tier", "").lower()
    if meta_tier in {"pro", "premium"}:
        return meta_tier

    nick = (price.get("nickname") or "").lower()
    if "premium" in nick:
        return "premium"
    if "pro" in nick:
        return "pro"

    return None


def sync_user_subscription_from_stripe(user: User, db: Session) -> bool:
    """
    Pull the user's latest subscription from Stripe and update user.subscription_tier.
    Returns True if the user's tier was changed, else False.
    """
    if not stripe:
        log.info("Stripe not configured; skipping sync.")
        return False

    customer_id = (user.stripe_customer_id or "").strip()
    if not customer_id:
        # Nothing we can do—checkout flow sets this; avoid creating here.
        log.info("No stripe_customer_id on user; skipping sync.")
        return False

    try:
        # IMPORTANT: don't over-expand; expand only price (NOT price.product)
        subs = stripe.Subscription.list(
            customer=customer_id,
            limit=10,
            expand=["data.items.data.price"],
            # do NOT pass status=... so we can see trialing/past_due right after payment
        )
    except Exception as e:
        log.error(f"Stripe subscriptions list failed for {customer_id}: {e}")
        return False

    if not subs.data:
        log.info(f"No subscriptions for customer {customer_id}")
        # If there are none, consider downgrading to free
        if user.subscription_tier and user.subscription_tier != "free":
            user.subscription_tier = "free"
            db.commit()
            db.refresh(user)
            return True
        return False

    # Pick the most recent, non-canceled subscription that still grants access
    subs_sorted = sorted(
        subs.data,
        key=lambda s: int(s.get("current_period_end") or s.get("created") or 0),
        reverse=True,
    )

    new_tier: Optional[str] = None
    for s in subs_sorted:
        status = s.get("status")
        if status not in ACTIVE_STATUSES:
            continue
        if s.get("cancel_at_period_end"):
            continue

        # Find a qualifying item (usually just one)
        items = (s.get("items") or {}).get("data") or []
        for it in items:
            price = it.get("price")
            tier = _tier_from_price(price)
            if tier in {"pro", "premium"}:
                new_tier = tier
                break
        if new_tier:
            break

    # If nothing matched, assume free
    if not new_tier:
        new_tier = "free"

    if (user.subscription_tier or "free") != new_tier:
        log.info(f"Updating user {user.id} tier {user.subscription_tier!r} -> {new_tier!r}")
        user.subscription_tier = new_tier
        db.commit()
        db.refresh(user)
        return True

    return False
