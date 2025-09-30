# backend/subscription_sync.py
import os
import logging
from typing import Optional, Tuple

from sqlalchemy.orm import Session
from models import User

# Stripe init (kept local here to avoid circular imports)
stripe = None
try:
    import stripe as _stripe  # type: ignore
    if os.getenv("STRIPE_SECRET_KEY"):
        _stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        stripe = _stripe
except Exception:
    stripe = None

logger = logging.getLogger("payment")

# Keep these in sync with your pricing setup
PRO_LOOKUP = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly")
PREM_LOOKUP = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly")

# Which Stripe statuses should grant entitlements
ACTIVE_LIKE = {"active", "trialing"}  # add "past_due" if you want a grace period

# Rank statuses when choosing a "best" subscription
STATUS_RANK = {
    "active": 100,
    "trialing": 90,
    "past_due": 70,        # not granting by default, but higher than incomplete
    "paused": 60,
    "unpaid": 50,
    "incomplete": 10,
    "incomplete_expired": 0,
    "canceled": -1,
}


def _ensure_customer_id(user: User) -> Optional[str]:
    """
    Try to ensure we have a Stripe customer id for this user.
    We do NOT create a customer here (that's handled during checkout).
    """
    if not stripe:
        return None

    cust_id = (user.stripe_customer_id or "").strip() or None
    if cust_id:
        try:
            c = stripe.Customer.retrieve(cust_id)
            if c and not c.get("deleted", False):
                return cust_id
        except Exception as e:
            logger.info("Customer %s not valid anymore: %s", cust_id, e)

    # Try lookup by email
    email = (user.email or "").strip().lower()
    if not email:
        return None
    try:
        found = stripe.Customer.list(email=email, limit=1)
        if found and found.data:
            return found.data[0].id
    except Exception as e:
        logger.warning("Customer lookup by email failed (%s): %s", email, e)
    return None


def _pick_best_subscription(subs) -> Optional[dict]:
    """
    Given a list of subscriptions, pick the best candidate.
    """
    if not subs:
        return None
    # Sort by (status rank, created desc)
    return sorted(
        subs,
        key=lambda s: (STATUS_RANK.get(s.get("status", ""), 0), s.get("created", 0)),
        reverse=True,
    )[0]


def _infer_tier_from_subscription(sub: dict) -> Optional[str]:
    """
    Map Stripe subscription → app tier using price.lookup_key primarily.
    """
    try:
        items = sub.get("items", {}).get("data", [])
        if not items:
            return None
        price = items[0].get("price", {}) or {}
        lookup = (price.get("lookup_key") or "").strip().lower()

        if lookup == PRO_LOOKUP.lower() or "pro" in lookup:
            return "pro"
        if lookup == PREM_LOOKUP.lower() or "premium" in lookup:
            return "premium"

        # Fallbacks: nickname/product name/price id heuristics
        nickname = (price.get("nickname") or "").lower()
        if "pro" in nickname:
            return "pro"
        if "premium" in nickname:
            return "premium"

        product = price.get("product")
        # If expanded product exists, try its name
        if isinstance(product, dict):
            name = (product.get("name") or "").lower()
            if "pro" in name:
                return "pro"
            if "premium" in name:
                return "premium"

        # Last-resort heuristic on price id
        pid = (price.get("id") or "").lower()
        if "pro" in pid:
            return "pro"
        if "prem" in pid or "premium" in pid:
            return "premium"
    except Exception as e:
        logger.warning("Failed to infer tier from subscription: %s", e)
    return None


def sync_user_subscription_from_stripe(db: Session, user: User) -> Tuple[str, Optional[str]]:
    """
    Pulls subscription state from Stripe and updates the local user record.
    Returns (tier, stripe_subscription_id).
    """
    if not stripe:
        logger.info("Stripe not configured; keeping tier=%s", getattr(user, "subscription_tier", "free"))
        return getattr(user, "subscription_tier", "free") or "free", getattr(user, "stripe_subscription_id", None)

    customer_id = _ensure_customer_id(user)
    if not customer_id:
        logger.info("No Stripe customer found for user %s; keeping local tier=%s",
                    user.id, getattr(user, "subscription_tier", "free"))
        return getattr(user, "subscription_tier", "free") or "free", getattr(user, "stripe_subscription_id", None)

    try:
        # ⚠️ Do NOT filter by status here; we’ll pick the best one ourselves.
        subs = stripe.Subscription.list(
            customer=customer_id,
            limit=10,
            expand=["data.items.data.price.product"]
        ).data
    except Exception as e:
        logger.error("Stripe subscriptions list failed for %s: %s", customer_id, e)
        return getattr(user, "subscription_tier", "free") or "free", getattr(user, "stripe_subscription_id", None)

    if not subs:
        logger.info("No subscriptions found for customer %s", customer_id)
        new_tier, sub_id = "free", None
    else:
        best = _pick_best_subscription(subs)
        status = best.get("status")
        sub_id = best.get("id")

        if status in ACTIVE_LIKE:
            tier = _infer_tier_from_subscription(best) or "pro"  # default to pro if ambiguous
            new_tier = tier
            logger.info("Customer %s subscription %s is %s → tier=%s",
                        customer_id, sub_id, status, new_tier)
        else:
            logger.info("Customer %s has subscription %s but status=%s (not active-like)",
                        customer_id, sub_id, status)
            new_tier, sub_id = "free", None

    # Persist if changed
    changed = False
    if getattr(user, "stripe_customer_id", None) != customer_id:
        user.stripe_customer_id = customer_id
        changed = True
    if getattr(user, "stripe_subscription_id", None) != sub_id:
        user.stripe_subscription_id = sub_id
        changed = True
    if getattr(user, "subscription_tier", "free") != new_tier:
        user.subscription_tier = new_tier
        changed = True

    if changed:
        try:
            db.commit()
            db.refresh(user)
        except Exception as e:
            logger.error("Failed to persist subscription sync for user %s: %s", user.id, e)
            db.rollback()

    return user.subscription_tier or "free", user.stripe_subscription_id
