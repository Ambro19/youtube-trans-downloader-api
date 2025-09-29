# backend/subscription_sync.py
from __future__ import annotations

import os
import logging
from typing import Optional, Tuple

import stripe
from sqlalchemy.orm import Session
from sqlalchemy.exc import OperationalError, SQLAlchemyError

# Adjust these imports to your project layout if needed
from models import User
try:
    # Optional: if you have a subscriptions table
    from models import Subscription  # type: ignore
    HAS_SUBSCRIPTION_MODEL = True
except Exception:
    Subscription = None  # type: ignore
    HAS_SUBSCRIPTION_MODEL = False

logger = logging.getLogger("payment")

STRIPE_API_KEY = os.getenv("STRIPE_SECRET_KEY") or os.getenv("STRIPE_API_KEY")
if STRIPE_API_KEY:
    stripe.api_key = STRIPE_API_KEY


# --- helpers ---------------------------------------------------------------

def _normalize_tier_from_price(price_obj) -> Optional[str]:
    """
    Map Stripe price to our app's tier.
    Prefers price.lookup_key, falls back to product/price metadata/name checks.
    """
    if not price_obj:
        return None

    lk = (getattr(price_obj, "lookup_key", None) or "").lower()
    if "premium" in lk:
        return "premium"
    if "pro" in lk:
        return "pro"

    # Try product name / metadata as a fallback
    try:
        prod = price_obj.get("product") if isinstance(price_obj, dict) else price_obj.product
        if isinstance(prod, str):
            prod = stripe.Product.retrieve(prod)
        name = (prod.get("name") if isinstance(prod, dict) else getattr(prod, "name", "")).lower()
        if "premium" in name:
            return "premium"
        if "pro" in name:
            return "pro"
    except Exception:
        pass

    return None


def _find_stripe_customer_id_for_user(user: User) -> Optional[str]:
    # In your app this may live on user.stripe_customer_id or in your Subscription row
    cid = getattr(user, "stripe_customer_id", None)
    if cid:
        return cid

    # Try a quick lookup by email
    try:
        # Works in most Stripe accounts without enabling the search beta
        lst = stripe.Customer.list(email=getattr(user, "email", None), limit=1)
        if lst and lst.data:
            return lst.data[0].id
    except Exception:
        pass

    return None


def _get_active_tier_from_stripe(customer_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Returns (tier, subscription_id) for an ACTIVE Stripe subscription, else (None, None).
    """
    if not customer_id or not STRIPE_API_KEY:
        return (None, None)

    subs = stripe.Subscription.list(customer=customer_id, status="active", limit=1)
    if not subs or not subs.data:
        return (None, None)

    sub = subs.data[0]
    price = None
    try:
        # First item is fine for single-plan subscriptions
        if sub.items and sub.items.data:
            price = sub.items.data[0].price
    except Exception:
        pass

    tier = _normalize_tier_from_price(price)
    return (tier, sub.id if tier else None)


# --- public API ------------------------------------------------------------

def sync_user_subscription_from_stripe(db: Session, user: User) -> Tuple[Optional[str], Optional[str]]:
    """
    Idempotently reconcile local subscription for `user` with Stripe.

    Returns:
      (new_tier, subscription_id) if we found/updated a tier, else (None, None).
    """
    try:
        customer_id = _find_stripe_customer_id_for_user(user)
        if not customer_id:
            logger.info(f"No Stripe customer found for {getattr(user, 'email', user.id)}")
            return (None, None)

        tier, sub_id = _get_active_tier_from_stripe(customer_id)
        if not tier:
            logger.info(f"No active Stripe subscription for customer {customer_id}")
            return (None, None)

        # Update user's tier if needed
        current_tier = getattr(user, "subscription_tier", "free") or "free"
        if current_tier != tier:
            setattr(user, "subscription_tier", tier)
            try:
                db.add(user)
                db.commit()
                db.refresh(user)
            except SQLAlchemyError as e:
                db.rollback()
                logger.warning(f"Failed to persist user tier change -> {e}")

        # Optional: update or create Subscription row if your schema supports it
        if HAS_SUBSCRIPTION_MODEL and Subscription is not None:
            try:
                sub_row = db.query(Subscription).filter(Subscription.user_id == user.id).first()
                if not sub_row:
                    sub_row = Subscription(user_id=user.id)  # type: ignore
                # Be defensive: some migrations may not have all columns yet
                for attr, value in {
                    "tier": tier,
                    "status": "active",
                    "stripe_subscription_id": sub_id,
                    "stripe_customer_id": customer_id,
                }.items():
                    if hasattr(sub_row, attr):
                        setattr(sub_row, attr, value)

                db.add(sub_row)
                db.commit()
            except OperationalError as e:
                # e.g. "no such column: subscriptions.stripe_customer_id"
                db.rollback()
                logger.warning(f"Skipping Subscription table update due to schema mismatch: {e}")
            except SQLAlchemyError as e:
                db.rollback()
                logger.warning(f"Failed to upsert Subscription row: {e}")

        return (tier, sub_id)
    except stripe.error.StripeError as e:
        logger.warning(f"Stripe sync error for user {getattr(user, 'email', user.id)}: {e}")
        return (None, None)
    except Exception as e:
        logger.warning(f"Unexpected sync error: {e}")
        return (None, None)
