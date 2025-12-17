# webhook_handler.py
"""
Stripe webhook handler for automatic subscription upgrades/downgrades.

Design:
- main.py verifies Stripe signature + idempotency and sets request.state.verified_event
- this handler processes the event and updates local user state
- tier/status/expiry are ultimately aligned by calling sync_user_subscription_from_stripe(user, db)
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional, Any, Dict

from fastapi import Request, HTTPException  # pyright: ignore[reportMissingImports]
from sqlalchemy.orm import Session  # pyright: ignore[reportMissingImports]

from models import User, SessionLocal
from subscription_sync import sync_user_subscription_from_stripe  # your updated sync

logger = logging.getLogger("payment")

# Stripe is already configured in many projects in main.py; keep this lightweight.
stripe = None
try:
    import stripe as _stripe  # type: ignore
    if os.getenv("STRIPE_SECRET_KEY"):
        _stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        stripe = _stripe
except Exception:
    stripe = None


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(int(value), tz=timezone.utc)
        except Exception:
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None
    return None


def _db() -> Session:
    return SessionLocal()


def _find_user_by_customer_id(db: Session, customer_id: str) -> Optional[User]:
    return db.query(User).filter(User.stripe_customer_id == customer_id).first()


def _find_user_fallback_by_email(db: Session, email: Optional[str]) -> Optional[User]:
    if not email:
        return None
    return db.query(User).filter(User.email == email).first()


def _set_user_stripe_fields_best_effort(user: User, *, status: Optional[str], period_end: Optional[datetime]) -> None:
    # These are the local timestamps/status fields you wanted.
    now = _utcnow()

    if hasattr(user, "stripe_subscription_status") and status:
        user.stripe_subscription_status = status.lower()

    if hasattr(user, "subscription_expires_at") and period_end:
        user.subscription_expires_at = period_end

    if hasattr(user, "subscription_updated_at"):
        user.subscription_updated_at = now


def _extract_customer_id(event_type: str, obj: Dict[str, Any]) -> Optional[str]:
    # For most subscription/invoice events:
    cid = obj.get("customer")
    if cid:
        return str(cid)

    # checkout.session.completed: customer is also here; sometimes nested
    if event_type == "checkout.session.completed":
        cid = obj.get("customer")
        if cid:
            return str(cid)

    return None


def _extract_email_from_checkout(obj: Dict[str, Any]) -> Optional[str]:
    # Stripe checkout sessions often include email in customer_details
    cd = obj.get("customer_details") or {}
    email = cd.get("email") or obj.get("customer_email")
    return str(email) if email else None


def _extract_period_end(event_type: str, obj: Dict[str, Any]) -> Optional[datetime]:
    # Subscription objects include current_period_end (unix seconds)
    if event_type.startswith("customer.subscription."):
        return _to_dt(obj.get("current_period_end"))

    # Invoice objects sometimes have lines with period end; but simplest:
    # rely on sync_user_subscription_from_stripe to pull canonical period_end.
    return None


def _extract_sub_status(event_type: str, obj: Dict[str, Any]) -> Optional[str]:
    if event_type.startswith("customer.subscription."):
        st = obj.get("status")
        return str(st) if st else None
    # invoice events don't directly represent subscription status reliably
    return None


RELEVANT_EVENTS = {
    "checkout.session.completed",
    "customer.subscription.created",
    "customer.subscription.updated",
    "customer.subscription.deleted",
    "invoice.payment_succeeded",
    "invoice.payment_failed",
}


async def handle_stripe_webhook(request: Request):
    """
    Main webhook handler. Expects main.py already:
    - verified signature
    - did idempotency check
    - stored event in request.state.verified_event
    """
    if not stripe or not os.getenv("STRIPE_SECRET_KEY"):
        raise HTTPException(status_code=503, detail="Stripe is not configured")

    event = getattr(request.state, "verified_event", None)
    if not event:
        raise HTTPException(status_code=400, detail="Missing verified webhook event")

    event_type = event.get("type")
    if not event_type:
        raise HTTPException(status_code=400, detail="Invalid event type")

    obj = (event.get("data") or {}).get("object") or {}
    logger.info(f"✅ Stripe webhook received: {event_type}")

    # Ignore irrelevant events safely
    if event_type not in RELEVANT_EVENTS:
        return {"status": "ok", "ignored": True, "event_type": event_type}

    db = _db()
    try:
        customer_id = _extract_customer_id(event_type, obj)
        user: Optional[User] = None

        if customer_id:
            user = _find_user_by_customer_id(db, customer_id)

        # Fallback for checkout events: attach customer_id to user found by email (first-time purchase)
        if not user and event_type == "checkout.session.completed":
            email = _extract_email_from_checkout(obj)
            user = _find_user_fallback_by_email(db, email)
            if user and customer_id and not (user.stripe_customer_id or "").strip():
                user.stripe_customer_id = customer_id

        if not user:
            logger.warning(f"Webhook {event_type}: could not map to a user (customer_id={customer_id!r}).")
            db.commit()  # commit any possible customer_id attach (rare path)
            return {"status": "ok", "processed": event_type, "mapped_user": False}

        # Best-effort store local status/expiry from webhook payload (if present)
        sub_status = _extract_sub_status(event_type, obj)
        period_end = _extract_period_end(event_type, obj)
        _set_user_stripe_fields_best_effort(user, status=sub_status, period_end=period_end)

        db.commit()
        db.refresh(user)

        # Single source of truth: pull Stripe state and set tier + subscription_expires_at/status/updated_at
        # (your sync function should set these fields too)
        try:
            sync_user_subscription_from_stripe(user, db)
        except Exception as e:
            logger.warning(f"Webhook Stripe sync failed (non-fatal): {e}")

        return {"status": "ok", "processed": event_type, "mapped_user": True, "user_id": user.id}

    except Exception as e:
        logger.error(f"❌ Webhook error: {e}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Webhook processing failed")
    finally:
        db.close()
