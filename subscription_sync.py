# payment.py (add near other Stripe helpers)
#subscription_sync.py
from typing import Optional, Tuple
import stripe
from sqlalchemy.orm import Session
from models import User, Subscription  # adjust if your names/paths differ
from datetime import datetime, timezone

ACTIVE_STATUSES = {"active", "trialing"}  # treat these as upgraded

def _map_lookup_key_to_tier(lookup_key: Optional[str]) -> Optional[str]:
    if not lookup_key:
        return None
    lk = lookup_key.lower()
    if "pro_monthly" in lk or lk == "pro":
        return "pro"
    if "premium_monthly" in lk or lk == "premium":
        return "premium"
    return None

def sync_user_subscription_from_stripe(db: Session, user: User) -> Tuple[str, Optional[str]]:
    """
    Idempotently sync local subscription row from Stripe.
    Returns: (tier, stripe_customer_id)
    """
    # 1) Find Stripe customer id
    cust_id = getattr(user, "stripe_customer_id", None)

    if not cust_id:
        # Try existing local Subscription row
        sub = db.query(Subscription).filter(Subscription.user_id == user.id).first()
        cust_id = getattr(sub, "stripe_customer_id", None) if sub else None

    if not cust_id:
        # Last resort: search Stripe by email
        candidates = stripe.Customer.list(email=user.email, limit=1)
        if candidates.data:
            cust_id = candidates.data[0].id

    if not cust_id:
        # no stripe customer known => user remains on free
        return ("free", None)

    # 2) Look up latest subscription for the customer
    subs = stripe.Subscription.list(customer=cust_id, status="all", limit=3, expand=["data.items.data.price"])
    best = None
    for s in subs.auto_paging_iter():
        if not best:
            best = s
        # prefer an active/trialing sub
        if s.status in ACTIVE_STATUSES:
            best = s
            break

    if not best:
        # no subscription object
        return ("free", cust_id)

    lookup_key = None
    price = None
    try:
        item = best["items"]["data"][0]
        price = item["price"]
        lookup_key = price.get("lookup_key")
    except Exception:
        pass

    tier = _map_lookup_key_to_tier(lookup_key) or "free"

    # 3) Upsert local Subscription row
    sub_row = db.query(Subscription).filter(Subscription.user_id == user.id).first()
    if not sub_row:
        sub_row = Subscription(user_id=user.id)

    sub_row.tier = tier
    sub_row.status = best.status
    sub_row.stripe_subscription_id = best.id
    sub_row.stripe_customer_id = cust_id
    sub_row.updated_at = datetime.now(timezone.utc)

    # set period end if present
    try:
        period_end = best["current_period_end"]
        if period_end:
            sub_row.expires_at = datetime.fromtimestamp(period_end, tz=timezone.utc)
    except Exception:
        pass

    db.add(sub_row)
    try:
        db.commit()
    except Exception:
        db.rollback()
        # keep going—worst case we still return the computed tier

    # 4) also persist customer id on user for future calls
    if getattr(user, "stripe_customer_id", None) != cust_id:
        try:
            setattr(user, "stripe_customer_id", cust_id)
            db.add(user)
            db.commit()
        except Exception:
            db.rollback()

    return (tier, cust_id)
