# subscription_sync.py
import os
import logging
from typing import Optional, Any
from datetime import datetime, timezone

from sqlalchemy.orm import Session  # pyright: ignore[reportMissingImports]
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

PRO_LOOKUP = os.getenv("STRIPE_PRO_LOOKUP_KEY", "pro_monthly").lower()
PREM_LOOKUP = os.getenv("STRIPE_PREMIUM_LOOKUP_KEY", "premium_monthly").lower()

# Treat "past_due" as entitled ONLY if the period hasn't ended yet (or within a grace period).
ENTITLED_STATUSES = {"active", "trialing", "past_due"}

# Optional grace window for "past_due" after period end (in hours). Keep small or 0.
PAST_DUE_GRACE_HOURS = int(os.getenv("STRIPE_PAST_DUE_GRACE_HOURS", "0") or "0")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _to_dt(value: Any) -> Optional[datetime]:
    """
    Parse common timestamp shapes into a timezone-aware UTC datetime.
    Supports:
      - int/float unix seconds (Stripe uses unix seconds)
      - ISO string
      - datetime
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        # Ensure tz-aware UTC
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
            # Python 3.11+ handles many ISO forms; fallback if missing tz
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None

    return None


def _tier_from_price(price: dict) -> Optional[str]:
    """
    Decide Pro/Premium from a Stripe Price object.
    Primary: Price.lookup_key
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


def _subscription_entitles_access(sub: dict) -> bool:
    """
    Decide whether this Stripe Subscription should grant access right now.

    Rules:
    - active/trialing => yes (unless period ended, which shouldn't happen, but we guard)
    - past_due => yes ONLY if current_period_end is still in the future (or within grace)
    - cancel_at_period_end => still entitled until period end, then no
    """
    status = (sub.get("status") or "").lower()
    if status not in ENTITLED_STATUSES:
        return False

    now = _utcnow()
    period_end = _to_dt(sub.get("current_period_end"))

    # If we can't read period_end, be conservative:
    # - active/trialing => allow
    # - past_due => do NOT allow (avoids infinite entitlement)
    if not period_end:
        return status in {"active", "trialing"}

    # cancel_at_period_end: allow only until period_end
    if sub.get("cancel_at_period_end"):
        return now < period_end

    if status in {"active", "trialing"}:
        return now < period_end

    # past_due:
    # allow until period_end (+ optional grace window), otherwise deny
    grace_seconds = max(0, PAST_DUE_GRACE_HOURS) * 3600
    if grace_seconds > 0:
        return now < (period_end.replace() if grace_seconds == 0 else period_end) and (
            (period_end.timestamp() + grace_seconds) > now.timestamp()
        )
    return now < period_end


def _pick_best_subscription(subs: list[dict]) -> Optional[dict]:
    """
    Pick the most relevant subscription:
    - Prefer ones that currently entitle access
    - Use the latest current_period_end (or created) as tiebreaker
    """
    def sort_key(s: dict) -> int:
        return int(s.get("current_period_end") or s.get("created") or 0)

    subs_sorted = sorted(subs, key=sort_key, reverse=True)

    # First pass: any that entitle access
    for s in subs_sorted:
        if _subscription_entitles_access(s):
            return s

    # Second pass: even if none entitle, return the most recent one (so we can downgrade correctly)
    return subs_sorted[0] if subs_sorted else None


def _extract_tier_from_subscription(sub: dict) -> Optional[str]:
    items = (sub.get("items") or {}).get("data") or []
    for it in items:
        price = it.get("price")
        tier = _tier_from_price(price)
        if tier in {"pro", "premium"}:
            return tier
    return None


def _apply_local_overdue_downgrade_if_possible(user: User, db: Session) -> bool:
    """
    If Stripe is unreachable, we can still downgrade if the User model contains
    a local period end / reset timestamp. This is best-effort and safe.
    """
    candidates = [
        "subscription_current_period_end",
        "current_period_end",
        "stripe_current_period_end",
        "subscription_expires_at",
        "next_reset",
        "subscription_next_reset",
    ]

    now = _utcnow()
    best_dt: Optional[datetime] = None
    best_attr: Optional[str] = None

    for attr in candidates:
        if hasattr(user, attr):
            dt = _to_dt(getattr(user, attr))
            if dt:
                # pick the most "authoritative" (latest) timestamp available
                if best_dt is None or dt > best_dt:
                    best_dt = dt
                    best_attr = attr

    if best_dt and now >= best_dt:
        if (user.subscription_tier or "free") != "free":
            log.warning(
                f"Stripe sync unavailable; local timestamp {best_attr} indicates overdue "
                f"({best_dt.isoformat()}). Downgrading user {user.id} to free."
            )
            user.subscription_tier = "free"
            db.commit()
            db.refresh(user)
            return True

    return False


def sync_user_subscription_from_stripe(user: User, db: Session) -> bool:
    """
    Pull the user's latest subscription from Stripe and update user.subscription_tier.

    Key behavior:
    - past_due is NOT unlimited access; it expires at current_period_end (plus optional grace).
    - cancel_at_period_end is honored until period_end; after that user becomes free.
    - If Stripe is unreachable, we still downgrade if we can infer overdue from local user fields.

    Returns True if the user's tier was changed, else False.
    """
    # If Stripe isn't configured, do best-effort downgrade using local timestamps
    if not stripe:
        log.info("Stripe not configured; skipping Stripe sync.")
        return _apply_local_overdue_downgngrade_if_possible(user, db)  # typo guard? see below

    customer_id = (user.stripe_customer_id or "").strip()
    if not customer_id:
        log.info("No stripe_customer_id on user; skipping sync.")
        return False

    try:
        # Expand only what we need: price for tier, and current period info is on subscription itself.
        subs = stripe.Subscription.list(
            customer=customer_id,
            limit=10,
            expand=["data.items.data.price"],
        )
    except Exception as e:
        log.error(f"Stripe subscriptions list failed for {customer_id}: {e}")
        # Stripe down: best-effort local overdue downgrade (if your User has timestamps)
        return _apply_local_overdue_downgrade_if_possible(user, db)

    if not getattr(subs, "data", None):
        log.info(f"No subscriptions for customer {customer_id}")
        # No subscriptions => free
        if user.subscription_tier and user.subscription_tier != "free":
            user.subscription_tier = "free"
            db.commit()
            db.refresh(user)
            return True
        return False

    chosen = _pick_best_subscription(list(subs.data))
    if not chosen:
        return False

    entitled = _subscription_entitles_access(chosen)
    tier_from_stripe = _extract_tier_from_subscription(chosen)

    # If not entitled, force free (this is the critical “overdue reset => downgrade” behavior)
    new_tier = tier_from_stripe if (entitled and tier_from_stripe) else "free"

    # Optional: store stripe period end/status locally if your User model has fields for it
    try:
        if hasattr(user, "stripe_subscription_status"):
            setattr(user, "stripe_subscription_status", (chosen.get("status") or "").lower())
        if hasattr(user, "stripe_current_period_end"):
            setattr(user, "stripe_current_period_end", chosen.get("current_period_end"))
    except Exception:
        pass

    old_tier = (user.subscription_tier or "free")
    if old_tier != new_tier:
        log.info(f"Updating user {user.id} tier {old_tier!r} -> {new_tier!r}")
        user.subscription_tier = new_tier
        db.commit()
        db.refresh(user)
        return True

    return False


# --- tiny safety fix: avoid NameError if you ever hit the non-stripe branch above
def _apply_local_overdue_downgngrade_if_possible(user: User, db: Session) -> bool:
    # backward-compatible alias if an older import/call path exists
    return _apply_local_overdue_downgrade_if_possible(user, db)
