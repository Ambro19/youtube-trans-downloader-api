# # backend/subscription_sync.py
# """
# Helper functions to sync user subscription status from Stripe.
# Used by /subscription_status endpoint with ?sync=1 parameter.
# """

# import os
# import logging
# from datetime import datetime
# from typing import Optional
# from sqlalchemy.orm import Session
# from models import User, Subscription

# logger = logging.getLogger("subscription_sync")

# try:
#     import stripe
#     stripe_key = os.getenv("STRIPE_SECRET_KEY")
#     if stripe_key:
#         stripe.api_key = stripe_key  # ← CRITICAL: Set the API key
#         STRIPE_ENABLED = True
#     else:
#         STRIPE_ENABLED = False
# except ImportError:
#     stripe = None
#     STRIPE_ENABLED = False


# def sync_user_subscription_from_stripe(user: User, db: Session) -> bool:
#     """
#     Check Stripe for active subscriptions and update user.subscription_tier.
#     Returns True if changes were made, False otherwise.
    
#     This is CRITICAL for fixing the UI discrepancy - it ensures the database
#     tier matches what Stripe knows about.
#     """
#     if not STRIPE_ENABLED or not stripe:
#         logger.debug("Stripe not enabled, skipping sync")
#         return False
    
#     if not user.stripe_customer_id:
#         logger.debug(f"User {user.username} has no Stripe customer ID")
#         return False
    
#     try:
#         # Fetch active subscriptions from Stripe
#         subscriptions = stripe.Subscription.list(
#             customer=user.stripe_customer_id,
#             status="active",
#             limit=10
#         )
        
#         if not subscriptions.data:
#             # No active subscriptions - should be on free tier
#             if user.subscription_tier != "free":
#                 logger.info(f"🔄 Downgrading {user.username} to free (no active Stripe subscription)")
#                 user.subscription_tier = "free"
#                 user.subscription_status = "inactive"
#                 db.commit()
#                 return True
#             return False
        
#         # User has active subscription(s) - get the highest tier
#         highest_tier = "pro"  # default
#         active_sub = subscriptions.data[0]  # use the first active one
        
#         try:
#             # Determine tier from price
#             price_id = active_sub.get("items", {}).get("data", [{}])[0].get("price", {}).get("id")
            
#             # Check environment variables for price IDs
#             PRO_PRICE_ID = os.getenv("STRIPE_PRO_PRICE_ID")
#             PREMIUM_PRICE_ID = os.getenv("STRIPE_PREMIUM_PRICE_ID")
            
#             if price_id == PREMIUM_PRICE_ID:
#                 highest_tier = "premium"
#             elif price_id == PRO_PRICE_ID:
#                 highest_tier = "pro"
#             else:
#                 # Fallback: check lookup key
#                 try:
#                     price = stripe.Price.retrieve(price_id)
#                     lookup_key = (price.get("lookup_key") or "").lower()
#                     if "premium" in lookup_key:
#                         highest_tier = "premium"
#                     elif "pro" in lookup_key:
#                         highest_tier = "pro"
#                 except Exception as e:
#                     logger.warning(f"Could not retrieve price details: {e}")
        
#         except Exception as e:
#             logger.warning(f"Error determining tier from subscription: {e}")
        
#         # Update user if tier changed
#         if user.subscription_tier != highest_tier:
#             logger.info(f"✅ Updating {user.username}: {user.subscription_tier} → {highest_tier}")
#             user.subscription_tier = highest_tier
#             user.subscription_status = "active"
            
#             # Also update/create subscription record
#             sub_record = (
#                 db.query(Subscription)
#                 .filter(Subscription.stripe_subscription_id == active_sub.id)
#                 .first()
#             )
            
#             if not sub_record:
#                 sub_record = Subscription(
#                     user_id=user.id,
#                     tier=highest_tier,
#                     status="active",
#                     stripe_subscription_id=active_sub.id,
#                     stripe_customer_id=user.stripe_customer_id,
#                     created_at=datetime.utcnow(),
#                     updated_at=datetime.utcnow()
#                 )
#                 db.add(sub_record)
#             else:
#                 sub_record.tier = highest_tier
#                 sub_record.status = "active"
#                 sub_record.updated_at = datetime.utcnow()
            
#             db.commit()
#             db.refresh(user)
#             return True
        
#         # Tier is already correct
#         logger.debug(f"User {user.username} tier is already correct: {highest_tier}")
#         return False
    
#     except Exception as e:
#         logger.error(f"Error syncing subscription for {user.username}: {e}", exc_info=True)
#         return False


# def get_stripe_customer_subscriptions(customer_id: str) -> Optional[list]:
#     """
#     Get all subscriptions for a Stripe customer.
#     Returns list of subscription dicts or None if error.
#     """
#     if not STRIPE_ENABLED or not stripe or not customer_id:
#         return None
    
#     try:
#         subscriptions = stripe.Subscription.list(
#             customer=customer_id,
#             limit=100
#         )
#         return subscriptions.data
#     except Exception as e:
#         logger.error(f"Error fetching subscriptions for customer {customer_id}: {e}")
#         return None


# def sync_all_users_with_stripe(db: Session) -> dict:
#     """
#     ONE-TIME SYNC: Check all users with Stripe customer IDs and sync their tiers.
#     This is useful for fixing existing discrepancies.
    
#     Returns: {"synced": int, "errors": int, "unchanged": int}
#     """
#     if not STRIPE_ENABLED or not stripe:
#         return {"error": "Stripe not enabled"}
    
#     users = db.query(User).filter(User.stripe_customer_id.isnot(None)).all()
    
#     stats = {"synced": 0, "errors": 0, "unchanged": 0}
    
#     for user in users:
#         try:
#             changed = sync_user_subscription_from_stripe(user, db)
#             if changed:
#                 stats["synced"] += 1
#             else:
#                 stats["unchanged"] += 1
#         except Exception as e:
#             logger.error(f"Error syncing user {user.username}: {e}")
#             stats["errors"] += 1
    
#     logger.info(f"✅ Sync complete: {stats}")
#     return stats


############################################
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
