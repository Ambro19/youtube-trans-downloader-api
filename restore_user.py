"""
Reset a user's password, mark for change on next login,
and optionally enforce a specific tier and ensure a subscription record.

Usage examples:
  python restore_user.py --user LovePets --password test123 --tier premium --ensure-sub
  python restore_user.py --user LovePets --password myNewTempPW
"""

# Standard Library
import argparse
from datetime import datetime, timezone
from typing import Optional

# Third-party
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

# Local Application
from models import SessionLocal, User, Subscription, initialize_database, engine
from main import get_password_hash  # use the same password hasher as the API


def _get_subscription_table_info():
    """Return list of (cid, name, type, notnull, dflt_value, pk) for subscriptions."""
    with engine.begin() as conn:
        return conn.exec_driver_sql("PRAGMA table_info(subscriptions)").fetchall()


def _get_subscription_columns() -> set[str]:
    """Return the set of actual columns present on the 'subscriptions' table."""
    return {r[1] for r in _get_subscription_table_info()}


def _get_latest_subscription(db, user_id: int) -> Optional[Subscription]:
    """Fetch latest subscription for user; order by created_at if available, fall back to id."""
    has_created = False
    try:
        cols = db.execute(text("PRAGMA table_info(subscriptions)")).fetchall()
        has_created = any(c[1] == "created_at" for c in cols)
    except Exception:
        pass

    q = db.query(Subscription).filter(Subscription.user_id == user_id)
    q = q.order_by(Subscription.created_at.desc()) if has_created else q.order_by(Subscription.id.desc())
    return q.first()


def _create_subscription_resilient(db, user_id: int, tier: str):
    """
    Create a subscription row with only available columns,
    automatically populating required NOT NULL fields.
    """
    cols_info = _get_subscription_table_info()
    cols = {c[1] for c in cols_info}
    now = datetime.now(timezone.utc)

    # Try ORM first with known columns
    try:
        sub = Subscription(
            user_id=user_id,
            tier=tier,
            status="active",
            created_at=now if "created_at" in cols else None,
            updated_at=now if "updated_at" in cols else None,
        )
        # Add start_date if ORM model supports it
        if hasattr(sub, "start_date") and "start_date" in cols:
            sub.start_date = now
        db.add(sub)
        db.commit()
        db.refresh(sub)
        print(f"✅ Created subscription record (ORM) for user_id={user_id} ({tier})")
        return sub
    except Exception as e:
        db.rollback()
        print(f"⚠️ ORM insert failed: {e} - falling back to raw SQL")

    # Fallback to raw SQL
    fields = []
    values = []
    for c in cols_info:
        name = c[1]
        notnull = c[3]
        default = c[4]

        if name == "id":  # skip PK
            continue
        if name == "user_id":
            fields.append(name)
            values.append(user_id)
            continue
        if name == "tier":
            fields.append(name)
            values.append(tier)
            continue
        if name == "status":
            fields.append(name)
            values.append("active")
            continue

        # For any date-like required fields without defaults
        if notnull == 1 and default is None:
            # guess a reasonable default
            if "date" in name or "time" in name:
                fields.append(name)
                values.append(now)
            else:
                # fallback to empty string for unknown text
                fields.append(name)
                values.append("")
        elif name in ("created_at", "updated_at", "start_date", "expiry_date"):
            fields.append(name)
            values.append(now)

    placeholders = ", ".join([f":{f}" for f in fields])
    fields_sql = ", ".join(fields)
    params = {f: v for f, v in zip(fields, values)}

    with engine.begin() as conn:
        conn.exec_driver_sql(
            f"INSERT INTO subscriptions ({fields_sql}) VALUES ({placeholders})",
            params,
        )
    print(f"✅ Created subscription record (raw SQL) for user_id={user_id} ({tier})")

    return _get_latest_subscription(db, user_id)


def _update_subscription_resilient(db, sub: Subscription, tier: str):
    """
    Update existing subscription to active/tier using ORM when possible,
    and fall back to a raw UPDATE with only present columns if needed.
    """
    try:
        changed = False
        if sub.tier != tier:
            sub.tier = tier
            changed = True
        if sub.status != "active":
            sub.status = "active"
            changed = True
        if changed:
            try:
                sub.updated_at = datetime.now(timezone.utc)
            except Exception:
                pass
            db.commit()
            print("🔄 Updated existing subscription to active/%s (ORM)" % tier)
        return
    except OperationalError:
        db.rollback()
        cols = _get_subscription_columns()
        sets = []
        params = {"id": sub.id}

        if "tier" in cols:
            sets.append("tier=:tier")
            params["tier"] = tier
        if "status" in cols:
            sets.append("status=:status")
            params["status"] = "active"
        if "updated_at" in cols:
            sets.append("updated_at=:updated_at")
            params["updated_at"] = datetime.now(timezone.utc)

        if sets:
            with engine.begin() as conn:
                conn.exec_driver_sql(
                    f"UPDATE subscriptions SET {', '.join(sets)} WHERE id=:id",
                    params,
                )
            print("🔄 Updated existing subscription to active/%s (raw SQL)" % tier)


def restore_user(username: str, password: str, tier: Optional[str], ensure_sub: bool):
    initialize_database()  # no-op if already migrated

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username.strip()).first()
        if not user:
            print(f"❌ User '{username}' not found.")
            return 1

        # 1) reset password + force change
        user.hashed_password = get_password_hash(password)
        try:
            setattr(user, "must_change_password", True)
        except Exception:
            pass

        # 2) optionally enforce tier (e.g., premium)
        if tier:
            old_tier = getattr(user, "subscription_tier", "free")
            user.subscription_tier = tier
            print(f"🔧 Tier: {old_tier} → {tier}")

        db.commit()
        db.refresh(user)

        # 3) optionally ensure a subscription row exists & is active
        if ensure_sub and tier and tier != "free":
            sub = _get_latest_subscription(db, user.id)
            if not sub:
                sub = _create_subscription_resilient(db, user.id, tier)
            else:
                _update_subscription_resilient(db, sub, tier)

        print(f"🎉 Restored user '{username}': password reset, must_change_password set.")
        if tier:
            print(f"   Tier enforced: {tier}")
        return 0
    except Exception as e:
        db.rollback()
        print(f"❌ Error: {e}")
        return 2
    finally:
        db.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True, help="username (e.g., LovePets)")
    ap.add_argument("--password", required=True, help="temporary password to set")
    ap.add_argument("--tier", choices=["free", "pro", "premium"], help="force tier")
    ap.add_argument("--ensure-sub", action="store_true", help="create/align a subscription row")
    args = ap.parse_args()

    raise SystemExit(
        restore_user(args.user, args.password, args.tier, args.ensure_sub)
    )
