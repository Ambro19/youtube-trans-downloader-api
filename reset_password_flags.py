# backend/reset_password_flags.py
import argparse
from models import SessionLocal, User, initialize_database

def mark_users_for_password_change(usernames=None, clear=False, only_missing=False):
    """
    Set (or clear) the must_change_password flag.
      - usernames: list[str] or None for all users
      - clear: set flag to False instead of True
      - only_missing: when setting True, only touch users where it's False/NULL
    """
    initialize_database()
    db = SessionLocal()
    try:
        q = db.query(User)
        if usernames:
            q = q.filter(User.username.in_(usernames))

        users = q.all()
        if not users:
            print("ℹ️ No matching users found.")
            return

        changed = 0
        for u in users:
            current = bool(getattr(u, "must_change_password", False))
            target = False if clear else True

            if only_missing and target is True and current is True:
                # Skip already-marked users if only_missing
                continue

            setattr(u, "must_change_password", target)
            changed += 1
            state = "cleared" if clear else "marked"
            print(f"🔒 {state} {u.username} ({u.email})")

        db.commit()
        verb = "cleared" if clear else "marked"
        print(f"✅ {verb} {changed} user(s).")
    except Exception as e:
        db.rollback()
        print(f"❌ Error: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Set/clear must_change_password for users.")
    ap.add_argument("--user", action="append", dest="users",
                    help="Username to update (can be used multiple times). Omit to affect ALL users.")
    ap.add_argument("--clear", action="store_true",
                    help="Clear the flag instead of setting it.")
    ap.add_argument("--only-missing", action="store_true",
                    help="When setting the flag, only mark users not already marked.")
    args = ap.parse_args()

    mark_users_for_password_change(
        usernames=args.users,
        clear=args.clear,
        only_missing=args.only_missing
    )
