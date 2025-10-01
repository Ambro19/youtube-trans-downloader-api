# reset_all_passwords.py
"""
Reset ALL users' passwords to a known value (default: 'test123').
Self-contained: does not import hashing from app modules.
"""

from sqlalchemy.orm import Session
from models import User, SessionLocal
from passlib.context import CryptContext

# ---------- config ----------
DEFAULT_PASSWORD = "test123"   # change if you like
EXCLUDE_USERNAMES = set()      # e.g. {"admin"} to skip certain accounts
# ----------------------------

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def reset_all_user_passwords():
    db: Session = SessionLocal()
    try:
        users = db.query(User).all()
        if not users:
            print("⚠️ No users found in the database.")
            return

        count = 0
        for user in users:
            if user.username in EXCLUDE_USERNAMES:
                print(f"⏭️  Skipping {user.username}")
                continue
            user.hashed_password = get_password_hash(DEFAULT_PASSWORD)
            print(f"✅ Reset password for {user.username} → '{DEFAULT_PASSWORD}'")
            count += 1

        db.commit()
        print(f"🎉 Done! Reset {count} user(s).")
    except Exception as e:
        db.rollback()
        print(f"❌ Error resetting passwords: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    reset_all_user_passwords()
