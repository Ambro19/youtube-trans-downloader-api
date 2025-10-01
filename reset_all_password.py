# reset_all_passwords.py

from sqlalchemy.orm import Session
from models import User, SessionLocal
from auth import get_password_hash

# Set the default password you'd like to apply to all users
DEFAULT_PASSWORD = "test123"

def reset_all_user_passwords():
    db: Session = SessionLocal()
    try:
        users = db.query(User).all()
        if not users:
            print("⚠️ No users found in the database.")
            return

        for user in users:
            user.hashed_password = get_password_hash(DEFAULT_PASSWORD)
            print(f"✅ Reset password for {user.username} → '{DEFAULT_PASSWORD}'")

        db.commit()
        print("🎉 All user passwords have been reset successfully.")
    except Exception as e:
        db.rollback()
        print(f"❌ Error resetting passwords: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    reset_all_user_passwords()


