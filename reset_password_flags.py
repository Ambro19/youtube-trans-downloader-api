# reset_password_flags.py
from models import SessionLocal, User

def mark_users_for_password_change():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        for user in users:
            user.must_change_password = True
            print(f"🔒 Marked {user.username} to change password on next login")
        db.commit()
    finally:
        db.close()

if __name__ == "__main__":
    mark_users_for_password_change()
