# reset_all_passwords.py
from sqlalchemy.orm import Session
from models import User, SessionLocal
from auth import get_password_hash

DEFAULT_PASSWORD = "test123"

db: Session = SessionLocal()

users = db.query(User).all()
for user in users:
    user.hashed_password = get_password_hash(DEFAULT_PASSWORD)
    print(f"✅ Reset password for {user.username} → {DEFAULT_PASSWORD}")
db.commit()
db.close()

