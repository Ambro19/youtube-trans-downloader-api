# backend/auth_deps.py
import os, jwt
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

# ✅ Load env here so SECRET_KEY/ALGORITHM match tokens created in main.py
try:
    from dotenv import load_dotenv, find_dotenv
    # local overrides first, then base .env (local wins)
    load_dotenv(find_dotenv(".env.local"), override=True)
    load_dotenv(find_dotenv(".env"), override=False)
except Exception:
    pass

from models import get_db, User

SECRET_KEY = os.getenv("SECRET_KEY", "devsecret")
ALGORITHM  = os.getenv("ALGORITHM", "HS256")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def _get_user(db: Session, username: str) -> Optional[User]:
    return db.query(User).filter(User.username == username).first()

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    cred_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if not username:
            raise cred_exc
    except Exception:
        raise cred_exc
    user = _get_user(db, username)
    if not user:
        raise cred_exc
    return user
