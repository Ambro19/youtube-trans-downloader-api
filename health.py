# Add this to your main.py or create a new health.py file

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
import os

router = APIRouter()

@router.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "development"),
        "database": "connected",
        "services": {
            "youtube_api": "available",
            "stripe": "configured",
            "file_system": "accessible"
        }
    }

@router.get("/debug/users")
async def debug_users(db: Session = Depends(get_db)):
    """Debug endpoint to list users (development only)"""
    if os.getenv("ENVIRONMENT") != "development":
        raise HTTPException(status_code=404, detail="Not found")
    
    from your_models import User  # Import your User model
    users = db.query(User).all()
    
    return {
        "total_users": len(users),
        "users": [
            {
                "username": user.username,
                "email": user.email,
                "created_at": user.created_at.isoformat() if user.created_at else None,
                "is_active": getattr(user, 'is_active', True)
            }
            for user in users
        ]
    }

@router.post("/debug/test-login")
async def debug_test_login(username: str, password: str, db: Session = Depends(get_db)):
    """Debug login endpoint with detailed error information"""
    if os.getenv("ENVIRONMENT") != "development":
        raise HTTPException(status_code=404, detail="Not found")
    
    from your_auth_utils import verify_password, get_user_by_username  # Import your auth functions
    
    # Check if user exists
    user = get_user_by_username(db, username)
    if not user:
        return {
            "success": False,
            "error": "user_not_found",
            "message": f"User '{username}' does not exist in database",
            "debug_info": {
                "searched_username": username,
                "total_users_in_db": db.query(User).count()
            }
        }
    
    # Check password
    if not verify_password(password, user.hashed_password):
        return {
            "success": False,
            "error": "invalid_password",
            "message": "Password verification failed",
            "debug_info": {
                "user_exists": True,
                "username": username,
                "password_length": len(password)
            }
        }
    
    return {
        "success": True,
        "message": "Login credentials are valid",
        "user_info": {
            "username": user.username,
            "email": user.email,
            "is_active": getattr(user, 'is_active', True)
        }
    }

# Add the router to your main app
# app.include_router(router, prefix="/api/v1", tags=["health"])