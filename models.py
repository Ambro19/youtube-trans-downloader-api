# models.py - Updated User Model with Subscription Fields

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import bcrypt

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)  # <-- CRUCIAL for frontend login
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    subscription_tier = Column(String(20), default='free', nullable=False)
    subscription_status = Column(String(20), default='inactive', nullable=False)
    subscription_id = Column(String(255), nullable=True)
    subscription_current_period_end = Column(DateTime, nullable=True)
    stripe_customer_id = Column(String(255), nullable=True)
    usage_clean_transcripts = Column(Integer, default=0, nullable=False)
    usage_unclean_transcripts = Column(Integer, default=0, nullable=False)
    usage_audio_downloads = Column(Integer, default=0, nullable=False)
    usage_video_downloads = Column(Integer, default=0, nullable=False)
    usage_reset_date = Column(DateTime, default=datetime.utcnow, nullable=False)
    timezone = Column(String(50), default='UTC')
    language = Column(String(10), default='en')
    notification_preferences = Column(Text, nullable=True)


    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"
 
    def set_password(self, password: str):
        salt = bcrypt.gensalt()
        self.hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def verify_password(self, password: str) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), self.hashed_password.encode('utf-8'))


    def is_subscription_active(self) -> bool:
        """Check if user has an active subscription"""
        if self.subscription_tier == 'free':
            return True
        
        if not self.subscription_current_period_end:
            return False
            
        return (
            self.subscription_status in ['active', 'trialing'] and
            self.subscription_current_period_end > datetime.utcnow()
        )

    def get_plan_limits(self) -> dict:
        """Get the usage limits for the user's current plan"""
        limits = {
            'free': {
                'clean_transcripts': 5,
                'unclean_transcripts': 3,
                'audio_downloads': 2,
                'video_downloads': 1
            },
            'pro': {
                'clean_transcripts': 100,
                'unclean_transcripts': 50,
                'audio_downloads': 50,
                'video_downloads': 20
            },
            'premium': {
                'clean_transcripts': float('inf'),
                'unclean_transcripts': float('inf'),
                'audio_downloads': float('inf'),
                'video_downloads': float('inf')
            }
        }
        return limits.get(self.subscription_tier, limits['free'])

    def get_current_usage(self) -> dict:
        """Get the user's current usage for this month"""
        return {
            'clean_transcripts': self.usage_clean_transcripts,
            'unclean_transcripts': self.usage_unclean_transcripts,
            'audio_downloads': self.usage_audio_downloads,
            'video_downloads': self.usage_video_downloads
        }

    def can_perform_action(self, action_type: str) -> bool:
        """Check if user can perform the specified action based on limits"""
        # Reset usage if it's a new month
        if self.usage_reset_date.month != datetime.utcnow().month:
            self.reset_monthly_usage()

        limits = self.get_plan_limits()
        current_usage = getattr(self, f'usage_{action_type}', 0)
        limit = limits.get(action_type, 0)

        if limit == float('inf'):
            return True

        return current_usage < limit

    def reset_monthly_usage(self):
        """Reset monthly usage counters"""
        self.usage_clean_transcripts = 0
        self.usage_unclean_transcripts = 0
        self.usage_audio_downloads = 0
        self.usage_video_downloads = 0
        self.usage_reset_date = datetime.utcnow()

    def increment_usage(self, action_type: str):
        """Increment usage counter for the specified action"""
        current_usage = getattr(self, f'usage_{action_type}', 0)
        setattr(self, f'usage_{action_type}', current_usage + 1)

    def to_dict(self) -> dict:
        """Convert user to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'email': self.email,
            'full_name': self.full_name,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'subscription_tier': self.subscription_tier,
            'subscription_status': self.subscription_status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
        }

# Additional model for storing subscription history (optional)
class SubscriptionHistory(Base):
    __tablename__ = "subscription_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    action = Column(String(50), nullable=False)  # 'upgraded', 'downgraded', 'cancelled', 'renewed'
    from_tier = Column(String(20), nullable=True)
    to_tier = Column(String(20), nullable=True)
    amount = Column(Float, nullable=True)
    stripe_subscription_id = Column(String(255), nullable=True)
    stripe_payment_intent_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    history_metadata = Column(Text, nullable=True)  # JSON string for additional data

    def __repr__(self):
        return f"<SubscriptionHistory(user_id={self.user_id}, action='{self.action}', from='{self.from_tier}', to='{self.to_tier}')>"

# Create tables function
def create_tables(engine):
    """Create all tables in the database"""
    Base.metadata.create_all(bind=engine)

#===========================================
# # models.py - Updated User Model with Subscription Fields

# from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float
# from sqlalchemy.ext.declarative import declarative_base
# from sqlalchemy.sql import func
# from datetime import datetime
# import bcrypt

# Base = declarative_base()

# class User(Base):
#     __tablename__ = "users"

#     # Primary key and basic fields
#     id = Column(Integer, primary_key=True, index=True)
#     email = Column(String(255), unique=True, index=True, nullable=False)
#     full_name = Column(String(255), nullable=True)
#     hashed_password = Column(String(255), nullable=False)
#     is_active = Column(Boolean, default=True)
#     is_verified = Column(Boolean, default=False)
    
#     # Timestamps
#     created_at = Column(DateTime, default=datetime.utcnow)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
#     last_login = Column(DateTime, nullable=True)

#     # Subscription fields
#     subscription_tier = Column(String(20), default='free', nullable=False)
#     subscription_status = Column(String(20), default='inactive', nullable=False)
#     subscription_id = Column(String(255), nullable=True)  # Stripe subscription ID
#     subscription_current_period_end = Column(DateTime, nullable=True)
#     stripe_customer_id = Column(String(255), nullable=True)

#     # Usage tracking fields (reset monthly)
#     usage_clean_transcripts = Column(Integer, default=0, nullable=False)
#     usage_unclean_transcripts = Column(Integer, default=0, nullable=False)
#     usage_audio_downloads = Column(Integer, default=0, nullable=False)
#     usage_video_downloads = Column(Integer, default=0, nullable=False)
#     usage_reset_date = Column(DateTime, default=datetime.utcnow, nullable=False)

#     # Additional user preferences
#     timezone = Column(String(50), default='UTC')
#     language = Column(String(10), default='en')
#     notification_preferences = Column(Text, nullable=True)  # JSON string

#     def __repr__(self):
#         return f"<User(id={self.id}, email='{self.email}', tier='{self.subscription_tier}')>"

#     def set_password(self, password: str):
#         """Hash and set the user's password"""
#         salt = bcrypt.gensalt()
#         self.hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

#     def verify_password(self, password: str) -> bool:
#         """Verify the user's password"""
#         return bcrypt.checkpw(
#             password.encode('utf-8'), 
#             self.hashed_password.encode('utf-8')
#         )

#     def is_subscription_active(self) -> bool:
#         """Check if user has an active subscription"""
#         if self.subscription_tier == 'free':
#             return True
        
#         if not self.subscription_current_period_end:
#             return False
            
#         return (
#             self.subscription_status in ['active', 'trialing'] and
#             self.subscription_current_period_end > datetime.utcnow()
#         )

#     def get_plan_limits(self) -> dict:
#         """Get the usage limits for the user's current plan"""
#         limits = {
#             'free': {
#                 'clean_transcripts': 5,
#                 'unclean_transcripts': 3,
#                 'audio_downloads': 2,
#                 'video_downloads': 1
#             },
#             'pro': {
#                 'clean_transcripts': 100,
#                 'unclean_transcripts': 50,
#                 'audio_downloads': 50,
#                 'video_downloads': 20
#             },
#             'premium': {
#                 'clean_transcripts': float('inf'),
#                 'unclean_transcripts': float('inf'),
#                 'audio_downloads': float('inf'),
#                 'video_downloads': float('inf')
#             }
#         }
#         return limits.get(self.subscription_tier, limits['free'])

#     def get_current_usage(self) -> dict:
#         """Get the user's current usage for this month"""
#         return {
#             'clean_transcripts': self.usage_clean_transcripts,
#             'unclean_transcripts': self.usage_unclean_transcripts,
#             'audio_downloads': self.usage_audio_downloads,
#             'video_downloads': self.usage_video_downloads
#         }

#     def can_perform_action(self, action_type: str) -> bool:
#         """Check if user can perform the specified action based on limits"""
#         # Reset usage if it's a new month
#         if self.usage_reset_date.month != datetime.utcnow().month:
#             self.reset_monthly_usage()

#         limits = self.get_plan_limits()
#         current_usage = getattr(self, f'usage_{action_type}', 0)
#         limit = limits.get(action_type, 0)

#         if limit == float('inf'):
#             return True

#         return current_usage < limit

#     def reset_monthly_usage(self):
#         """Reset monthly usage counters"""
#         self.usage_clean_transcripts = 0
#         self.usage_unclean_transcripts = 0
#         self.usage_audio_downloads = 0
#         self.usage_video_downloads = 0
#         self.usage_reset_date = datetime.utcnow()

#     def increment_usage(self, action_type: str):
#         """Increment usage counter for the specified action"""
#         current_usage = getattr(self, f'usage_{action_type}', 0)
#         setattr(self, f'usage_{action_type}', current_usage + 1)

#     def to_dict(self) -> dict:
#         """Convert user to dictionary for JSON serialization"""
#         return {
#             'id': self.id,
#             'email': self.email,
#             'full_name': self.full_name,
#             'is_active': self.is_active,
#             'is_verified': self.is_verified,
#             'subscription_tier': self.subscription_tier,
#             'subscription_status': self.subscription_status,
#             'created_at': self.created_at.isoformat() if self.created_at else None,
#             'last_login': self.last_login.isoformat() if self.last_login else None,
#         }

# # Additional model for storing subscription history (optional)
# class SubscriptionHistory(Base):
#     __tablename__ = "subscription_history"

#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, nullable=False)
#     action = Column(String(50), nullable=False)  # 'upgraded', 'downgraded', 'cancelled', 'renewed'
#     from_tier = Column(String(20), nullable=True)
#     to_tier = Column(String(20), nullable=True)
#     amount = Column(Float, nullable=True)
#     stripe_subscription_id = Column(String(255), nullable=True)
#     stripe_payment_intent_id = Column(String(255), nullable=True)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     history_metadata = Column(Text, nullable=True)  # JSON string for additional data

#     def __repr__(self):
#         return f"<SubscriptionHistory(user_id={self.user_id}, action='{self.action}', from='{self.from_tier}', to='{self.to_tier}')>"

# # Create tables function
# def create_tables(engine):
#     """Create all tables in the database"""
#     Base.metadata.create_all(bind=engine)