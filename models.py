from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import bcrypt

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    # Primary fields
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255), nullable=True)
    hashed_password = Column(String(255), nullable=False)

    # Status and timestamps
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    # Subscription tracking
    subscription_tier = Column(String(20), default='free', nullable=False)
    subscription_status = Column(String(20), default='inactive', nullable=False)
    subscription_id = Column(String(255), nullable=True)
    subscription_current_period_end = Column(DateTime, nullable=True)
    stripe_customer_id = Column(String(255), nullable=True)

    # Usage counters
    usage_clean_transcripts = Column(Integer, default=0, nullable=False)
    usage_unclean_transcripts = Column(Integer, default=0, nullable=False)
    usage_audio_downloads = Column(Integer, default=0, nullable=False)
    usage_video_downloads = Column(Integer, default=0, nullable=False)
    usage_reset_date = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Preferences
    timezone = Column(String(50), default='UTC')
    language = Column(String(10), default='en')
    notification_preferences = Column(Text, nullable=True)  # JSON string

    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}', tier='{self.subscription_tier}')>"

    def set_password(self, password: str):
        salt = bcrypt.gensalt()
        self.hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def verify_password(self, password: str) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), self.hashed_password.encode('utf-8'))

    def is_subscription_active(self) -> bool:
        if self.subscription_tier == 'free':
            return True
        return (
            self.subscription_status in ['active', 'trialing'] and
            self.subscription_current_period_end and
            self.subscription_current_period_end > datetime.utcnow()
        )

    def get_plan_limits(self) -> dict:
        return {
            'free':     {'clean_transcripts': 5, 'unclean_transcripts': 3, 'audio_downloads': 2, 'video_downloads': 1},
            'pro':      {'clean_transcripts': 100, 'unclean_transcripts': 50, 'audio_downloads': 50, 'video_downloads': 20},
            'premium':  {'clean_transcripts': float('inf'), 'unclean_transcripts': float('inf'),
                         'audio_downloads': float('inf'), 'video_downloads': float('inf')}
        }.get(self.subscription_tier, {})

    def get_current_usage(self) -> dict:
        return {
            'clean_transcripts': self.usage_clean_transcripts,
            'unclean_transcripts': self.usage_unclean_transcripts,
            'audio_downloads': self.usage_audio_downloads,
            'video_downloads': self.usage_video_downloads
        }

    def can_perform_action(self, action_type: str) -> bool:
        if self.usage_reset_date.month != datetime.utcnow().month:
            self.reset_monthly_usage()
        limits = self.get_plan_limits()
        current_usage = getattr(self, f'usage_{action_type}', 0)
        limit = limits.get(action_type, 0)
        return limit == float('inf') or current_usage < limit

    def reset_monthly_usage(self):
        """Reset monthly usage counters and update reset date"""
        self.usage_clean_transcripts = 0
        self.usage_unclean_transcripts = 0
        self.usage_audio_downloads = 0
        self.usage_video_downloads = 0
        self.usage_reset_date = datetime.utcnow()

    def increment_usage(self, action_type: str):
        """Increment usage counter for a specific action type"""
        if hasattr(self, f'usage_{action_type}'):
            current = getattr(self, f'usage_{action_type}', 0)
            setattr(self, f'usage_{action_type}', current + 1)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'is_active': self.is_active,
            'is_verified': self.is_verified,
            'subscription_tier': self.subscription_tier,
            'subscription_status': self.subscription_status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
        }

# Optional: Store historical subscription events
class SubscriptionHistory(Base):
    __tablename__ = "subscription_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    action = Column(String(50), nullable=False)
    from_tier = Column(String(20), nullable=True)
    to_tier = Column(String(20), nullable=True)
    amount = Column(Float, nullable=True)
    stripe_subscription_id = Column(String(255), nullable=True)
    stripe_payment_intent_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    history_metadata = Column(Text, nullable=True)

    def __repr__(self):
        return f"<SubscriptionHistory(user_id={self.user_id}, action='{self.action}', from='{self.from_tier}', to='{self.to_tier}')>"

# Database table creation utility
def create_tables(engine):
    Base.metadata.create_all(bind=engine)

