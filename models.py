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

    # Enhanced usage counters with audio/video support
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
        """Set user password with bcrypt hashing"""
        salt = bcrypt.gensalt()
        self.hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

    def verify_password(self, password: str) -> bool:
        """Verify password against stored hash"""
        return bcrypt.checkpw(password.encode('utf-8'), self.hashed_password.encode('utf-8'))

    def is_subscription_active(self) -> bool:
        """Check if user has an active subscription"""
        if self.subscription_tier == 'free':
            return True
        return (
            self.subscription_status in ['active', 'trialing'] and
            self.subscription_current_period_end and
            self.subscription_current_period_end > datetime.utcnow()
        )

    def get_plan_limits(self) -> dict:
        """Get usage limits based on subscription tier"""
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
        """Get current usage counts for all features"""
        return {
            'clean_transcripts': self.usage_clean_transcripts,
            'unclean_transcripts': self.usage_unclean_transcripts,
            'audio_downloads': self.usage_audio_downloads,
            'video_downloads': self.usage_video_downloads
        }

    def can_perform_action(self, action_type: str) -> bool:
        """Check if user can perform a specific action based on limits"""
        # Check if monthly reset is needed
        if self.usage_reset_date.month != datetime.utcnow().month:
            self.reset_monthly_usage()
        
        limits = self.get_plan_limits()
        current_usage = getattr(self, f'usage_{action_type}', 0)
        limit = limits.get(action_type, 0)
        
        return limit == float('inf') or current_usage < limit

    def get_remaining_usage(self, action_type: str) -> int:
        """Get remaining usage for a specific action type"""
        limits = self.get_plan_limits()
        current_usage = getattr(self, f'usage_{action_type}', 0)
        limit = limits.get(action_type, 0)
        
        if limit == float('inf'):
            return float('inf')
        
        return max(0, limit - current_usage)

    def get_usage_percentage(self, action_type: str) -> float:
        """Get usage percentage for a specific action type"""
        limits = self.get_plan_limits()
        current_usage = getattr(self, f'usage_{action_type}', 0)
        limit = limits.get(action_type, 0)
        
        if limit == float('inf') or limit == 0:
            return 0.0
        
        return min(100.0, (current_usage / limit) * 100)

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

    def get_plan_features(self) -> dict:
        """Get detailed plan features and descriptions"""
        features = {
            'free': {
                'name': 'Free',
                'price': 0,
                'features': [
                    '5 transcript downloads',
                    '2 audio downloads', 
                    '1 video download',
                    'Basic formats (TXT, SRT)',
                    'Community support',
                    'No priority processing'
                ]
            },
            'pro': {
                'name': 'Pro',
                'price': 9.99,
                'features': [
                    '100 transcript downloads',
                    '50 audio downloads',
                    '20 video downloads', 
                    'All formats (TXT, SRT, VTT)',
                    'Priority processing',
                    'Email support',
                    'Advanced transcript cleaning'
                ]
            },
            'premium': {
                'name': 'Premium',
                'price': 19.99,
                'features': [
                    'Unlimited downloads',
                    'Unlimited audio & video',
                    'All formats + API access',
                    'Fastest processing',
                    'Priority support',
                    'Batch processing',
                    'Custom integrations'
                ]
            }
        }
        return features.get(self.subscription_tier, features['free'])

    def needs_upgrade_for(self, action_type: str) -> bool:
        """Check if user needs to upgrade to perform an action"""
        return not self.can_perform_action(action_type)

    def to_dict(self) -> dict:
        """Convert user object to dictionary for API responses"""
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
            'usage': self.get_current_usage(),
            'limits': self.get_plan_limits()
        }

# Enhanced subscription history tracking
class SubscriptionHistory(Base):
    __tablename__ = "subscription_history"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    action = Column(String(50), nullable=False)  # 'upgrade', 'downgrade', 'cancel', 'renew'
    from_tier = Column(String(20), nullable=True)
    to_tier = Column(String(20), nullable=True)
    amount = Column(Float, nullable=True)
    stripe_subscription_id = Column(String(255), nullable=True)
    stripe_payment_intent_id = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    history_metadata = Column(Text, nullable=True)  # JSON string for additional data

    def __repr__(self):
        return f"<SubscriptionHistory(user_id={self.user_id}, action='{self.action}', from='{self.from_tier}', to='{self.to_tier}')>"

# Enhanced download tracking with audio/video support
class Download(Base):
    __tablename__ = "downloads"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    youtube_id = Column(String(20), nullable=False, index=True)
    download_type = Column(String(20), nullable=False)  # 'transcript', 'audio', 'video'
    download_format = Column(String(20), nullable=True)  # 'clean', 'srt', 'vtt', 'mp3', 'mp4'
    quality = Column(String(20), nullable=True)  # For audio/video: 'high', 'medium', 'low', '1080p', '720p', etc.
    file_size = Column(Integer, nullable=True)  # Size in bytes
    processing_time = Column(Integer, nullable=True)  # Time in milliseconds
    download_method = Column(String(50), nullable=True)  # 'youtube-transcript-api', 'yt-dlp'
    status = Column(String(20), default='completed', nullable=False)  # 'completed', 'failed', 'processing'
    error_message = Column(Text, nullable=True)
    language = Column(String(10), default='en')
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # File storage information (for production use)
    file_path = Column(String(500), nullable=True)  # Local or cloud storage path
    download_url = Column(String(500), nullable=True)  # Public download URL
    expires_at = Column(DateTime, nullable=True)  # When download link expires

    def __repr__(self):
        return f"<Download(id={self.id}, user_id={self.user_id}, youtube_id='{self.youtube_id}', type='{self.download_type}')>"

    def to_dict(self) -> dict:
        """Convert download object to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'youtube_id': self.youtube_id,
            'download_type': self.download_type,
            'download_format': self.download_format,
            'quality': self.quality,
            'file_size': self.file_size,
            'processing_time': self.processing_time,
            'status': self.status,
            'language': self.language,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'download_url': self.download_url,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None
        }

# Legacy support - keeping old TranscriptDownload for backwards compatibility
class TranscriptDownload(Base):
    __tablename__ = "transcript_downloads"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False, index=True)
    youtube_id = Column(String(20), nullable=False, index=True)
    transcript_type = Column(String(20), nullable=False)  # 'clean', 'unclean', 'audio', 'video'
    created_at = Column(DateTime, nullable=False)
    
    # Enhanced tracking fields
    file_size = Column(Integer, nullable=True)
    processing_time = Column(Integer, nullable=True)
    download_method = Column(String(50), nullable=True)
    quality = Column(String(20), nullable=True)
    language = Column(String(10), default='en')

    def __repr__(self):
        return f"<TranscriptDownload(id={self.id}, user_id={self.user_id}, youtube_id='{self.youtube_id}', type='{self.transcript_type}')>"

# Usage analytics for admin dashboard
class UsageAnalytics(Base):
    __tablename__ = "usage_analytics"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=False, index=True)
    total_users = Column(Integer, default=0)
    active_users = Column(Integer, default=0)
    new_registrations = Column(Integer, default=0)
    total_downloads = Column(Integer, default=0)
    transcript_downloads = Column(Integer, default=0)
    audio_downloads = Column(Integer, default=0)
    video_downloads = Column(Integer, default=0)
    subscription_upgrades = Column(Integer, default=0)
    subscription_cancellations = Column(Integer, default=0)
    revenue = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<UsageAnalytics(date={self.date.strftime('%Y-%m-%d')}, total_users={self.total_users}, total_downloads={self.total_downloads})>"

# Database table creation utility
def create_tables(engine):
    """Create all database tables"""
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ All database tables created successfully")
        return True
    except Exception as e:
        print(f"❌ Error creating tables: {str(e)}")
        return False

# Database migration utilities
def add_audio_video_columns(engine):
    """Add audio/video columns to existing users table (for migrations)"""
    try:
        from sqlalchemy import text
        
        with engine.connect() as connection:
            # Check if columns already exist
            result = connection.execute(text("PRAGMA table_info(users)"))
            columns = [row[1] for row in result.fetchall()]
            
            if 'usage_audio_downloads' not in columns:
                connection.execute(text("ALTER TABLE users ADD COLUMN usage_audio_downloads INTEGER DEFAULT 0"))
                print("✅ Added usage_audio_downloads column")
            
            if 'usage_video_downloads' not in columns:
                connection.execute(text("ALTER TABLE users ADD COLUMN usage_video_downloads INTEGER DEFAULT 0"))
                print("✅ Added usage_video_downloads column")
                
            connection.commit()
        
        return True
    except Exception as e:
        print(f"❌ Error adding audio/video columns: {str(e)}")
        return False

#===========================================================
# from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Float
# from sqlalchemy.ext.declarative import declarative_base
# from datetime import datetime
# import bcrypt

# Base = declarative_base()

# class User(Base):
#     __tablename__ = "users"

#     # Primary fields
#     id = Column(Integer, primary_key=True, index=True)
#     username = Column(String(150), unique=True, index=True, nullable=False)
#     email = Column(String(255), unique=True, index=True, nullable=False)
#     full_name = Column(String(255), nullable=True)
#     hashed_password = Column(String(255), nullable=False)

#     # Status and timestamps
#     is_active = Column(Boolean, default=True)
#     is_verified = Column(Boolean, default=False)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
#     last_login = Column(DateTime, nullable=True)

#     # Subscription tracking
#     subscription_tier = Column(String(20), default='free', nullable=False)
#     subscription_status = Column(String(20), default='inactive', nullable=False)
#     subscription_id = Column(String(255), nullable=True)
#     subscription_current_period_end = Column(DateTime, nullable=True)
#     stripe_customer_id = Column(String(255), nullable=True)

#     # Usage counters
#     usage_clean_transcripts = Column(Integer, default=0, nullable=False)
#     usage_unclean_transcripts = Column(Integer, default=0, nullable=False)
#     usage_audio_downloads = Column(Integer, default=0, nullable=False)
#     usage_video_downloads = Column(Integer, default=0, nullable=False)
#     usage_reset_date = Column(DateTime, default=datetime.utcnow, nullable=False)

#     # Preferences
#     timezone = Column(String(50), default='UTC')
#     language = Column(String(10), default='en')
#     notification_preferences = Column(Text, nullable=True)  # JSON string

#     def __repr__(self):
#         return f"<User(id={self.id}, username='{self.username}', email='{self.email}', tier='{self.subscription_tier}')>"

#     def set_password(self, password: str):
#         salt = bcrypt.gensalt()
#         self.hashed_password = bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

#     def verify_password(self, password: str) -> bool:
#         return bcrypt.checkpw(password.encode('utf-8'), self.hashed_password.encode('utf-8'))

#     def is_subscription_active(self) -> bool:
#         if self.subscription_tier == 'free':
#             return True
#         return (
#             self.subscription_status in ['active', 'trialing'] and
#             self.subscription_current_period_end and
#             self.subscription_current_period_end > datetime.utcnow()
#         )

#     def get_plan_limits(self) -> dict:
#         return {
#             'free':     {'clean_transcripts': 5, 'unclean_transcripts': 3, 'audio_downloads': 2, 'video_downloads': 1},
#             'pro':      {'clean_transcripts': 100, 'unclean_transcripts': 50, 'audio_downloads': 50, 'video_downloads': 20},
#             'premium':  {'clean_transcripts': float('inf'), 'unclean_transcripts': float('inf'),
#                          'audio_downloads': float('inf'), 'video_downloads': float('inf')}
#         }.get(self.subscription_tier, {})

#     def get_current_usage(self) -> dict:
#         return {
#             'clean_transcripts': self.usage_clean_transcripts,
#             'unclean_transcripts': self.usage_unclean_transcripts,
#             'audio_downloads': self.usage_audio_downloads,
#             'video_downloads': self.usage_video_downloads
#         }

#     def can_perform_action(self, action_type: str) -> bool:
#         if self.usage_reset_date.month != datetime.utcnow().month:
#             self.reset_monthly_usage()
#         limits = self.get_plan_limits()
#         current_usage = getattr(self, f'usage_{action_type}', 0)
#         limit = limits.get(action_type, 0)
#         return limit == float('inf') or current_usage < limit

#     def reset_monthly_usage(self):
#         """Reset monthly usage counters and update reset date"""
#         self.usage_clean_transcripts = 0
#         self.usage_unclean_transcripts = 0
#         self.usage_audio_downloads = 0
#         self.usage_video_downloads = 0
#         self.usage_reset_date = datetime.utcnow()

#     def increment_usage(self, action_type: str):
#         """Increment usage counter for a specific action type"""
#         if hasattr(self, f'usage_{action_type}'):
#             current = getattr(self, f'usage_{action_type}', 0)
#             setattr(self, f'usage_{action_type}', current + 1)

#     def to_dict(self) -> dict:
#         return {
#             'id': self.id,
#             'username': self.username,
#             'email': self.email,
#             'full_name': self.full_name,
#             'is_active': self.is_active,
#             'is_verified': self.is_verified,
#             'subscription_tier': self.subscription_tier,
#             'subscription_status': self.subscription_status,
#             'created_at': self.created_at.isoformat() if self.created_at else None,
#             'last_login': self.last_login.isoformat() if self.last_login else None,
#         }

# # Optional: Store historical subscription events
# class SubscriptionHistory(Base):
#     __tablename__ = "subscription_history"

#     id = Column(Integer, primary_key=True, index=True)
#     user_id = Column(Integer, nullable=False)
#     action = Column(String(50), nullable=False)
#     from_tier = Column(String(20), nullable=True)
#     to_tier = Column(String(20), nullable=True)
#     amount = Column(Float, nullable=True)
#     stripe_subscription_id = Column(String(255), nullable=True)
#     stripe_payment_intent_id = Column(String(255), nullable=True)
#     created_at = Column(DateTime, default=datetime.utcnow)
#     history_metadata = Column(Text, nullable=True)

#     def __repr__(self):
#         return f"<SubscriptionHistory(user_id={self.user_id}, action='{self.action}', from='{self.from_tier}', to='{self.to_tier}')>"

# # Database table creation utility
# def create_tables(engine):
#     Base.metadata.create_all(bind=engine)

