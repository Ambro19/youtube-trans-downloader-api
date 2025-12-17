# backend/models.py
# Modern SQLAlchemy base + helpful compound index + lightweight SQLite migrations
import os
from datetime import datetime
from sqlalchemy import ( # pyright: ignore[reportMissingImports]
    Column, Integer, String, DateTime, Boolean, Float, Text, ForeignKey,
    create_engine, Index
)  # pyright: ignore[reportMissingImports]
from sqlalchemy.orm import sessionmaker, relationship  # pyright: ignore[reportMissingImports]

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./youtube_trans_downloader.db")

engine = create_engine(
    DATABASE_URL,
    future=True,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)

from sqlalchemy import event # pyright: ignore[reportMissingImports]

if "sqlite" in DATABASE_URL:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, _):
        cur = dbapi_conn.cursor()
        cur.execute("PRAGMA foreign_keys=ON")
        cur.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

# New (v2-forward compatible with fallback):
try:
    from sqlalchemy.orm import DeclarativeBase # pyright: ignore[reportMissingImports]
    class Base(DeclarativeBase):
        pass
except Exception:  # SQLAlchemy < 2
    from sqlalchemy.ext.declarative import declarative_base # pyright: ignore[reportMissingImports]
    Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)

    # ✅ force-password-change flag (0/1 boolean in SQLite)
    must_change_password = Column(Boolean, nullable=False, default=False, server_default="0")

    # ✅ Keep tier (current) + legacy status (compat)
    subscription_tier = Column(String, default="free")
    subscription_status = Column(  # legacy column so inserts don’t fail
        String, nullable=False, default="inactive", server_default="inactive"
    )

    stripe_customer_id = Column(String, nullable=True, unique=True)

    # ------------------------------------------------------------------
    # ✅ NEW: Local Stripe sync + entitlement timestamps (for offline compare)
    # ------------------------------------------------------------------
    # Stripe status mirror (active/trialing/past_due/canceled/unpaid/etc.)
    stripe_subscription_status = Column(String, nullable=True)

    # Stripe "current_period_end" from Subscription (unix seconds)
    stripe_current_period_end = Column(Integer, nullable=True)

    # Same value but stored as DateTime UTC for easy comparisons
    stripe_current_period_end_dt = Column(DateTime, nullable=True)

    # Your app’s own entitlement end time: if now >= this => must be free
    # (Populate from Stripe current_period_end, or set to now for immediate downgrade)
    subscription_expires_at = Column(DateTime, nullable=True)

    # Track last successful sync time (useful for debugging/staleness checks)
    subscription_updated_at = Column(DateTime, nullable=True)

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------
    usage_clean_transcripts = Column(Integer, default=0)
    usage_unclean_transcripts = Column(Integer, default=0)
    usage_audio_downloads = Column(Integer, default=0)
    usage_video_downloads = Column(Integer, default=0)

    # Existing field (kept)
    usage_reset_date = Column(DateTime, default=datetime.utcnow)

    # New clearer alias (optional for future use; doesn’t break old code)
    usage_reset_at = Column(DateTime, nullable=True)

    subscriptions = relationship("Subscription", back_populates="user")
    transcript_downloads = relationship("TranscriptDownload", back_populates="user")


class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    tier = Column(String, nullable=False)
    status = Column(String, default="active")

    stripe_subscription_id = Column(String, nullable=True, unique=True)
    stripe_customer_id = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    cancelled_at = Column(DateTime, nullable=True)

    # NOTE: you already have this here — good.
    # You can treat Subscription.expires_at as your historical record,
    # and User.subscription_expires_at as your fast “current entitlement” source.
    expires_at = Column(DateTime, nullable=True)

    extra_data = Column(Text, nullable=True)

    user = relationship("User", back_populates="subscriptions")


class TranscriptDownload(Base):
    __tablename__ = "transcript_downloads"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)

    youtube_id = Column(String, nullable=False, index=True)
    transcript_type = Column(String, nullable=False)

    quality = Column(String, nullable=True)
    file_format = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)
    file_path = Column(String, nullable=True)

    processing_time = Column(Float, nullable=True)
    status = Column(String, default="completed")
    language = Column(String, default="en")

    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    error_message = Column(Text, nullable=True)

    user = relationship("User", back_populates="transcript_downloads")

    __table_args__ = (
        Index("ix_transcripts_user_created", "user_id", "created_at"),
    )


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def initialize_database():
    """Create tables and apply light SQLite compatibility patches."""
    try:
        db_path = DATABASE_URL.replace("sqlite:///", "").replace("./", "")
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            print(f"📁 Created database directory: {db_dir}")

        # Create tables normally (covers fresh installs)
        Base.metadata.create_all(bind=engine)

        # ✅ Lightweight SQLite migrations (add columns if missing)
        if DATABASE_URL.startswith("sqlite"):
            with engine.begin() as conn:
                cols = conn.exec_driver_sql("PRAGMA table_info(users)").fetchall()
                colnames = {c[1] for c in cols}

                # --- legacy column kept for backwards compat ---
                if "subscription_status" not in colnames:
                    conn.exec_driver_sql(
                        "ALTER TABLE users ADD COLUMN subscription_status TEXT NOT NULL DEFAULT 'inactive'"
                    )
                conn.exec_driver_sql(
                    "UPDATE users SET subscription_status='inactive' WHERE subscription_status IS NULL"
                )

                # --- force-password-change flag ---
                if "must_change_password" not in colnames:
                    conn.exec_driver_sql(
                        "ALTER TABLE users ADD COLUMN must_change_password INTEGER NOT NULL DEFAULT 0"
                    )
                conn.exec_driver_sql(
                    "UPDATE users SET must_change_password=0 WHERE must_change_password IS NULL"
                )

                # ----------------------------------------------------------
                # ✅ NEW: Stripe/local entitlement fields for offline compare
                # ----------------------------------------------------------
                if "stripe_subscription_status" not in colnames:
                    conn.exec_driver_sql(
                        "ALTER TABLE users ADD COLUMN stripe_subscription_status TEXT"
                    )

                if "stripe_current_period_end" not in colnames:
                    conn.exec_driver_sql(
                        "ALTER TABLE users ADD COLUMN stripe_current_period_end INTEGER"
                    )

                if "stripe_current_period_end_dt" not in colnames:
                    conn.exec_driver_sql(
                        "ALTER TABLE users ADD COLUMN stripe_current_period_end_dt DATETIME"
                    )

                if "subscription_expires_at" not in colnames:
                    conn.exec_driver_sql(
                        "ALTER TABLE users ADD COLUMN subscription_expires_at DATETIME"
                    )

                if "subscription_updated_at" not in colnames:
                    conn.exec_driver_sql(
                        "ALTER TABLE users ADD COLUMN subscription_updated_at DATETIME"
                    )

                if "usage_reset_at" not in colnames:
                    conn.exec_driver_sql(
                        "ALTER TABLE users ADD COLUMN usage_reset_at DATETIME"
                    )

                # Optional: backfill usage_reset_at from usage_reset_date if present
                # (Safe no-op if usage_reset_date is already being used)
                try:
                    conn.exec_driver_sql(
                        "UPDATE users SET usage_reset_at = usage_reset_date "
                        "WHERE usage_reset_at IS NULL AND usage_reset_date IS NOT NULL"
                    )
                except Exception:
                    pass

        print(f"✅ Database tables created successfully at: {db_path}")
        return True
    except Exception as e:
        print(f"❌ Error creating database tables: {e}")
        raise



