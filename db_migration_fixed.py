"""
Database Migration Script - FIXED VERSION
=========================================

This script fixes database schema issues by adding missing columns
to your EXISTING database (youtube_trans_downloader.db).

Run this script to fix the database schema errors.
"""

import sqlite3
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("db_migration")

def migrate_database(db_path="./youtube_trans_downloader.db"):  # ← FIXED: Use your existing DB
    """
    Migrate the database to add missing columns
    """
    try:
        # Check if the database exists
        if not os.path.exists(db_path):
            logger.error(f"❌ Database file not found: {db_path}")
            logger.info("Available database files:")
            for file in os.listdir("."):
                if file.endswith(".db"):
                    logger.info(f"  - {file}")
            return False
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        logger.info(f"🔄 Starting database migration for: {db_path}")
        
        # Check current tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        logger.info(f"📊 Existing tables: {tables}")
        
        # Add missing columns to transcript_downloads table
        migrations = [
            ("file_size", "INTEGER"),
            ("processing_time", "REAL"), 
            ("download_method", "VARCHAR(20)"),
            ("quality", "VARCHAR(20)"),
            ("language", "VARCHAR(10) DEFAULT 'en'"),
            ("file_format", "VARCHAR(10)"),
            ("download_url", "TEXT"),
            ("expires_at", "DATETIME"),
            ("status", "VARCHAR(20) DEFAULT 'completed'"),
            ("error_message", "TEXT"),
            ("video_title", "VARCHAR(200)"),
            ("video_duration", "INTEGER"),
            ("ip_address", "VARCHAR(45)"),
            ("user_agent", "VARCHAR(500)")
        ]
        
        logger.info("📊 Adding missing columns to transcript_downloads table...")
        
        for column_name, column_type in migrations:
            try:
                cursor.execute(f"ALTER TABLE transcript_downloads ADD COLUMN {column_name} {column_type}")
                logger.info(f"✅ Added column: {column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    logger.info(f"ℹ️ Column {column_name} already exists")
                else:
                    logger.warning(f"⚠️ Error adding column {column_name}: {e}")
        
        # Add missing columns to users table
        user_migrations = [
            ("usage_audio_downloads", "INTEGER DEFAULT 0"),
            ("usage_video_downloads", "INTEGER DEFAULT 0"),
            ("total_downloads", "INTEGER DEFAULT 0"),
            ("stripe_customer_id", "VARCHAR(100)"),
            ("stripe_subscription_id", "VARCHAR(100)"),
            ("subscription_status", "VARCHAR(20) DEFAULT 'inactive'"),
            ("subscription_start_date", "DATETIME"),
            ("subscription_end_date", "DATETIME"),
            ("is_active", "BOOLEAN DEFAULT 1"),
            ("last_login", "DATETIME")
        ]
        
        logger.info("👤 Adding missing columns to users table...")
        
        for column_name, column_type in user_migrations:
            try:
                cursor.execute(f"ALTER TABLE users ADD COLUMN {column_name} {column_type}")
                logger.info(f"✅ Added column: {column_name}")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e).lower():
                    logger.info(f"ℹ️ Column {column_name} already exists")
                else:
                    logger.warning(f"⚠️ Error adding column {column_name}: {e}")
        
        # Create new tables if they don't exist
        logger.info("🆕 Creating new tables...")
        
        # Payment records table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS payment_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                stripe_payment_intent_id VARCHAR(100) UNIQUE NOT NULL,
                amount REAL NOT NULL,
                currency VARCHAR(3) DEFAULT 'usd',
                plan_type VARCHAR(20) NOT NULL,
                status VARCHAR(20) NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                stripe_customer_id VARCHAR(100),
                payment_method VARCHAR(50)
            )
        """)
        logger.info("✅ Payment records table ready")
        
        # System analytics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATETIME DEFAULT CURRENT_TIMESTAMP,
                total_users INTEGER DEFAULT 0,
                new_users INTEGER DEFAULT 0,
                active_users INTEGER DEFAULT 0,
                transcript_downloads INTEGER DEFAULT 0,
                audio_downloads INTEGER DEFAULT 0,
                video_downloads INTEGER DEFAULT 0,
                free_users INTEGER DEFAULT 0,
                pro_users INTEGER DEFAULT 0,
                premium_users INTEGER DEFAULT 0,
                daily_revenue REAL DEFAULT 0.0
            )
        """)
        logger.info("✅ System analytics table ready")
        
        # Update existing data
        logger.info("📝 Updating existing data...")
        
        # Set default values for new columns
        cursor.execute("UPDATE transcript_downloads SET language = 'en' WHERE language IS NULL")
        cursor.execute("UPDATE transcript_downloads SET status = 'completed' WHERE status IS NULL")
        cursor.execute("UPDATE transcript_downloads SET download_method = 'api' WHERE download_method IS NULL AND transcript_type IN ('clean', 'unclean')")
        cursor.execute("UPDATE transcript_downloads SET download_method = 'ytdlp' WHERE download_method IS NULL AND transcript_type IN ('audio', 'video')")
        
        # Set default values for users
        cursor.execute("UPDATE users SET usage_audio_downloads = 0 WHERE usage_audio_downloads IS NULL")
        cursor.execute("UPDATE users SET usage_video_downloads = 0 WHERE usage_video_downloads IS NULL")
        cursor.execute("UPDATE users SET total_downloads = 0 WHERE total_downloads IS NULL")
        cursor.execute("UPDATE users SET subscription_status = 'inactive' WHERE subscription_status IS NULL")
        cursor.execute("UPDATE users SET is_active = 1 WHERE is_active IS NULL")
        
        # Commit changes
        conn.commit()
        
        # Verify the migration
        logger.info("🔍 Verifying migration...")
        
        # Check transcript_downloads table structure
        cursor.execute("PRAGMA table_info(transcript_downloads)")
        columns = [row[1] for row in cursor.fetchall()]
        
        required_columns = ['file_size', 'processing_time', 'download_method', 'quality', 'language', 
                          'file_format', 'download_url', 'expires_at', 'status']
        
        missing_columns = [col for col in required_columns if col not in columns]
        
        if missing_columns:
            logger.warning(f"⚠️ Still missing columns: {missing_columns}")
        else:
            logger.info("✅ All required columns present in transcript_downloads table")
        
        # Check users table structure
        cursor.execute("PRAGMA table_info(users)")
        user_columns = [row[1] for row in cursor.fetchall()]
        
        required_user_columns = ['usage_audio_downloads', 'usage_video_downloads', 'total_downloads']
        missing_user_columns = [col for col in required_user_columns if col not in user_columns]
        
        if missing_user_columns:
            logger.warning(f"⚠️ Still missing user columns: {missing_user_columns}")
        else:
            logger.info("✅ All required columns present in users table")
        
        # Show user data to confirm we're working with the right database
        cursor.execute("SELECT COUNT(*) FROM users")
        user_count = cursor.fetchone()[0]
        logger.info(f"👥 Found {user_count} users in database")
        
        if user_count > 0:
            cursor.execute("SELECT username FROM users LIMIT 3")
            users = cursor.fetchall()
            logger.info(f"📋 Sample users: {[user[0] for user in users]}")
        
        conn.close()
        logger.info("✅ Database migration completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

def check_database_files():
    """Check which database files exist and their sizes"""
    logger.info("🔍 Checking database files...")
    
    db_files = [
        "youtube_trans_downloader.db",           # Your original database
        "youtube_transcript_downloader.db"       # New database created by migration
    ]
    
    for db_file in db_files:
        if os.path.exists(db_file):
            size = os.path.getsize(db_file)
            logger.info(f"📄 {db_file}: {size:,} bytes")
            
            # Check user count
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM users")
                user_count = cursor.fetchone()[0]
                logger.info(f"   👥 Users: {user_count}")
                conn.close()
            except:
                logger.info(f"   ❌ Could not read user data")
        else:
            logger.info(f"❌ {db_file}: Not found")

def fix_database_naming():
    """Fix the database naming inconsistency"""
    logger.info("🔧 Fixing database naming inconsistency...")
    
    # Check which databases exist
    old_db = "youtube_trans_downloader.db"
    new_db = "youtube_transcript_downloader.db"
    
    old_exists = os.path.exists(old_db)
    new_exists = os.path.exists(new_db)
    
    logger.info(f"📄 {old_db} exists: {old_exists}")
    logger.info(f"📄 {new_db} exists: {new_exists}")
    
    if old_exists and new_exists:
        # Both exist - need to choose which one to use
        old_size = os.path.getsize(old_db)
        new_size = os.path.getsize(new_db)
        
        logger.info(f"📊 {old_db}: {old_size:,} bytes")
        logger.info(f"📊 {new_db}: {new_size:,} bytes")
        
        # Count users in each
        try:
            conn = sqlite3.connect(old_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users")
            old_users = cursor.fetchone()[0]
            conn.close()
        except:
            old_users = 0
            
        try:
            conn = sqlite3.connect(new_db)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM users")
            new_users = cursor.fetchone()[0]
            conn.close()
        except:
            new_users = 0
            
        logger.info(f"👥 {old_db}: {old_users} users")
        logger.info(f"👥 {new_db}: {new_users} users")
        
        # The one with more users is probably the right one
        if old_users > new_users:
            logger.info(f"✅ {old_db} has more user data - this is your main database")
            return old_db
        else:
            logger.info(f"✅ {new_db} has more user data - this is your main database")
            return new_db
    elif old_exists:
        logger.info(f"✅ Using {old_db} (only one that exists)")
        return old_db
    elif new_exists:
        logger.info(f"✅ Using {new_db} (only one that exists)")
        return new_db
    else:
        logger.error("❌ No database files found!")
        return None

if __name__ == "__main__":
    print("🔧 YouTube Transcript Downloader - Database Migration Tool (FIXED)")
    print("=" * 70)
    
    # Check current database files
    print("\n1. Checking database files...")
    check_database_files()
    
    # Fix naming inconsistency
    print("\n2. Identifying correct database...")
    correct_db = fix_database_naming()
    
    if not correct_db:
        print("❌ No database found to migrate!")
        exit(1)
    
    # Ask user what to do
    print(f"\n3. Ready to migrate: {correct_db}")
    print("\nChoose an option:")
    print("1. Migrate the identified database (recommended)")
    print("2. Exit")
    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    if choice == "1":
        print(f"\n4. Running migration on {correct_db}...")
        if migrate_database(correct_db):
            print("\n✅ Migration completed successfully!")
            print("Now update your models.py to use the same database name.")
        else:
            print("\n❌ Migration failed. Check the logs above.")
    else:
        print("👋 Goodbye!")
        
    print("\n" + "=" * 70)