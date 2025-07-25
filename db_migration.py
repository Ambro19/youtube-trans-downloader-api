"""
Database Migration Script
========================

This script fixes database schema issues by adding missing columns
and ensuring the database structure matches the updated models.

Run this script to fix the database schema errors.
"""

import sqlite3
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("db_migration")

def migrate_database(db_path="./youtube_transcript_downloader.db"):
    """
    Migrate the database to add missing columns
    """
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        logger.info(f"🔄 Starting database migration for: {db_path}")
        
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
        
        conn.close()
        logger.info("✅ Database migration completed successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        return False

def backup_database(db_path="./youtube_transcript_downloader.db"):
    """Create a backup of the database before migration"""
    try:
        if os.path.exists(db_path):
            backup_path = f"{db_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Copy database file
            with open(db_path, 'rb') as src, open(backup_path, 'wb') as dst:
                dst.write(src.read())
            
            logger.info(f"✅ Database backup created: {backup_path}")
            return backup_path
        else:
            logger.info("ℹ️ No existing database to backup")
            return None
            
    except Exception as e:
        logger.error(f"❌ Backup failed: {e}")
        return None

def reset_database_completely(db_path="./youtube_transcript_downloader.db"):
    """
    DANGEROUS: Completely reset the database (delete and recreate)
    This will delete all data!
    """
    try:
        logger.warning("⚠️ RESETTING DATABASE - ALL DATA WILL BE LOST!")
        
        # Create backup first
        backup_path = backup_database(db_path)
        if backup_path:
            logger.info(f"💾 Backup created at: {backup_path}")
        
        # Delete database file
        if os.path.exists(db_path):
            os.remove(db_path)
            logger.info("🗑️ Old database deleted")
        
        logger.info("✅ Database reset completed. The application will create a new database on next startup.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Reset failed: {e}")
        return False

def check_database_status(db_path="./youtube_transcript_downloader.db"):
    """Check the current database status"""
    try:
        if not os.path.exists(db_path):
            logger.info("ℹ️ No database file exists")
            return False
            
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if main tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        logger.info(f"📊 Existing tables: {tables}")
        
        # Check transcript_downloads structure
        if 'transcript_downloads' in tables:
            cursor.execute("PRAGMA table_info(transcript_downloads)")
            columns = [row[1] for row in cursor.fetchall()]
            logger.info(f"📋 transcript_downloads columns: {columns}")
            
            # Check for problematic columns
            required_columns = ['file_size', 'processing_time', 'download_method', 'quality', 
                              'language', 'file_format', 'download_url', 'expires_at']
            missing = [col for col in required_columns if col not in columns]
            
            if missing:
                logger.warning(f"⚠️ Missing columns: {missing}")
                return False
            else:
                logger.info("✅ All required columns present")
                return True
        else:
            logger.warning("⚠️ transcript_downloads table not found")
            return False
            
        conn.close()
        
    except Exception as e:
        logger.error(f"❌ Error checking database: {e}")
        return False

if __name__ == "__main__":
    print("🔧 YouTube Transcript Downloader - Database Migration Tool")
    print("=" * 60)
    
    # Check current status
    print("\n1. Checking current database status...")
    check_database_status()
    
    # Ask user what to do
    print("\nChoose an option:")
    print("1. Migrate existing database (recommended)")
    print("2. Reset database completely (⚠️ DELETES ALL DATA)")
    print("3. Just check status")
    print("4. Exit")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        print("\n2. Creating backup...")
        backup_database()
        
        print("\n3. Running migration...")
        if migrate_database():
            print("\n✅ Migration completed successfully!")
            print("You can now restart your application.")
        else:
            print("\n❌ Migration failed. Check the logs above.")
            
    elif choice == "2":
        confirm = input("\n⚠️ This will DELETE ALL DATA. Type 'DELETE' to confirm: ")
        if confirm == "DELETE":
            print("\n2. Resetting database...")
            if reset_database_completely():
                print("\n✅ Database reset completed!")
        else:
            print("❌ Reset cancelled.")
            
    elif choice == "3":
        print("\n2. Checking database status...")
        check_database_status()
        
    else:
        print("👋 Goodbye!")
        
    print("\n" + "=" * 60)