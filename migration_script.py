# migration_script.py - Run this to add Stripe fields to existing database

import sqlite3
from datetime import datetime

def migrate_database():
    """
    Add new columns to existing database for Stripe integration
    Run this script once to update your existing database
    """
    try:
        # Connect to your existing database
        conn = sqlite3.connect('youtube_transcript.db')
        cursor = conn.cursor()
        
        print("🔄 Starting database migration...")
        
        # Add new columns to users table
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN stripe_customer_id TEXT')
            print("✅ Added stripe_customer_id to users table")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("📝 stripe_customer_id column already exists")
            else:
                print(f"❌ Error adding stripe_customer_id: {e}")
        
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN stripe_subscription_id TEXT')
            print("✅ Added stripe_subscription_id to users table")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("📝 stripe_subscription_id column already exists")
            else:
                print(f"❌ Error adding stripe_subscription_id: {e}")
                
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN full_name TEXT')
            print("✅ Added full_name to users table")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("📝 full_name column already exists")
            else:
                print(f"❌ Error adding full_name: {e}")
        
        try:
            cursor.execute('ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT 1')
            print("✅ Added is_active to users table")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("📝 is_active column already exists")
            else:
                print(f"❌ Error adding is_active: {e}")
        
        # Add new columns to subscriptions table
        try:
            cursor.execute('ALTER TABLE subscriptions ADD COLUMN stripe_subscription_id TEXT')
            print("✅ Added stripe_subscription_id to subscriptions table")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("📝 stripe_subscription_id column already exists in subscriptions")
            else:
                print(f"❌ Error adding stripe_subscription_id to subscriptions: {e}")
        
        try:
            cursor.execute('ALTER TABLE subscriptions ADD COLUMN stripe_price_id TEXT')
            print("✅ Added stripe_price_id to subscriptions table")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("📝 stripe_price_id column already exists")
            else:
                print(f"❌ Error adding stripe_price_id: {e}")
                
        try:
            cursor.execute('ALTER TABLE subscriptions ADD COLUMN status TEXT DEFAULT "active"')
            print("✅ Added status to subscriptions table")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("📝 status column already exists")
            else:
                print(f"❌ Error adding status: {e}")
        
        # Create payment history table
        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS payment_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    stripe_payment_intent_id TEXT NOT NULL,
                    stripe_customer_id TEXT NOT NULL,
                    amount INTEGER NOT NULL,
                    currency TEXT DEFAULT 'usd',
                    status TEXT NOT NULL,
                    subscription_tier TEXT NOT NULL,
                    created_at DATETIME NOT NULL,
                    metadata TEXT
                )
            ''')
            print("✅ Created payment_history table")
        except sqlite3.Error as e:
            print(f"❌ Error creating payment_history table: {e}")
        
        # Add enhanced columns to transcript_downloads table
        try:
            cursor.execute('ALTER TABLE transcript_downloads ADD COLUMN file_size INTEGER')
            print("✅ Added file_size to transcript_downloads table")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("📝 file_size column already exists")
            else:
                print(f"❌ Error adding file_size: {e}")
                
        try:
            cursor.execute('ALTER TABLE transcript_downloads ADD COLUMN download_method TEXT')
            print("✅ Added download_method to transcript_downloads table")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                print("📝 download_method column already exists")
            else:
                print(f"❌ Error adding download_method: {e}")
        
        # Commit all changes
        conn.commit()
        print("🎉 Database migration completed successfully!")
        
        # Show updated table structure
        cursor.execute("PRAGMA table_info(users)")
        users_columns = cursor.fetchall()
        print("\n📋 Updated users table structure:")
        for column in users_columns:
            print(f"   - {column[1]} ({column[2]})")
            
        cursor.execute("PRAGMA table_info(subscriptions)")
        subscriptions_columns = cursor.fetchall()
        print("\n📋 Updated subscriptions table structure:")
        for column in subscriptions_columns:
            print(f"   - {column[1]} ({column[2]})")
        
    except Exception as e:
        print(f"❌ Migration failed: {str(e)}")
    finally:
        conn.close()
        print("\n🔒 Database connection closed")

if __name__ == "__main__":
    migrate_database()